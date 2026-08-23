#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Fused MiniMax-M3 RoPE + clamp + cast (no RMSNorm; RMSNorm done by npu_rms_norm).

Inputs are already RMSNormed Q/K/V/index_Q/index_K tensors.
The kernel applies NeoX RoPE, optional fp8 clamp +-448, and dtype cast.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.triton.triton_utils import (
    extract_slice,
    get_element,
    get_vectorcore_num,
    insert_slice,
)

_A5_UB_RESERVE = 8 * 1024
_UB_KB_A2 = 192
_UB_KB_A5 = 256


@lru_cache(maxsize=1)
def _ub_size_bytes() -> int:
    kb: int | None = None
    try:
        from triton.backends.ascend.runtime import utils as npu_utils
        kb = int(npu_utils.ub_size_in_kbytes)
    except Exception:
        kb = None
    if kb is None:
        name = ""
        try:
            name = str(torch.npu.get_device_name(0) or "")
        except Exception:
            name = ""
        arch = name.lower()
        is_a5 = any(m in arch for m in ("910_95", "91095", "ascend950", "950pr", "950dt", "dav-c310"))
        kb = _UB_KB_A5 if is_a5 else _UB_KB_A2
    nbytes = kb * 1024
    if kb >= _UB_KB_A5:
        nbytes -= _A5_UB_RESERVE
    return nbytes


def _tokens_per_iter(elem_size: int, elems_per_token: int, *, cap: int = 2) -> int:
    n = int((_ub_size_bytes() // 4) / max(elem_size, 1)) // max(int(elems_per_token), 1)
    return max(1, min(cap, n))


@triton.jit
def qkv_index_rope_clamp_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    index_q_ptr,
    index_k_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    index_q_out_ptr,
    index_k_out_ptr,
    positions_gm_ptr,
    cos_sin_cache_gm_ptr,
    batch_size,
    q_hidden_size: tl.constexpr,
    kv_hidden_size: tl.constexpr,
    index_q_size: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IDX_HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HALF_CACHE: tl.constexpr,
    ATTN_HALF: tl.constexpr,
    IDX_HALF: tl.constexpr,
    num_vectorcore: tl.constexpr,
    batch_tile: tl.constexpr,
    q_head_num: tl.constexpr,
    kv_head_num: tl.constexpr,
    v_batch_tile: tl.constexpr,
    idx_batch_tile: tl.constexpr,
    index_q_head_num: tl.constexpr,
    ATTN_OUT_FP8: tl.constexpr,
    INDEX_OUT_FP8: tl.constexpr,
):
    row_pid = tl.program_id(0)

    batch_size_per_vec = tl.cdiv(batch_size, num_vectorcore)
    iter_num_per_vec = tl.cdiv(batch_size_per_vec, batch_tile)
    v_iter_num_per_vec = tl.cdiv(batch_size_per_vec, v_batch_tile)
    idx_iter_num_per_vec = tl.cdiv(batch_size_per_vec, idx_batch_tile)
    input_batch_offset = row_pid * batch_size_per_vec
    input_batch_offset_end = min(input_batch_offset + batch_size_per_vec, batch_size)

    sin_cos_range = tl.arange(0, ROPE_DIM)
    cos_sin_cache_offset = cos_sin_cache_gm_ptr + sin_cos_range

    # === main QK: RoPE + clamp + cast ===
    mblk_idx = tl.arange(0, batch_tile) + input_batch_offset
    pos_indices = input_batch_offset + tl.arange(0, batch_tile)
    output_q_nblk_idx = tl.arange(0, q_hidden_size)
    output_q_nmask = output_q_nblk_idx < q_hidden_size
    output_kv_nblk_idx = tl.arange(0, kv_hidden_size)
    output_kv_nmask = output_kv_nblk_idx < kv_hidden_size

    for iter in tl.range(iter_num_per_vec):
        pos_offset = iter * batch_tile
        mmask = (mblk_idx + pos_offset) < input_batch_offset_end
        x = tl.load(
            positions_gm_ptr + pos_indices + pos_offset,
            mask=(pos_indices + pos_offset) < input_batch_offset_end,
        )
        row64 = (mblk_idx + pos_offset).to(tl.int64)

        # gather cos/sin
        cache_rows = tl.zeros((batch_tile, ROPE_DIM), dtype=tl.float32)
        for i in tl.range(batch_tile):
            pos = get_element(x, (i,))
            cache_rows = insert_slice(
                cache_rows,
                tl.load(pos * ROPE_DIM + cos_sin_cache_offset[:, None]).reshape(1, ROPE_DIM).to(tl.float32),
                offsets=(i, 0), sizes=(1, ROPE_DIM), strides=(1, 1),
            )
        cache_rows = cache_rows.reshape(batch_tile, 1, ROPE_DIM)
        cos = extract_slice(cache_rows, offsets=(0, 0, 0), sizes=(batch_tile, 1, ATTN_HALF), strides=(1, 1, 1))
        sin = extract_slice(cache_rows, offsets=(0, 0, HALF_CACHE), sizes=(batch_tile, 1, ATTN_HALF), strides=(1, 1, 1))

        # Q: load (already normed) -> RoPE -> clamp -> store
        q_heads = tl.load(q_ptr + row64[:, None] * q_hidden_size + tl.arange(0, q_hidden_size)[None, :],
                          mask=mmask[:, None] & (tl.arange(0, q_hidden_size)[None, :] < q_hidden_size)).reshape(
            batch_tile, q_head_num, HEAD_DIM)
        q_x1 = extract_slice(q_heads, offsets=(0, 0, 0), sizes=(batch_tile, q_head_num, ATTN_HALF), strides=(1, 1, 1))
        q_x2 = extract_slice(q_heads, offsets=(0, 0, ATTN_HALF), sizes=(batch_tile, q_head_num, ATTN_HALF), strides=(1, 1, 1))
        q_heads = insert_slice(q_heads, q_x1 * cos - q_x2 * sin, offsets=(0, 0, 0), sizes=(batch_tile, q_head_num, ATTN_HALF), strides=(1, 1, 1))
        q_heads = insert_slice(q_heads, q_x2 * cos + q_x1 * sin, offsets=(0, 0, ATTN_HALF), sizes=(batch_tile, q_head_num, ATTN_HALF), strides=(1, 1, 1))
        if ATTN_OUT_FP8:
            q_heads = tl.minimum(tl.maximum(q_heads, -448.0), 448.0)
        q_output_idx = output_q_nblk_idx[None, :] + row64[:, None] * q_hidden_size
        q_store_mask = (mmask[:, None]) & (output_q_nmask[None, :])
        tl.store(q_out_ptr + q_output_idx, q_heads.reshape(batch_tile, q_hidden_size).to(q_out_ptr.dtype.element_ty), mask=q_store_mask)

        # K: load (already normed) -> RoPE -> clamp -> store
        k_heads = tl.load(k_ptr + row64[:, None] * kv_hidden_size + tl.arange(0, kv_hidden_size)[None, :],
                          mask=mmask[:, None] & (tl.arange(0, kv_hidden_size)[None, :] < kv_hidden_size)).reshape(
            batch_tile, kv_head_num, HEAD_DIM)
        k_x1 = extract_slice(k_heads, offsets=(0, 0, 0), sizes=(batch_tile, kv_head_num, ATTN_HALF), strides=(1, 1, 1))
        k_x2 = extract_slice(k_heads, offsets=(0, 0, ATTN_HALF), sizes=(batch_tile, kv_head_num, ATTN_HALF), strides=(1, 1, 1))
        k_heads = insert_slice(k_heads, k_x1 * cos - k_x2 * sin, offsets=(0, 0, 0), sizes=(batch_tile, kv_head_num, ATTN_HALF), strides=(1, 1, 1))
        k_heads = insert_slice(k_heads, k_x2 * cos + k_x1 * sin, offsets=(0, 0, ATTN_HALF), sizes=(batch_tile, kv_head_num, ATTN_HALF), strides=(1, 1, 1))
        if ATTN_OUT_FP8:
            k_heads = tl.minimum(tl.maximum(k_heads, -448.0), 448.0)
        kv_output_idx = output_kv_nblk_idx[None, :] + row64[:, None] * kv_hidden_size
        k_store_mask = (mmask[:, None]) & (output_kv_nmask[None, :])
        tl.store(k_out_ptr + kv_output_idx, k_heads.reshape(batch_tile, kv_hidden_size).to(k_out_ptr.dtype.element_ty), mask=k_store_mask)

    # === V: clamp + cast (no RoPE) ===
    mblk_idx = tl.arange(0, v_batch_tile) + input_batch_offset
    out_nblk_idx = tl.arange(0, kv_hidden_size)
    out_nmask = out_nblk_idx < kv_hidden_size
    for _ in tl.range(v_iter_num_per_vec):
        mmask = mblk_idx < input_batch_offset_end
        row64 = mblk_idx.to(tl.int64)
        values = tl.load(v_ptr + row64[:, None] * kv_hidden_size + out_nblk_idx[None, :], mask=mmask[:, None] & out_nmask[None, :])
        if ATTN_OUT_FP8:
            values = tl.minimum(tl.maximum(values.to(tl.float32), -448.0), 448.0)
        out_idx = row64[:, None] * kv_hidden_size + out_nblk_idx[None, :]
        out_mask = (mmask[:, None]) & (out_nmask[None, :])
        tl.store(v_out_ptr + out_idx, values.to(v_out_ptr.dtype.element_ty), mask=out_mask)
        mblk_idx += v_batch_tile

    # === indexer Q/K: RoPE + clamp + cast ===
    idx_mblk = tl.arange(0, idx_batch_tile) + input_batch_offset
    idx_pos = input_batch_offset + tl.arange(0, idx_batch_tile)
    out_iq_nblk = tl.arange(0, index_q_size)
    out_iq_nmask = out_iq_nblk < index_q_size
    out_ik_nblk = tl.arange(0, IDX_HEAD_DIM)
    out_ik_nmask = out_ik_nblk < IDX_HEAD_DIM
    index_qk_head_num = index_q_head_num + 1

    for iter in tl.range(idx_iter_num_per_vec):
        pos_offset = iter * idx_batch_tile
        mmask = (idx_mblk + pos_offset) < input_batch_offset_end
        x = tl.load(positions_gm_ptr + idx_pos + pos_offset, mask=(idx_pos + pos_offset) < input_batch_offset_end)
        row64 = (idx_mblk + pos_offset).to(tl.int64)

        cache_rows = tl.zeros((idx_batch_tile, ROPE_DIM), dtype=tl.float32)
        for i in tl.range(idx_batch_tile):
            pos = get_element(x, (i,))
            cache_rows = insert_slice(cache_rows, tl.load(pos * ROPE_DIM + cos_sin_cache_offset[:, None]).reshape(1, ROPE_DIM).to(tl.float32),
                                      offsets=(i, 0), sizes=(1, ROPE_DIM), strides=(1, 1))
        cache_rows = cache_rows.reshape(idx_batch_tile, 1, ROPE_DIM)
        cos = extract_slice(cache_rows, offsets=(0, 0, 0), sizes=(idx_batch_tile, 1, IDX_HALF), strides=(1, 1, 1))
        sin = extract_slice(cache_rows, offsets=(0, 0, HALF_CACHE), sizes=(idx_batch_tile, 1, IDX_HALF), strides=(1, 1, 1))

        # index_q
        iq_heads = tl.load(index_q_ptr + row64[:, None] * index_q_size + out_iq_nblk[None, :],
                           mask=mmask[:, None] & out_iq_nmask[None, :]).reshape(idx_batch_tile, index_q_head_num, IDX_HEAD_DIM)
        iq_x1 = extract_slice(iq_heads, offsets=(0, 0, 0), sizes=(idx_batch_tile, index_q_head_num, IDX_HALF), strides=(1, 1, 1))
        iq_x2 = extract_slice(iq_heads, offsets=(0, 0, IDX_HALF), sizes=(idx_batch_tile, index_q_head_num, IDX_HALF), strides=(1, 1, 1))
        iq_heads = insert_slice(iq_heads, iq_x1 * cos - iq_x2 * sin, offsets=(0, 0, 0), sizes=(idx_batch_tile, index_q_head_num, IDX_HALF), strides=(1, 1, 1))
        iq_heads = insert_slice(iq_heads, iq_x2 * cos + iq_x1 * sin, offsets=(0, 0, IDX_HALF), sizes=(idx_batch_tile, index_q_head_num, IDX_HALF), strides=(1, 1, 1))
        if INDEX_OUT_FP8:
            iq_heads = tl.minimum(tl.maximum(iq_heads, -448.0), 448.0)
        iq_idx = out_iq_nblk[None, :] + row64[:, None] * index_q_size
        iq_mask = (mmask[:, None]) & (out_iq_nmask[None, :])
        tl.store(index_q_out_ptr + iq_idx, iq_heads.reshape(idx_batch_tile, index_q_size).to(index_q_out_ptr.dtype.element_ty), mask=iq_mask)

        # index_k (shared single head)
        ik_heads = tl.load(index_k_ptr + row64[:, None] * IDX_HEAD_DIM + out_ik_nblk[None, :],
                           mask=mmask[:, None] & out_ik_nmask[None, :]).reshape(idx_batch_tile, 1, IDX_HEAD_DIM)
        ik_x1 = extract_slice(ik_heads, offsets=(0, 0, 0), sizes=(idx_batch_tile, 1, IDX_HALF), strides=(1, 1, 1))
        ik_x2 = extract_slice(ik_heads, offsets=(0, 0, IDX_HALF), sizes=(idx_batch_tile, 1, IDX_HALF), strides=(1, 1, 1))
        ik_heads = insert_slice(ik_heads, ik_x1 * cos - ik_x2 * sin, offsets=(0, 0, 0), sizes=(idx_batch_tile, 1, IDX_HALF), strides=(1, 1, 1))
        ik_heads = insert_slice(ik_heads, ik_x2 * cos + ik_x1 * sin, offsets=(0, 0, IDX_HALF), sizes=(idx_batch_tile, 1, IDX_HALF), strides=(1, 1, 1))
        if INDEX_OUT_FP8:
            ik_heads = tl.minimum(tl.maximum(ik_heads, -448.0), 448.0)
        ik_idx = out_ik_nblk[None, :] + row64[:, None] * IDX_HEAD_DIM
        ik_mask = (mmask[:, None]) & (out_ik_nmask[None, :])
        tl.store(index_k_out_ptr + ik_idx, ik_heads.reshape(idx_batch_tile, IDX_HEAD_DIM).to(index_k_out_ptr.dtype.element_ty), mask=ik_mask)


def qkv_index_rope_clamp_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    index_q_size: int,
    head_dim: int,
    idx_head_dim: int,
    attn_out_fp8: bool = False,
    indexer_out_fp8: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    index_q = index_q.contiguous()
    index_k = index_k.contiguous()
    cos_sin_cache = cos_sin_cache.contiguous()
    positions = positions.contiguous()

    num_vectorcore = get_vectorcore_num()
    batch_size = q.shape[0]
    cache_dim = int(cos_sin_cache.shape[-1])
    attn_rope_dim = min(cache_dim, int(head_dim))
    idx_rope_dim = min(cache_dim, int(idx_head_dim))
    attn_dtype = torch.float8_e4m3fn if attn_out_fp8 else q.dtype
    index_dtype = torch.float8_e4m3fn if indexer_out_fp8 else index_q.dtype

    q_out = torch.empty(batch_size, q_hidden_size, device=q.device, dtype=attn_dtype)
    k_out = torch.empty(batch_size, kv_hidden_size, device=q.device, dtype=attn_dtype)
    v_out = torch.empty(batch_size, kv_hidden_size, device=q.device, dtype=attn_dtype)
    index_q_out = torch.empty(batch_size, index_q_size, device=q.device, dtype=index_dtype)
    index_k_out = torch.empty(batch_size, idx_head_dim, device=q.device, dtype=index_dtype)

    q_head_num = q_hidden_size // head_dim
    kv_head_num = kv_hidden_size // head_dim
    index_q_head_num = index_q_size // idx_head_dim

    elem = q.element_size()
    qk_factor = 5 * q_hidden_size + 3 * kv_hidden_size + cache_dim * 4 + q_head_num * attn_rope_dim
    idx_factor = 5 * index_q_size + 3 * idx_head_dim + cache_dim * 4 + index_q_head_num * idx_rope_dim
    batch_tile = _tokens_per_iter(elem, qk_factor)
    idx_batch_tile = _tokens_per_iter(elem, idx_factor)
    v_batch_tile = _tokens_per_iter(elem, kv_hidden_size + 1, cap=4)

    grid = (num_vectorcore,)
    qkv_index_rope_clamp_kernel[grid](
        q, k, v, index_q, index_k,
        q_out, k_out, v_out, index_q_out, index_k_out,
        positions, cos_sin_cache,
        batch_size,
        q_hidden_size, kv_hidden_size, index_q_size,
        head_dim, idx_head_dim,
        cache_dim, cache_dim // 2,
        attn_rope_dim // 2, idx_rope_dim // 2,
        num_vectorcore,
        int(batch_tile),
        q_head_num, kv_head_num,
        int(v_batch_tile),
        int(idx_batch_tile),
        index_q_head_num,
        attn_out_fp8, indexer_out_fp8,
    )
    return q_out, k_out, v_out, index_q_out, index_k_out


def qkv_index_rope_clamp_impl_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    index_q_size: int,
    head_dim: int,
    idx_head_dim: int,
    attn_out_fp8: bool = False,
    indexer_out_fp8: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = q.shape[0]
    attn_dtype = torch.float8_e4m3fn if attn_out_fp8 else q.dtype
    index_dtype = torch.float8_e4m3fn if indexer_out_fp8 else index_q.dtype
    return (
        torch.empty(batch_size, int(q_hidden_size), device=q.device, dtype=attn_dtype),
        torch.empty(batch_size, int(kv_hidden_size), device=q.device, dtype=attn_dtype),
        torch.empty(batch_size, int(kv_hidden_size), device=q.device, dtype=attn_dtype),
        torch.empty(batch_size, int(index_q_size), device=q.device, dtype=index_dtype),
        torch.empty(batch_size, int(idx_head_dim), device=q.device, dtype=index_dtype),
    )


direct_register_custom_op(
    op_name="qkv_index_rope_clamp",
    op_func=qkv_index_rope_clamp_impl,
    fake_impl=qkv_index_rope_clamp_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
