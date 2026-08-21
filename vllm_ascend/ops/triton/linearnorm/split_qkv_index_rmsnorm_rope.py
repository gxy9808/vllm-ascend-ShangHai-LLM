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
"""Fused MiniMax-M3 sparse prepare: split + Gemma RMSNorm + Neox RoPE.

Concat layout ``[q | k | v | index_q | index_k]``.
``out_fp8=True`` 时 indexer 先 clamp ±448 再写成 e4m3；main Q/K/V 保持输入 dtype。
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

# * 910_95 / 950 物理 UB=256KB，编译器预留 8KB。
_A5_UB_RESERVE = 8 * 1024
_UB_KB_A2 = 192
_UB_KB_A5 = 256


@lru_cache(maxsize=1)
def _ub_size_bytes() -> int:
    """按当前 NPU 卡型取可用 UB 字节数。

    A2 / 910B / 910_93：192KB；A5 / 910_95 / 950：256KB - 8KB 编译器预留。
    优先读 Triton 运行时 ``ub_size_in_kbytes``（与编译器同源）。
    """
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
        is_a5 = any(
            m in arch
            for m in ("910_95", "91095", "ascend950", "950pr", "950dt", "dav-c310")
        )
        kb = _UB_KB_A5 if is_a5 else _UB_KB_A2
    nbytes = kb * 1024
    if kb >= _UB_KB_A5:
        nbytes -= _A5_UB_RESERVE
    return nbytes


def _tokens_per_iter(elem_size: int, elems_per_token: int, *, cap: int = 2) -> int:
    """三路 loop 的 UB 会被同时计入，再叠加 multibuffer，按 1/4 UB 估 tile。"""
    n = int((_ub_size_bytes() // 4) / max(elem_size, 1)) // max(int(elems_per_token), 1)
    return max(1, min(cap, n))


@triton.jit
def split_qkv_index_rmsnorm_rope_kernel(
    input_gm_ptr,  # * concat QKV 输入 [q|k|v|index_q|index_k]
    q_gm_ptr,  # * main Q 输出
    k_gm_ptr,  # * main K 输出
    v_gm_ptr,  # * main V 输出（只切分，不 norm/RoPE）
    index_q_gm_ptr,  # * indexer Q 输出
    index_k_gm_ptr,  # * indexer K 输出（共享单头）
    q_weight_ptr,  # * main Q Gemma RMSNorm 的 1+w
    q_bias_ptr,  # * main Q 可选 bias（BIAS=False 时不读）
    k_weight_ptr,  # * main K Gemma RMSNorm 的 1+w
    k_bias_ptr,  # * main K 可选 bias（BIAS=False 时不读）
    index_q_weight_ptr,  # * indexer Q Gemma RMSNorm 的 1+w
    index_k_weight_ptr,  # * indexer K Gemma RMSNorm 的 1+w
    positions_gm_ptr,  # * token 位置，用来索引 cos_sin_cache
    cos_sin_cache_gm_ptr,  # * RoPE cache [max_pos, ROPE_DIM]=concat(cos, sin)
    batch_size,  # * token 数
    q_hidden_size: tl.constexpr,  # * q_size = q_head_num * HEAD_DIM
    kv_hidden_size: tl.constexpr,  # * kv_size = kv_head_num * HEAD_DIM
    index_q_size: tl.constexpr,  # * index_q_head_num * IDX_HEAD_DIM
    total_hidden_size: tl.constexpr,  # * concat 最后一维总宽
    index_offset: tl.constexpr,  # * index_q 在 concat 中的起点 = q+2*kv
    index_qk_hidden: tl.constexpr,  # * index_q_size + IDX_HEAD_DIM
    eps: tl.constexpr,  # * RMSNorm epsilon
    BIAS: tl.constexpr,  # * 是否给 main Q/K 加 bias
    HEAD_DIM: tl.constexpr,  # * main 头维；RoPE 按此 view
    IDX_HEAD_DIM: tl.constexpr,  # * indexer 头维；RMSNorm 按此归约
    ROPE_DIM: tl.constexpr,  # * cache 最后一维（cos∥sin 拼接长度）
    HALF_CACHE: tl.constexpr,  # * ROPE_DIM/2，sin 从这里开始
    ATTN_HALF: tl.constexpr,  # * main partial RoPE 半宽 = attn_rope_dim/2
    IDX_HALF: tl.constexpr,  # * indexer partial RoPE 半宽 = idx_rope_dim/2
    num_vectorcore: tl.constexpr,  # * Vector Core 数（grid）
    batch_size_per_iter_per_vec: tl.constexpr,  # * main QK 循环每轮 token tile
    qk_head_nums_per_iter_per_vec: tl.constexpr,  # * tile * (q_head+kv_head)，reshape 用
    q_head_num: tl.constexpr,  # * main Q 头数
    kv_head_num: tl.constexpr,  # * main KV 头数
    qk_head_num_sum: tl.constexpr,  # * q_head_num + kv_head_num
    v_batch_size_per_iter_per_vec: tl.constexpr,  # * V 拷贝每轮 token tile
    idx_batch_size_per_iter_per_vec: tl.constexpr,  # * indexer 循环每轮 token tile
    idx_qk_heads_per_iter: tl.constexpr,  # * tile * index_qk_head_num，reshape 用
    index_q_head_num: tl.constexpr,  # * indexer Q 头数
    index_qk_head_num: tl.constexpr,  # * indexer Q 头数 + 1（共享 index_k）
    ATTN_OUT_FP8: tl.constexpr,  # * main Q/K/V clamp -> e4m3 (attn_out forced bf16 by empty_like)
    INDEX_OUT_FP8: tl.constexpr,  # * indexer 输出 clamp ±448 再存 e4m3
):
    row_pid = tl.program_id(0)

    # * Gemma 1+w（及可选 bias）驻核，三路 loop 共用
    q_weight_values = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
    k_weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
    index_q_weight_values = tl.load(index_q_weight_ptr + tl.arange(0, IDX_HEAD_DIM)).to(
        tl.float32
    )
    index_k_weight_values = tl.load(index_k_weight_ptr + tl.arange(0, IDX_HEAD_DIM)).to(
        tl.float32
    )
    if BIAS:
        q_bias_values = tl.load(q_bias_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
        k_bias_values = tl.load(k_bias_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
    # * 按 Vector Core 切 token；QK / V / indexer 各自一轮 tile
    batch_size_per_vec = tl.cdiv(batch_size, num_vectorcore)
    iter_num_per_vec = tl.cdiv(batch_size_per_vec, batch_size_per_iter_per_vec)
    v_iter_num_per_vec = tl.cdiv(batch_size_per_vec, v_batch_size_per_iter_per_vec)
    idx_iter_num_per_vec = tl.cdiv(batch_size_per_vec, idx_batch_size_per_iter_per_vec)
    input_batch_offset = row_pid * batch_size_per_vec
    input_batch_offset_end = min(input_batch_offset + batch_size_per_vec, batch_size)

    # * main QK 列范围：concat 前缀 [q|k]
    mblk_idx = tl.arange(0, batch_size_per_iter_per_vec) + input_batch_offset
    nblk_idx = tl.arange(0, q_hidden_size + kv_hidden_size)
    nmask = nblk_idx < total_hidden_size
    pos_indices = input_batch_offset + tl.arange(0, batch_size_per_iter_per_vec)
    output_q_nblk_idx = tl.arange(0, q_hidden_size)
    output_q_nmask = output_q_nblk_idx < q_hidden_size
    output_kv_nblk_idx = tl.arange(0, kv_hidden_size)
    output_kv_nmask = output_kv_nblk_idx < kv_hidden_size
    sin_cos_range = tl.arange(0, ROPE_DIM)
    cos_sin_cache_offset = cos_sin_cache_gm_ptr + sin_cos_range
    # * 1 main QK：load [q|k] → Gemma RMSNorm → NeoX RoPE
    for iter in tl.range(iter_num_per_vec):
        pos_offset = iter * batch_size_per_iter_per_vec
        mmask = (mblk_idx + pos_offset) < input_batch_offset_end
        x = tl.load(
            positions_gm_ptr + pos_indices + pos_offset,
            mask=(pos_indices + pos_offset) < input_batch_offset_end,
        )
        mask = (mmask[:, None]) & (nmask[None, :])
        # ! T * hidden 在 64×16k 会超过 int32；GM 偏移用 int64。
        row64 = (mblk_idx + pos_offset).to(tl.int64)
        idx = row64[:, None] * total_hidden_size + nblk_idx[None, :]
        values_tmp1 = tl.load(input_gm_ptr + idx, mask=mask).reshape(
            qk_head_nums_per_iter_per_vec, HEAD_DIM
        )

        cache_rows = tl.zeros(
            (batch_size_per_iter_per_vec, ROPE_DIM), dtype=tl.float32
        )
        # * 按 token position 收集 cos/sin 行（scalar 索引，无法向量 load）
        for i in tl.range(batch_size_per_iter_per_vec):
            pos = get_element(x, (i,))
            cache_rows = insert_slice(
                cache_rows,
                tl.load(pos * ROPE_DIM + cos_sin_cache_offset[:, None])
                .reshape(1, ROPE_DIM)
                .to(tl.float32),
                offsets=(i, 0),
                sizes=(1, ROPE_DIM),
                strides=(1, 1),
            )
        cache_rows = cache_rows.reshape(batch_size_per_iter_per_vec, 1, ROPE_DIM)
        # * cache=concat(cos,sin)；partial RoPE 只取前 ATTN_HALF
        cos = extract_slice(
            cache_rows,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, 1, ATTN_HALF),
            strides=(1, 1, 1),
        )
        sin = extract_slice(
            cache_rows,
            offsets=(0, 0, HALF_CACHE),
            sizes=(batch_size_per_iter_per_vec, 1, ATTN_HALF),
            strides=(1, 1, 1),
        )

        # * Gemma RMSNorm：按 HEAD_DIM 归约，Q/K 头拼在一起算 rstd
        # ! 用逐元素顺序累加替代 tl.sum，保证确定性（tl.sum 的硬件 reduction
        # ! 树在不同运行时线程调度可能不同，导致浮点累加顺序不确定）
        x32 = values_tmp1.to(tl.float32)
        sq = x32 * x32
        sum_sq = tl.zeros((qk_head_nums_per_iter_per_vec,), dtype=tl.float32)
        for d in tl.range(HEAD_DIM):
            sum_sq += sq[:, d]
        rstd = tl.rsqrt(sum_sq / HEAD_DIM + eps).reshape(
            qk_head_nums_per_iter_per_vec, 1
        )
        normalized_values = (x32 * rstd).reshape(
            batch_size_per_iter_per_vec, qk_head_num_sum, HEAD_DIM
        )

        # * Q：×(1+w) → NeoX [x1*cos-x2*sin | x2*cos+x1*sin]，尾维不转
        q_heads = extract_slice(
            normalized_values,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HEAD_DIM),
            strides=(1, 1, 1),
        )
        if BIAS:
            q_heads = q_heads * q_weight_values + q_bias_values
        else:
            q_heads = q_heads * q_weight_values

        q_x1 = extract_slice(
            q_heads,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, q_head_num, ATTN_HALF),
            strides=(1, 1, 1),
        )
        q_x2 = extract_slice(
            q_heads,
            offsets=(0, 0, ATTN_HALF),
            sizes=(batch_size_per_iter_per_vec, q_head_num, ATTN_HALF),
            strides=(1, 1, 1),
        )
        q_heads = insert_slice(
            q_heads,
            q_x1 * cos - q_x2 * sin,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, q_head_num, ATTN_HALF),
            strides=(1, 1, 1),
        )
        q_heads = insert_slice(
            q_heads,
            q_x2 * cos + q_x1 * sin,
            offsets=(0, 0, ATTN_HALF),
            sizes=(batch_size_per_iter_per_vec, q_head_num, ATTN_HALF),
            strides=(1, 1, 1),
        )
        # * FP8：RoPE 后 clamp ±448，再按输出 dtype store
        if ATTN_OUT_FP8:
            q_heads = tl.minimum(tl.maximum(q_heads, -448.0), 448.0)
        q_output_idx = output_q_nblk_idx[None, :] + row64[:, None] * q_hidden_size
        q_store_mask = (mmask[:, None]) & (output_q_nmask[None, :])
        tl.store(
            q_gm_ptr + q_output_idx,
            q_heads.reshape(batch_size_per_iter_per_vec, q_hidden_size).to(
                q_gm_ptr.dtype.element_ty
            ),
            mask=q_store_mask,
        )

        # * K：与 Q 同一套 norm 结果，从 q_head_num 起切
        k_heads = extract_slice(
            normalized_values,
            offsets=(0, q_head_num, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HEAD_DIM),
            strides=(1, 1, 1),
        )
        if BIAS:
            k_heads = k_heads * k_weight_values + k_bias_values
        else:
            k_heads = k_heads * k_weight_values

        k_x1 = extract_slice(
            k_heads,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, ATTN_HALF),
            strides=(1, 1, 1),
        )
        k_x2 = extract_slice(
            k_heads,
            offsets=(0, 0, ATTN_HALF),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, ATTN_HALF),
            strides=(1, 1, 1),
        )
        k_heads = insert_slice(
            k_heads,
            k_x1 * cos - k_x2 * sin,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, ATTN_HALF),
            strides=(1, 1, 1),
        )
        k_heads = insert_slice(
            k_heads,
            k_x2 * cos + k_x1 * sin,
            offsets=(0, 0, ATTN_HALF),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, ATTN_HALF),
            strides=(1, 1, 1),
        )
        if ATTN_OUT_FP8:
            k_heads = tl.minimum(tl.maximum(k_heads, -448.0), 448.0)
        kv_output_idx = output_kv_nblk_idx[None, :] + row64[:, None] * kv_hidden_size
        k_store_mask = (mmask[:, None]) & (output_kv_nmask[None, :])
        tl.store(
            k_gm_ptr + kv_output_idx,
            k_heads.reshape(batch_size_per_iter_per_vec, kv_hidden_size).to(
                k_gm_ptr.dtype.element_ty
            ),
            mask=k_store_mask,
        )

    # * V：concat 中段原样拷贝，不 norm / 不 RoPE
    mblk_idx = tl.arange(0, v_batch_size_per_iter_per_vec) + input_batch_offset
    nblk_idx = (q_hidden_size + kv_hidden_size) + tl.arange(0, kv_hidden_size)
    nmask = nblk_idx < total_hidden_size
    out_nblk_idx = tl.arange(0, kv_hidden_size)
    out_nmask = out_nblk_idx < kv_hidden_size
    for _ in tl.range(v_iter_num_per_vec):
        mmask = mblk_idx < input_batch_offset_end
        mask = (mmask[:, None]) & (nmask[None, :])
        row64 = mblk_idx.to(tl.int64)
        idx = row64[:, None] * total_hidden_size + nblk_idx[None, :]
        values = tl.load(input_gm_ptr + idx, mask=mask)
        # * V 无 RoPE，FP8 仍先 clamp 再按输出 dtype store
        if ATTN_OUT_FP8:
            values = tl.minimum(tl.maximum(values.to(tl.float32), -448.0), 448.0)
        out_idx = row64[:, None] * kv_hidden_size + out_nblk_idx[None, :]
        out_mask = (mmask[:, None]) & (out_nmask[None, :])
        tl.store(
            v_gm_ptr + out_idx,
            values.to(v_gm_ptr.dtype.element_ty),
            mask=out_mask,
        )
        mblk_idx += v_batch_size_per_iter_per_vec

    # * indexer：concat 尾部 [index_q | index_k]，index_k 为共享单头
    idx_mblk = tl.arange(0, idx_batch_size_per_iter_per_vec) + input_batch_offset
    idx_nblk = index_offset + tl.arange(0, index_qk_hidden)
    idx_nmask = idx_nblk < total_hidden_size
    idx_pos = input_batch_offset + tl.arange(0, idx_batch_size_per_iter_per_vec)
    out_iq_nblk = tl.arange(0, index_q_size)
    out_iq_nmask = out_iq_nblk < index_q_size
    out_ik_nblk = tl.arange(0, IDX_HEAD_DIM)
    out_ik_nmask = out_ik_nblk < IDX_HEAD_DIM

    # * indexer：load → Gemma RMSNorm(IDX_HEAD_DIM) → NeoX RoPE → 可选 FP8 clamp
    for iter in tl.range(idx_iter_num_per_vec):
        pos_offset = iter * idx_batch_size_per_iter_per_vec
        mmask = (idx_mblk + pos_offset) < input_batch_offset_end
        x = tl.load(
            positions_gm_ptr + idx_pos + pos_offset,
            mask=(idx_pos + pos_offset) < input_batch_offset_end,
        )
        mask = (mmask[:, None]) & (idx_nmask[None, :])
        row64 = (idx_mblk + pos_offset).to(tl.int64)
        idx = row64[:, None] * total_hidden_size + idx_nblk[None, :]
        values_idx = tl.load(input_gm_ptr + idx, mask=mask).reshape(
            idx_qk_heads_per_iter, IDX_HEAD_DIM
        )

        cache_rows = tl.zeros(
            (idx_batch_size_per_iter_per_vec, ROPE_DIM), dtype=tl.float32
        )
        # * 与 main 相同：按 position 收集 cache 行
        for i in tl.range(idx_batch_size_per_iter_per_vec):
            pos = get_element(x, (i,))
            cache_rows = insert_slice(
                cache_rows,
                tl.load(pos * ROPE_DIM + cos_sin_cache_offset[:, None])
                .reshape(1, ROPE_DIM)
                .to(tl.float32),
                offsets=(i, 0),
                sizes=(1, ROPE_DIM),
                strides=(1, 1),
            )
        cache_rows = cache_rows.reshape(idx_batch_size_per_iter_per_vec, 1, ROPE_DIM)
        # * indexer partial RoPE 半宽用 IDX_HALF；sin 仍从 HALF_CACHE 起
        cos = extract_slice(
            cache_rows,
            offsets=(0, 0, 0),
            sizes=(idx_batch_size_per_iter_per_vec, 1, IDX_HALF),
            strides=(1, 1, 1),
        )
        sin = extract_slice(
            cache_rows,
            offsets=(0, 0, HALF_CACHE),
            sizes=(idx_batch_size_per_iter_per_vec, 1, IDX_HALF),
            strides=(1, 1, 1),
        )

        # * Gemma RMSNorm：index_q 多头 + index_k 一头拼在一起按 IDX_HEAD_DIM 归约
        # ! 用逐元素顺序累加替代 tl.sum，保证确定性
        x32 = values_idx.to(tl.float32)
        sq = x32 * x32
        sum_sq = tl.zeros((idx_qk_heads_per_iter,), dtype=tl.float32)
        for d in tl.range(IDX_HEAD_DIM):
            sum_sq += sq[:, d]
        rstd = tl.rsqrt(sum_sq / IDX_HEAD_DIM + eps).reshape(
            idx_qk_heads_per_iter, 1
        )
        normalized_idx = (x32 * rstd).reshape(
            idx_batch_size_per_iter_per_vec, index_qk_head_num, IDX_HEAD_DIM
        )

        # * index_q：×(1+w) + NeoX RoPE
        iq_heads = extract_slice(
            normalized_idx,
            offsets=(0, 0, 0),
            sizes=(idx_batch_size_per_iter_per_vec, index_q_head_num, IDX_HEAD_DIM),
            strides=(1, 1, 1),
        )
        iq_heads = iq_heads * index_q_weight_values
        iq_x1 = extract_slice(
            iq_heads,
            offsets=(0, 0, 0),
            sizes=(idx_batch_size_per_iter_per_vec, index_q_head_num, IDX_HALF),
            strides=(1, 1, 1),
        )
        iq_x2 = extract_slice(
            iq_heads,
            offsets=(0, 0, IDX_HALF),
            sizes=(idx_batch_size_per_iter_per_vec, index_q_head_num, IDX_HALF),
            strides=(1, 1, 1),
        )
        iq_heads = insert_slice(
            iq_heads,
            iq_x1 * cos - iq_x2 * sin,
            offsets=(0, 0, 0),
            sizes=(idx_batch_size_per_iter_per_vec, index_q_head_num, IDX_HALF),
            strides=(1, 1, 1),
        )
        iq_heads = insert_slice(
            iq_heads,
            iq_x2 * cos + iq_x1 * sin,
            offsets=(0, 0, IDX_HALF),
            sizes=(idx_batch_size_per_iter_per_vec, index_q_head_num, IDX_HALF),
            strides=(1, 1, 1),
        )

        # * index_k：共享单头，RoPE 与 index_q 共用同一份 cos/sin
        ik_heads = extract_slice(
            normalized_idx,
            offsets=(0, index_q_head_num, 0),
            sizes=(idx_batch_size_per_iter_per_vec, 1, IDX_HEAD_DIM),
            strides=(1, 1, 1),
        )
        ik_heads = ik_heads * index_k_weight_values
        ik_x1 = extract_slice(
            ik_heads,
            offsets=(0, 0, 0),
            sizes=(idx_batch_size_per_iter_per_vec, 1, IDX_HALF),
            strides=(1, 1, 1),
        )
        ik_x2 = extract_slice(
            ik_heads,
            offsets=(0, 0, IDX_HALF),
            sizes=(idx_batch_size_per_iter_per_vec, 1, IDX_HALF),
            strides=(1, 1, 1),
        )
        ik_heads = insert_slice(
            ik_heads,
            ik_x1 * cos - ik_x2 * sin,
            offsets=(0, 0, 0),
            sizes=(idx_batch_size_per_iter_per_vec, 1, IDX_HALF),
            strides=(1, 1, 1),
        )
        ik_heads = insert_slice(
            ik_heads,
            ik_x2 * cos + ik_x1 * sin,
            offsets=(0, 0, IDX_HALF),
            sizes=(idx_batch_size_per_iter_per_vec, 1, IDX_HALF),
            strides=(1, 1, 1),
        )

        # * FP8：indexer RoPE 后 clamp ±448 再按输出 dtype store
        if INDEX_OUT_FP8:
            iq_heads = tl.minimum(tl.maximum(iq_heads, -448.0), 448.0)
            ik_heads = tl.minimum(tl.maximum(ik_heads, -448.0), 448.0)


        iq_idx = out_iq_nblk[None, :] + row64[:, None] * index_q_size
        ik_idx = out_ik_nblk[None, :] + row64[:, None] * IDX_HEAD_DIM
        iq_mask = (mmask[:, None]) & (out_iq_nmask[None, :])
        ik_mask = (mmask[:, None]) & (out_ik_nmask[None, :])
        tl.store(
            index_q_gm_ptr + iq_idx,
            iq_heads.reshape(idx_batch_size_per_iter_per_vec, index_q_size).to(
                index_q_gm_ptr.dtype.element_ty
            ),
            mask=iq_mask,
        )
        tl.store(
            index_k_gm_ptr + ik_idx,
            ik_heads.reshape(idx_batch_size_per_iter_per_vec, IDX_HEAD_DIM).to(
                index_k_gm_ptr.dtype.element_ty
            ),
            mask=ik_mask,
        )


def split_qkv_index_rmsnorm_rope_impl(
    input: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    index_q_weight: torch.Tensor,
    index_k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    index_q_size: int,
    head_dim: int,
    idx_head_dim: int,
    eps: float,
    attn_out_fp8: bool = False,
    indexer_out_fp8: bool = False,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused split → Gemma RMSNorm → Neox RoPE（attn + indexer）。

    Concat 布局 ``[q | k | v | index_q | index_k]``。
    ``attn_out_fp8=True`` 时 main Q/K/V clamp ±448 再写成 e4m3；
    ``indexer_out_fp8=True`` 时 indexer 同样 clamp ±448 再写成 e4m3。
    """
    input = input.contiguous()
    positions = positions.contiguous()
    q_weight = q_weight.contiguous()
    k_weight = k_weight.contiguous()
    index_q_weight = index_q_weight.contiguous()
    index_k_weight = index_k_weight.contiguous()
    cos_sin_cache = cos_sin_cache.contiguous()

    num_vectorcore = get_vectorcore_num()
    batch_size = input.shape[0]
    cache_dim = int(cos_sin_cache.shape[-1])
    attn_rope_dim = min(cache_dim, int(head_dim))
    idx_rope_dim = min(cache_dim, int(idx_head_dim))
    bias = q_bias is not None
    attn_dtype = torch.float8_e4m3fn if attn_out_fp8 else input.dtype
    index_dtype = torch.float8_e4m3fn if indexer_out_fp8 else input.dtype

    q_out = torch.empty(batch_size, q_hidden_size, device=input.device, dtype=attn_dtype)
    k_out = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=attn_dtype)
    v_out = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=attn_dtype)
    index_q_out = torch.empty(batch_size, index_q_size, device=input.device, dtype=index_dtype)
    index_k_out = torch.empty(batch_size, idx_head_dim, device=input.device, dtype=index_dtype)

    q_head_num = q_hidden_size // head_dim
    kv_head_num = kv_hidden_size // head_dim
    index_q_head_num = index_q_size // idx_head_dim
    index_qk_head_num = index_q_head_num + 1  # * +1 = 共享 index_k
    index_qk_hidden = index_qk_head_num * idx_head_dim
    index_offset = q_hidden_size + 2 * kv_hidden_size
    total_hidden_size = index_offset + index_qk_hidden
    qk_head_num_sum = q_head_num + kv_head_num

    elem = input.element_size()
    qk_factor = 5 * q_hidden_size + 3 * kv_hidden_size + cache_dim * 4 + q_head_num * attn_rope_dim
    idx_factor = (
        5 * index_q_size
        + 3 * idx_head_dim
        + cache_dim * 4
        + index_q_head_num * idx_rope_dim
    )
    batch_tile = _tokens_per_iter(elem, qk_factor)
    idx_batch_tile = _tokens_per_iter(elem, idx_factor)
    v_batch_tile = _tokens_per_iter(elem, kv_hidden_size + 1, cap=4)

    dummy = q_weight
    q_bias = q_bias.contiguous() if q_bias is not None else dummy
    k_bias = k_bias.contiguous() if k_bias is not None else dummy

    grid = (num_vectorcore,)
    split_qkv_index_rmsnorm_rope_kernel[grid](
        input,
        q_out,
        k_out,
        v_out,
        index_q_out,
        index_k_out,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        index_q_weight,
        index_k_weight,
        positions,
        cos_sin_cache,
        batch_size,
        q_hidden_size,
        kv_hidden_size,
        index_q_size,
        total_hidden_size,
        index_offset,
        index_qk_hidden,
        eps,
        bias,
        head_dim,
        idx_head_dim,
        cache_dim,
        cache_dim // 2,
        attn_rope_dim // 2,
        idx_rope_dim // 2,
        num_vectorcore,
        int(batch_tile),
        int(batch_tile * qk_head_num_sum),
        q_head_num,
        kv_head_num,
        qk_head_num_sum,
        int(v_batch_tile),
        int(idx_batch_tile),
        int(idx_batch_tile * index_qk_head_num),
        index_q_head_num,
        index_qk_head_num,
        attn_out_fp8,
        indexer_out_fp8,
    )
    return q_out, k_out, v_out, index_q_out, index_k_out


def split_qkv_index_rmsnorm_rope_impl_fake(
    input: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    index_q_weight: torch.Tensor,
    index_k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    index_q_size: int,
    head_dim: int,
    idx_head_dim: int,
    eps: float,
    attn_out_fp8: bool = False,
    indexer_out_fp8: bool = False,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = input.shape[0]
    attn_dtype = torch.float8_e4m3fn if attn_out_fp8 else input.dtype
    index_dtype = torch.float8_e4m3fn if indexer_out_fp8 else input.dtype
    q_output = torch.empty(batch_size, int(q_hidden_size), device=input.device, dtype=attn_dtype)
    k_output = torch.empty(batch_size, int(kv_hidden_size), device=input.device, dtype=attn_dtype)
    v_output = torch.empty(batch_size, int(kv_hidden_size), device=input.device, dtype=attn_dtype)
    index_q_output = torch.empty(
        batch_size, int(index_q_size), device=input.device, dtype=index_dtype
    )
    index_k_output = torch.empty(
        batch_size, int(idx_head_dim), device=input.device, dtype=index_dtype
    )
    return q_output, k_output, v_output, index_q_output, index_k_output


direct_register_custom_op(
    op_name="qkv_index_rmsnorm_rope",
    op_func=split_qkv_index_rmsnorm_rope_impl,
    fake_impl=split_qkv_index_rmsnorm_rope_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
