# SPDX-License-Identifier: Apache-2.0
"""Step4 DSA kernels for Ascend, vendored from the step4-hf multi-hardware
release (``inference/kernel.py``, which is Apache-2.0 and device-portable
Triton). Only the operators the vLLM-Ascend port needs are copied:

- ``csa_compress_regions`` (Ascend variant): per-region softmax-weighted mean
  of the sparse-indexer k, producing the fp8 summary the selector scores.
- ``indexer_logits``: weighted-ReLU indexer scores (bf16 tl.dot; the e4m3
  activation rounding is part of the semantics and kept).
- ``sparse_attention_prefill`` / ``sparse_attention_decode``: token-wise
  sparse GQA over per-row selected regions (online softmax, no causal mask --
  causality is encoded in the packed metadata).

The top-k selector and metadata packing are reimplemented in pure PyTorch
below (the Triton versions rely on ``tl.histogram`` / ``tl.sort``; the sort is
known to be unavailable on triton-ascend, and step4-hf itself uses a torch
fallback on NPU for the decode pack).
"""

import torch
import triton
import triton.language as tl

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass

# Step-4 compresses every 8 consecutive tokens into one summary vector.
REGION_BLOCK_SIZE = 8
PROXY_DIM = 256

REGION_VALID_SHIFT = tl.constexpr(24)
REGION_ID_MASK = tl.constexpr((1 << 24) - 1)
DECODE_PHYS_SHIFT = tl.constexpr(32)
DECODE_LOW_MASK = tl.constexpr((1 << 32) - 1)
NEGATIVE_INFINITY = tl.constexpr(float("-inf"))


# ---------------------------------------------------------------------------
# CSA region compression (Ascend variant)
# ---------------------------------------------------------------------------

_ASCEND_VEC_CORES: int | None = None


def _ascend_vector_core_count() -> int:
    """Physical AIV count for contiguous region partitioning. Cached per process."""
    global _ASCEND_VEC_CORES
    if _ASCEND_VEC_CORES is not None:
        return _ASCEND_VEC_CORES
    count = 64
    try:
        import triton.runtime.driver as driver

        device = torch.npu.current_device()
        props = driver.active.utils.get_device_properties(device)
        raw = props.get("num_vectorcore", props.get("num_vector_core"))
        if raw:
            count = int(raw)
    except Exception:
        pass
    _ASCEND_VEC_CORES = max(1, count)
    return _ASCEND_VEC_CORES


@triton.jit
def _csa_compress_regions_kernel_ascend(
    index_k_ptr,
    index_z_ptr,
    token_start_ptr,
    token_count_ptr,
    summary_ptr,
    stride_k_token,
    stride_k_head,
    stride_z_token,
    stride_z_head,
    stride_summary_region,
    stride_summary_head,
    n_jobs,
    num_heads: tl.constexpr,
    proxy_dim: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    JOBS_PER_CORE: tl.constexpr,
) -> None:
    """Ascend CSA compress: one hardware program owns a *contiguous* job range.

    128k/64 AIV means ~256 regions per core. Interleaved ``pid, n, n_cores`` would
    jump 64 regions (512 tokens) between iterations and keep MTE2 on tiny strided
    loads. Contiguous slices keep packed-sequence tokens sequential.

    Each job is still one (region, head) with a one-shot [T, D] tile. The shift is a
    scalar max over both axes, matching the H200 / torch reference.
    """
    pid = tl.program_id(0)
    job_begin = pid * JOBS_PER_CORE

    tokens = tl.arange(0, BLOCK_T)
    dims = tl.arange(0, BLOCK_D)
    dim_ok = dims < proxy_dim

    for off in range(JOBS_PER_CORE):
        job = job_begin + off
        if job < n_jobs:
            region = job // num_heads
            head = job - region * num_heads

            start = tl.load(token_start_ptr + region).to(tl.int32)
            count = tl.load(token_count_ptr + region).to(tl.int32)

            token_ok = tokens < count
            mask = token_ok[:, None] & dim_ok[None, :]

            token_idx = start + tokens
            z_off = (
                token_idx[:, None] * stride_z_token
                + head * stride_z_head
                + dims[None, :]
            )
            k_off = (
                token_idx[:, None] * stride_k_token
                + head * stride_k_head
                + dims[None, :]
            )
            logits = tl.load(index_z_ptr + z_off, mask=mask, other=float("-inf")).to(
                tl.float32
            )
            values = tl.load(index_k_ptr + k_off, mask=mask, other=0.0).to(tl.float32)

            row_max = tl.max(tl.where(mask, logits, float("-inf")), axis=0)
            shift = tl.max(row_max, axis=0)
            weights = tl.where(mask, tl.exp(logits - shift), 0.0)
            denominator = tl.sum(weights, axis=0)
            numerator = tl.sum(weights * values, axis=0)
            summary = tl.where(
                denominator > 0.0, numerator / tl.maximum(denominator, 1e-20), 0.0
            )

            out = region * stride_summary_region + head * stride_summary_head + dims
            tl.store(summary_ptr + out, summary, mask=dim_ok)


def csa_compress_regions(
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    token_start: torch.Tensor,
    token_count: torch.Tensor,
    *,
    region_size: int = REGION_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress packed tokens into per-region summaries (Ascend path).

    Args:
        index_k: ``[tokens, heads, proxy_dim]`` proxy keys.
        index_z: ``[tokens, heads, proxy_dim]`` per-dimension softmax logits.
        token_start: ``[regions]`` int32, first token of each region.
        token_count: ``[regions]`` int32, tokens present in each region; 0 = skip.

    Returns:
        ``(summary_fp32, summary_fp8)``, both ``[regions, heads, proxy_dim]``.
    """
    if index_k.shape != index_z.shape:
        raise ValueError(
            f"index_k and index_z must match, got {tuple(index_k.shape)} and "
            f"{tuple(index_z.shape)}"
        )
    if index_k.ndim != 3:
        raise ValueError(
            f"index_k must be [tokens, heads, proxy_dim], got {tuple(index_k.shape)}"
        )
    if index_k.stride(2) != 1:
        raise ValueError("index_k must be contiguous in the proxy dimension")

    regions = int(token_start.numel())
    _, heads, proxy_dim = index_k.shape
    summary = torch.zeros(
        (regions, heads, proxy_dim), device=index_k.device, dtype=torch.float32
    )
    block_t = triton.next_power_of_2(region_size)
    block_d = triton.next_power_of_2(proxy_dim)

    n_jobs = regions * heads
    vec_cores = _ascend_vector_core_count()
    jobs_per_core = triton.cdiv(n_jobs, min(n_jobs, vec_cores))
    grid_size = triton.cdiv(n_jobs, jobs_per_core)
    _csa_compress_regions_kernel_ascend[(grid_size,)](
        index_k,
        index_z,
        token_start,
        token_count,
        summary,
        index_k.stride(0),
        index_k.stride(1),
        index_z.stride(0),
        index_z.stride(1),
        summary.stride(0),
        summary.stride(1),
        n_jobs,
        num_heads=heads,
        proxy_dim=proxy_dim,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        JOBS_PER_CORE=jobs_per_core,
    )
    # Host-side double rounding (fp32 -> bf16 -> e4m3) matches the deployed
    # quantize_summary_e4m3 byte-for-byte.
    mean = summary.to(torch.bfloat16).to(torch.float8_e4m3fn)
    return summary, mean


# ---------------------------------------------------------------------------
# Indexer weighted-ReLU logits
# ---------------------------------------------------------------------------


@triton.jit
def _indexer_logits_kernel(
    index_q_ptr,
    index_k_ptr,
    weights_ptr,
    out_ptr,
    seq_q,
    seq_k,
    stride_q_token,
    stride_q_group,
    stride_q_head,
    stride_k_token,
    stride_w_token,
    stride_w_group,
    stride_out_row,
    heads_per_group: tl.constexpr,
    proxy_dim: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Weighted-ReLU indexer scores for one (query tile, key tile, provider group).

        score[q, r] = sum_h relu(dot(q_qh, k_r)) * w_qh

    The ReLU is per head and *inside* the sum, so the heads cannot be folded into a
    single matmul -- hence the loop.
    """
    q_block = tl.program_id(0)
    k_block = tl.program_id(1)
    group = tl.program_id(2)

    queries = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
    keys = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    dims = tl.arange(0, proxy_dim)
    query_ok = queries < seq_q
    key_ok = keys < seq_k

    key_tile = tl.load(
        index_k_ptr + dims[:, None] + keys[None, :] * stride_k_token,
        mask=key_ok[None, :],
        other=0.0,
    )

    accumulator = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
    for head in tl.static_range(heads_per_group):
        query_tile = tl.load(
            index_q_ptr
            + queries[:, None] * stride_q_token
            + group * stride_q_group
            + head * stride_q_head
            + dims[None, :],
            mask=query_ok[:, None],
            other=0.0,
        )
        head_weight = tl.load(
            weights_ptr + queries * stride_w_token + group * stride_w_group + head,
            mask=query_ok,
            other=0.0,
        ).to(tl.float32)
        scores = tl.dot(query_tile, key_tile, out_dtype=tl.float32)
        accumulator += tl.maximum(scores, 0.0) * head_weight[:, None]

    rows = group * seq_q + queries
    tl.store(
        out_ptr + rows[:, None] * stride_out_row + keys[None, :],
        accumulator,
        mask=query_ok[:, None] & key_ok[None, :],
    )


def round_activations_e4m3(tensor: torch.Tensor) -> torch.Tensor:
    """Apply the indexer's e4m3 activation rounding, then widen back to bfloat16.

    The deployed kernel quantizes indexer activations to e4m3 before the score matmul,
    even though the checkpoint stores these weights in bf16. Skipping the rounding
    changes which regions get selected, so it is part of the semantics.
    """
    return tensor.to(torch.float8_e4m3fn).to(torch.bfloat16)


def indexer_logits(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    *,
    block_q: int | None = None,
    block_k: int | None = None,
) -> torch.Tensor:
    """Weighted-ReLU indexer scores over proxy keys.

    Args:
        index_q: ``[seq_q, groups, heads_per_group, proxy_dim]``.
        weights: ``[seq_q, groups, heads_per_group]``, already carrying the
            ``heads_per_group ** -0.5`` prescale.
        index_k: ``[seq_k, 1, proxy_dim]`` -- one shared key head (MQA).

    Returns:
        ``[groups * seq_q, seq_k]`` float32, group-major then query.
    """
    if index_k.shape[1] != 1:
        raise ValueError(f"indexer keys are MQA; expected one head, got {index_k.shape[1]}")
    seq_q, groups, heads_per_group, proxy_dim = index_q.shape
    seq_k = index_k.shape[0]

    # The H200 tiles (64x128) overflow the Ascend 950 UB once the compiler's
    # multi-buffering is accounted for; smaller tiles fit with margin.
    on_npu = index_q.device.type == "npu"
    if block_q is None:
        block_q = 16 if on_npu else 64
    if block_k is None:
        block_k = 64 if on_npu else 128

    quantized_q = round_activations_e4m3(index_q)
    quantized_k = round_activations_e4m3(index_k)
    out = torch.empty((groups * seq_q, seq_k), device=index_q.device, dtype=torch.float32)

    _indexer_logits_kernel[
        (triton.cdiv(seq_q, block_q), triton.cdiv(seq_k, block_k), groups)
    ](
        quantized_q,
        quantized_k,
        weights,
        out,
        seq_q,
        seq_k,
        quantized_q.stride(0),
        quantized_q.stride(1),
        quantized_q.stride(2),
        quantized_k.stride(0),
        weights.stride(0),
        weights.stride(1),
        out.stride(0),
        heads_per_group=heads_per_group,
        proxy_dim=proxy_dim,
        BLOCK_Q=block_q,
        BLOCK_K=block_k,
    )
    return out


# ---------------------------------------------------------------------------
# Sparse attention over selected regions
# ---------------------------------------------------------------------------


@triton.jit
def _sparse_attn_prefill_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    packed_ptr,
    counts_ptr,
    out_ptr,
    lse_ptr,
    stride_q_token,
    stride_q_head,
    stride_k_token,
    stride_k_group,
    stride_v_token,
    stride_v_group,
    stride_packed_row,
    stride_out_token,
    stride_out_head,
    stride_lse_token,
    softmax_scale,
    total_q,
    topk: tl.constexpr,
    heads_per_group: tl.constexpr,
    head_dim: tl.constexpr,
    region_size: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
) -> None:
    """Attention for one query against one KV group's selected regions.

    Online softmax over region tiles; K/V rows come from a gather driven by the
    packed region list. There is no causal mask: causality is already encoded in
    ``valid_tokens`` and the selector's candidate range.
    """
    query = tl.program_id(0)
    group = tl.program_id(1)
    row = group * total_q + query
    count = tl.load(counts_ptr + row).to(tl.int32)

    heads = tl.arange(0, BLOCK_H)
    dims = tl.arange(0, BLOCK_D)
    head_ok = heads < heads_per_group
    dim_ok = dims < head_dim
    tile_ok = head_ok[:, None] & dim_ok[None, :]

    queries = tl.load(
        q_ptr
        + query * stride_q_token
        + (group * heads_per_group + heads)[:, None] * stride_q_head
        + dims[None, :],
        mask=tile_ok,
        other=0.0,
    )

    running_max = tl.full((BLOCK_H,), NEGATIVE_INFINITY, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    offsets_in_region = tl.arange(0, region_size)
    for base in tl.range(0, topk, BLOCK_R):
        slots = base + tl.arange(0, BLOCK_R)
        slot_ok = slots < count
        meta = tl.load(
            packed_ptr + row * stride_packed_row + slots, mask=slot_ok, other=-1
        )
        physical = meta & REGION_ID_MASK
        valid = meta >> REGION_VALID_SHIFT
        live = slot_ok & (meta >= 0) & (valid > 0)

        tokens = tl.reshape(
            physical[:, None] * region_size + offsets_in_region[None, :],
            (BLOCK_R * region_size,),
        )
        token_ok = tl.reshape(
            live[:, None] & (offsets_in_region[None, :] < valid[:, None]),
            (BLOCK_R * region_size,),
        )
        safe_tokens = tl.where(token_ok, tokens, 0)

        keys = tl.load(
            k_ptr
            + safe_tokens[:, None] * stride_k_token
            + group * stride_k_group
            + dims[None, :],
            mask=token_ok[:, None] & dim_ok[None, :],
            other=0.0,
        )
        values = tl.load(
            v_ptr
            + safe_tokens[:, None] * stride_v_token
            + group * stride_v_group
            + dims[None, :],
            mask=token_ok[:, None] & dim_ok[None, :],
            other=0.0,
        )

        scores = tl.dot(queries, tl.trans(keys), out_dtype=tl.float32) * softmax_scale
        scores = tl.where(token_ok[None, :] & head_ok[:, None], scores, NEGATIVE_INFINITY)

        tile_max = tl.maximum(running_max, tl.max(scores, axis=1))
        # A tile can be entirely masked; substituting a finite shift keeps the
        # arithmetic well defined and leaves the running state untouched.
        empty = tile_max == NEGATIVE_INFINITY
        shift = tl.where(empty, 0.0, tile_max)
        rescale = tl.where(empty, 1.0, tl.exp(running_max - shift))

        probabilities = tl.exp(scores - shift[:, None])
        running_sum = running_sum * rescale + tl.sum(probabilities, axis=1)
        accumulator = accumulator * rescale[:, None] + tl.dot(
            probabilities.to(values.dtype), values, out_dtype=tl.float32
        )
        running_max = tile_max

    selected = running_sum > 0
    output = tl.where(selected[:, None], accumulator / running_sum[:, None], 0.0)
    tl.store(
        out_ptr
        + query * stride_out_token
        + (group * heads_per_group + heads)[:, None] * stride_out_head
        + dims[None, :],
        output.to(out_ptr.dtype.element_ty),
        mask=tile_ok,
    )
    tl.store(
        lse_ptr + query * stride_lse_token + group * heads_per_group + heads,
        tl.where(selected, running_max + tl.log(running_sum), NEGATIVE_INFINITY),
        mask=head_ok,
    )


def sparse_attention_prefill(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    packed_regions: torch.Tensor,
    region_counts: torch.Tensor,
    *,
    num_kv_groups: int,
    region_size: int = REGION_BLOCK_SIZE,
    softmax_scale: float | None = None,
    block_regions: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse attention over per-query selected regions.

    Args:
        query: ``[total_q, num_heads, head_dim]``.
        key_cache, value_cache: ``[tokens, num_kv_groups, head_dim]`` (pages flattened).
        packed_regions: ``[num_kv_groups * total_q, topk]`` int32, each slot
            ``phys_region | (valid_tokens << 24)``, ``-1`` padding the tail.
        region_counts: ``[num_kv_groups * total_q]`` int32, positional truncation.

    Returns:
        ``(out, lse)`` shaped ``[total_q, num_heads, head_dim]`` / ``[total_q, num_heads]``.
    """
    total_q, num_heads, head_dim = query.shape
    if num_heads % num_kv_groups:
        raise ValueError(f"{num_heads} heads do not divide into {num_kv_groups} groups")
    if packed_regions.shape[0] != num_kv_groups * total_q:
        raise ValueError(
            f"expected {num_kv_groups * total_q} metadata rows, got {packed_regions.shape[0]}"
        )

    heads_per_group = num_heads // num_kv_groups
    topk = int(packed_regions.shape[1])
    scale = softmax_scale if softmax_scale is not None else head_dim**-0.5

    on_npu = query.device.type == "npu"
    block_h = triton.next_power_of_2(heads_per_group)
    if not on_npu:
        block_h = max(16, block_h)
    block_r = 1 if on_npu else min(block_regions, triton.next_power_of_2(topk))

    out = torch.empty_like(query)
    lse = torch.empty((total_q, num_heads), device=query.device, dtype=torch.float32)
    _sparse_attn_prefill_kernel[(total_q, num_kv_groups)](
        query,
        key_cache,
        value_cache,
        packed_regions,
        region_counts,
        out,
        lse,
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        value_cache.stride(0),
        value_cache.stride(1),
        packed_regions.stride(0),
        out.stride(0),
        out.stride(1),
        lse.stride(0),
        scale,
        total_q,
        topk=topk,
        heads_per_group=heads_per_group,
        head_dim=head_dim,
        region_size=region_size,
        BLOCK_H=block_h,
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_R=block_r,
    )
    return out, lse


@triton.jit
def _sparse_attn_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    packed_ptr,
    counts_ptr,
    seqlens_ptr,
    out_ptr,
    lse_ptr,
    stride_q_token,
    stride_q_head,
    stride_k_token,
    stride_k_group,
    stride_v_token,
    stride_v_group,
    stride_packed_row,
    stride_out_split,
    stride_out_token,
    stride_out_head,
    stride_lse_split,
    stride_lse_token,
    softmax_scale,
    num_reqs,
    topk: tl.constexpr,
    heads_per_group: tl.constexpr,
    head_dim: tl.constexpr,
    region_size: tl.constexpr,
    slots_per_split: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
) -> None:
    """One decode query against one KV group's selected regions, over one slot window.

    Decode packs ``start_token | (phys_region << 32)`` in int64 and recomputes the
    visible length as ``clamp(kv_seqlen - start_token, 0, region_size)``.
    """
    request = tl.program_id(0)
    group = tl.program_id(1)
    split = tl.program_id(2)
    row = group * num_reqs + request
    count = tl.load(counts_ptr + row).to(tl.int32)
    kv_seqlen = tl.load(seqlens_ptr + request).to(tl.int64)

    heads = tl.arange(0, BLOCK_H)
    dims = tl.arange(0, BLOCK_D)
    head_ok = heads < heads_per_group
    dim_ok = dims < head_dim
    tile_ok = head_ok[:, None] & dim_ok[None, :]

    queries = tl.load(
        q_ptr
        + request * stride_q_token
        + (group * heads_per_group + heads)[:, None] * stride_q_head
        + dims[None, :],
        mask=tile_ok,
        other=0.0,
    )

    running_max = tl.full((BLOCK_H,), NEGATIVE_INFINITY, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    offsets_in_region = tl.arange(0, region_size)
    window_begin = split * slots_per_split
    for base in tl.range(window_begin, window_begin + slots_per_split, BLOCK_R):
        slots = base + tl.arange(0, BLOCK_R)
        slot_ok = (slots < count) & (slots < topk)
        meta = tl.load(
            packed_ptr + row * stride_packed_row + slots, mask=slot_ok, other=-1
        )
        physical = (meta >> DECODE_PHYS_SHIFT) & DECODE_LOW_MASK
        start_token = meta & DECODE_LOW_MASK
        valid = tl.minimum(tl.maximum(kv_seqlen - start_token, 0), region_size)
        live = slot_ok & (valid > 0)

        safe_physical = tl.where(live, physical, 0)
        tokens = tl.reshape(
            safe_physical[:, None] * region_size + offsets_in_region[None, :],
            (BLOCK_R * region_size,),
        )
        token_ok = tl.reshape(
            live[:, None] & (offsets_in_region[None, :] < valid[:, None]),
            (BLOCK_R * region_size,),
        )

        keys = tl.load(
            k_ptr + tokens[:, None] * stride_k_token + group * stride_k_group + dims[None, :],
            mask=token_ok[:, None] & dim_ok[None, :],
            other=0.0,
        )
        values = tl.load(
            v_ptr + tokens[:, None] * stride_v_token + group * stride_v_group + dims[None, :],
            mask=token_ok[:, None] & dim_ok[None, :],
            other=0.0,
        )

        scores = tl.dot(queries, tl.trans(keys), out_dtype=tl.float32) * softmax_scale
        scores = tl.where(token_ok[None, :] & head_ok[:, None], scores, NEGATIVE_INFINITY)

        tile_max = tl.maximum(running_max, tl.max(scores, axis=1))
        empty = tile_max == NEGATIVE_INFINITY
        shift = tl.where(empty, 0.0, tile_max)
        rescale = tl.where(empty, 1.0, tl.exp(running_max - shift))

        probabilities = tl.exp(scores - shift[:, None])
        running_sum = running_sum * rescale + tl.sum(probabilities, axis=1)
        accumulator = accumulator * rescale[:, None] + tl.dot(
            probabilities.to(values.dtype), values, out_dtype=tl.float32
        )
        running_max = tile_max

    selected = running_sum > 0
    output = tl.where(selected[:, None], accumulator / running_sum[:, None], 0.0)
    tl.store(
        out_ptr
        + split * stride_out_split
        + request * stride_out_token
        + (group * heads_per_group + heads)[:, None] * stride_out_head
        + dims[None, :],
        output.to(out_ptr.dtype.element_ty),
        mask=tile_ok,
    )
    tl.store(
        lse_ptr
        + split * stride_lse_split
        + request * stride_lse_token
        + group * heads_per_group
        + heads,
        tl.where(selected, running_max + tl.log(running_sum), NEGATIVE_INFINITY),
        mask=head_ok,
    )


def sparse_attention_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    packed_regions: torch.Tensor,
    region_counts: torch.Tensor,
    kv_seqlens: torch.Tensor,
    *,
    num_kv_groups: int,
    region_size: int = REGION_BLOCK_SIZE,
    softmax_scale: float | None = None,
    block_regions: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse attention for one query token per request (num_splits == 1 only).

    Args:
        query: ``[num_reqs, num_heads, head_dim]``.
        key_cache, value_cache: ``[tokens, num_kv_groups, head_dim]``, pages flattened.
        packed_regions: ``[num_kv_groups * num_reqs, topk]`` int64, each slot
            ``start_token | (phys_region << 32)``; tail zero-padded.
        region_counts: ``[num_kv_groups * num_reqs]`` int32, live slot count.
        kv_seqlens: ``[num_reqs]`` int32, context length including the current token.

    Returns:
        ``(out [num_reqs, num_heads, head_dim], lse [num_reqs, num_heads])``.
    """
    num_reqs, num_heads, head_dim = query.shape
    if num_heads % num_kv_groups:
        raise ValueError(f"{num_heads} heads do not divide into {num_kv_groups} groups")
    if packed_regions.shape[0] != num_kv_groups * num_reqs:
        raise ValueError(
            f"expected {num_kv_groups * num_reqs} metadata rows, got {packed_regions.shape[0]}"
        )
    if packed_regions.dtype != torch.int64:
        raise ValueError(f"decode metadata must be int64, got {packed_regions.dtype}")

    heads_per_group = num_heads // num_kv_groups
    topk = int(packed_regions.shape[1])
    scale = softmax_scale if softmax_scale is not None else head_dim**-0.5

    on_npu = query.device.type == "npu"
    block_h = triton.next_power_of_2(heads_per_group)
    if not on_npu:
        block_h = max(16, block_h)
    block_r = 1 if on_npu else min(block_regions, triton.next_power_of_2(topk))

    out = torch.empty((1, num_reqs, num_heads, head_dim), device=query.device, dtype=query.dtype)
    lse = torch.empty((1, num_reqs, num_heads), device=query.device, dtype=torch.float32)
    _sparse_attn_decode_kernel[(num_reqs, num_kv_groups, 1)](
        query,
        key_cache,
        value_cache,
        packed_regions,
        region_counts,
        kv_seqlens,
        out,
        lse,
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        value_cache.stride(0),
        value_cache.stride(1),
        packed_regions.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lse.stride(0),
        lse.stride(1),
        scale,
        num_reqs,
        topk=topk,
        heads_per_group=heads_per_group,
        head_dim=head_dim,
        region_size=region_size,
        slots_per_split=topk,
        BLOCK_H=block_h,
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_R=block_r,
    )
    return out[0], lse[0]


# ---------------------------------------------------------------------------
# Pure-torch region selection and metadata packing
# ---------------------------------------------------------------------------


def region_topk_ids_torch(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    *,
    topk: int,
) -> torch.Tensor:
    """Top-k region selection, logical ids ascending, ``-1`` padding.

    Selection contract: order the visible scores by ``(-score, id)`` and keep the
    first ``min(topk, visible)`` -- ties take a prefix of the tied group (ties are
    load-bearing: post-ReLU scores are exactly zero ~6% of the time).
    """
    if logits.dtype != torch.float32:
        raise ValueError(f"selector expects float32 scores, got {logits.dtype}")
    rows, seq_regions = logits.shape
    device = logits.device
    region_ids = torch.arange(seq_regions, device=device)
    visible = lengths.to(torch.int64).clamp(min=0, max=seq_regions)

    # Sort key: -score ascending; invisible regions sort last; stable sort keeps
    # tied ids ascending.
    sort_key = (-logits).masked_fill(region_ids[None, :] >= visible[:, None], float("inf"))
    order = torch.argsort(sort_key, dim=-1, stable=True)
    budget = torch.minimum(visible, torch.full_like(visible, topk))
    width = min(topk, seq_regions)
    take = order[:, :width].to(torch.int64)
    take = torch.where(
        torch.arange(width, device=device)[None, :] < budget[:, None],
        take,
        torch.full_like(take, seq_regions),
    )
    # Emit in ascending region order (the contract consumers rely on).
    take = torch.sort(take, dim=-1).values
    take = torch.where(take < seq_regions, take, torch.full_like(take, -1))
    if width < topk:
        take = torch.cat(
            [take, torch.full((rows, topk - width), -1, device=device, dtype=torch.int64)],
            dim=1,
        )
    return take.to(torch.int32)


def prefill_sparse_meta_torch(
    logits: torch.Tensor,
    q_positions: torch.Tensor,
    block_table: torch.Tensor,
    request_indices: torch.Tensor,
    *,
    topk: int,
    region_size: int = REGION_BLOCK_SIZE,
    regions_per_page: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select regions and build prefill sparse metadata (pure PyTorch).

    Returns ``(packed [rows, topk + 1] int32, counts [rows] int32)`` where each
    history slot is ``phys_region | (valid_tokens << 24)`` with ``-1`` holes kept
    in place, the query's current region is appended at index
    ``min(current_region, topk)``, and ``counts = min(current_region, topk) + 1``.
    """
    rows = int(logits.shape[0])
    device = logits.device
    q_positions = q_positions.to(torch.int64)
    request_indices = request_indices.to(torch.int64)
    # History excludes the region containing the query itself.
    history_lengths = torch.div(q_positions, region_size, rounding_mode="floor")

    ids = region_topk_ids_torch(logits, history_lengths.to(torch.int32), topk=topk)
    safe_ids = ids.to(torch.int64).clamp_min(0)
    pages = block_table[request_indices[:, None], safe_ids // regions_per_page]
    live = (ids >= 0) & (pages >= 0)
    physical = pages * regions_per_page + safe_ids % regions_per_page
    valid_tokens = (q_positions[:, None] + 1 - safe_ids * region_size).clamp(
        min=0, max=region_size
    )
    history = torch.where(
        live,
        (physical | (valid_tokens << 24)).to(torch.int32),
        torch.full_like(ids, -1),
    )

    packed = torch.cat(
        [history, torch.full((rows, 1), -1, device=device, dtype=torch.int32)], dim=1
    )
    current_region = torch.div(q_positions, region_size, rounding_mode="floor").clamp_min(0)
    history_count = torch.minimum(
        current_region, torch.full_like(current_region, topk)
    )
    current_page = block_table[
        request_indices, current_region // regions_per_page
    ]
    current_physical = current_page * regions_per_page + current_region % regions_per_page
    current_valid = (q_positions + 1 - current_region * region_size).clamp(
        min=0, max=region_size
    )
    current_packed = torch.where(
        current_page >= 0,
        (current_physical | (current_valid << 24)).to(torch.int32),
        torch.full_like(current_page, -1, dtype=torch.int32),
    )
    packed[torch.arange(rows, device=device), history_count] = current_packed
    counts = (history_count + 1).to(torch.int32)
    return packed, counts


def decode_sparse_meta_torch(
    logits: torch.Tensor,
    kv_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    request_indices: torch.Tensor,
    *,
    topk: int,
    region_size: int = REGION_BLOCK_SIZE,
    regions_per_page: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select regions and build decode sparse metadata (pure PyTorch).

    Returns ``(packed [rows, topk + 1] int64, counts [rows] int32)`` where each
    live slot is ``start_token | (phys_region << 32)``, invalid slots are
    compacted out, the tail is zero-padded, and the current region is appended
    at the surviving count.
    """
    rows = int(logits.shape[0])
    device = logits.device
    req = request_indices.to(torch.int64)
    kv_seqlen = kv_seqlens.to(torch.int64)[req]
    valid_regions = (kv_seqlen + region_size - 1) // region_size
    history_lengths = torch.div(
        (kv_seqlen - 1).clamp_min(0), region_size, rounding_mode="floor"
    )

    ids = region_topk_ids_torch(logits, history_lengths.to(torch.int32), topk=topk)
    ids_i = ids.to(torch.int64)
    live = (ids >= 0) & (ids_i < valid_regions[:, None])
    safe_ids = torch.where(live, ids_i, torch.zeros_like(ids_i))
    pages = block_table[req[:, None], safe_ids // regions_per_page]
    live = live & (pages >= 0)

    physical = pages * regions_per_page + safe_ids % regions_per_page
    packed = (safe_ids * region_size) | (physical << DECODE_PHYS_SHIFT.value)
    sentinel = 1 << 62
    packed = torch.where(live, packed, torch.full_like(packed, sentinel))
    packed, _ = torch.sort(packed, dim=-1, descending=False)
    count = live.sum(dim=-1, dtype=torch.int64)

    out = torch.zeros((rows, topk + 1), device=device, dtype=torch.int64)
    keep = torch.arange(topk, device=device).unsqueeze(0) < count.unsqueeze(1)
    out[:, :topk] = torch.where(keep, packed, torch.zeros_like(packed))

    current_region = torch.div((kv_seqlen - 1).clamp_min(0), region_size, rounding_mode="floor")
    current_page = block_table[req, current_region // regions_per_page]
    current_live = (current_region < valid_regions) & (current_page >= 0)
    current_physical = current_page * regions_per_page + current_region % regions_per_page
    current_packed = (current_region * region_size) | (current_physical << DECODE_PHYS_SHIFT.value)
    current_packed = torch.where(current_live, current_packed, torch.zeros_like(current_packed))
    out[torch.arange(rows, device=device), count] = current_packed
    counts = (count + current_live.to(torch.int64)).to(torch.int32)
    return out, counts
