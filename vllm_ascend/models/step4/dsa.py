# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step4 DSA (dynamic sparse attention) core for Ascend NPUs.

Torch-native port of the standalone Step-4 minimal inference DSA path
(``step4-inference/inference/kernel.py``): per-region CSA summaries,
weighted-ReLU indexer scoring, top-k region selection, and sparse attention
over the selected regions. The core integrates with the vLLM v1 engine via
the ``AttentionLayerBase`` + custom backend pattern used by the Ascend
MiniMax-M3 sparse attention; projections, QK-norm, and RoPE of the main
attention stay on the parent ``Step4Attention`` so checkpoint weight names
map directly.

Correctness-first notes:

- Summaries are stored per *physical* region
  (``block_id * regions_per_block + offset``), so they are stable across
  request rescheduling, chunked prefill, and prefix-cache block sharing. A
  summary is a pure function of the region's tokens, which makes cross-request
  block reuse safe.
- The incomplete tail region of each request is buffered per physical block;
  a region is compressed exactly once, when its 8th token arrives.
- Region selection uses ``torch.topk``; the standalone selector uses a radix
  sort whose tie-breaking on exact-zero scores may differ on rare boundary
  rows.
- Only one local provider/KV group per rank is supported (TP >= provider
  groups), which covers the validated TP4/TP8 deployments.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from vllm_ascend.device.device_op import DeviceOperator

logger = init_logger(__name__)

REGION_BLOCK_SIZE = 8
_NEG_INF = float("-inf")


def _quantize_e4m3(tensor: torch.Tensor) -> torch.Tensor:
    """e4m3 activation rounding, widened back to bf16 (standalone semantics)."""
    return tensor.to(torch.float8_e4m3fn).to(torch.bfloat16)


def csa_compress_regions(
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    token_start: torch.Tensor,
    token_count: torch.Tensor,
    region_size: int = REGION_BLOCK_SIZE,
) -> torch.Tensor:
    """Compress packed tokens into per-region e4m3 summaries.

    ``summary = softmax(z) . k`` per (region, head): the softmax shift is a
    scalar max over both the token and proxy dims, matching the standalone
    kernel. The result is double-rounded fp32 -> bf16 -> e4m3.

    Args:
        index_k/index_z: ``[tokens, 1, proxy_dim]``.
        token_start/token_count: ``[regions]`` int; count 0 skips the region.

    Returns:
        ``[regions, 1, proxy_dim]`` float8_e4m3fn.
    """
    regions = int(token_start.numel())
    proxy_dim = index_k.shape[-1]
    device = index_k.device
    offs = torch.arange(region_size, device=device)
    token_idx = token_start[:, None] + offs[None, :]
    valid = (offs[None, :] < token_count[:, None]) & (token_count[:, None] > 0)
    safe_idx = torch.where(valid, token_idx, 0).reshape(-1)

    z = index_z[safe_idx].reshape(regions, region_size, proxy_dim).float()
    k = index_k[safe_idx].reshape(regions, region_size, proxy_dim).float()
    z = torch.where(valid[..., None], z, _NEG_INF)

    shift = z.amax(dim=(1, 2))
    weights = torch.exp(z - shift[:, None, None])
    weights = torch.where(valid[..., None], weights, 0.0)
    denom = weights.sum(dim=1)
    numer = (weights * k).sum(dim=1)
    summary = torch.where(denom > 0, numer / denom.clamp_min(1e-20), 0.0)
    return summary.to(torch.bfloat16).to(torch.float8_e4m3fn).reshape(
        regions, 1, proxy_dim
    )


def indexer_logits(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
) -> torch.Tensor:
    """Weighted-ReLU indexer scores against region summaries.

    ``score[t, r] = sum_h relu(e4m3(q_th) . e4m3(k_r)) * w_th``, fp32.

    Args:
        index_q: ``[tokens, heads_per_group, proxy_dim]`` (one local group).
        weights: ``[tokens, heads_per_group]`` fp32, already carrying the
            ``heads_per_group ** -0.5`` prescale.
        index_k: ``[regions, 1, proxy_dim]`` summaries (MQA key).

    Returns:
        ``[tokens, regions]`` float32.
    """
    qq = _quantize_e4m3(index_q).float()
    kk = _quantize_e4m3(index_k).float()
    dots = torch.einsum("thd,rd->thr", qq, kk)
    return (torch.relu(dots) * weights.unsqueeze(-1)).sum(dim=1)


def _gather_region_summaries(
    summary_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    num_regions: int,
    regions_per_block: int,
) -> torch.Tensor:
    """Logical regions ``[0, num_regions)`` -> summaries ``[num_regions, D]`` bf16."""
    device = block_table_row.device
    logical = torch.arange(num_regions, device=device)
    pages = logical // regions_per_block
    offsets = logical % regions_per_block
    phys = block_table_row[pages].long() * regions_per_block + offsets
    return summary_cache[phys].to(torch.bfloat16).squeeze(1)


def _select_topk_regions(
    scores: torch.Tensor,
    candidate_counts: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Top-k over history regions, ascending region-id order, ``-1`` padding.

    The candidate range of a query at absolute position ``p`` is
    ``[0, p // region_size)`` -- strictly past regions; the region containing
    the query is appended by the caller and never competes.
    """
    rows, max_regions = scores.shape
    device = scores.device
    ar = torch.arange(max_regions, device=device)
    valid = ar[None, :] < candidate_counts[:, None]
    masked = torch.where(valid, scores, _NEG_INF)
    k = min(int(topk), max_regions)
    _, idx = torch.topk(masked, k, dim=-1)
    picked = torch.where(
        (ar[:k] < candidate_counts[:, None]), idx, torch.full_like(idx, -1)
    )
    return torch.sort(picked, dim=-1).values


def sparse_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
    selected_regions: torch.Tensor,
    region_valid: torch.Tensor,
    *,
    scale: float,
    region_size: int = REGION_BLOCK_SIZE,
    row_chunk: int = 32,
) -> torch.Tensor:
    """Attention over per-query selected regions, gathered from the paged cache.

    Args:
        query: ``[tokens, heads, head_dim]``.
        kv_cache: ``[2, num_blocks, block_size, kv_heads, head_dim]``.
        block_table: ``[num_reqs, pages]`` int32.
        request_ids: ``[tokens]`` int, owning request of each query token.
        selected_regions: ``[tokens, sel]`` int64 logical region ids, ``-1`` pad.
        region_valid: ``[tokens, sel]`` int, visible tokens per region.

    Returns:
        ``[tokens, heads, head_dim]`` (fp32 softmax, bf16 output). Rows whose
        selection is empty produce zeros.
    """
    tokens, num_heads, head_dim = query.shape
    device = query.device
    num_blocks, block_size = kv_cache.shape[1], kv_cache.shape[2]
    key_cache, value_cache = kv_cache[0], kv_cache[1]
    flat_k = key_cache.reshape(num_blocks * block_size, *key_cache.shape[2:])
    flat_v = value_cache.reshape(num_blocks * block_size, *value_cache.shape[2:])

    out = torch.zeros_like(query)
    offsets = torch.arange(region_size, device=device)
    token_idx = selected_regions[:, :, None] * region_size + offsets[None, None, :]
    token_mask = offsets[None, None, :] < region_valid[:, :, None]
    token_mask &= selected_regions[:, :, None] >= 0
    safe_token = torch.where(token_mask, token_idx, torch.zeros_like(token_idx))
    pages = block_table[request_ids[:, None, None], safe_token // block_size]
    slots = (pages * block_size + safe_token % block_size).reshape(tokens, -1)
    mask2d = token_mask.reshape(tokens, -1)

    for begin in range(0, tokens, row_chunk):
        end = min(begin + row_chunk, tokens)
        q = query[begin:end].float()
        m = mask2d[begin:end]
        if not bool(m.any()):
            continue
        s = slots[begin:end]
        keys = flat_k[s].float()
        values = flat_v[s].float()
        scores = torch.einsum("thd,tkhd->htk", q, keys) * scale
        scores = scores.masked_fill(~m[:, None, :], _NEG_INF)
        probs = torch.softmax(scores, dim=-1)
        out[begin:end] = torch.einsum("htk,tkhd->thd", probs, values).to(query.dtype)
    return out


@dataclass
class Step4DSADecodeMetadata:
    seq_lens: torch.Tensor
    block_table: torch.Tensor


@dataclass
class Step4DSAPrefillMetadata:
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor


@dataclass
class Step4DSAMetadata(AttentionMetadata):
    num_actual_tokens: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    max_seq_len: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    decode: Step4DSADecodeMetadata | None = None
    prefill: Step4DSAPrefillMetadata | None = None


class Step4DSAMetadataBuilder(AttentionMetadataBuilder[Step4DSAMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Step4DSAMetadata:
        num_tokens = common_attn_metadata.num_actual_tokens
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )
        decode_md = None
        if num_decodes > 0:
            decode_md = Step4DSADecodeMetadata(
                seq_lens=common_attn_metadata.seq_lens[:num_decodes],
                block_table=common_attn_metadata.block_table_tensor[:num_decodes],
            )
        prefill_md = None
        if num_prefills > 0:
            prefill_md = Step4DSAPrefillMetadata(
                seq_lens=common_attn_metadata.seq_lens[num_decodes:],
                block_table=common_attn_metadata.block_table_tensor[num_decodes:],
                query_start_loc=(
                    common_attn_metadata.query_start_loc[num_decodes:]
                    - num_decode_tokens
                ),
            )
        return Step4DSAMetadata(
            num_actual_tokens=num_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_table=common_attn_metadata.block_table_tensor,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            decode=decode_md,
            prefill=prefill_md,
        )


class Step4DSABackend(AttentionBackend):
    supported_dtypes: ClassVar[list] = [torch.bfloat16, torch.float16]
    supported_kv_cache_dtypes: ClassVar[list] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "STEP4_DSA_ASCEND"

    @staticmethod
    def get_impl_cls() -> type:
        return Step4DSAAttentionImpl

    @staticmethod
    def get_builder_cls() -> type:
        return Step4DSAMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128, 192]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            raise NotImplementedError
        return (0, 1, 2, 3, 4)


def _dsa_is_graph_capturing() -> bool:
    """Whether an ACL/CUDA graph capture is in progress on this stream.

    vllm-ascend shims ``torch.cuda.is_current_stream_capturing`` onto
    ``torch.npu`` in the model runner, so the torch.cuda spelling works on
    both platforms.
    """
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


class Step4DSAAttentionImpl:
    """Sparse-attention execution over layer-computed region selections."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        kv_cache_dtype: str = "auto",
        *,
        region_size: int = REGION_BLOCK_SIZE,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.region_size = region_size

    def forward(
        self,
        layer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        selection: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        from vllm.forward_context import get_forward_context

        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict) or _dsa_is_graph_capturing():
            return output
        md = attn_metadata[layer.layer_name]
        assert isinstance(md, Step4DSAMetadata)
        regions, region_valid, request_ids = selection
        return sparse_attention(
            query,
            kv_cache,
            md.block_table,
            request_ids,
            regions,
            region_valid,
            scale=self.scale,
            region_size=self.region_size,
        )


class Step4IndexerLinear(nn.Module):
    """Indexer projection sharded by provider group over its output rows."""

    def __init__(
        self,
        input_size: int,
        total_output: int,
        *,
        index_tp_rank: int,
        index_tp_size: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.index_tp_rank = index_tp_rank
        self.index_tp_size = index_tp_size
        if total_output % index_tp_size:
            raise ValueError(
                f"{prefix}: output {total_output} is not divisible by "
                f"index_tp_size {index_tp_size}."
            )
        self.output_per_rank = total_output // index_tp_size
        self.weight = nn.Parameter(
            torch.empty(self.output_per_rank, input_size, dtype=params_dtype)
        )
        self.weight.weight_loader = self._weight_loader

    def _weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        start = self.index_tp_rank * self.output_per_rank
        param.data.copy_(loaded_weight.narrow(0, start, self.output_per_rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)


class Step4ReplicatedLinear(nn.Module):
    """Replicated single-head projection (indexer k/z), full row copy."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=params_dtype)
        )
        self.weight.weight_loader = self._weight_loader

    def _weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)


class Step4DSACore(nn.Module, AttentionLayerBase):
    """Step4 DSA indexer + summary + selection + sparse attention.

    Constructed by ``Step4Attention`` for full-attention layers of a
    DSA-capable checkpoint. The parent keeps the main q/k/v projections and
    their norm/RoPE; this module owns the indexer parameters (whose checkpoint
    names, ``self_attn.sparse_indexer_*``, map directly) and the sidecar state.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        sparse_config,
        prefix: str,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
        rms_norm_eps: float,
        indexer_rope_parameters: dict[str, Any],
        indexer_rope_theta: float | None,
        max_position: int,
        dtype: torch.dtype,
        topk: int,
        ssmax_s_size: int,
        proxy_dim: int = 256,
        rope_dim: int = 32,
        region_size: int = REGION_BLOCK_SIZE,
    ) -> None:
        super().__init__()
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        self.layer_name = f"{prefix}.attn"
        self.prefix = prefix
        self.vllm_config = vllm_config
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.scaling = head_dim**-0.5
        self.proxy_dim = proxy_dim
        self.topk = topk
        self.region_size = region_size
        self.rope_dim = rope_dim

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        total_provider_groups = int(sparse_config.index_tp_size)
        if num_kv_heads != 1 or tp_size < total_provider_groups:
            raise ValueError(
                "Step4 DSA on Ascend currently requires exactly one local "
                f"KV/provider group per rank (tp >= provider groups), got "
                f"local_kv_heads={num_kv_heads}, tp={tp_size}, "
                f"groups={total_provider_groups}."
            )
        if tp_size % total_provider_groups:
            raise ValueError(
                "Step4 DSA requires the provider-group count to divide the "
                f"tensor-parallel size, got groups={total_provider_groups}, "
                f"tp={tp_size}."
            )
        self.index_tp_rank = tp_rank // (tp_size // total_provider_groups)
        total_indexer_heads = int(sparse_config.sparse_indexer_num_heads)
        if total_indexer_heads % total_provider_groups:
            raise ValueError(
                "Step4 indexer heads must divide into provider groups, got "
                f"{total_indexer_heads} heads / {total_provider_groups} groups."
            )
        self.heads_per_group = total_indexer_heads // total_provider_groups
        self.weight_prescale = float(self.heads_per_group) ** -0.5

        self.sparse_indexer_q = Step4IndexerLinear(
            hidden_size,
            total_indexer_heads * proxy_dim,
            index_tp_rank=self.index_tp_rank,
            index_tp_size=total_provider_groups,
            params_dtype=dtype,
            prefix=f"{prefix}.sparse_indexer_q",
        )
        self.sparse_indexer_k = Step4ReplicatedLinear(
            hidden_size,
            proxy_dim,
            params_dtype=dtype,
            prefix=f"{prefix}.sparse_indexer_k",
        )
        self.sparse_indexer_z = Step4ReplicatedLinear(
            hidden_size,
            proxy_dim,
            params_dtype=dtype,
            prefix=f"{prefix}.sparse_indexer_z",
        )
        self.sparse_indexer_w = Step4IndexerLinear(
            hidden_size,
            total_indexer_heads,
            index_tp_rank=self.index_tp_rank,
            index_tp_size=total_provider_groups,
            params_dtype=dtype,
            prefix=f"{prefix}.sparse_indexer_w",
        )
        # Dormant checkpoint tensor: the deployed indexer scores with weighted
        # ReLU, so ssmax_s is loaded for checkpoint completeness only.
        self.ssmax_s = nn.Parameter(
            torch.zeros(ssmax_s_size, dtype=torch.float32), requires_grad=False
        )

        from .layernorm import OptimusLayerNorm, OptimusRMSNorm

        self.sparse_indexer_q_norm = OptimusRMSNorm(
            proxy_dim, eps=rms_norm_eps, zero_centered=True, dtype=torch.float32
        )
        self.sparse_indexer_k_norm = OptimusLayerNorm(proxy_dim, eps=rms_norm_eps)

        indexer_rope = dict(indexer_rope_parameters)
        indexer_rope.setdefault("rope_type", "default")
        if indexer_rope_theta is not None:
            indexer_rope["rope_theta"] = indexer_rope_theta
        indexer_rope["partial_rotary_factor"] = rope_dim / proxy_dim
        self.indexer_rotary_emb = get_rope(
            head_size=proxy_dim,
            max_position=max_position,
            rope_parameters=indexer_rope,
            dtype=dtype,
        )
        cos_sin = self.indexer_rotary_emb.cos_sin_cache
        self.indexer_rope_cos, self.indexer_rope_sin = cos_sin.chunk(2, dim=-1)

        self.attn_backend = Step4DSABackend
        self.impl = Step4DSAAttentionImpl(
            num_heads,
            head_dim,
            self.scaling,
            num_kv_heads,
            region_size=region_size,
        )
        self.kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self.summary_cache: torch.Tensor | None = None
        self._pending: dict[int, torch.Tensor] = {}
        self._regions_per_block: int = 0

        compilation_config = vllm_config.compilation_config
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self

    # -- AttentionLayerBase contract ------------------------------------

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        self.kv_cache = (kv_cache[0], kv_cache[1])

    def get_attn_backend(self) -> type[Step4DSABackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        block_size = vllm_config.cache_config.block_size
        if block_size % self.region_size:
            raise ValueError(
                "Step4 DSA requires the KV block size to be a multiple of the "
                f"region size {self.region_size}, got {block_size}."
            )
        kv_dtype = (
            torch.bfloat16
            if vllm_config.cache_config.cache_dtype == "auto"
            else vllm_config.cache_config.cache_dtype
        )
        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=kv_dtype,
        )

    # -- sidecar state ---------------------------------------------------

    def _ensure_sidecars(self) -> None:
        if self.summary_cache is not None or self.kv_cache is None:
            return
        key_cache, _ = self.kv_cache
        num_blocks, block_size = key_cache.shape[0], key_cache.shape[1]
        self._regions_per_block = block_size // self.region_size
        self.summary_cache = torch.zeros(
            (num_blocks * self._regions_per_block, 1, self.proxy_dim),
            device=key_cache.device,
            dtype=torch.float8_e4m3fn,
        )
        logger.info_once(
            "Step4 DSA sidecar for %s: %.1f MB summary buffer.",
            self.layer_name,
            self.summary_cache.numel() / 1e6,
        )

    def _request_ids(self, md: Step4DSAMetadata) -> torch.Tensor:
        qsl_cpu = md.query_start_loc.cpu()
        lens = (qsl_cpu[1:] - qsl_cpu[:-1]).clamp(min=0)
        ids = torch.repeat_interleave(
            torch.arange(len(lens)), lens.to(torch.int64)
        )
        return ids.to(md.seq_lens.device, non_blocking=True)

    # -- per-step sidecar update ------------------------------------------

    def _update_summaries(
        self,
        index_k: torch.Tensor,
        index_z: torch.Tensor,
        md: Step4DSAMetadata,
        positions: torch.Tensor,
        request_ids: torch.Tensor,
    ) -> None:
        """Compress every region this step completed into the summary cache.

        Regions fully inside the current chunk are compressed from the packed
        indexer tensors; a region spanning a chunk/decode boundary is rebuilt
        from its per-block pending buffer, which this method also maintains.
        """
        assert self.summary_cache is not None and self.kv_cache is not None
        key_cache, _ = self.kv_cache
        block_size = key_cache.shape[1]
        device = index_k.device
        rs = self.region_size
        seq_cpu = md.seq_lens.cpu().tolist()
        qsl_cpu = md.query_start_loc.cpu().tolist()

        for row, total in enumerate(seq_cpu):
            begin = int(qsl_cpu[row])
            end = int(qsl_cpu[row + 1])
            length = end - begin
            past = total - length

            # 1. Fill the region that was incomplete at the chunk start.
            if past % rs != 0:
                region_begin = (past // rs) * rs
                phys = self._phys_region(md, row, region_begin, block_size)
                fill = min(length, rs - past % rs)
                pending = self._pending_for(phys, device)
                pending[past % rs : past % rs + fill, 0] = (
                    index_k[begin : begin + fill].to(torch.bfloat16)
                )
                pending[past % rs : past % rs + fill, 1] = (
                    index_z[begin : begin + fill].to(torch.bfloat16)
                )
                if past % rs + fill == rs:
                    self._compress_pending(phys)

            # 2. Regions fully inside this chunk, compressed from packed tokens.
            full_start = past if past % rs == 0 else (past // rs + 1) * rs
            for lr in range(full_start // rs, total // rs):
                region_begin = lr * rs
                mask = (positions >= region_begin) & (positions < region_begin + rs)
                idx = mask.nonzero(as_tuple=True)[0]
                self.summary_cache[self._phys_region(md, row, region_begin, block_size)] = (
                    csa_compress_regions(
                        index_k[idx],
                        index_z[idx],
                        torch.zeros(1, dtype=torch.int32, device=device),
                        torch.full((1,), rs, dtype=torch.int32, device=device),
                        region_size=rs,
                    )[0]
                )

            # 3. Stash this chunk's tokens of the (new) incomplete tail region.
            tail = total % rs
            if tail != 0:
                region_begin = (total // rs) * rs
                phys = self._phys_region(md, row, region_begin, block_size)
                src_start = max(past, region_begin)
                pending = self._pending_for(phys, device)
                pending[src_start % rs : total % rs, 0] = index_k[
                    begin + (src_start - past) : end
                ].to(torch.bfloat16)
                pending[src_start % rs : total % rs, 1] = index_z[
                    begin + (src_start - past) : end
                ].to(torch.bfloat16)

    def _phys_region(
        self, md: Step4DSAMetadata, row: int, logical_token_begin: int, block_size: int
    ) -> int:
        block_id = int(md.block_table[row, logical_token_begin // block_size])
        return block_id * self._regions_per_block + (
            logical_token_begin % block_size
        ) // self.region_size

    def _pending_for(self, phys: int, device: torch.device) -> torch.Tensor:
        pending = self._pending.get(phys)
        if pending is None:
            pending = torch.zeros(
                (self.region_size, 2, self.proxy_dim),
                device=device,
                dtype=torch.bfloat16,
            )
            self._pending[phys] = pending
        return pending

    def _compress_pending(self, phys: int) -> None:
        pending = self._pending.pop(phys)
        self.summary_cache[phys] = csa_compress_regions(
            pending[:, 0].contiguous(),
            pending[:, 1].contiguous(),
            torch.zeros(1, dtype=torch.int32, device=pending.device),
            torch.full(
                (1,), self.region_size, dtype=torch.int32, device=pending.device
            ),
            region_size=self.region_size,
        )[0]

    # -- selection ---------------------------------------------------------

    def _score_and_select(
        self,
        index_q: torch.Tensor,
        weights: torch.Tensor,
        md: Step4DSAMetadata,
        positions: torch.Tensor,
        request_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-query history scores -> top-k regions + the current region."""
        assert self.summary_cache is not None and self.kv_cache is not None
        key_cache, _ = self.kv_cache
        block_size = key_cache.shape[1]
        rpb = self._regions_per_block
        device = index_q.device
        rs = self.region_size
        tokens = index_q.shape[0]

        regions_out = torch.full(
            (tokens, self.topk + 1), -1, dtype=torch.long, device=device
        )
        valid_out = torch.zeros((tokens, self.topk + 1), dtype=torch.long, device=device)

        seq_cpu = md.seq_lens.cpu().tolist()
        req_cpu = request_ids.cpu().tolist()
        by_req: dict[int, list[int]] = {}
        for t, r in enumerate(req_cpu):
            by_req.setdefault(r, []).append(t)

        for req, token_rows in by_req.items():
            total = int(seq_cpu[req])
            max_regions = (total + rs - 1) // rs
            if max_regions == 0:
                continue
            summaries = _gather_region_summaries(
                self.summary_cache, md.block_table[req], max_regions, rpb
            ).squeeze(1)
            rows = torch.tensor(token_rows, device=device)
            pos = positions[rows]
            hist = pos // rs
            candidates = int(hist.max())
            picked = torch.full(
                (len(token_rows), self.topk), -1, dtype=torch.long, device=device
            )
            if candidates > 0:
                scores = indexer_logits(
                    index_q[rows], weights[rows], summaries[:candidates]
                )
                picked = _select_topk_regions(scores, hist, self.topk)
            regions_out[rows, : picked.shape[1]] = picked
            valid_out[rows, : picked.shape[1]] = torch.where(
                picked >= 0,
                (pos[:, None] + 1 - picked * rs).clamp(min=0, max=rs),
                0,
            )
            # The region containing the query is appended unconditionally.
            own = pos // rs
            regions_out[rows, self.topk] = own
            valid_out[rows, self.topk] = pos - own * rs + 1
        return regions_out, valid_out

    # -- forward -----------------------------------------------------------

    @torch.compiler.disable(recursive=True)
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Run the DSA path for pre-normed q/k/v; returns ``[tokens, H*D]``.

        Disabled under torch.compile: the sidecar update and per-request
        selection loops are host-driven in this correctness-first port, so
        the region runs eagerly (and outside ACL graph capture).
        """
        from vllm.forward_context import get_forward_context

        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict) or _dsa_is_graph_capturing():
            # Profile/dummy runs and graph capture execute without scheduled
            # metadata. Memory accounting only needs a shape-correct output;
            # the DSA region itself runs eagerly at replay (it is excluded
            # from compiled/captured graphs via torch.compiler.disable).
            return torch.zeros(
                (hidden_states.shape[0], self.num_heads * self.head_dim),
                dtype=query.dtype,
                device=query.device,
            )
        md = attn_metadata[self.layer_name]
        assert isinstance(md, Step4DSAMetadata)
        self._ensure_sidecars()

        num_tokens = md.num_actual_tokens
        self._write_kv(key, value, md)
        index_q, index_k, index_z, weights = self._project_indexer(
            hidden_states, positions
        )
        request_ids = self._request_ids(md)
        self._update_summaries(index_k, index_z, md, positions, request_ids)
        regions, region_valid = self._score_and_select(
            index_q, weights, md, positions, request_ids
        )
        return self.impl.forward(
            self,
            query.view(num_tokens, self.num_heads, self.head_dim),
            self.kv_cache,
            (regions, region_valid, request_ids),
        )

    def _project_indexer(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from .kernels import fused_indexer_norm_rope_forward_impl

        tokens = hidden_states.shape[0]
        index_q = self.sparse_indexer_q(hidden_states)
        index_k = self.sparse_indexer_k(hidden_states)
        index_z = self.sparse_indexer_z(hidden_states)

        index_q, index_k, index_z = fused_indexer_norm_rope_forward_impl(
            index_q.contiguous(),
            index_k.contiguous(),
            index_z.contiguous(),
            self.sparse_indexer_q_norm.weight.to(dtype=index_q.dtype),
            self.sparse_indexer_k_norm.weight.to(dtype=index_k.dtype),
            self.sparse_indexer_k_norm.bias.to(dtype=index_k.dtype),
            self.indexer_rope_cos,
            self.indexer_rope_sin,
            positions,
            self.proxy_dim,
            self.heads_per_group,
            1,
            self.rope_dim // 2,
            self.sparse_indexer_q_norm.variance_epsilon,
            1.0,
        )
        weights = torch.nn.functional.linear(
            hidden_states, self.sparse_indexer_w.weight.to(dtype=hidden_states.dtype)
        ).float()
        weights = weights.view(tokens, self.heads_per_group)
        weights = weights * self.weight_prescale
        index_q = index_q.view(tokens, self.heads_per_group, self.proxy_dim)
        return index_q, index_k, index_z, weights

    def _write_kv(self, key: torch.Tensor, value: torch.Tensor, md: Step4DSAMetadata) -> None:
        assert self.kv_cache is not None
        key_cache, value_cache = self.kv_cache
        DeviceOperator.reshape_and_cache(
            key[: md.num_actual_tokens].contiguous().view(
                -1, self.num_kv_heads, self.head_dim
            ),
            value[: md.num_actual_tokens].contiguous().view(
                -1, self.num_kv_heads, self.head_dim
            ),
            key_cache,
            value_cache,
            md.slot_mapping[: md.num_actual_tokens],
        )
