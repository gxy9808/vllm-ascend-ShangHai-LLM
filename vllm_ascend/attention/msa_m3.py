# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 sparse attention and indexer backends for Ascend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.distributed as dist
import torch_npu
from torch import nn
from torch.nn.parameter import Parameter
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
    adjust_block_scale_shard,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.parameter import BasevLLMParameter, BlockQuantScaleParameter
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImplBase,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

# ============================================================================
# Determinism verification: compare fused kernel outputs across calls
# ============================================================================
_DETERMINISM_CALL_COUNT = 0
_DETERMINISM_MAX_COMPARE = 4
_DETERMINISM_BASELINE = {}  # saved inputs/outputs from first call

def _det_save_and_compare(tag, outputs, inputs_dict):
    global _DETERMINISM_CALL_COUNT, _DETERMINISM_BASELINE
    _DETERMINISM_CALL_COUNT += 1
    call_idx = _DETERMINISM_CALL_COUNT

    if call_idx == 1:
        # First call: save baseline
        for name, tensor in inputs_dict.items():
            _DETERMINISM_BASELINE[f"input_{name}"] = tensor.clone().cpu().to(torch.float32)
        for i, out in enumerate(outputs):
            _DETERMINISM_BASELINE[f"output_{i}"] = out.clone().cpu().to(torch.float32)
        out_shapes = [('q_out', outputs[0].shape), ('k_out', outputs[1].shape),
                       ('v_out', outputs[2].shape), ('iq_out', outputs[3].shape),
                       ('ik_out', outputs[4].shape)]
        print(f"[DET] {tag} call #{call_idx}: baseline saved. outputs={out_shapes}")
        return

    if call_idx > _DETERMINISM_MAX_COMPARE:
        return

    # Compare inputs
    for name, tensor in inputs_dict.items():
        key = f"input_{name}"
        if key in _DETERMINISM_BASELINE:
            base = _DETERMINISM_BASELINE[key]
            cur = tensor.clone().cpu().to(torch.float32)
            if not torch.equal(base, cur):
                diff = (base - cur).abs()
                print(f"[DET] {tag} call #{call_idx}: INPUT '{name}' MISMATCH! "
                      f"max_diff={diff.max().item():.8f}, "
                      f"n_diff={(diff > 0).sum().item()}")

    # Compare outputs
    names = ["q_out", "k_out", "v_out", "index_q_out", "index_k_out"]
    all_match = True
    for i, (name, out) in enumerate(zip(names, outputs)):
        key = f"output_{i}"
        if key not in _DETERMINISM_BASELINE:
            continue
        base = _DETERMINISM_BASELINE[key]
        cur = out.clone().cpu().to(torch.float32)
        if not torch.equal(base, cur):
            diff = (base - cur).abs()
            n_diff = (diff > 0).sum().item()
            total = base.numel()
            max_diff = diff.max().item()
            print(f"[DET] {tag} call #{call_idx}: OUTPUT '{name}' MISMATCH! "
                  f"max_diff={max_diff:.8f}, n_diff={n_diff}/{total}")
            # Show first diff location
            if n_diff > 0:
                flat_diff = diff.flatten()
                first_idx = (flat_diff > 0).nonzero()[0].item()
                print(f"  first diff at flat idx {first_idx}: "
                      f"base={base.flatten()[first_idx].item():.8f}, "
                      f"cur={cur.flatten()[first_idx].item():.8f}")
            all_match = False
        else:
            print(f"[DET] {tag} call #{call_idx}: OUTPUT '{name}' identical")

    if all_match:
        print(f"[DET] {tag} call #{call_idx}: all outputs bitwise identical")

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    get_kv_quant_mode,
)
from vllm_ascend.attention.msa_m3_triton import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_index_decode,
    minimax_m3_index_score,
    minimax_m3_index_topk,
)
from vllm_ascend.attention.msa_m3_npu import (
    minimax_m3_sparse_attn as minimax_m3_sparse_attn_ascendc_legacy,
)
from vllm_ascend.attention.msa_m3_npu_new import (
    minimax_m3_sparse_attn as minimax_m3_sparse_attn_ascendc_pp8,
    minimax_m3_sparse_attn_decode as minimax_m3_sparse_attn_decode_ascendc,
)
from vllm_ascend.attention.msa_m3_ops import (
    minimax_m3_sparse_attn_torch as minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode_torch as minimax_m3_sparse_attn_decode,
)
from vllm_ascend.distributed.parallel_state import get_kv_head_replica_group
import vllm_ascend.ops.minimax_m3_sparse  # noqa: F401
from vllm_ascend.ops.linear import AscendColumnParallelLinear
from vllm_ascend.ops.linear_op import get_parallel_op


logger = init_logger(__name__)

_SPARSE_ATTN_NEW_OP_PP_SIZE = 8

_SPARSE_ATTN_LOGGED = False
FP8_E4M3_MAX = 448.0
_INDEX_LOCAL_CP_QUERY_TILE_SIZE = 128



def _resolve_indexer_kv_dtype(vllm_config: VllmConfig) -> str:
    """Resolve indexer cache dtype without requiring upstream AttentionConfig field.

    Priority:
      1. ``attention_config.indexer_kv_dtype`` (newer vLLM)
      2. ``additional_config["indexer_kv_dtype"]`` (Ascend-only, no vLLM change)
      3. default ``"bf16"``
    """
    attn = getattr(vllm_config, "attention_config", None)
    dtype = getattr(attn, "indexer_kv_dtype", None) if attn is not None else None
    if dtype is None:
        additional = getattr(vllm_config, "additional_config", None) or {}
        dtype = additional.get("indexer_kv_dtype", "bf16")
    return str(dtype)


def _scatter_index_cache(
    cache: torch.Tensor,
    updates: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Write index keys while safely ignoring graph/parallel padding slots.

    ``updates`` must already match ``cache.dtype`` (cast at the call site /
    before insert). Uses CANN ScatterPaCache
    (``torch_npu.npu_scatter_pa_cache``) scene-1:
    ``key [T,1,D]`` -> ``keyCache [B, block_size, 1, D]``.

    For ``float8_e4m3fn``, the PA scatter op accepts e4m3 natively (no uint8
    bitcast). Padding slots (``slot < 0``) are ignored by the op.
    """
    slots = slot_mapping.reshape(-1)
    if slots.numel() == 0:
        return

    updates = updates.reshape(slots.shape[0], cache.shape[-1])
    if updates.dtype != cache.dtype:
        updates = updates.to(cache.dtype)

    head_dim = cache.shape[-1]
    slots = slots.contiguous()
    num_tokens = slots.numel()
    if cache.ndim == 2:
        num_blocks = cache.shape[0] // SPARSE_BLOCK_SIZE
        pa_cache = cache.view(num_blocks, SPARSE_BLOCK_SIZE, 1, head_dim)
    elif cache.ndim == 3:
        pa_cache = cache.unsqueeze(2)
    elif cache.ndim == 4:
        pa_cache = cache
    else:
        raise ValueError(f"Unexpected index cache ndim: {cache.ndim}")
    key = updates.reshape(num_tokens, 1, head_dim).contiguous()
    torch_npu.npu_scatter_pa_cache(key, slots, key_cache=pa_cache)


def _select_num_idx_from_topk(topk_idx: torch.Tensor) -> torch.Tensor:
    return (topk_idx >= 0).sum(dim=-1).to(torch.int32).contiguous()


def _as_index_triton_kv_cache(
    index_kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    if isinstance(index_kv_cache, (tuple, list)):
        index_kv_cache = index_kv_cache[0]
    if index_kv_cache.ndim == 5 and index_kv_cache.shape[0] == 2:
        index_kv_cache = index_kv_cache[0]
    if index_kv_cache.ndim == 4:
        if index_kv_cache.shape[2] != 1:
            raise ValueError(
                f"Unexpected index cache shape: {tuple(index_kv_cache.shape)}"
            )
        index_kv_cache = index_kv_cache.squeeze(2)
    if index_kv_cache.ndim != 3:
        raise ValueError(f"Unexpected index cache ndim: {index_kv_cache.ndim}")
    return index_kv_cache


def _active_decode_num_reqs(
    num_decodes: int,
    num_decode_tokens: int,
    decode_query_len: int,
) -> int:
    """Return the number of real decode requests, ignoring FIA/graph padding."""
    if decode_query_len <= 0:
        return 0
    return min(num_decodes, num_decode_tokens // decode_query_len)


def _active_prefill_num_reqs(
    num_prefills: int,
    num_prefill_tokens: int,
    query_start_loc_cpu: torch.Tensor,
    num_decodes: int,
) -> int:
    """Return real prefill requests, ignoring FIA/SP tail padding segments."""
    if num_prefills <= 0 or num_prefill_tokens <= 0:
        return 0
    qsl_cpu = query_start_loc_cpu.detach().cpu()
    num_reqs_fia = int(qsl_cpu.shape[0] - 1)
    active = 0
    tokens_accounted = 0
    for i in range(num_decodes, min(num_reqs_fia, num_decodes + num_prefills)):
        query_len = int(qsl_cpu[i + 1] - qsl_cpu[i])
        if query_len <= 0:
            continue
        if tokens_accounted + query_len > num_prefill_tokens:
            break
        tokens_accounted += query_len
        active += 1
    if active > 0:
        return active
    return min(1, num_prefills)


def _build_local_cp_query_tile_layout(
    query_lens_cpu: torch.Tensor,
    local_cp_size: int,
    local_cp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pack round-robin query tiles as independent local pseudo requests.

    Returns query indices into the original flattened prefill tensor, original
    request indices and offsets for every pseudo request, its cu-seqlens, and
    the maximum local pseudo-request length.
    """
    query_index_parts: list[torch.Tensor] = []
    request_indices: list[int] = []
    request_offsets: list[int] = []
    local_cu_seqlens = [0]
    max_local_query_len = 0
    query_base = 0

    for request_index, query_len in enumerate(query_lens_cpu.tolist()):
        num_tiles = (query_len + _INDEX_LOCAL_CP_QUERY_TILE_SIZE - 1) // _INDEX_LOCAL_CP_QUERY_TILE_SIZE
        for tile_index in range(local_cp_rank, num_tiles, local_cp_size):
            request_offset = tile_index * _INDEX_LOCAL_CP_QUERY_TILE_SIZE
            tile_len = min(
                _INDEX_LOCAL_CP_QUERY_TILE_SIZE,
                query_len - request_offset,
            )
            query_index_parts.append(
                torch.arange(
                    query_base + request_offset,
                    query_base + request_offset + tile_len,
                    dtype=torch.int64,
                )
            )
            request_indices.append(request_index)
            request_offsets.append(request_offset)
            local_cu_seqlens.append(local_cu_seqlens[-1] + tile_len)
            max_local_query_len = max(max_local_query_len, tile_len)
        query_base += query_len

    query_indices = (
        torch.cat(query_index_parts)
        if query_index_parts
        else torch.empty(0, dtype=torch.int64)
    )
    return (
        query_indices,
        torch.tensor(request_indices, dtype=torch.int64),
        torch.tensor(request_offsets, dtype=torch.int32),
        torch.tensor(local_cu_seqlens, dtype=torch.int32),
        max_local_query_len,
    )


class AscendMiniMaxM3IndexerBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16, torch.float16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_name() -> str:
        return "ASCEND_MINIMAX_M3_SPARSE_INDEXER"

    @staticmethod
    def get_impl_cls() -> type["AscendMiniMaxM3IndexerImpl"]:
        return AscendMiniMaxM3IndexerImpl

    @staticmethod
    def get_builder_cls() -> type["AscendMiniMaxM3IndexerMetadataBuilder"]:
        return AscendMiniMaxM3IndexerMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [128]

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del num_kv_heads, cache_dtype_str
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            raise NotImplementedError
        return (0, 1, 2)


class AscendMiniMaxM3IndexerCache(nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        prefix: str,
        cache_config: CacheConfig | None = None,
        indexer_kv_dtype: str = "bf16",
    ) -> None:
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        if indexer_kv_dtype in ("fp8", "fp8_e4m3"):
            self.dtype = torch.float8_e4m3fn
        elif indexer_kv_dtype == "bf16":
            self.dtype = torch.bfloat16
        else:
            raise NotImplementedError(
                f"indexer_kv_dtype={indexer_kv_dtype!r} is not supported "
                "(only 'bf16' or 'fp8'/'fp8_e4m3')."
            )
        self.indexer_kv_dtype = indexer_kv_dtype
        self.prefix = prefix
        self.cache_config = cache_config
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # Key-only: MLAAttentionSpec budgets one vector/token (not 2x for K+V).
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
        )

    def forward(self) -> None: ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return AscendMiniMaxM3IndexerBackend


@dataclass
class AscendMiniMaxM3IndexerLocalCPMetadata:
    query_indices: torch.Tensor
    cu_seqlens_q: torch.Tensor
    seq_lens: torch.Tensor
    context_lens: torch.Tensor
    block_table: torch.Tensor
    max_query_len: int


@dataclass
class AscendMiniMaxM3IndexerPrefillMetadata:
    cu_seqlens_q: torch.Tensor
    seq_lens: torch.Tensor
    context_lens: torch.Tensor
    block_table: torch.Tensor
    max_query_len: int
    max_seq_len: int
    local_cp: AscendMiniMaxM3IndexerLocalCPMetadata | None = None


@dataclass
class AscendMiniMaxM3IndexerDecodeMetadata:
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    max_seq_len: int
    decode_query_len: int
    max_decode_query_len: int


@dataclass
class AscendMiniMaxM3IndexerMetadata(AttentionMetadata):
    seq_lens: torch.Tensor
    max_seq_len: int
    slot_mapping: torch.Tensor
    num_actual_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    prefill: AscendMiniMaxM3IndexerPrefillMetadata | None = None
    decode: AscendMiniMaxM3IndexerDecodeMetadata | None = None


class AscendMiniMaxM3IndexerMetadataBuilder(
    AttentionMetadataBuilder[AscendMiniMaxM3IndexerMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.device = device
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        self.max_decode_query_len = self.reorder_batch_threshold
        self.context_len_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=device,
        )
        total_num_kv_heads = vllm_config.model_config.get_total_num_kv_heads()
        tensor_parallel_size = vllm_config.parallel_config.tensor_parallel_size
        self.local_cp_size = 1
        self.local_cp_rank = 0
        if (
            isinstance(total_num_kv_heads, int)
            and total_num_kv_heads > 0
            and tensor_parallel_size > total_num_kv_heads
            and tensor_parallel_size % total_num_kv_heads == 0
        ):
            self.local_cp_size = tensor_parallel_size // total_num_kv_heads
            local_cp_group = get_kv_head_replica_group()
            if local_cp_group.world_size != self.local_cp_size:
                raise RuntimeError(
                    "M3 indexer metadata head-replica group size mismatch: "
                    f"expected {self.local_cp_size}, got {local_cp_group.world_size}"
                )
            self.local_cp_rank = local_cp_group.rank_in_group

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendMiniMaxM3IndexerMetadata:
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor
        qsl_cpu = common_attn_metadata.query_start_loc_cpu

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )

        prefill_metadata: AscendMiniMaxM3IndexerPrefillMetadata | None = None
        active_prefills = 0
        if num_prefills > 0:
            active_prefills = _active_prefill_num_reqs(
                num_prefills, num_prefill_tokens, qsl_cpu, num_decodes
            )
            prefill_end = num_decodes + active_prefills
            prefill_query_lens_cpu = (
                qsl_cpu[num_decodes + 1 : prefill_end + 1]
                - qsl_cpu[num_decodes:prefill_end]
            )
            prefill_seq_lens_cpu = (
                seq_lens[num_decodes:prefill_end]
                .detach()
                .cpu()
                .to(torch.int32)
            )
            prefill_context_lens_cpu = (
                prefill_seq_lens_cpu - prefill_query_lens_cpu
            )
            prefill_context_lens = self.context_len_buffer[num_decodes:prefill_end]
            prefill_context_lens.copy_(
                prefill_context_lens_cpu.to(
                    device=self.context_len_buffer.device,
                    dtype=torch.int32,
                    non_blocking=True,
                ),
                non_blocking=True,
            )
            cu_seqlens_q = (
                query_start_loc[num_decodes : prefill_end + 1] - num_decode_tokens
            ).to(torch.int32)
            local_cp_metadata: AscendMiniMaxM3IndexerLocalCPMetadata | None = None
            if self.local_cp_size > 1:
                (
                    local_query_indices_cpu,
                    local_request_indices_cpu,
                    local_request_offsets_cpu,
                    local_cu_seqlens_cpu,
                    local_max_query_len,
                ) = _build_local_cp_query_tile_layout(
                    prefill_query_lens_cpu,
                    self.local_cp_size,
                    self.local_cp_rank,
                )
                local_query_indices = local_query_indices_cpu.to(
                    device=self.device,
                    non_blocking=True,
                )
                local_request_indices = local_request_indices_cpu.to(
                    device=self.device,
                    non_blocking=True,
                )
                local_request_offsets = local_request_offsets_cpu.to(
                    device=self.device,
                    non_blocking=True,
                )
                local_cu_seqlens = local_cu_seqlens_cpu.to(
                    device=self.device,
                    non_blocking=True,
                )
                prefill_seq_lens = seq_lens[num_decodes:prefill_end]
                prefill_block_table = block_table[num_decodes:prefill_end]
                num_local_tiles = local_request_indices.shape[0]
                # seq_lens is addressed as a flat pointer by the Triton kernel,
                # so it must be materialized rather than stride-0 expanded.
                local_seq_lens = torch.index_select(
                    prefill_seq_lens,
                    0,
                    local_request_indices,
                )
                if active_prefills == 1:
                    # block_table supplies its row stride to the kernel, so a
                    # stride-0 view safely avoids duplicating a 1M-context table.
                    local_block_table = prefill_block_table.expand(
                        num_local_tiles,
                        -1,
                    )
                else:
                    local_block_table = torch.index_select(
                        prefill_block_table,
                        0,
                        local_request_indices,
                    )
                local_context_lens = torch.index_select(
                    prefill_context_lens,
                    0,
                    local_request_indices,
                ) + local_request_offsets
                local_cp_metadata = AscendMiniMaxM3IndexerLocalCPMetadata(
                    query_indices=local_query_indices,
                    cu_seqlens_q=local_cu_seqlens,
                    seq_lens=local_seq_lens,
                    context_lens=local_context_lens,
                    block_table=local_block_table,
                    max_query_len=local_max_query_len,
                )
            prefill_metadata = AscendMiniMaxM3IndexerPrefillMetadata(
                cu_seqlens_q=cu_seqlens_q,
                seq_lens=seq_lens[num_decodes:prefill_end],
                context_lens=prefill_context_lens,
                block_table=block_table[num_decodes:prefill_end],
                max_query_len=common_attn_metadata.max_query_len,
                max_seq_len=common_attn_metadata.max_seq_len,
                local_cp=local_cp_metadata,
            )

        decode_metadata: AscendMiniMaxM3IndexerDecodeMetadata | None = None
        active_decodes = 0
        if num_decodes > 0:
            query_lens_cpu = qsl_cpu[1 : num_decodes + 1] - qsl_cpu[:num_decodes]
            decode_query_len = int(query_lens_cpu[0].item())
            active_decodes = _active_decode_num_reqs(
                num_decodes, num_decode_tokens, decode_query_len
            )
            decode_metadata = AscendMiniMaxM3IndexerDecodeMetadata(
                seq_lens=seq_lens[:active_decodes],
                block_table=block_table[:active_decodes],
                max_seq_len=common_attn_metadata.max_seq_len,
                decode_query_len=decode_query_len,
                max_decode_query_len=self.max_decode_query_len,
            )

        return AscendMiniMaxM3IndexerMetadata(
            seq_lens=seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_actual_tokens=num_tokens,
            num_decodes=active_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=active_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )


class AscendMiniMaxM3IndexerImpl(nn.Module):
    def __init__(
        self,
        *,
        num_kv_heads: int,
        scale: float,
        topk_blocks: int,
        sparse_block_size: int,
        num_index_heads: int,
        index_head_dim: int,
        prefix: str,
        init_blocks: int = 0,
        local_blocks: int = 0,
        cache_config: CacheConfig | None = None,
        indexer_kv_dtype: str = "bf16",
        head_replica_size: int = 1,
    ) -> None:
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.scale = scale
        self.topk_blocks = topk_blocks
        self.block_size = sparse_block_size
        self.init_blocks = init_blocks
        self.local_blocks = local_blocks
        self.num_index_heads = num_index_heads
        self.index_head_dim = index_head_dim
        self.local_cp_size = head_replica_size
        self.local_cp_rank = 0
        self.local_cp_group = None
        if self.local_cp_size > 1:
            self.local_cp_group = get_kv_head_replica_group()
            if self.local_cp_group.world_size != self.local_cp_size:
                raise RuntimeError(
                    "M3 indexer head-replica group size mismatch: "
                    f"expected {self.local_cp_size}, got {self.local_cp_group.world_size}"
                )
            self.local_cp_rank = self.local_cp_group.rank_in_group
        self.index_cache = AscendMiniMaxM3IndexerCache(
            head_dim=index_head_dim,
            prefix=f"{prefix}.index_cache",
            cache_config=cache_config,
            indexer_kv_dtype=indexer_kv_dtype,
        )

    def forward(
        self,
        index_query: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None, None, None
        index_md = attn_metadata[self.index_cache.prefix]
        assert isinstance(index_md, AscendMiniMaxM3IndexerMetadata)
        num_tokens = index_md.num_actual_tokens
        nd = index_md.num_decode_tokens
        iq = index_query[:num_tokens].view(
            -1, self.num_index_heads, self.index_head_dim
        )
        kv = _as_index_triton_kv_cache(self.index_cache.kv_cache)

        decode_topk: torch.Tensor | None = None
        prefill_topk: torch.Tensor | None = None
        decode_select_num_idx: torch.Tensor | None = None
        if index_md.num_decodes > 0:
            d = index_md.decode
            assert d is not None
            decode_topk, decode_select_num_idx = minimax_m3_index_decode(
                iq[:nd],
                kv,
                d.block_table,
                d.seq_lens,
                d.max_seq_len,
                self.topk_blocks,
                self.init_blocks,
                self.local_blocks,
                self.num_kv_heads,
                d.decode_query_len,
                d.max_decode_query_len,
                sm_scale=self.scale,
            )
        if index_md.num_prefills > 0:
            p = index_md.prefill
            assert p is not None
            local_cp = p.local_cp
            if local_cp is None:
                score = minimax_m3_index_score(
                    iq[nd:],
                    kv,
                    p.block_table,
                    p.cu_seqlens_q,
                    p.seq_lens,
                    p.context_lens,
                    p.max_query_len,
                    p.max_seq_len,
                    self.num_kv_heads,
                )
                prefill_topk = minimax_m3_index_topk(
                    score,
                    p.cu_seqlens_q,
                    p.context_lens,
                    p.max_query_len,
                    self.topk_blocks,
                    self.init_blocks,
                    self.local_blocks,
                )
            else:
                num_prefill_tokens = iq.shape[0] - nd
                prefill_topk = torch.full(
                    (
                        self.num_index_heads,
                        num_prefill_tokens,
                        self.topk_blocks,
                    ),
                    -1,
                    dtype=torch.int32,
                    device=iq.device,
                )
                if local_cp.query_indices.numel() > 0:
                    prefill_iq = iq[nd:]
                    if prefill_iq.dtype == torch.float8_e4m3fn:
                        # aclnnIndexSelect does not support FP8 E4M3. Both
                        # dtypes are one byte, so select the unchanged payload
                        # as uint8 and reinterpret it as FP8 afterwards.
                        local_iq = torch.index_select(
                            prefill_iq.view(torch.uint8),
                            0,
                            local_cp.query_indices,
                        ).view(torch.float8_e4m3fn)
                    else:
                        local_iq = torch.index_select(
                            prefill_iq,
                            0,
                            local_cp.query_indices,
                        )
                    local_score = minimax_m3_index_score(
                        local_iq,
                        kv,
                        local_cp.block_table,
                        local_cp.cu_seqlens_q,
                        local_cp.seq_lens,
                        local_cp.context_lens,
                        local_cp.max_query_len,
                        p.max_seq_len,
                        self.num_kv_heads,
                    )
                    local_topk = minimax_m3_index_topk(
                        local_score,
                        local_cp.cu_seqlens_q,
                        local_cp.context_lens,
                        local_cp.max_query_len,
                        self.topk_blocks,
                        self.init_blocks,
                        self.local_blocks,
                    )
                    prefill_topk.index_copy_(
                        1,
                        local_cp.query_indices,
                        local_topk,
                    )
                assert self.local_cp_group is not None
                # Every pseudo request is owned by one local-CP rank. The
                # remaining rows stay -1, so MAX restores the full top-k tensor.
                dist.all_reduce(
                    prefill_topk,
                    op=dist.ReduceOp.MAX,
                    group=self.local_cp_group.device_group,
                )
        return decode_topk, prefill_topk, decode_select_num_idx


class AscendMiniMaxM3Indexer(nn.Module):
    def __init__(
        self,
        *,
        num_kv_heads: int,
        scale: float,
        topk_blocks: int,
        sparse_block_size: int,
        num_index_heads: int,
        index_head_dim: int,
        prefix: str,
        init_blocks: int = 0,
        local_blocks: int = 0,
        cache_config: CacheConfig | None = None,
        indexer_kv_dtype: str = "bf16",
        head_replica_size: int = 1,
    ) -> None:
        super().__init__()
        self.impl = AscendMiniMaxM3IndexerImpl(
            num_kv_heads=num_kv_heads,
            scale=scale,
            topk_blocks=topk_blocks,
            sparse_block_size=sparse_block_size,
            num_index_heads=num_index_heads,
            index_head_dim=index_head_dim,
            prefix=prefix,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            cache_config=cache_config,
            indexer_kv_dtype=indexer_kv_dtype,
            head_replica_size=head_replica_size,
        )

    @property
    def index_cache(self) -> AscendMiniMaxM3IndexerCache:
        return self.impl.index_cache

    def forward(
        self,
        index_query: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        return self.impl(index_query)


class AscendMiniMaxM3SparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16, torch.float16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_name() -> str:
        return "ASCEND_MINIMAX_M3_SPARSE"

    @staticmethod
    def get_impl_cls() -> type["AscendMiniMaxM3SparseImpl"]:
        return AscendMiniMaxM3SparseImpl

    @staticmethod
    def get_builder_cls() -> type["AscendMiniMaxM3SparseMetadataBuilder"]:
        return AscendMiniMaxM3SparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [128]

    @classmethod
    def is_sparse(cls) -> bool:
        return True

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


@dataclass
class AscendMiniMaxM3SparsePrefillMetadata:
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    seq_lens: torch.Tensor
    actual_seq_lengths: torch.Tensor
    actual_seq_lengths_kv: torch.Tensor
    context_lens: torch.Tensor
    block_table: torch.Tensor
    max_query_len: int
    max_seq_len: int


@dataclass
class AscendMiniMaxM3SparseDecodeMetadata:
    seq_lens: torch.Tensor
    actual_seq_lengths: torch.Tensor
    actual_seq_lengths_kv: torch.Tensor
    block_table: torch.Tensor
    max_seq_len: int
    decode_query_len: int


@dataclass
class AscendMiniMaxM3SparseMetadata(AttentionMetadata):
    seq_lens: torch.Tensor
    max_seq_len: int
    slot_mapping: torch.Tensor
    num_actual_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    prefill: AscendMiniMaxM3SparsePrefillMetadata | None = None
    decode: AscendMiniMaxM3SparseDecodeMetadata | None = None


class AscendMiniMaxM3SparseMetadataBuilder(
    AttentionMetadataBuilder[AscendMiniMaxM3SparseMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        self.context_len_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendMiniMaxM3SparseMetadata:
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor
        qsl_cpu = common_attn_metadata.query_start_loc_cpu

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )

        prefill_metadata: AscendMiniMaxM3SparsePrefillMetadata | None = None
        active_prefills = 0
        if num_prefills > 0:
            active_prefills = _active_prefill_num_reqs(
                num_prefills, num_prefill_tokens, qsl_cpu, num_decodes
            )
            prefill_end = num_decodes + active_prefills
            prefill_kv_lens = seq_lens[num_decodes:prefill_end]
            prefill_cu_seqlens_q = (
                query_start_loc[num_decodes : prefill_end + 1] - num_decode_tokens
            ).to(torch.int32)
            prefill_cu_seqlens_k = torch.empty(
                active_prefills + 1, dtype=torch.int32, device=seq_lens.device
            )
            prefill_cu_seqlens_k[0] = 0
            torch.cumsum(prefill_kv_lens, dim=0, out=prefill_cu_seqlens_k[1:])
            prefill_query_lens_cpu = (
                qsl_cpu[num_decodes + 1 : prefill_end + 1]
                - qsl_cpu[num_decodes:prefill_end]
            )
            prefill_context_lens = self.context_len_buffer[num_decodes:prefill_end]
            prefill_context_lens.copy_(
                (
                    prefill_kv_lens.detach().cpu() - prefill_query_lens_cpu
                ).to(
                    device=self.context_len_buffer.device,
                    dtype=torch.int32,
                    non_blocking=True,
                ),
                non_blocking=True,
            )
            prefill_actual_seq_lengths = (
                prefill_cu_seqlens_q[1:] - prefill_cu_seqlens_q[:-1]
            ).to(torch.int64).contiguous()
            prefill_actual_seq_lengths_kv = prefill_kv_lens.to(
                torch.int64
            ).contiguous()
            prefill_metadata = AscendMiniMaxM3SparsePrefillMetadata(
                cu_seqlens_q=prefill_cu_seqlens_q,
                cu_seqlens_k=prefill_cu_seqlens_k,
                seq_lens=prefill_kv_lens,
                actual_seq_lengths=prefill_actual_seq_lengths,
                actual_seq_lengths_kv=prefill_actual_seq_lengths_kv,
                context_lens=prefill_context_lens,
                block_table=block_table[num_decodes:prefill_end].contiguous(),
                max_query_len=common_attn_metadata.max_query_len,
                max_seq_len=common_attn_metadata.max_seq_len,
            )

        decode_metadata: AscendMiniMaxM3SparseDecodeMetadata | None = None
        active_decodes = 0
        if num_decodes > 0:
            query_lens_cpu = qsl_cpu[1 : num_decodes + 1] - qsl_cpu[:num_decodes]
            decode_query_len = int(query_lens_cpu[0].item())
            active_decodes = _active_decode_num_reqs(
                num_decodes, num_decode_tokens, decode_query_len
            )
            decode_seq_lens = seq_lens[:active_decodes]
            decode_actual_seq_lengths = torch.full(
                (active_decodes,),
                decode_query_len,
                dtype=torch.int32,
                device=seq_lens.device,
            )
            decode_actual_seq_lengths_kv = decode_seq_lens.to(
                torch.int64
            ).contiguous()
            decode_metadata = AscendMiniMaxM3SparseDecodeMetadata(
                seq_lens=decode_seq_lens,
                actual_seq_lengths=decode_actual_seq_lengths,
                actual_seq_lengths_kv=decode_actual_seq_lengths_kv,
                block_table=block_table[:active_decodes].contiguous(),
                max_seq_len=common_attn_metadata.max_seq_len,
                decode_query_len=decode_query_len,
            )

        return AscendMiniMaxM3SparseMetadata(
            seq_lens=seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_actual_tokens=num_tokens,
            num_decodes=active_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=active_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )


class AscendMiniMaxM3SparseImpl(AttentionImplBase[AscendMiniMaxM3SparseMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        kv_cache_dtype: str = "auto",
        *,
        topk_blocks: int,
        sparse_block_size: int,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.topk_blocks = topk_blocks
        self.block_size = sparse_block_size
        try:
            pp_size = (
                get_current_vllm_config().parallel_config.pipeline_parallel_size
            )
        except Exception:
            pp_size = 1
        self.minimax_m3_sparse_attn_ascendc = (
            minimax_m3_sparse_attn_ascendc_pp8
            if pp_size == _SPARSE_ATTN_NEW_OP_PP_SIZE
            else minimax_m3_sparse_attn_ascendc_legacy
        )
        self._dequant_scale_buf: torch.Tensor | None = None

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_idx: tuple[
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
        ],
        output: torch.Tensor,
    ) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return output
        main_md = attn_metadata[layer.layer_name]
        assert isinstance(main_md, AscendMiniMaxM3SparseMetadata)
        decode_topk, prefill_topk, decode_select_num_idx = topk_idx

        nd = main_md.num_decode_tokens
        num_tokens = main_md.num_actual_tokens
        hd = self.head_size
        q = query[:num_tokens].view(-1, self.num_heads, hd)
        out = output[:num_tokens].view(-1, self.num_heads, hd)

        # if main_md.num_decodes > 0:
        #     d = main_md.decode
        #     assert d is not None and decode_topk is not None
        #     minimax_m3_sparse_attn_decode(
        #         q[:nd],
        #         kv_cache,
        #         decode_topk,
        #         d.block_table,
        #         d.seq_lens,
        #         self.num_kv_heads,
        #         self.scale,
        #         out[:nd],
        #         d.decode_query_len,
        #     )

        # if main_md.num_prefills > 0:
        #     p = main_md.prefill
        #     assert p is not None and prefill_topk is not None
        #     minimax_m3_sparse_attn(
        #         q[nd:],
        #         kv_cache,
        #         prefill_topk,
        #         p.block_table,
        #         p.cu_seqlens_q,
        #         p.seq_lens,
        #         p.context_lens,
        #         p.max_query_len,
        #         self.num_kv_heads,
        #         self.scale,
        #         out[nd:],
        #     )
        # key_cache, value_cache = kv_cache[0], kv_cache[1]

        # if main_md.num_decodes > 0:
        #     d = main_md.decode
        #     assert d is not None and decode_topk is not None
        #     decode_topk = decode_topk.contiguous()
        #     decode_out = torch.ops._C_ascend.npu_sparse_attention_score(
        #         q[:nd],
        #         key_cache,
        #         value_cache,
        #         decode_topk,
        #         d.block_table,
        #         self.num_kv_heads,
        #         self.scale,
        #         self.block_size,
        #         self.topk_blocks,
        #         4,
        #         select_num_idx=None,
        #         actual_seq_lengths=d.actual_seq_lengths,
        #         actual_seq_lengths_kv=d.actual_seq_lengths_kv,
        #     )
        #     output[:nd].view(-1, self.num_heads, hd).copy_(decode_out)

        # if main_md.num_prefills > 0:
        #     p = main_md.prefill
        #     assert p is not None and prefill_topk is not None
        #     prefill_q = q[nd:num_tokens]
        #     prefill_topk = prefill_topk.contiguous()
        #     prefill_out = torch.ops._C_ascend.npu_sparse_attention_score(
        #         prefill_q,
        #         key_cache,
        #         value_cache,
        #         prefill_topk,
        #         p.block_table,
        #         self.num_kv_heads,
        #         self.scale,
        #         self.block_size,
        #         self.topk_blocks,
        #         4,
        #         select_num_idx=None,
        #         actual_seq_lengths=p.actual_seq_lengths,
        #         actual_seq_lengths_kv=p.actual_seq_lengths_kv,
        #     )
        #     output[nd:num_tokens].view(-1, self.num_heads, hd).copy_(prefill_out)

        if main_md.num_decodes > 0:
            d = main_md.decode
            assert d is not None and decode_topk is not None
            assert decode_select_num_idx is not None
            if self._dequant_scale_buf is None:
                self._dequant_scale_buf = torch.ones(
                    1, dtype=torch.float32, device=query.device
                )
            minimax_m3_sparse_attn_decode_ascendc(
                q[:nd],
                kv_cache,
                decode_topk,
                decode_select_num_idx,
                d.block_table,
                d.actual_seq_lengths,
                d.seq_lens,
                self.num_kv_heads,
                self.scale,
                out[:nd],
                d.decode_query_len,
                self._dequant_scale_buf,
                block_size=self.block_size,
            )

        if main_md.num_prefills > 0:
            p = main_md.prefill
            assert p is not None and prefill_topk is not None
            self.minimax_m3_sparse_attn_ascendc(
                q[nd:],
                kv_cache,
                prefill_topk,
                p.block_table,
                p.cu_seqlens_q,
                p.seq_lens,
                p.context_lens,
                p.max_query_len,
                self.num_kv_heads,
                self.scale,
                out[nd:],
                block_size=self.block_size,
            )
        return output

        


class AscendMinimaxM3QKVParallelLinearWithIndexer(QKVParallelLinear):
    """Fused [q | k | v | index_q | index_k] column-parallel GEMM for M3 sparse layers."""

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        total_num_index_heads: int,
        index_head_size: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        assert total_num_index_heads == total_num_kv_heads, (
            "AscendMinimaxM3QKVParallelLinearWithIndexer requires "
            "total_num_index_heads == total_num_kv_heads"
        )
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.v_head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.total_num_index_heads = total_num_index_heads
        self.index_head_size = index_head_size

        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        self.num_index_heads = self.num_kv_heads

        q = self.num_heads * self.head_size
        kv = self.num_kv_heads * self.head_size
        iq = self.num_index_heads * self.index_head_size
        ik = self.index_head_size
        self.output_sizes = [
            q * tp_size,
            kv * tp_size,
            kv * tp_size,
            iq * tp_size,
            ik * tp_size,
        ]

        self.custom_op, _, _ = get_parallel_op(False, prefix, self, "column")
        AscendColumnParallelLinear.__init__(
            self,
            input_size=self.hidden_size,
            output_size=sum(self.output_sizes),
            bias=bias,
            gather_output=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def forward(self, input_):
        if self.custom_op is not None:
            return self.custom_op.apply(input_)
        return super().forward(input_)

    def validate_shard_id(self, loaded_shard_id: str | None) -> None:
        if loaded_shard_id is None:
            return
        if loaded_shard_id not in ("q", "k", "v", "index_q", "index_k"):
            raise ValueError(
                "Shard id for AscendMinimaxM3QKVParallelLinearWithIndexer must be "
                "one of 'q', 'k', 'v', 'index_q', 'index_k'; got "
                f"{loaded_shard_id}."
            )

    def _get_shard_offset_mapping(self, loaded_shard_id: str) -> int | None:
        h = self.head_size
        nq, nkv, nidx = self.num_heads, self.num_kv_heads, self.num_index_heads
        return {
            "q": 0,
            "k": nq * h,
            "v": (nq + nkv) * h,
            "index_q": (nq + 2 * nkv) * h,
            "index_k": (nq + 2 * nkv + nidx) * h,
        }.get(loaded_shard_id)

    def _get_shard_size_mapping(self, loaded_shard_id: str) -> int | None:
        h = self.head_size
        return {
            "q": self.num_heads * h,
            "k": self.num_kv_heads * h,
            "v": self.num_kv_heads * h,
            "index_q": self.num_index_heads * h,
            "index_k": self.index_head_size,
        }.get(loaded_shard_id)

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight,
        loaded_shard_id: str | None = None,
    ) -> None:
        self.validate_shard_id(loaded_shard_id)
        assert loaded_shard_id in ("q", "k", "v", "index_q", "index_k")

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)
        assert shard_offset is not None and shard_size is not None
        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = getattr(self, "weight_block_size", None)
            shard_size, shard_offset = adjust_block_scale_shard(
                weight_block_size, shard_size, shard_offset
            )

        num_heads = (
            self.tp_size if loaded_shard_id == "index_k" else self.num_kv_head_replicas
        )
        param.load_qkv_weight(
            loaded_weight=loaded_weight,
            num_heads=num_heads,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
            tp_rank=self.tp_rank,
        )

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight,
        loaded_shard_id: str | None = None,
    ) -> None:
        self.validate_shard_id(loaded_shard_id)
        assert loaded_shard_id in ("q", "k", "v", "index_q", "index_k")
        output_dim = getattr(param, "output_dim", None)
        assert output_dim is not None

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)
        assert shard_offset is not None and shard_size is not None
        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = getattr(self, "weight_block_size", None)
            shard_size, shard_offset = adjust_block_scale_shard(
                weight_block_size, shard_size, shard_offset
            )

        param_data = param.data.narrow(output_dim, shard_offset, shard_size)
        if loaded_shard_id == "q":
            shard_rank = self.tp_rank
        elif loaded_shard_id == "index_k":
            shard_rank = 0
        else:
            shard_rank = self.tp_rank // self.num_kv_head_replicas
        loaded_weight = loaded_weight.narrow(
            output_dim, shard_rank * shard_size, shard_size
        )
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class MiniMaxM3SparseAttention(nn.Module, AttentionLayerBase):
    """Block-sparse attention with lightning indexer on Ascend."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rotary_dim: int,
        rope_parameters: dict[str, Any] | None = None,
        attn_window_size: int | None = None,
        max_position_embeddings: int = 8192,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        sparse_cfg: dict[str, Any] | None = None,
        disable_index_value: bool = False,
    ) -> None:
        super().__init__()
        assert sparse_cfg is not None
        self.hidden_size = hidden_size
        self.disable_index_value = disable_index_value

        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        self.total_idx_heads = sparse_cfg["sparse_num_index_heads"]
        self.idx_head_dim = sparse_cfg["sparse_index_dim"]
        assert self.total_idx_heads == self.total_num_kv_heads, (
            "MiniMax M3 sparse attention requires "
            "sparse_num_index_heads == num_key_value_heads"
        )
        self.num_idx_heads = self.num_kv_heads
        self.index_q_size = self.num_idx_heads * self.idx_head_dim

        self.qkv_proj = AscendMinimaxM3QKVParallelLinearWithIndexer(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            self.total_idx_heads,
            self.idx_head_dim,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=True,
            prefix=f"{prefix}.o_proj",
        )

        if rope_parameters is not None and "partial_rotary_factor" not in rope_parameters:
            rope_parameters["partial_rotary_factor"] = rotary_dim / self.head_dim
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
        )

        self.index_q_norm = GemmaRMSNorm(self.idx_head_dim, eps=rms_norm_eps)
        self.index_k_norm = GemmaRMSNorm(self.idx_head_dim, eps=rms_norm_eps)

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)

        vllm_config = get_current_vllm_config()
        self.layer_name = f"{prefix}.attn"
        self.indexer_kv_dtype = _resolve_indexer_kv_dtype(vllm_config)
        self.kv_cache_dtype = (
            cache_config.cache_dtype if cache_config is not None else "auto"
        )
        # self.kv_cache_dtype = "bfloat16"
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, vllm_config.model_config
        )
        self.attn_backend = AscendMiniMaxM3SparseBackend
        self.impl = AscendMiniMaxM3SparseImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype,
            topk_blocks=sparse_cfg["sparse_topk_blocks"],
            sparse_block_size=sparse_cfg["sparse_block_size"],
        )
        self.topk_blocks = sparse_cfg["sparse_topk_blocks"]
        self.sparse_block_size = sparse_cfg["sparse_block_size"]
        self.indexer = AscendMiniMaxM3Indexer(
            num_kv_heads=self.num_kv_heads,
            scale=self.scaling,
            topk_blocks=self.topk_blocks,
            sparse_block_size=self.sparse_block_size,
            num_index_heads=self.num_idx_heads,
            index_head_dim=self.idx_head_dim,
            prefix=self.layer_name,
            init_blocks=sparse_cfg.get("sparse_init_block", 0),
            local_blocks=sparse_cfg.get("sparse_local_block", 0),
            cache_config=cache_config,
            indexer_kv_dtype=self.indexer_kv_dtype,
            head_replica_size=self.qkv_proj.num_kv_head_replicas,
        )

        compilation_config = vllm_config.compilation_config
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self
        self.kv_cache = torch.tensor([])

        global _SPARSE_ATTN_LOGGED
        if not _SPARSE_ATTN_LOGGED:
            logger.warning(
                "MiniMax M3 sparse attention enabled "
                "(topk_blocks=%d, block_size=%d, local_index_cp=%d)",
                sparse_cfg["sparse_topk_blocks"],
                sparse_cfg["sparse_block_size"],
                self.qkv_proj.num_kv_head_replicas,
            )
            _SPARSE_ATTN_LOGGED = True

    def get_attn_backend(self) -> type[AscendMiniMaxM3SparseBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    def _insert_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        index_key: torch.Tensor,
    ) -> None:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return
        main_meta = attn_metadata[self.layer_name]
        index_meta = attn_metadata[self.indexer.index_cache.prefix]
        assert isinstance(main_meta, AscendMiniMaxM3SparseMetadata)
        assert isinstance(index_meta, AscendMiniMaxM3IndexerMetadata)

        key_cache, value_cache = self.kv_cache[0], self.kv_cache[1]
        num_tokens = main_meta.num_actual_tokens
        k_insert = key[:num_tokens].view(-1, self.num_kv_heads, self.head_dim)
        v_insert = value[:num_tokens].view(-1, self.num_kv_heads, self.head_dim)
        if k_insert.dtype == torch.float8_e4m3fn:
            k_fp8 = k_insert
            v_fp8 = v_insert
        else:
            k_fp8 = k_insert.clamp(min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX).to(
                torch.float8_e4m3fn
            )
            v_fp8 = v_insert.clamp(min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX).to(
                torch.float8_e4m3fn
            )
        from vllm_ascend.device.device_op import DeviceOperator

        DeviceOperator.reshape_and_cache(
            k_fp8,
            v_fp8,
            key_cache,
            value_cache,
            main_meta.slot_mapping[:num_tokens],
        )

        idx_cache = self.indexer.index_cache.kv_cache
        if isinstance(idx_cache, (tuple, list)):
            idx_cache = idx_cache[0]
        flat = idx_cache.view(-1, self.idx_head_dim)
        cache_dtype = flat.dtype
        src = index_key[:num_tokens]
        if src.ndim == 3:
            src = src.reshape(num_tokens, -1)
        src = src.to(dtype=cache_dtype)
        _scatter_index_cache(
            flat,
            src,
            index_meta.slot_mapping[:num_tokens],
        )

    def _sparse_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv, _ = self.qkv_proj(hidden_states)

        #* 融合路径
        if (
            HAS_TRITON
            and qkv.device.type == "npu"
            and qkv.dtype == torch.bfloat16
            and positions.ndim == 1
            and getattr(self.rotary_emb, "is_neox_style", True)
        ):
            q_weight = getattr(self.q_norm, "weight_plus_one", None)
            k_weight = getattr(self.k_norm, "weight_plus_one", None)
            index_q_weight = getattr(self.index_q_norm, "weight_plus_one", None)
            index_k_weight = getattr(self.index_k_norm, "weight_plus_one", None)
            if q_weight is None:
                q_weight = self.q_norm.weight + 1.0
            if k_weight is None:
                k_weight = self.k_norm.weight + 1.0
            if index_q_weight is None:
                index_q_weight = self.index_q_norm.weight + 1.0
            if index_k_weight is None:
                index_k_weight = self.index_k_norm.weight + 1.0
            _fused_outputs = torch.ops.vllm.qkv_index_rmsnorm_rope(
                input=qkv.contiguous(),
                cos_sin_cache=self.rotary_emb.cos_sin_cache,
                positions=positions,
                q_weight=q_weight,
                k_weight=k_weight,
                index_q_weight=index_q_weight,
                index_k_weight=index_k_weight,
                q_hidden_size=self.q_size,
                kv_hidden_size=self.kv_size,
                index_q_size=self.index_q_size,
                head_dim=self.head_dim,
                idx_head_dim=self.idx_head_dim,
                eps=self.q_norm.variance_epsilon,
                attn_out_fp8=True,
                indexer_out_fp8=self.indexer_kv_dtype in ("fp8", "fp8_e4m3"),
                q_bias=None,
                k_bias=None,
            )
            _det_save_and_compare("fused", _fused_outputs, {
                "qkv": qkv.contiguous(),
                "positions": positions,
                "q_weight": q_weight,
                "k_weight": k_weight,
            })
            return _fused_outputs

        main_qkv_size = self.q_size + 2 * self.kv_size
        main_qkv = qkv.narrow(-1, 0, main_qkv_size)
        index_q = qkv.narrow(-1, main_qkv_size, self.index_q_size)
        index_k = qkv.narrow(-1, main_qkv_size + self.index_q_size, self.idx_head_dim)

        if (
            True
            or main_qkv.device.type != "npu"
            or main_qkv.dtype != torch.bfloat16
            or positions.ndim != 1
            or not getattr(self.rotary_emb, "is_neox_style", True)
        ):
            q, k, v = main_qkv.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
            q, k = self._qk_norm(q, k)
            q, k = self.rotary_emb(positions, q, k)
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
        else:
            q, k, v = torch.ops.vllm.qkv_rmsnorm_rope(
                input=main_qkv.contiguous(),
                q_weight=self.q_norm.weight_plus_one,
                k_weight=self.k_norm.weight_plus_one,
                q_hidden_size=self.q_size,
                kv_hidden_size=self.kv_size,
                head_dim=self.head_dim,
                eps=self.q_norm.variance_epsilon,
                q_bias=None,
                k_bias=None,
                cos_sin_cache=self.rotary_emb.cos_sin_cache,
                positions=positions,
            )

        index_q, index_k = self._index_qk_norm(index_q, index_k)
        # Indexer FP8: rotary_emb(..., out_dtype=e4m3) → Triton FP8 RoPE.
        # Main Q/K above stay bf16. index_q/index_k then feed insert_kv + score.
        if self.indexer_kv_dtype in ("fp8", "fp8_e4m3"):
            index_q, index_k = self.rotary_emb(
                positions,
                index_q,
                index_k,
                out_dtype=torch.float8_e4m3fn,
            )
        else:
            index_q, index_k = self.rotary_emb(positions, index_q, index_k)

        _det_save_and_compare("fallback", (q, k, v, index_q, index_k), {
                "qkv": qkv.contiguous(),
                "positions": positions,
            })
        return q, k, v, index_q, index_k

    def _run_sparse_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        index_query: torch.Tensor,
        index_key: torch.Tensor,
        attn_output: torch.Tensor,
    ) -> None:
        """Insert KV, build sparse top-k indices, then run sparse attention."""
        self._insert_kv(key, value, index_key)
        if not get_forward_context().capturing:
            torch.npu.current_stream().synchronize()
        topk_idx = self.indexer(index_query)
        self.impl.forward(self, query, self.kv_cache, topk_idx, attn_output)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q, k, v, index_q, index_k = self._sparse_prepare(positions, hidden_states)
        attn_out = torch.empty_like(q, dtype=torch.bfloat16)
        torch.ops.vllm.minimax_m3_sparse_forward(
            q,
            k,
            v,
            index_q,
            index_k,
            attn_out,
            self.layer_name,
        )

        projected, _ = self.o_proj(attn_out)
        return projected

    def _qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_shape = q.shape
        k_shape = k.shape
        q = q.reshape(-1, self.head_dim).contiguous()
        k = k.reshape(-1, self.head_dim).contiguous()
        q = self.q_norm(q).reshape(q_shape)
        k = self.k_norm(k).reshape(k_shape)
        return q, k

    def _index_qk_norm(
        self, idx_q: torch.Tensor, idx_k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx_q_shape = idx_q.shape
        idx_k_shape = idx_k.shape
        idx_q = idx_q.reshape(-1, self.idx_head_dim)
        idx_k = idx_k.reshape(-1, self.idx_head_dim)
        idx_q = self.index_q_norm(idx_q).reshape(idx_q_shape)
        idx_k = self.index_k_norm(idx_k).reshape(idx_k_shape)
        return idx_q, idx_k
