# SPDX-License-Identifier: Apache-2.0
"""Step4 DSA (DeepSeek-style sparse attention) backend for Ascend.

The compute kernels live in ``dsa_kernels.py`` (vendored from the step4-hf
multi-hardware release). This module wires them into vLLM:

- ``AscendStep4DSAAttentionBackend`` / ``AscendStep4DSAAttentionImpl`` are the
  custom attention backend the model passes to ``Attention(attn_backend=...)``.
- The main KV cache uses the split layout ``(2, num_blocks, block_size,
  num_kv_heads, head_dim)``; block_size must be a multiple of the region size
  (8).
- Per layer, three side caches (allocated lazily when the KV cache is bound)
  persist the sparse-indexer state across steps:

    * ``idx_k_cache`` / ``idx_z_cache`` ``[num_blocks * block_size, proxy_dim]``
      bf16 -- the projected indexer k/z of every token, paged exactly like the
      main KV cache (same slot mapping).
    * ``summary_cache`` ``[num_blocks * regions_per_block, proxy_dim]`` fp8 --
      one softmax-weighted mean per *completed* region. A region is only ever
      compressed once all its 8 tokens exist, so the streaming result matches
      the batch result.

- Per-request progress (how many regions have been summarized) is tracked
  host-side per batch slot; a slot resets when its request starts from
  position 0.

Limitations of this first Ascend version: eager mode only, no prefix caching
(a prefix hit cannot reconstruct the indexer k/z of cached tokens), no
speculative decoding.
"""

from typing import Any, ClassVar

import torch
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
)

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionMetadataBuilder,
    AscendMetadata,
)

from .dsa_kernels import (
    REGION_BLOCK_SIZE,
    csa_compress_regions,
    decode_sparse_meta_torch,
    indexer_logits,
    prefill_sparse_meta_torch,
    sparse_attention_decode,
    sparse_attention_prefill,
)

logger = init_logger(__name__)


class AscendStep4DSAMetadataBuilder(AscendAttentionMetadataBuilder):
    # The impl classifies rows by query length itself; no batch reordering.
    reorder_batch_threshold: int = 1


class AscendStep4DSAAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto", "bfloat16", "float16"]

    @staticmethod
    def get_name() -> str:
        # ``Attention.__init__`` reverse-looks the backend up in
        # ``AttentionBackendEnum`` by name; CUSTOM is the member reserved for
        # out-of-tree backends (already registered by vllm-ascend itself).
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["AscendStep4DSAAttentionImpl"]:
        return AscendStep4DSAAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[AscendStep4DSAMetadataBuilder]:
        return AscendStep4DSAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % REGION_BLOCK_SIZE != 0:
            raise ValueError(
                f"Step4 DSA requires block_size to be a multiple of "
                f"{REGION_BLOCK_SIZE}, got {block_size}."
            )
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension: bool = False):
        # Physical layout is the logical layout.
        raise NotImplementedError

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # Match the Ascend platform convention (128-token blocks); the impl
        # derives regions_per_block from the bound cache, so any multiple of
        # the region size works.
        return [128]


class AscendStep4DSAAttentionImpl(AttentionImpl):
    """Per-layer Step4 DSA attention on Ascend (eager path)."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        if alibi_slopes is not None or sliding_window is not None or logits_soft_cap:
            raise ValueError("Step4 DSA does not support alibi/sliding-window/softcap.")
        if kv_sharing_target_layer_name is not None:
            raise ValueError("Step4 DSA does not support KV sharing.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads  # local KV groups
        self.kv_cache_dtype = kv_cache_dtype

        sparse_config = kwargs.get("sparse_config")
        if sparse_config is None:
            raise ValueError("Step4 DSA requires sparse_config.")
        self.proxy_dim = int(sparse_config["proxy_dim"])
        self.topk = int(sparse_config["topk"])
        self.region_size = REGION_BLOCK_SIZE
        num_provider_groups = int(sparse_config["num_provider_groups"])
        num_index_heads = int(sparse_config["sparse_indexer_num_heads"])
        self.index_heads_per_group = num_index_heads // num_provider_groups
        self.prefix_caching_enabled = bool(
            kwargs.get("prefix_caching_enabled", False)
        )

        # Allocated lazily on the first forward that sees the real KV cache.
        self._bound = False
        self.key_cache: torch.Tensor | None = None
        self.value_cache: torch.Tensor | None = None
        self.idx_k_cache: torch.Tensor | None = None
        self.idx_z_cache: torch.Tensor | None = None
        self.summary_cache: torch.Tensor | None = None
        # Host-side per-batch-slot progress: how many regions of the resident
        # request have been summarized into summary_cache.
        self._summarized: list[int] = []

    # -- lifecycle -----------------------------------------------------------

    def _bind_caches(self, kv_cache: torch.Tensor | tuple) -> None:
        if self._bound:
            return
        # The Ascend runner hands over either a fused (2, nb, bs, h, d) tensor
        # or a (k_cache, v_cache) tuple of (nb, bs, h, d) tensors.
        if isinstance(kv_cache, tuple):
            key_cache, value_cache = kv_cache[0], kv_cache[1]
        else:
            key_cache, value_cache = kv_cache[0], kv_cache[1]
        num_blocks, block_size, num_kv_heads, head_size = key_cache.shape
        if num_blocks == 0:
            # Profile/warmup passes see an empty cache; bind later.
            return
        if num_kv_heads != self.num_kv_heads or head_size != self.head_size:
            raise ValueError(
                "Step4 DSA KV cache shape mismatch: "
                f"{tuple(key_cache.shape)} vs heads=({self.num_kv_heads}, {self.head_size})"
            )
        self.block_size = block_size
        self.regions_per_block = block_size // self.region_size
        self.key_cache = key_cache
        self.value_cache = value_cache
        device = key_cache.device
        self.idx_k_cache = torch.zeros(
            num_blocks * block_size,
            1,
            self.proxy_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        self.idx_z_cache = torch.zeros_like(self.idx_k_cache)
        self.summary_cache = torch.zeros(
            num_blocks * self.regions_per_block,
            1,
            self.proxy_dim,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        self._bound = True

    # -- helpers --------------------------------------------------------------

    def _write_caches(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        index_k: torch.Tensor,
        index_z: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        slots = slot_mapping.long()
        live = slots >= 0
        slots = slots[live]
        if slots.numel() == 0:
            return
        self.key_cache.view(-1, self.num_kv_heads, self.head_size)[slots] = key[live]
        self.value_cache.view(-1, self.num_kv_heads, self.head_size)[slots] = value[live]
        self.idx_k_cache.view(-1, self.proxy_dim)[slots] = index_k[live].view(
            -1, self.proxy_dim
        )
        self.idx_z_cache.view(-1, self.proxy_dim)[slots] = index_z[live].view(
            -1, self.proxy_dim
        )

    def _summarize_completed_regions(
        self,
        block_table: torch.Tensor,
        seq_lens: list[int],
        query_lens: list[int],
    ) -> None:
        """Incrementally compress regions that became complete this step."""
        rpb = self.regions_per_block
        device = self.summary_cache.device
        new_token_slots: list[torch.Tensor] = []
        new_summary_slots: list[torch.Tensor] = []
        for req, (seq_len, q_len) in enumerate(zip(seq_lens, query_lens)):
            past = seq_len - q_len
            while len(self._summarized) <= req:
                self._summarized.append(0)
            if past == 0:
                # A fresh (or preempted-and-restarted) request occupies this slot.
                self._summarized[req] = 0
            complete = seq_len // self.region_size
            done = self._summarized[req]
            if done > complete:
                raise RuntimeError(
                    "Step4 DSA summary state is inconsistent for batch slot "
                    f"{req}: {done} summarized regions but only {complete} "
                    "complete regions exist."
                )
            if done == complete:
                continue
            if self.prefix_caching_enabled and past > 0 and done == 0:
                raise RuntimeError(
                    "Step4 DSA on Ascend does not support prefix caching: the "
                    "indexer state of cached tokens cannot be reconstructed. "
                    "Restart with --no-enable-prefix-caching."
                )
            regions = torch.arange(done, complete, device=device, dtype=torch.int64)
            pages = block_table[req].to(torch.int64)[regions // rpb]
            # Physical token slots of the regions' tokens.
            token_slots = (pages * rpb + regions % rpb).unsqueeze(
                1
            ) * self.region_size + torch.arange(
                self.region_size, device=device
            ).unsqueeze(0)
            new_token_slots.append(token_slots.reshape(-1))
            new_summary_slots.append(pages * rpb + regions % rpb)
            self._summarized[req] = complete
        if not new_token_slots:
            return
        token_slots = torch.cat(new_token_slots)
        summary_slots = torch.cat(new_summary_slots)
        num_regions = int(summary_slots.numel())
        k = self.idx_k_cache.view(-1, self.proxy_dim)[token_slots]
        z = self.idx_z_cache.view(-1, self.proxy_dim)[token_slots]
        starts = (
            torch.arange(num_regions, device=device, dtype=torch.int32)
            * self.region_size
        )
        counts = torch.full(
            (num_regions,), self.region_size, device=device, dtype=torch.int32
        )
        _, mean_fp8 = csa_compress_regions(
            k.view(num_regions * self.region_size, 1, self.proxy_dim),
            z.view(num_regions * self.region_size, 1, self.proxy_dim),
            starts,
            counts,
            region_size=self.region_size,
        )
        # fp8 tensors cannot be index-assigned on NPU; go through a uint8 view.
        self.summary_cache.view(torch.uint8).view(-1, self.proxy_dim)[
            summary_slots
        ] = mean_fp8.view(torch.uint8).view(num_regions, self.proxy_dim)

    def _gather_summaries(
        self, req: int, num_regions: int, block_table: torch.Tensor
    ) -> torch.Tensor:
        """Collect a request's summarized region means, ``[num_regions, 1, proxy]``."""
        rpb = self.regions_per_block
        device = self.summary_cache.device
        regions = torch.arange(num_regions, device=device, dtype=torch.int64)
        pages = block_table[req].to(torch.int64)[regions // rpb]
        slots = pages * rpb + regions % rpb
        # aclnnIndex does not support fp8 tensors; gather through a uint8 view.
        gathered = self.summary_cache.view(torch.uint8).view(-1, self.proxy_dim)[slots]
        return gathered.view(torch.float8_e4m3fn).to(torch.bfloat16).unsqueeze(1)

    # -- main entry ------------------------------------------------------------

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AscendMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        dsa_proxy_query: torch.Tensor | None = None,
        dsa_proxy_key: torch.Tensor | None = None,
        dsa_proxy_weights: torch.Tensor | None = None,
        dsa_proxy_z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("Step4 DSA does not support output quantization.")
        if attn_metadata is None:
            return output.fill_(0)
        self._bind_caches(kv_cache)

        num_tokens = query.shape[0]
        # AscendMetadata.actual_seq_lengths_q is query_start_loc[1:] -- the
        # *cumulative* query-token count per request, possibly with a dummy
        # padding request appended for graph/SP layout constraints.
        cum_ends = [int(x) for x in attn_metadata.actual_seq_lengths_q]
        while cum_ends and cum_ends[-1] > num_tokens:
            cum_ends.pop()
        query_lens = [cum_ends[0]] + [
            cum_ends[i] - cum_ends[i - 1] for i in range(1, len(cum_ends))
        ]
        num_reqs = min(len(attn_metadata.seq_lens_list), len(query_lens))
        query_lens = query_lens[:num_reqs]
        seq_lens = [int(s) for s in attn_metadata.seq_lens_list[:num_reqs]]
        if sum(query_lens) != num_tokens:
            raise RuntimeError(
                "Step4 DSA metadata mismatch: query_lens "
                f"{query_lens} sum to {sum(query_lens)} but the batch has "
                f"{num_tokens} tokens."
            )
        block_table = attn_metadata.block_tables
        slot_mapping = attn_metadata.slot_mapping[:num_tokens]

        key = key.view(num_tokens, self.num_kv_heads, self.head_size)
        value = value.view(num_tokens, self.num_kv_heads, self.head_size)
        self._write_caches(key, value, dsa_proxy_key, dsa_proxy_z, slot_mapping)
        self._summarize_completed_regions(block_table, seq_lens, query_lens)

        groups = self.num_kv_heads
        total_q = num_tokens
        region_size = self.region_size
        device = query.device

        # Rows with a single query token take the decode path; its metadata
        # contract is identical to a one-token prefill at the same position.
        decode_reqs = [r for r in range(num_reqs) if query_lens[r] == 1]
        prefill_reqs = [r for r in range(num_reqs) if query_lens[r] > 1]

        # ---- score every query row against its request's region summaries ----
        # A row at absolute position p sees regions [0, p // 8).
        width = max(((s - 1) // region_size for s in seq_lens), default=0)
        index_q = dsa_proxy_query  # [T, groups, heads_per_group, proxy]
        weights = dsa_proxy_weights  # [T, groups, heads_per_group] fp32
        logits = torch.zeros(
            (groups * total_q, max(width, 1)), device=device, dtype=torch.float32
        )
        token_offset = 0
        for req in range(num_reqs):
            q_len = query_lens[req]
            regions = (seq_lens[req] - 1) // region_size
            if regions > 0 and q_len > 0:
                means = self._gather_summaries(req, regions, block_table)
                block = indexer_logits(
                    index_q[token_offset : token_offset + q_len].contiguous(),
                    weights[token_offset : token_offset + q_len].contiguous(),
                    means,
                )
                for group in range(groups):
                    rows = slice(
                        group * total_q + token_offset,
                        group * total_q + token_offset + q_len,
                    )
                    logits[rows, :regions] = block[group * q_len : (group + 1) * q_len]
            token_offset += q_len

        key_cache_flat = self.key_cache.view(-1, self.num_kv_heads, self.head_size)
        value_cache_flat = self.value_cache.view(-1, self.num_kv_heads, self.head_size)

        # ---- decode rows ----
        if decode_reqs:
            decode_token_idx = torch.tensor(
                [sum(query_lens[:r]) for r in decode_reqs],
                device=device,
                dtype=torch.int64,
            )
            num_decode = len(decode_reqs)
            decode_logits = torch.zeros(
                (groups * num_decode, max(width, 1)), device=device, dtype=torch.float32
            )
            for group in range(groups):
                decode_logits[group * num_decode : (group + 1) * num_decode] = logits[
                    group * total_q + decode_token_idx
                ]
            kv_seqlens = torch.tensor(
                [seq_lens[r] for r in decode_reqs], device=device, dtype=torch.int32
            )
            request_indices = torch.tensor(
                decode_reqs, device=device, dtype=torch.int32
            ).repeat(groups)
            packed, counts = decode_sparse_meta_torch(
                decode_logits,
                kv_seqlens,
                block_table,
                request_indices,
                topk=self.topk,
                region_size=region_size,
                regions_per_page=self.regions_per_block,
            )
            decode_out, _ = sparse_attention_decode(
                query[decode_token_idx].contiguous(),
                key_cache_flat,
                value_cache_flat,
                packed,
                counts,
                kv_seqlens,
                num_kv_groups=groups,
                region_size=region_size,
                softmax_scale=self.scale,
            )
            output[decode_token_idx] = decode_out

        # ---- prefill rows ----
        if prefill_reqs:
            prefill_token_idx: list[int] = []
            q_positions: list[int] = []
            prefill_requests: list[int] = []
            offset = 0
            for req in range(num_reqs):
                q_len = query_lens[req]
                if q_len > 1:
                    past = seq_lens[req] - q_len
                    prefill_token_idx.extend(range(offset, offset + q_len))
                    q_positions.extend(range(past, past + q_len))
                    prefill_requests.extend([req] * q_len)
                offset += q_len
            prefill_token_idx_t = torch.tensor(
                prefill_token_idx, device=device, dtype=torch.int64
            )
            q_positions_t = torch.tensor(
                q_positions, device=device, dtype=torch.int32
            )
            prefill_requests_t = torch.tensor(
                prefill_requests, device=device, dtype=torch.int32
            )
            num_prefill = len(prefill_token_idx)
            prefill_logits = torch.zeros(
                (groups * num_prefill, max(width, 1)), device=device, dtype=torch.float32
            )
            for group in range(groups):
                prefill_logits[
                    group * num_prefill : (group + 1) * num_prefill
                ] = logits[group * total_q + prefill_token_idx_t]
            packed, counts = prefill_sparse_meta_torch(
                prefill_logits,
                q_positions_t.repeat(groups),
                block_table,
                prefill_requests_t.repeat(groups),
                topk=self.topk,
                region_size=region_size,
                regions_per_page=self.regions_per_block,
            )
            prefill_out, _ = sparse_attention_prefill(
                query[prefill_token_idx_t].contiguous(),
                key_cache_flat,
                value_cache_flat,
                packed,
                counts,
                num_kv_groups=groups,
                region_size=region_size,
                softmax_scale=self.scale,
            )
            output[prefill_token_idx_t] = prefill_out

        return output
