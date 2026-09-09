# SPDX-License-Identifier: Apache-2.0
"""Ascend Step-4 DSA (DeepSeek-style sparse attention) backend.

The triton kernels (indexer scoring, region top-k, sparse attention) come from
the step4-hf minimal inference release (``dsa_kernels.py``, compiled by
triton-ascend).  K/V live in the standard paged KV pool shaped
``(2, num_blocks, block_size, num_kv_heads, head_size)``; the kernels address
them through the flattened per-page token space plus the block table, with
``regions_per_page = block_size // region_size``.

Per-layer region summaries are self-managed here (the upstream CUDA backend
routes them through the vLLM KV-cache extra-budget machinery, which is
CUDA-only).  They are keyed by the request's row in the persistent input batch:
while a request is running its row is stable, and a fresh request in a row is
detected by ``past_len == 0`` at forward time, which resets the pending-tail
buffer.  Stale summary rows beyond a request's current length are never read
because region scoring is bounded by the live history length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
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
from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec, KVCacheSpec

from vllm_ascend.models.step4.dsa_kernels import (
    append_regions,
    csa_compress_regions,
    decode_sparse_meta,
    indexer_logits,
    indexer_logits_rows,
    prefill_sparse_meta,
    sparse_attention_decode,
    sparse_attention_prefill,
    write_pending_tail,
)


_SHARED_DECODE_LOGITS: dict = {}


def _get_shared_decode_logits(rows: int, cols: int, device) -> torch.Tensor:
    """Single fixed-capacity indexer-score buffer shared by all DSA layers.

    Always allocated at the maximum (groups * max_num_seqs, max_regions)
    shape on first use so later, larger capture buckets never reallocate:
    a realloc would invalidate the pointers already baked into earlier
    captured ACL graphs. Layers run sequentially on one stream, so sharing
    one buffer is safe -- each layer consumes its scores (top-k selection)
    before the next layer overwrites them.
    """
    key = str(device)
    buf = _SHARED_DECODE_LOGITS.get(key)
    if buf is None or buf.shape[0] < rows or buf.shape[1] < cols:
        buf = torch.empty((rows, cols), device=device, dtype=torch.float32)
        _SHARED_DECODE_LOGITS[key] = buf
    return buf


def _active_decode_num_reqs(
    num_decodes: int,
    num_decode_tokens: int,
    decode_query_len: int,
) -> int:
    """Return the number of real decode requests, ignoring padding."""
    if decode_query_len <= 0:
        return 0
    return min(num_decodes, num_decode_tokens // decode_query_len)


def _active_prefill_num_reqs(
    num_prefills: int,
    num_prefill_tokens: int,
    query_start_loc_cpu: torch.Tensor,
    num_decodes: int,
) -> int:
    """Return real prefill requests, ignoring tail padding segments."""
    if num_prefills <= 0 or num_prefill_tokens <= 0:
        return 0
    qsl_cpu = query_start_loc_cpu.detach().cpu()
    num_reqs = int(qsl_cpu.shape[0] - 1)
    active = 0
    tokens_accounted = 0
    for i in range(num_decodes, min(num_reqs, num_decodes + num_prefills)):
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


class AscendStep4DSABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ASCEND_STEP4_DSA"

    @staticmethod
    def get_impl_cls():
        return AscendStep4DSAImpl

    @staticmethod
    def get_builder_cls():
        return AscendStep4DSAMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128, 192]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # The region arithmetic needs block_size % region_size == 0 (region
        # size 8); every vllm-ascend block size satisfies that.
        return [16, 32, 64, 128]

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
class AscendStep4DSAPrefillMetadata:
    seq_lens: torch.Tensor
    """(num_prefills,) tokens contributed by each prefill request this step."""

    context_lens: torch.Tensor
    """(num_prefills,) tokens already cached before this chunk."""

    block_table: torch.Tensor
    """(num_prefills, max_pages) int32 rows for the prefill requests."""

    query_start_loc: torch.Tensor
    """(num_prefills + 1,) start of each prefill request within the query."""

    max_query_len: int

    num_query_tokens: int = 0
    """Total prefill query tokens, filled from the CPU start-loc copy so the
    step never pays a device sync for it."""

    seq_lens_host: list = None
    context_lens_host: list = None
    """CPU copies produced by the builder's existing seq_lens transfer; the
    impl reads these instead of calling ``.tolist()`` on device tensors."""

    @property
    def query_lens(self) -> torch.Tensor:
        return self.query_start_loc[1:] - self.query_start_loc[:-1]

    @property
    def num_query_tokens_padded(self) -> int:
        return int(self.query_start_loc[-1])


@dataclass
class AscendStep4DSADecodeMetadata:
    seq_lens: torch.Tensor
    """(num_decodes,) context length including the current token."""

    block_table: torch.Tensor
    """(num_decodes, max_pages) int32 rows for the decode requests."""

    query_len: int = 1
    """Uniform decode query length K (1 for plain decode, 1+num_spec for
    MTP/EAGLE verify batches)."""


@dataclass
class AscendStep4DSAMetadata(AttentionMetadata):
    num_actual_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    slot_mapping: torch.Tensor
    prefill: AscendStep4DSAPrefillMetadata | None = None
    decode: AscendStep4DSADecodeMetadata | None = None


class AscendStep4DSAMetadataBuilder(
    AttentionMetadataBuilder[AscendStep4DSAMetadata]
):
    # Decode forward is fully tensor-driven (see AscendStep4DSAImpl
    # ._decode_step / _uniform_decode_step): static shapes per capture
    # bucket, all data-dependent choices realized with device-side
    # select/scatter. Uniform single-token decode AND uniform multi-token
    # spec-decode verify batches (K = 1 + num_speculative_tokens, K <= 8)
    # are ACL-graph capturable under FULL_DECODE_ONLY; mixed prefill+decode
    # batches still run eager -- the contract of UNIFORM_BATCH.
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_BATCH
    )
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
    ) -> AscendStep4DSAMetadata:
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

        prefill_metadata: AscendStep4DSAPrefillMetadata | None = None
        active_prefills = 0
        if num_prefills > 0:
            active_prefills = _active_prefill_num_reqs(
                num_prefills, num_prefill_tokens, qsl_cpu, num_decodes
            )
            prefill_end = num_decodes + active_prefills
            prefill_seq_lens = seq_lens[num_decodes:prefill_end]
            prefill_query_lens = qsl_cpu[num_decodes + 1 : prefill_end + 1] - qsl_cpu[
                num_decodes:prefill_end
            ]
            # One CPU round trip that already existed for context_lens: keep the
            # host lists around so the prefill step never calls .tolist() on a
            # device tensor again (each of those stalls the NPU pipeline).
            seq_lens_host_t = seq_lens[num_decodes:prefill_end].detach().cpu()
            context_lens_host_t = seq_lens_host_t - prefill_query_lens
            prefill_context_lens = self.context_len_buffer[num_decodes:prefill_end]
            prefill_context_lens.copy_(
                context_lens_host_t.to(
                    device=self.context_len_buffer.device,
                    dtype=torch.int32,
                    non_blocking=True,
                ),
                non_blocking=True,
            )
            prefill_metadata = AscendStep4DSAPrefillMetadata(
                seq_lens=prefill_seq_lens,
                context_lens=prefill_context_lens,
                block_table=block_table[num_decodes:prefill_end],
                query_start_loc=query_start_loc[num_decodes : prefill_end + 1]
                - num_decode_tokens,
                max_query_len=int(prefill_query_lens.max()),
                num_query_tokens=int(prefill_query_lens.sum()),
                seq_lens_host=seq_lens_host_t.tolist(),
                context_lens_host=context_lens_host_t.tolist(),
            )

        decode_metadata: AscendStep4DSADecodeMetadata | None = None
        active_decodes = 0
        if num_decodes > 0:
            query_lens_cpu = qsl_cpu[1 : num_decodes + 1] - qsl_cpu[:num_decodes]
            decode_query_len = int(query_lens_cpu[0].item())
            active_decodes = _active_decode_num_reqs(
                num_decodes, num_decode_tokens, decode_query_len
            )
            decode_metadata = AscendStep4DSADecodeMetadata(
                seq_lens=seq_lens[:active_decodes],
                block_table=block_table[:active_decodes],
                query_len=decode_query_len,
            )

        return AscendStep4DSAMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            num_decodes=active_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=active_prefills,
            num_prefill_tokens=num_prefill_tokens,
            slot_mapping=common_attn_metadata.slot_mapping,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )


class AscendStep4DSAState:
    """Self-managed per-layer DSA summary state, keyed by batch row.

    ``summary`` holds one fp8 proxy vector per completed 8-token region;
    ``pending_*`` buffer the ragged tail that has not filled a region yet.
    ``pending_len[row] == past_len % region_size`` is the maintained
    invariant, so a row only needs an explicit reset when a fresh request
    starts (``past_len == 0``).
    """

    def __init__(
        self,
        *,
        max_num_seqs: int,
        max_model_len: int,
        proxy_dim: int,
        region_size: int,
        device: torch.device,
    ) -> None:
        self.max_num_seqs = max_num_seqs
        self.region_size = region_size
        self.max_regions = -(-max_model_len // region_size)
        # +1 trash column: the graph-safe decode append scatters every row's
        # compressed region somewhere; rows whose region did not complete
        # write column ``max_regions`` instead of branching on the host.
        self.summary = torch.zeros(
            (max_num_seqs, self.max_regions + 1, 1, proxy_dim),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        # +1 trash slot per row: the graph-safe uniform-K decode writes each
        # tail token through a device-computed slot index; masked-out lanes
        # write slot ``region_size`` instead of branching on the host. Live
        # tails only ever occupy slots [0, region_size).
        self.pending_key = torch.zeros(
            (max_num_seqs, region_size + 1, 1, proxy_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        self.pending_z = torch.zeros_like(self.pending_key)
        self.pending_len = torch.zeros((max_num_seqs,), device=device, dtype=torch.int32)


class AscendStep4DSAImpl(AttentionImplBase[AscendStep4DSAMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        kv_cache_dtype: str = "auto",
        *,
        topk_regions: int,
        region_size: int,
        proxy_dim: int,
        max_num_seqs: int,
        max_model_len: int,
        block_size: int,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.topk_regions = topk_regions
        self.region_size = region_size
        self.proxy_dim = proxy_dim
        self.block_size = block_size
        self.regions_per_page = block_size // region_size
        self.num_kv_groups = self.num_kv_heads
        self.state: AscendStep4DSAState | None = None
        self._state_args = dict(
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            proxy_dim=proxy_dim,
            region_size=region_size,
        )

    def _get_state(self, device: torch.device) -> AscendStep4DSAState:
        if self.state is None:
            self.state = AscendStep4DSAState(device=device, **self._state_args)
        return self.state

    def _score_regions(
        self,
        state: AscendStep4DSAState,
        index_q: torch.Tensor,
        weights: torch.Tensor,
        num_tokens: int,
        num_rows: int,
        seq_lens_host: list,
        context_lens_host: list,
        summary_row_offset: int,
    ) -> torch.Tensor:
        """Indexer scores for the first ``num_rows`` batch rows.

        ``seq_lens_host``/``context_lens_host`` are the builder's CPU copies
        (no device sync here); ``summary_row_offset`` maps a relative row to
        its batch row (decode requests sit at offset 0, prefill requests after
        the decodes).  Mirrors the reference ``score_regions``: a
        [groups * total_q, width] fp32 tensor in group-major row order, width
        sized to the live history rather than the capacity.  Each request's
        kernel writes its block straight into the buffer at the right
        group-major offset -- no per-request result copies, and the fp8
        summary is read directly (widened in-register).
        """
        groups = self.num_kv_groups
        # ``seq_lens`` from the common attention metadata hold each request's
        # FULL history (past + this chunk's query tokens); the per-row query
        # length is the difference. Chunked-prefill rows have past > 0, and
        # treating the total as the query length makes ``regions`` overflow
        # the summary capacity (regions > max_regions) on the second chunk.
        query_lens_list = [
            int(seq_lens_host[row]) - int(context_lens_host[row])
            for row in range(num_rows)
        ]
        last_positions = [
            int(seq_lens_host[row]) - 1
            for row in range(num_rows)
            if query_lens_list[row] > 0
        ]
        width = max(
            (position // self.region_size for position in last_positions), default=0
        )
        logits = torch.zeros(
            (groups * num_tokens, max(width, 1)),
            device=index_q.device,
            dtype=torch.float32,
        )

        offset = 0
        for row in range(num_rows):
            length = query_lens_list[row]
            if length == 0:
                continue
            regions = (int(seq_lens_host[row]) - 1) // self.region_size
            if regions == 0:
                offset += length
                continue
            batch_row = summary_row_offset + row
            indexer_logits(
                index_q[offset : offset + length],
                weights[offset : offset + length],
                state.summary[batch_row, :regions],
                out=logits,
                out_row_offset=offset,
                out_q_stride=num_tokens,
            )
            offset += length
        return logits

    def _decode_step(
        self,
        layer: AttentionLayer,
        state: AscendStep4DSAState,
        md: AscendStep4DSAMetadata,
        query: torch.Tensor,
        key_flat: torch.Tensor,
        value_flat: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        index_z: torch.Tensor,
        weights: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        d = md.decode
        assert d is not None
        num_reqs = md.num_decodes
        rs = self.region_size
        groups = self.num_kv_groups

        # Graph-safe decode: no host sync, no data-dependent shapes. All
        # row-dependent choices (did the region complete? where does the
        # token go?) are device-side select/scatter so an ACL graph can
        # capture this routine once per bucket size and replay it with new
        # data in the persistent buffers. Semantics are identical to the
        # eager per-row loop in AscendStep4DSAState.append_chunk:
        #   plen_eff == past % region_size (fresh rows reset to 0)
        #   full     <=> plen_eff == region_size - 1 (this token completes a
        #                region; its index is past // region_size)
        #   not full <=> token appended to the pending tail at plen_eff
        seq_lens = d.seq_lens
        past = (seq_lens - 1).clamp_min(0)
        # Rejection repair: spec-decode verify can leave the pending tail
        # advanced past the confirmed length (rejected draft tokens).  The
        # true tail is the prefix ending at `past`, i.e. min(plen, past %
        # region_size); truncated slots are re-appended by this step's own
        # batch, and the region rewritten last step (the only one that can
        # hold rejected tokens) is exactly this step's dst = past // rs,
        # so the normal append self-heals it.
        plen = state.pending_len[:num_reqs]
        plen_eff = torch.minimum(plen, torch.remainder(past, rs))
        full = plen_eff == (rs - 1)

        # Fixed-shape (num_reqs, region_size, 1, proxy) window: the pending
        # tail plus the current token. For rows that complete a region this
        # is exactly the region content; other rows produce unused values.
        reg_key = torch.cat(
            (state.pending_key[:num_reqs, : rs - 1], index_k[:num_reqs].unsqueeze(1)),
            dim=1,
        )
        reg_z = torch.cat(
            (state.pending_z[:num_reqs, : rs - 1], index_z[:num_reqs].unsqueeze(1)),
            dim=1,
        )
        token_start = (
            torch.arange(num_reqs, device=reg_key.device, dtype=torch.int32) * rs
        )
        token_count = torch.full(
            (num_reqs,), rs, device=reg_key.device, dtype=torch.int32
        )
        _, reg_summary = csa_compress_regions(
            reg_key.reshape(num_reqs * rs, 1, -1),
            reg_z.reshape(num_reqs * rs, 1, -1),
            token_start,
            token_count,
            region_size=rs,
            write_fp32=False,
        )
        rows_idx = torch.arange(num_reqs, device=reg_key.device)
        dst_region = torch.div(past, rs, rounding_mode="floor").to(torch.int64)
        trash = torch.full_like(dst_region, state.max_regions)
        # NPU index_put_ has no fp8 destination kernel; float8_e4m3fn and
        # uint8 share itemsize 1, so scatter the raw bytes through a view.
        # The 5D shape is preserved by the dtype view, the (row, col) indices
        # stay valid, and the fp8 bit pattern is copied verbatim.
        state.summary.view(torch.uint8).index_put_(
            (rows_idx, torch.where(full, dst_region, trash)),
            reg_summary.view(torch.uint8),
        )

        # Pending-tail update: a completed region empties the tail (the
        # token lives in the summary); otherwise the token lands at
        # plen_eff. Writing position 0 on completion is harmless scratch:
        # plen becomes 0 and the next append overwrites it first.
        pos = torch.where(full, torch.zeros_like(plen_eff), plen_eff).to(torch.int64)
        state.pending_key.index_put_(
            (rows_idx, pos), index_k[:num_reqs].to(state.pending_key.dtype)
        )
        state.pending_z.index_put_(
            (rows_idx, pos), index_z[:num_reqs].to(state.pending_z.dtype)
        )
        state.pending_len[:num_reqs] = torch.where(
            full, torch.zeros_like(plen_eff), plen_eff + 1
        )

        # Region scoring over the full static capacity, the whole batch in
        # one kernel launch (it used to be one tiny launch per request).
        # Dead columns hold stale scores, but region_topk_ids only reads
        # columns below each row's live history length, so selection matches
        # the eager live-width path exactly. The fp8 summary is read
        # directly -- the per-row fp8->bf16 staging casts are gone.
        cap = state.max_regions
        logits = _get_shared_decode_logits(
            groups * state.max_num_seqs * rs, cap, index_q.device
        )[: groups * num_reqs]
        indexer_logits_rows(
            index_q[:num_reqs],
            weights[:num_reqs],
            state.summary,
            logits,
            q_per_row=1,
        )
        seq_lens_i32 = d.seq_lens.to(torch.int32)
        packed, counts = decode_sparse_meta(
            logits,
            seq_lens_i32,
            d.block_table,
            topk=self.topk_regions,
            region_size=self.region_size,
            regions_per_page=self.regions_per_page,
        )
        out, _ = sparse_attention_decode(
            query,
            key_flat,
            value_flat,
            packed,
            counts,
            seq_lens_i32,
            num_kv_groups=self.num_kv_groups,
            region_size=self.region_size,
            softmax_scale=self.scale,
            block_regions=2,
        )
        output[:num_reqs] = out.to(output.dtype)

    def _uniform_decode_step(
        self,
        layer: AttentionLayer,
        state: AscendStep4DSAState,
        md: AscendStep4DSAMetadata,
        query: torch.Tensor,
        key_flat: torch.Tensor,
        value_flat: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        index_z: torch.Tensor,
        weights: torch.Tensor,
        output: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Graph-safe uniform multi-token decode (MTP/EAGLE verify batches).

        Every request contributes exactly ``k`` query tokens (k static per
        capture bucket, k <= region_size). Same math as the eager
        ``_prefill_step`` -- append the k tokens, prefill-style region
        selection, sparse_attention_prefill -- with all row-dependent
        choices realized as device-side select/scatter so an ACL graph can
        capture and replay it: at most one region completes per row per
        step (pending < region_size, k <= region_size), and the completed
        region is exactly the last ``region_size`` tokens of the fixed
        window [pending(7) | new(k)].
        """
        d = md.decode
        assert d is not None
        n = md.num_decodes
        k = d.query_len
        rs = self.region_size
        groups = self.num_kv_groups
        assert 1 < k <= rs, f"uniform decode supports 1 < k <= {rs}, got {k}"

        seq_lens = d.seq_lens  # includes the k current tokens
        past = (seq_lens - k).clamp_min(0)
        # Rejection repair (see _decode_step): clamp the pending tail to
        # the confirmed length; fresh rows clamp to 0 the same way.
        plen = state.pending_len[:n]
        plen_eff = torch.minimum(plen, torch.remainder(past, rs))

        index_k2d = index_k.view(n, k, *index_k.shape[1:])
        index_z2d = index_z.view(n, k, *index_z.shape[1:])

        # Fixed window: pending tail (rs-1 slots) + the k new tokens.
        # Pending tokens are stored left-aligned (slot 0 = oldest); slide
        # them to the right edge of the window so the last-rs static slice
        # below covers the true last `rs` tokens when a region completes.
        # Slots left of the real pending tokens gather from the trash slot
        # (index rs) so the gather is always in-bounds.
        gidx = (
            torch.arange(rs - 1, device=plen_eff.device, dtype=torch.int64)[None, :]
            + plen_eff.to(torch.int64)[:, None]
            - (rs - 1)
        )
        gidx = torch.where(gidx >= 0, gidx, torch.full_like(gidx, rs))
        gidx = gidx[:, :, None, None].expand(
            n, rs - 1, *state.pending_key.shape[2:]
        )
        win_k = torch.cat(
            (
                state.pending_key[:n].gather(1, gidx),
                index_k2d.to(state.pending_key.dtype),
            ),
            dim=1,
        )
        win_z = torch.cat(
            (
                state.pending_z[:n].gather(1, gidx),
                index_z2d.to(state.pending_z.dtype),
            ),
            dim=1,
        )

        total_after = past + k
        crossed = (
            torch.div(total_after, rs, rounding_mode="floor")
            > torch.div(past, rs, rounding_mode="floor")
        )

        # Completed region = the 0-aligned block that just filled: the rs
        # tokens starting at the oldest pending token, i.e. window slot
        # rs-1-plen in the right-aligned window (plen+k >= rs when crossed).
        # Non-crossed rows clamp the start so the gather stays in-bounds;
        # their garbage slice scatters to the trash column below.
        w = win_k.shape[1]
        start = (rs - 1 - plen_eff.to(torch.int64)).clamp_(0, w - rs)
        g2 = start[:, None, None, None] + torch.arange(
            rs, device=seq_lens.device, dtype=torch.int64
        )[None, :, None, None]
        g2 = g2.expand(n, rs, *state.pending_key.shape[2:])
        reg_key = win_k.gather(1, g2).reshape(n * rs, 1, -1)
        reg_z = win_z.gather(1, g2).reshape(n * rs, 1, -1)
        token_start = (
            torch.arange(n, device=seq_lens.device, dtype=torch.int32) * rs
        )
        token_count = torch.full((n,), rs, device=seq_lens.device, dtype=torch.int32)
        _, reg_summary = csa_compress_regions(
            reg_key, reg_z, token_start, token_count, region_size=rs, write_fp32=False
        )
        rows_idx = torch.arange(n, device=seq_lens.device)
        dst_region = torch.div(past, rs, rounding_mode="floor").to(torch.int64)
        trash = torch.full_like(dst_region, state.max_regions)
        state.summary.view(torch.uint8).index_put_(
            (rows_idx, torch.where(crossed, dst_region, trash)),
            reg_summary.view(torch.uint8),
        )

        # Pending tail = last (total_after % rs) tokens of the window,
        # refreshed with one vectorized gather per side buffer (it used to
        # be rs-1 iterations of where + index_put_ pairs). Slots past the
        # tail receive clamped garbage that is never read: every access to
        # the pending buffer is bounded by pending_len.
        w = win_k.shape[1]
        tail_len = torch.remainder(total_after, rs).to(torch.int64)
        src = (w - tail_len)[:, None] + torch.arange(
            rs, device=seq_lens.device, dtype=torch.int64
        )[None, :]
        src = src.clamp_(0, w - 1)
        src4 = src[:, :, None, None].expand(n, rs, *state.pending_key.shape[2:])
        state.pending_key[:n, :rs] = win_k.gather(1, src4)
        state.pending_z[:n, :rs] = win_z.gather(1, src4)
        state.pending_len[:n] = tail_len.to(torch.int32)

        # Prefill-style scoring over the static region capacity: the whole
        # batch in one kernel launch (q_per_row = k), writing straight into
        # the group-major logits buffer.
        cap = state.max_regions
        logits = _get_shared_decode_logits(
            groups * state.max_num_seqs * rs, cap, index_q.device
        )[: groups * n * k]
        indexer_logits_rows(
            index_q[: n * k],
            weights[: n * k],
            state.summary,
            logits,
            q_per_row=k,
        )

        q_positions = positions[: n * k].to(torch.int32)
        requests = torch.div(
            torch.arange(n * k, device=index_q.device, dtype=torch.int32),
            k,
            rounding_mode="floor",
        )
        if groups > 1:
            q_positions = q_positions.repeat(groups).contiguous()
            requests = requests.repeat(groups).contiguous()
        packed, counts = prefill_sparse_meta(
            logits,
            q_positions,
            d.block_table,
            requests,
            topk=self.topk_regions,
            region_size=rs,
            regions_per_page=self.regions_per_page,
        )
        out, _ = sparse_attention_prefill(
            query,
            key_flat,
            value_flat,
            packed,
            counts,
            num_kv_groups=groups,
            region_size=rs,
            softmax_scale=self.scale,
            block_regions=2,
        )
        output[: n * k] = out.to(output.dtype)

    def _prefill_step(
        self,
        layer: AttentionLayer,
        state: AscendStep4DSAState,
        md: AscendStep4DSAMetadata,
        query: torch.Tensor,
        key_flat: torch.Tensor,
        value_flat: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        index_z: torch.Tensor,
        weights: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
        token_offset: int,
    ) -> None:
        p = md.prefill
        assert p is not None
        num_reqs = md.num_prefills
        rs = self.region_size
        device = index_q.device
        num_query_tokens = p.num_query_tokens

        # ---- batched append: compress every request's newly completed regions
        # with one launch, reading pending tail and chunk tokens in place (no
        # per-request cat/contiguous/host sync). Per-request bookkeeping:
        #   plen_eff = min(pending_len, past % rs)   (fresh rows clamp to 0)
        #   r0      = first newly completed region, delta = stream misalignment
        #   n       = (past + q_len) // rs - r0 completed inside this append
        rows_t = torch.arange(
            md.num_decodes, md.num_decodes + num_reqs, device=device, dtype=torch.int32
        )
        rows_long = rows_t.long()
        past = p.context_lens
        plen_eff = torch.minimum(state.pending_len[rows_long], torch.remainder(past, rs))
        # NB: integer torch.div without rounding_mode is true division -- the
        # float results would truncate on the int32 casts below and corrupt
        # every derived offset.
        r0 = torch.div(past - plen_eff + (rs - 1), rs, rounding_mode="floor")
        delta = r0 * rs - (past - plen_eff)
        q_lens_dev = p.query_start_loc[1:] - p.query_start_loc[:-1]
        ts = p.query_start_loc[:num_reqs]
        n_new = torch.div(past + q_lens_dev, rs, rounding_mode="floor") - r0
        meta = torch.stack(
            (rows_t, r0.to(torch.int32), delta.to(torch.int32), plen_eff, ts, n_new.to(torch.int32), q_lens_dev),
            dim=1,
        ).contiguous()
        max_new = max(
            (int(s) - int(c)) // rs + 2
            for s, c in zip(p.seq_lens_host, p.context_lens_host)
        )
        append_regions(
            state.pending_key,
            state.pending_z,
            index_k,
            index_z,
            meta,
            state.summary,
            region_size=rs,
            max_new_regions=max_new,
        )
        write_pending_tail(
            state.pending_key,
            state.pending_z,
            index_k,
            index_z,
            meta,
            region_size=rs,
        )
        state.pending_len[rows_long] = torch.remainder(past + q_lens_dev, rs)

        logits = self._score_regions(
            state,
            index_q,
            weights,
            num_query_tokens,
            num_reqs,
            p.seq_lens_host,
            p.context_lens_host,
            summary_row_offset=md.num_decodes,
        )
        groups = self.num_kv_groups
        q_positions = positions[token_offset : token_offset + num_query_tokens].to(
            torch.int32
        )
        # Token -> request map straight from the device start-loc (replaces a
        # host q_lens list -> H2D tensor -> repeat_interleave chain).
        requests = (
            torch.searchsorted(
                p.query_start_loc,
                torch.arange(num_query_tokens, device=device, dtype=p.query_start_loc.dtype),
                right=True,
            )
            - 1
        ).to(torch.int32)
        if groups > 1:
            q_positions = q_positions.repeat(groups).contiguous()
            requests = requests.repeat(groups).contiguous()
        packed, counts = prefill_sparse_meta(
            logits,
            q_positions,
            p.block_table,
            requests,
            topk=self.topk_regions,
            region_size=self.region_size,
            regions_per_page=self.regions_per_page,
        )
        out, _ = sparse_attention_prefill(
            query,
            key_flat,
            value_flat,
            packed,
            counts,
            num_kv_groups=self.num_kv_groups,
            region_size=self.region_size,
            softmax_scale=self.scale,
            block_regions=2,
        )
        output[token_offset : token_offset + num_query_tokens] = out.to(output.dtype)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        index_z: torch.Tensor,
        weights: torch.Tensor,
        output: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return output
        md = attn_metadata[layer.layer_name]
        if not isinstance(md, AscendStep4DSAMetadata):
            return output

        num_tokens = md.num_actual_tokens
        state = self._get_state(query.device)

        if isinstance(kv_cache, (tuple, list)):
            key_cache, value_cache = kv_cache[0], kv_cache[1]
        else:
            key_cache, value_cache = kv_cache[0], kv_cache[1]
        key_flat = key_cache.view(-1, self.num_kv_heads, self.head_size)
        value_flat = value_cache.view(-1, self.num_kv_heads, self.head_size)

        if md.num_decodes > 0:
            n = md.num_decodes
            k = md.decode.query_len if md.decode is not None else 1
            if k > 1:
                self._uniform_decode_step(
                    layer,
                    state,
                    md,
                    query[: n * k].contiguous(),
                    key_flat,
                    value_flat,
                    index_q[: n * k],
                    index_k[: n * k],
                    index_z[: n * k],
                    weights[: n * k],
                    output,
                    positions,
                )
            else:
                self._decode_step(
                    layer,
                    state,
                    md,
                    query[:n].contiguous(),
                    key_flat,
                    value_flat,
                    index_q[:n],
                    index_k[:n],
                    index_z[:n],
                    weights[:n],
                    output,
                )

        if md.num_prefills > 0:
            start = md.num_decode_tokens
            self._prefill_step(
                layer,
                state,
                md,
                query[start:num_tokens].contiguous(),
                key_flat,
                value_flat,
                index_q[start:num_tokens],
                index_k[start:num_tokens],
                index_z[start:num_tokens],
                weights[start:num_tokens],
                positions,
                output,
                token_offset=start,
            )
        return output
