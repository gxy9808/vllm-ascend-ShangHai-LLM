# SPDX-License-Identifier: Apache-2.0
"""Ascend Step-4 model fork.

Sliding-window layers reuse the upstream dense ``Step4Attention`` unchanged
(sparse_config=None, model_has_dsa_layers=False) and run on the
platform-default Ascend attention backend, which supports sliding windows.

Full-attention (DSA) layers use ``AscendStep4DSAAttention``, a
minimax_m3-style custom attention module whose impl runs the step4-hf
reference triton kernels through triton-ascend: region summaries are
self-managed per layer while K/V live in the standard paged KV pool.

Weight loading, MoE, norms and the causal-LM shell are inherited from the
upstream vLLM implementation, so module/parameter names here mirror it
exactly (qkv_indexer_proj, sparse_indexer_*, ssmax_s, ...).
"""

from __future__ import annotations

import copy
import os

_DBG_STEP = [0]
from typing import Any

import torch
from torch import nn
from torch.nn.parameter import Parameter

from vllm.config import VllmConfig
from vllm.distributed import (
    get_dp_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from vllm.models.step4.kernels import (
    Step4SparseConfig,
    checkpoint_has_step4_sparse_config,
    get_step4_sparse_config,
)
from vllm.models.step4.layernorm import OptimusLayerNorm
from vllm.models.step4.model import (
    FusedMoEBlock,
    RMSNormFactory,
    Step4Attention,
    Step4DecoderLayer,
    Step4ForCausalLM,
    Step4FusedQKVIndexerLinear,
    Step4MLP,
    Step4Model,
    Step4SparseIndexerIndexTPLinear,
    _get_step4_moe_layer_indices,
    _is_step4_full_attention_layer,
    _per_layer_value,
    _require_resolved_valid_vocab_size,
    _set_step4_moe_protocol_metadata,
    _step_layer_types,
    _validate_step4_dsa_parallel_geometry,
    get_norm_dtype,
)

import vllm_ascend.ops  # noqa: F401  (loads before device_op to avoid its import cycle)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.models.step4.dsa_attention import (
    AscendStep4DSABackend,
    AscendStep4DSAImpl,
)

logger = init_logger(__name__)


def _kv_cache_torch_dtype(cache_config, vllm_config) -> torch.dtype:
    cache_dtype = cache_config.cache_dtype if cache_config is not None else "auto"
    if cache_dtype in ("auto", None, ""):
        return vllm_config.model_config.dtype
    if cache_dtype in ("bfloat16", "bf16"):
        return torch.bfloat16
    if cache_dtype in ("float16", "fp16", "half"):
        return torch.float16
    if cache_dtype in ("float", "fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported step4-ascend KV cache dtype: {cache_dtype}")


class AscendStep4DSAAttention(nn.Module, AttentionLayerBase):
    """Step-4 DSA layer on Ascend: reference triton kernels, paged K/V pool."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float | list[float] | None = 10000,
        cache_config: Any | None = None,
        quant_config: Any | None = None,
        rope_scaling: dict[str, Any] | None = None,
        sliding_window: int | None = None,
        prefix: str = "",
        use_head_wise_attn_gate: bool = False,
        layer_types: list | None = None,
        use_rope_layers: list | None = None,
        yarn_only_types: list | None = None,
        swa_num_attention_heads: int | None = None,
        partial_rotary_factor: float = 1.0,
        zero_centered: bool = True,
        vllm_config: VllmConfig | None = None,
        sparse_config: Step4SparseConfig | None = None,
        norm_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert sparse_config is not None and vllm_config is not None
        # Both are sliding-layer options the shared kwargs carry; DSA layers
        # are full-attention so they are accepted and ignored.
        del sliding_window, swa_num_attention_heads
        if qkv_bias:
            raise ValueError(
                "Step4 DSA layers fuse qkv with the sparse indexer into one "
                "bias-free GEMM, but attention_bias is enabled."
            )
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.prefix = prefix
        self.layer_idx = extract_layer_index(prefix)
        tp_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_size
        self.rank = get_tensor_model_parallel_rank()

        layer_type = _per_layer_value(
            layer_types,
            self.layer_idx,
            name="layer_types",
            default="full_attention",
        )
        if yarn_only_types and layer_type not in yarn_only_types:
            rope_scaling = None
        if isinstance(rope_theta, list):
            rope_theta = _per_layer_value(
                rope_theta, self.layer_idx, name="rope_theta", default=None
            )

        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.partial_rotary_factor = float(partial_rotary_factor)
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.zero_centered = zero_centered
        self.use_head_wise_attn_gate = use_head_wise_attn_gate
        self.use_rope = bool(
            _per_layer_value(
                use_rope_layers, self.layer_idx, name="use_rope_layers", default=True
            )
        )
        if norm_dtype is None:
            norm_dtype = get_norm_dtype(vllm_config.model_config.hf_config)
        linear_quant_config = (
            quant_config
            if quant_config is None or quant_config.get_name() != "fp8"
            else None
        )

        # --- upstream DSA geometry validation -------------------------------
        if self.num_kv_heads != 1:
            raise ValueError(
                f"Step4 DSA requires exactly one local KV head, got {self.num_kv_heads}"
            )
        if self.head_dim not in (128, 192):
            raise ValueError(
                f"Step4 DSA head_dim must be 128/192, got {self.head_dim}"
            )
        _validate_step4_dsa_parallel_geometry(
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            indexer_num_heads=int(sparse_config.sparse_indexer_num_heads),
            index_tp_size=int(sparse_config.index_tp_size),
            tp_size=tp_size,
        )

        # --- projections -----------------------------------------------------
        proxy_dim = int(sparse_config.proxy_dim)
        self.sparse_config = sparse_config
        self.qkv_indexer_proj = Step4FusedQKVIndexerLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            indexer_q_output_size=(
                int(sparse_config.sparse_indexer_num_heads) * proxy_dim
            ),
            indexer_kv_output_size=(
                int(sparse_config.sparse_indexer_num_k_heads) * proxy_dim
            ),
            gate_output_size=(
                self.total_num_heads if use_head_wise_attn_gate else None
            ),
            proxy_dim=proxy_dim,
            index_tp_size=int(sparse_config.index_tp_size),
            quant_config=linear_quant_config,
            prefix=f"{prefix}.qkv_indexer_proj",
        )
        self.params_dtype = self.qkv_indexer_proj.params_dtype
        self.sparse_indexer_w = Step4SparseIndexerIndexTPLinear(
            hidden_size,
            int(sparse_config.sparse_indexer_num_heads),
            params_dtype=self.params_dtype,
            index_tp_size=int(sparse_config.index_tp_size),
        )
        # Dormant in the deployed scorer; registered for explicit loading.
        self.ssmax_s = Parameter(
            torch.zeros(self.total_num_heads, dtype=torch.float32),
            requires_grad=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=linear_quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # --- main attention norms + rope -------------------------------------
        rope_parameters: dict[str, Any] = (
            dict(rope_scaling) if rope_scaling is not None else {}
        )
        rope_parameters.setdefault("rope_type", "default")
        if self.rope_theta is not None:
            rope_parameters["rope_theta"] = self.rope_theta
        rope_parameters["partial_rotary_factor"] = partial_rotary_factor

        self._sparse_indexer_base_rope_parameters = dict(rope_parameters)
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            dtype=self.params_dtype,
        )
        self.q_norm = RMSNormFactory(
            self.head_dim,
            eps=rms_norm_eps,
            zero_centered=zero_centered,
            dtype=norm_dtype,
        )
        self.k_norm = RMSNormFactory(
            self.head_dim,
            eps=rms_norm_eps,
            zero_centered=zero_centered,
            dtype=norm_dtype,
        )

        # --- indexer norms + rope (upstream eager path) -----------------------
        sparse_params = dict(self._sparse_indexer_base_rope_parameters)
        if str(sparse_params.get("rope_type", "default")).strip().lower() == "none":
            sparse_params["rope_type"] = "default"
        sparse_params["partial_rotary_factor"] = float(
            int(sparse_config.sparse_indexer_rope_dim) / proxy_dim
        )
        self.sparse_indexer_rotary_emb = get_rope(
            head_size=proxy_dim,
            max_position=max_position,
            rope_parameters=sparse_params,
            dtype=self.params_dtype,
        )
        self.sparse_indexer_q_norm = RMSNormFactory(
            proxy_dim,
            eps=rms_norm_eps,
            zero_centered=zero_centered,
        )
        self.sparse_indexer_k_norm = OptimusLayerNorm(
            proxy_dim,
            eps=rms_norm_eps,
        )

        # --- vLLM attention-layer plumbing (minimax_m3 pattern) ---------------
        self.layer_name = f"{prefix}.attn"
        self.kv_cache_dtype = (
            cache_config.cache_dtype if cache_config is not None else "auto"
        )
        self.kv_cache_torch_dtype = _kv_cache_torch_dtype(cache_config, vllm_config)
        self.attn_backend = AscendStep4DSABackend
        self.impl = AscendStep4DSAImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype,
            topk_regions=int(sparse_config.topk),
            region_size=int(sparse_config.region_block_size),
            proxy_dim=proxy_dim,
            max_num_seqs=int(vllm_config.scheduler_config.max_num_seqs),
            max_model_len=int(vllm_config.model_config.max_model_len),
            block_size=int(vllm_config.cache_config.block_size),
        )
        compilation_config = vllm_config.compilation_config
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self
        self.kv_cache = torch.tensor([])
        self.max_position_embeddings = max_position

    def get_attn_backend(self) -> type[AscendStep4DSABackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
        )

    def _apply_sparse_indexer_rope(
        self,
        positions: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        proxy_dim = int(index_q.shape[-1])
        positions = positions.reshape(-1).to(device=index_q.device, dtype=torch.long)
        q_shape = tuple(index_q.shape)
        k_shape = tuple(index_k.shape)
        q = index_q.reshape(q_shape[0], -1, proxy_dim)
        k = index_k.reshape(k_shape[0], -1, proxy_dim)
        q, k = self.sparse_indexer_rotary_emb(positions, q, k)
        return q.reshape(q_shape), k.reshape(k_shape)

    def _finish_indexer(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        index_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Indexer norm+rope (eager path) and head scores, all local shapes.

        Returns index_q [t, groups, heads_per_group, proxy], index_k [t, 1, proxy],
        index_z [t, 1, proxy], weights [t, groups, heads_per_group] fp32.
        """
        num_tokens = int(hidden_states.shape[0])
        proxy_dim = int(self.sparse_config.proxy_dim)
        weights, _ = self.sparse_indexer_w(hidden_states)

        num_index_q_heads = int(index_q.shape[-1]) // proxy_dim
        num_index_k_heads = int(index_k.shape[-1]) // proxy_dim
        heads_per_group = num_index_q_heads // self.num_kv_heads

        index_k = self.sparse_indexer_k_norm(index_k.contiguous())
        index_q = index_q.view(
            num_tokens, self.num_kv_heads, heads_per_group, proxy_dim
        ).contiguous()
        index_q = self.sparse_indexer_q_norm(index_q)
        index_k = index_k.view(num_tokens, num_index_k_heads, proxy_dim).contiguous()
        index_q, index_k = self._apply_sparse_indexer_rope(
            positions,
            index_q.reshape(num_tokens, -1, proxy_dim),
            index_k,
        )
        index_q = index_q.reshape(
            num_tokens, self.num_kv_heads, heads_per_group, proxy_dim
        )
        index_z = index_z.view(num_tokens, num_index_k_heads, proxy_dim).contiguous()

        weights = weights.to(dtype=torch.float32)
        weights = weights.view(num_tokens, self.num_kv_heads, heads_per_group)
        weights = weights * (float(heads_per_group) ** -0.5)
        return index_q, index_k, index_z, weights.contiguous()

    def _write_kv(self, key: torch.Tensor, value: torch.Tensor) -> None:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return
        md = attn_metadata[self.layer_name]
        num_tokens = md.num_actual_tokens
        k = key[:num_tokens].view(-1, self.num_kv_heads, self.head_dim)
        v = value[:num_tokens].view(-1, self.num_kv_heads, self.head_dim)
        if isinstance(self.kv_cache, (tuple, list)):
            key_cache, value_cache = self.kv_cache[0], self.kv_cache[1]
        else:
            key_cache, value_cache = self.kv_cache[0], self.kv_cache[1]
        DeviceOperator.reshape_and_cache(
            k, v, key_cache, value_cache, md.slot_mapping[:num_tokens]
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        reduce_scatter_output: bool = False,
    ) -> torch.Tensor:
        qkv, qkzg = self.qkv_indexer_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_by_head = q.view(*q.shape[:-1], self.num_heads, self.head_dim)
        q = self.q_norm(q_by_head.contiguous()).view(q.shape)
        k_by_head = k.view(*k.shape[:-1], self.num_kv_heads, self.head_dim)
        k = self.k_norm(k_by_head.contiguous()).view(k.shape)
        if self.use_rope:
            q, k = self.rotary_emb(positions, q, k)

        index_q_raw, index_k_raw, index_z_raw, gate = (
            self.qkv_indexer_proj.split_indexer(qkzg)
        )
        index_q, index_k, index_z, weights = self._finish_indexer(
            positions, hidden_states, index_q_raw, index_k_raw, index_z_raw
        )

        num_tokens = int(hidden_states.shape[0])
        query = q.view(num_tokens, self.num_heads, self.head_dim)
        output = torch.empty(
            (num_tokens, self.num_heads, self.head_dim),
            dtype=q.dtype,
            device=q.device,
        )
        self._write_kv(k, v)
        self.impl.forward(
            self,
            query,
            self.kv_cache,
            index_q,
            index_k,
            index_z,
            weights,
            output,
            positions,
        )
        attn_output = output.view(num_tokens, self.num_heads * self.head_dim)
        if gate is not None:
            attn_output = (
                attn_output.view(num_tokens, self.num_heads, self.head_dim)
                * gate.unsqueeze(-1).sigmoid()
            ).view(num_tokens, -1)
        if reduce_scatter_output:
            output_tensor, _ = self.o_proj(
                attn_output, reduce_scatter_results=True, reduce_scatter_dim=0
            )
        else:
            output_tensor, _ = self.o_proj(attn_output)
        return output_tensor


class AscendStep4DecoderLayer(Step4DecoderLayer):
    """Upstream decoder layer with the attention swapped per layer type.

    DSA (full-attention) layers get the Ascend DSA module; sliding layers get
    the upstream dense Step4Attention on the platform-default backend (the
    construction the proven VLLM_STEP4_SPARSE=0 path used).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.fp32_residual_connection = config.fp32_residual_connection
        layer_idx = extract_layer_index(prefix)
        self.layer_idx = layer_idx
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        sliding_window = getattr(config, "sliding_window", None)
        if sliding_window is None:
            sliding_window = getattr(
                vllm_config.model_config.hf_text_config, "sliding_window", None
            )
        if sliding_window is None and cache_config is not None:
            sliding_window = cache_config.sliding_window
        if cache_config is not None:
            cache_config = copy.copy(cache_config)
            cache_config.sliding_window = None

        sparse_config = get_step4_sparse_config(config)
        checkpoint_has_dsa_layers = (
            checkpoint_has_step4_sparse_config(config) and sparse_config is not None
        )
        use_dsa_for_layer = (
            sparse_config is not None
            and _is_step4_full_attention_layer(config, layer_idx, sparse_config)
        )
        del checkpoint_has_dsa_layers  # dense layers must not opt into DSA wiring
        # Dense-fallback switch for A5 bring-up debugging: VLLM_STEP4_SPARSE=0
        # routes DSA layers through the platform dense backend as well.
        use_dsa_for_layer = use_dsa_for_layer and (
            os.environ.get("VLLM_STEP4_SPARSE", "1") != "0"
        )

        norm_dtype = get_norm_dtype(config)
        layer_types = _step_layer_types(config)
        partial_rotary_factors = getattr(config, "partial_rotary_factors", [])
        partial_rotary_factor = float(
            _per_layer_value(
                partial_rotary_factors,
                layer_idx,
                name="partial_rotary_factors",
                default=1.0,
            )
        )
        max_position = getattr(config, "max_position_embeddings", None)
        if max_position is None:
            max_position = vllm_config.model_config.max_model_len

        # Both layer families share 64 heads / 4 kv groups / head_dim 192 in
        # this checkpoint (num_attention_heads == num_sliding_attention_heads
        # == attention_other_setting), so config defaults resolve correctly.
        attn_kwargs = dict(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=max_position,
            num_kv_heads=config.num_attention_groups,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=getattr(config, "rope_scaling", None),
            sliding_window=sliding_window,
            use_head_wise_attn_gate=getattr(config, "use_head_wise_attn_gate", False),
            layer_types=layer_types,
            use_rope_layers=getattr(config, "use_rope_layers", []),
            yarn_only_types=getattr(config, "yarn_only_types", []),
            swa_num_attention_heads=getattr(
                config, "swa_num_attention_heads", None
            ),
            partial_rotary_factor=partial_rotary_factor,
            prefix=f"{prefix}.self_attn",
            zero_centered=config.zero_centered,
            vllm_config=vllm_config,
            norm_dtype=norm_dtype,
        )
        if use_dsa_for_layer:
            self.self_attn = AscendStep4DSAAttention(
                sparse_config=sparse_config,
                **attn_kwargs,
            )
        else:
            self.self_attn = Step4Attention(
                sparse_config=None,
                model_has_dsa_layers=False,
                **attn_kwargs,
            )

        self.use_moe = False
        self.tp_group = get_tp_group()
        self.use_fused_all_reduce = (
            get_tensor_model_parallel_world_size() > 1
            and get_dp_group().world_size == 1
        )

        moe_layers_idx = _get_step4_moe_layer_indices(config)
        if layer_idx in moe_layers_idx:
            self.moe = FusedMoEBlock(
                vllm_config,
                prefix=f"{prefix}.moe",
            )
            self.use_moe = True
        else:
            self.mlp = Step4MLP(
                config=config,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act="silu",
                quant_config=quant_config
                if quant_config and quant_config.get_name() != "fp8"
                else None,
                reduce_results=True,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNormFactory(
            config.hidden_size,
            eps=config.rms_norm_eps,
            zero_centered=config.zero_centered,
            dtype=norm_dtype,
        )
        self.post_attention_layernorm = RMSNormFactory(
            config.hidden_size,
            eps=config.rms_norm_eps,
            zero_centered=config.zero_centered,
            dtype=norm_dtype,
        )
        self.prefix = prefix
        self.use_attention_o_proj_reduce_scatter = False
        if os.environ.get("VLLM_STEP4_DEBUG_NORMS") == "1":
            self._dbg_calls = 0
            if layer_idx in (1, 2, 3, 4, 20, 60, 88, 91):
                rk0 = get_tensor_model_parallel_rank()

                def res_pre_hook(module, args, _li=layer_idx, _rk=rk0):
                    if args[0].shape[0] == 6:
                        torch.save(
                            {"res": args[0].detach().cpu()},
                            f"/tmp/res_dump_r{_rk}_l{_li}.pt",
                        )
                        print(f"RES_DUMPED r{_rk} l{_li}", flush=True)

                self.input_layernorm.register_forward_pre_hook(res_pre_hook)
            targets = [("attn", self.self_attn)]
            if self.use_moe:
                targets.append(("moe", self.moe))
            else:
                targets.append(("mlp", self.mlp))
            for tag, mod in targets:
                mod.register_forward_hook(
                    self._make_norm_hook(layer_idx, tag), with_kwargs=True
                )
            if layer_idx == 3 and self.use_moe:
                rk3 = get_tensor_model_parallel_rank()
                for tag, mod in [
                    ("gate", self.moe.gate),
                    ("exp", self.moe.experts),
                    ("shex", self.moe.share_expert),
                ]:
                    mod.register_forward_hook(
                        self._make_norm_hook(layer_idx, tag), with_kwargs=True
                    )
            if layer_idx == 0 and not self.use_moe:
                m = self.mlp
                qm_g = getattr(getattr(m, "gate_up_proj", None), "quant_method", None)
                qm_d = getattr(getattr(m, "down_proj", None), "quant_method", None)
                w_g = getattr(m.gate_up_proj, "weight", None)
                w_d = getattr(m.down_proj, "weight", None)
                print(
                    f"MLPDBG quant={type(qm_g).__name__}/{type(qm_d).__name__} "
                    f"limit={getattr(m, chr(108)+chr(105)+chr(109)+chr(105)+chr(116), None)} "
                    f"wg={None if w_g is None else (str(w_g.dtype), tuple(w_g.shape))} "
                    f"wd={None if w_d is None else (str(w_d.dtype), tuple(w_d.shape))}",
                    flush=True,
                )
                rk_m = get_tensor_model_parallel_rank()
                for tag, mod in [("gu", m.gate_up_proj), ("dproj", m.down_proj)]:
                    mod.register_forward_hook(
                        self._make_norm_hook(layer_idx, tag), with_kwargs=True
                    )
            if layer_idx == 0:
                attn_mod = self.self_attn
                sub = [("qkv", getattr(attn_mod, "qkvg_proj", None) or attn_mod.qkv_proj),
                       ("core", attn_mod.attn),
                       ("gproj", getattr(attn_mod, "g_proj", None)),
                       ("oproj", attn_mod.o_proj)]
                for tag, mod in [("qn", attn_mod.q_norm), ("kn", attn_mod.k_norm),
                                 ("rope", getattr(attn_mod, "rotary_emb", None))]:
                    if mod is not None:
                        mod.register_forward_hook(
                            self._make_norm_hook(layer_idx, tag), with_kwargs=True
                        )
                for tag, mod in sub:
                    if mod is not None:
                        mod.register_forward_hook(
                            self._make_norm_hook(layer_idx, tag), with_kwargs=True
                        )


    def _forward_ffn(
        self,
        hidden_states: torch.Tensor,
        input_is_sequence_parallel: bool = False,
        residual: torch.Tensor | None = None,
        orig_num_tokens: int | None = None,
    ) -> torch.Tensor:
        if (
            not self.use_moe
            or not self.moe.fuse_all_reduce
            or self.moe.experts.moe_config.is_sequence_parallel
        ):
            return super()._forward_ffn(
                hidden_states,
                input_is_sequence_parallel=input_is_sequence_parallel,
                residual=residual,
                orig_num_tokens=orig_num_tokens,
            )
        shared_output, moe_output = self.moe(
            hidden_states, input_is_sequence_parallel=input_is_sequence_parallel
        )
        # On Ascend the MoE comm impls (MC2 / AllGather / All2All) fuse the
        # EP/TP reduction into the token combine, so the routed output is
        # already complete even though FusedMoEBlock requested
        # reduce_results=False (upstream expects a partial here and reduces
        # shared + routed together). Only the shared expert (RowParallelLinear
        # with reduce_results=False) is partial w.r.t. TP: reduce it alone,
        # then add the complete routed output. Reducing the sum instead would
        # count the routed contribution once per TP rank (measured 8x).
        shared_output = self._cast_for_residual(shared_output)
        moe_output = self._cast_for_residual(moe_output)
        if self.use_fused_all_reduce:
            shared_output = self.tp_group.all_reduce(shared_output)
        else:
            shared_output = tensor_model_parallel_all_reduce(shared_output)
        return moe_output + shared_output

    @staticmethod
    def _make_norm_hook(layer_idx: int, tag: str):
        rk = get_tensor_model_parallel_rank()

        def hook(module, inputs, kwargs, output):
            if layer_idx == 0 and tag == "attn":
                _DBG_STEP[0] += 1
            t = output[0] if isinstance(output, tuple) else output
            if t is None or not torch.is_tensor(t):
                return
            src_t = kwargs.get("hidden_states") if kwargs else None
            if src_t is None:
                src_t = inputs[0] if inputs else t
            if _DBG_STEP[0] > 4:
                return
            if layer_idx == 0 and tag == "core" and t.shape[0] == 6:
                torch.save(
                    {"q": inputs[0].detach().cpu(), "k": inputs[1].detach().cpu(),
                     "v": inputs[2].detach().cpu(), "out": t.detach().cpu()},
                    f"/tmp/core_dump_r{rk}.pt",
                )
                print(f"CORE_DUMPED r{rk}", flush=True)
            if layer_idx == 0 and tag in ("qn", "kn", "rope") and t.shape[0] == 6:
                torch.save(
                    {"in": (inputs[0].detach().cpu() if inputs else None),
                     "out": t.detach().cpu()},
                    f"/tmp/{tag}_dump_r{rk}.pt",
                )
                print(f"{tag.upper()}_DUMPED r{rk}", flush=True)
            if layer_idx == 0 and tag == "qkv" and t.shape[0] == 6:
                torch.save({"out": t.detach().cpu()}, f"/tmp/qkvw_dump_r{rk}.pt")
                print(f"QKVW_DUMPED r{rk}", flush=True)
            if layer_idx == 3 and tag in ("gate", "exp", "shex") and t.shape[0] == 6:
                torch.save(
                    {"in": [a.detach().cpu() for a in inputs if torch.is_tensor(a)],
                     "kw": [a.detach().cpu() for a in (kwargs or {}).values() if torch.is_tensor(a)],
                     "out": t.detach().cpu()},
                    f"/tmp/moe_{tag}_r{rk}.pt",
                )
                print(f"MOE_{tag.upper()}_DUMPED r{rk}", flush=True)
            if layer_idx == 0 and tag in ("gu", "dproj") and t.shape[0] == 6:
                torch.save(
                    {"in": (inputs[0].detach().cpu() if inputs else None),
                     "out": t.detach().cpu(),
                     "w": module.weight.detach().cpu() if hasattr(module, "weight") else None},
                    f"/tmp/{tag}_dump_r{rk}.pt",
                )
                print(f"{tag.upper()}_DUMPED r{rk}", flush=True)
            if layer_idx == 0 and tag == "gproj" and t.shape[0] == 6:
                torch.save(
                    {"h": src_t[:8].detach().cpu(), "logits": t[:8].detach().cpu(),
                     "w_runtime": module.weight.detach().cpu()},
                    f"/tmp/gproj_dump_r{rk}.pt",
                )
                print(f"GPROJ_DUMPED r{rk} w=", tuple(module.weight.shape), flush=True)
            extra = ""
            if t.dim() == 2 and tag == "gproj":
                flat = t.float().flatten().sort().values
                n = flat.numel()
                dec = [flat[i * (n - 1) // 10].item() for i in range(11)]
                extra = (
                    f" deciles={[round(v, 2) for v in dec]} "
                    f"sig_raw={t.sigmoid().float().mean().item():.5f}"
                )
            print(
                f"NORMDBG r{rk} layer={layer_idx} {tag} call={_DBG_STEP[0]} "
                f"in={src_t.float().abs().mean():.4f} "
                f"out={t.float().abs().mean():.4f} max={t.float().abs().max():.4f}{extra}",
                flush=True,
            )
        return hook


class AscendStep4Model(Step4Model):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        # The @support_torch_compile wrapper on the upstream __init__ sets
        # this; bypassing it here means we set it ourselves. The DSA modules
        # hold python-side per-request state, so this fork stays eager.
        self.do_not_compile = True
        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        self.vocab_size = config.vocab_size
        self.config = config
        self.fp32_residual_connection = config.fp32_residual_connection
        self.moe_num_experts = config.moe_num_experts
        self.parallel_config = vllm_config.parallel_config

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: AscendStep4DecoderLayer(
                vllm_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            norm_dtype = get_norm_dtype(config)
            self.norm = RMSNormFactory(
                config.hidden_size,
                eps=config.rms_norm_eps,
                zero_centered=config.zero_centered,
                dtype=norm_dtype,
            )
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )


class AscendStep4ForCausalLM(Step4ForCausalLM):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        nn.Module.__init__(self)
        self.vllm_config = vllm_config
        model_config = vllm_config.model_config
        valid_vocab_size = _require_resolved_valid_vocab_size(model_config)
        config = model_config.hf_config
        self.config = config
        self.fp32_residual_connection = config.fp32_residual_connection
        self.model = AscendStep4Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config
                if vllm_config.quant_config
                and vllm_config.quant_config.get_name() != "fp8"
                else None,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
            self.logits_processor = LogitsProcessor(
                config.vocab_size,
                valid_vocab_size=valid_vocab_size,
            )
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.moe_layers: list[Any] = []
        example_layer: FusedMoEBlock | None = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if hasattr(layer, "moe") and isinstance(layer.moe, FusedMoEBlock):
                example_layer = layer.moe
                self.moe_layers.append(layer.moe.experts)
        _set_step4_moe_protocol_metadata(self, example_layer)
