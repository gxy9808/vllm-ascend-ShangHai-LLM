# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend Step4 model.

Ported from the CUDA/Optimus implementation in ``vllm.models.step4`` with the
following Ascend-specific adaptations:

- The DSA sparse-attention path (CuTeDSL SM90 kernels) is not available on
  NPU. Full-attention layers fall back to dense attention and the
  sparse-indexer weights (``sparse_indexer_*`` / ``ssmax_s``) are skipped
  during weight loading.
- The fused QK-norm + RoPE CuTeDSL kernels are replaced by the eager
  per-head RMSNorm + rotary-embedding path.
- The Triton router-bias top-k kernel is replaced by a pure PyTorch
  implementation with identical semantics.
- Routed experts run the checkpoint's FP8 e4m3 block quantization directly
  (128x128 weight scales + dynamic per-128-group activation quantization),
  mirroring the step4-hf reference: kernels are vendored in
  ``fp8_kernels.py`` and experts are sharded whole (EP-style, contiguous
  expert slices with the full inner dim) so the fp8 K-block grid stays
  aligned. Layers listed in ``modules_to_not_convert`` keep their bf16
  experts and take the plain bf16 path.
"""

import copy
import typing
from collections.abc import Callable, Iterable
from typing import Any

import regex as re
import torch
from torch import nn

from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (
    get_dp_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul, SwigluStepAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import MixtureOfExperts, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    get_spec_layer_idx_from_weight_name,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType

from .dsa import AscendStep4DSAAttentionBackend
from .fp8_kernels import linear_fp8_or_bf16

logger = init_logger(__name__)

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

# Checkpoint tensors with no consumer on Ascend. ``ssmax_s`` is a dormant
# calibration tensor (the deployed indexer scores with weighted ReLU, not a
# scalable softmax), registered by neither the Ascend nor any runtime path.
_DSA_ONLY_WEIGHT_MARKERS = ("ssmax_s",)


def _require_resolved_valid_vocab_size(model_config: ModelConfig) -> int:
    if model_config.valid_vocab_size is None:
        raise ValueError(
            "Step4 requires valid_vocab_size to be resolved from the tokenizer "
            "before model construction. Call VllmConfig.resolve_valid_vocab_size() "
            "or pass --valid-vocab-size when tokenizer initialization is skipped."
        )
    return model_config.get_valid_vocab_size()


def _step_layer_types(config: Any) -> list[str]:
    """Per-layer attention types spanning the dense stack and the MTP layers.

    `config.layer_types` only covers the dense stack so that transformers' own
    length validation passes; the MTP block indexes past it.
    """
    return (
        getattr(config, "layer_types_with_mtp", None)
        or getattr(config, "layer_types", None)
        or []
    )


def _parse_step4_layer_indices(
    value: str | Iterable[int] | None,
    *,
    name: str,
) -> set[int] | None:
    if value is None:
        return None
    raw_values = value.split(",") if isinstance(value, str) else value
    try:
        indices = [int(item) for item in raw_values if str(item).strip()]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Step4 {name} must contain integer layer indices.") from exc
    if len(indices) != len(set(indices)):
        raise ValueError(f"Step4 {name} contains duplicate layer indices.")
    return set(indices)


def _get_step4_moe_layer_indices(config: Any) -> set[int]:
    enum_indices = _parse_step4_layer_indices(
        getattr(config, "moe_layers_enum", None),
        name="moe_layers_enum",
    )
    list_indices = _parse_step4_layer_indices(
        getattr(config, "moe_layer_list", None),
        name="moe_layer_list",
    )
    if (
        enum_indices is not None
        and list_indices is not None
        and enum_indices != list_indices
    ):
        raise ValueError(
            "Step4 moe_layers_enum and moe_layer_list describe different layers."
        )

    indices = enum_indices if enum_indices is not None else list_indices
    if indices is None:
        indices = set(range(1, int(config.num_hidden_layers)))
    total_layers = int(config.num_hidden_layers) + int(
        getattr(config, "num_nextn_predict_layers", 0) or 0
    )
    invalid = sorted(index for index in indices if not 0 <= index < total_layers)
    if invalid:
        raise ValueError(
            f"Step4 MoE layer indices must be in [0, {total_layers}), got {invalid}."
        )
    return indices


def _set_step4_moe_protocol_metadata(
    model: Any,
    example_layer: Any | None,
) -> None:
    """Populate the MoE protocol for the layers local to this PP rank."""
    model.num_moe_layers = len(model.moe_layers)
    model.num_expert_groups = 1
    model.num_shared_experts = 0
    if example_layer is None:
        # A valid pipeline stage can contain only dense layers. Reporting zero
        # local MoE layers keeps that rank out of EPLB while other stages still
        # expose their local expert runners.
        model.num_logical_experts = 0
        model.num_physical_experts = 0
        model.num_local_physical_experts = 0
        model.num_routed_experts = 0
        model.num_redundant_experts = 0
        return

    model.num_logical_experts = example_layer.n_logical_experts
    model.num_physical_experts = example_layer.n_physical_experts
    model.num_local_physical_experts = example_layer.n_local_physical_experts
    model.num_routed_experts = example_layer.n_routed_experts
    model.num_redundant_experts = example_layer.n_redundant_experts


def _per_layer_value(
    values: list[Any] | tuple[Any, ...] | None,
    layer_idx: int,
    *,
    name: str,
    default: Any,
) -> Any:
    if not values:
        return default
    if layer_idx >= len(values):
        raise ValueError(
            f"Step4 {name} has {len(values)} entries, but layer {layer_idx} "
            "requires an entry."
        )
    return values[layer_idx]


_NORM_DTYPE_TO_TORCH_DTYPE = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "float": torch.float32,
}


def get_norm_dtype(config: Any) -> torch.dtype:
    norm_dtype = config.norm_dtype.lower()
    if norm_dtype in _NORM_DTYPE_TO_TORCH_DTYPE:
        return _NORM_DTYPE_TO_TORCH_DTYPE[norm_dtype]
    raise ValueError(f"Unknown norm_dtype: {norm_dtype!r}")


class Step4RMSNorm(nn.Module):
    """Pure PyTorch RMSNorm with optional zero-centered weight.

    Replaces the Optimus RMSNorm CUDA kernels. The math matches the Optimus
    eager fallback: fp32 internal compute, scale = weight (+1 when the
    checkpoint stores zero-centered weights), output cast back to the input
    dtype.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        zero_centered: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps
        self.zero_centered = zero_centered

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute = x.float()
        variance = compute.pow(2).mean(dim=-1, keepdim=True)
        compute = compute * torch.rsqrt(variance + self.variance_epsilon)
        scale = self.weight.float()
        if self.zero_centered:
            scale = scale + 1.0
        return (compute * scale).to(x.dtype)


def step4_router_bias_eager(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    *,
    router_bias: torch.Tensor,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch equivalent of the Step4 Triton router-bias top-k kernel.

    Selection scores are sigmoid(logits) + bias; the selected expert weights
    are the bias-free sigmoid probabilities, renormalized over the top-k and
    scaled by ``routed_scaling_factor``.
    """
    assert renormalize
    del hidden_states
    gate_prob = torch.sigmoid(gating_output.to(torch.float32))
    select_scores = gate_prob + router_bias.to(torch.float32)
    _, topk_ids = torch.topk(select_scores, topk, dim=-1)
    topk_weights = gate_prob.gather(-1, topk_ids)
    weight_sum = topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights / (weight_sum + 1e-20)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights, topk_ids.to(torch.int32)


class FP32ReplicatedLinear(ReplicatedLinear):
    """Router gate computed in FP32 for higher precision."""

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.nn.Parameter | None]:
        router_logits = torch.nn.functional.linear(
            x.to(torch.float32), self.weight.to(torch.float32)
        )
        return router_logits, None


def clamped_swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """``silu(gate).clamp(max=limit) * up.clamp(-limit, limit)``, in fp32.

    The clamps bound the activation's magnitude so the fp8 expert GEMMs
    downstream keep their dynamic range. The gate has no lower clamp because
    ``silu`` already bounds it from below at about -0.28. The fp32 compute
    with a single rounding back to bf16 matches the deployed (Triton) kernel
    rather than a bf16-native evaluation.
    """
    activated = torch.nn.functional.silu(gate.float()).clamp(max=limit)
    bounded = up.float().clamp(-limit, limit)
    return (activated * bounded).to(gate.dtype)


def _step4_fp8_expert_layout(config: Any) -> tuple[set[int], int]:
    """Return (bf16 expert layer indices, quant block size).

    A Step4 checkpoint either declares ``quantization_config.fp8`` with
    128x128 weight blocks -- in which case every MoE layer is fp8 except
    those under ``modules_to_not_convert`` -- or declares nothing, in which
    case every expert is bf16.
    """
    quant_config = getattr(config, "quantization_config", None)
    if not quant_config:
        return set(), 0
    quant_method = quant_config.get("quant_method")
    if quant_method not in ("fp8", "compressed-tensors"):
        raise ValueError(
            f"Step4 Ascend fp8 experts support quant_method=fp8, got "
            f"{quant_method!r}."
        )
    block = quant_config.get("weight_block_size") or [128, 128]
    if list(block) != [128, 128]:
        raise ValueError(
            "Step4 Ascend fp8 experts require 128x128 weight blocks, got "
            f"{list(block)}."
        )
    bf16_layers: set[int] = set()
    for entry in quant_config.get("modules_to_not_convert") or []:
        match = re.search(r"layers\.(\d+)\.", entry)
        if match and ".moe." in entry:
            bf16_layers.add(int(match.group(1)))
    return bf16_layers, 128


class Step4MLP(nn.Module):
    def __init__(
        self,
        config: Any,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=None,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=None,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )

        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn: nn.Module = SiluAndMul()
        layer_idx = extract_layer_index(prefix)
        swiglu_limit = _per_layer_value(
            getattr(config, "swiglu_limits_shared", None),
            layer_idx,
            name="swiglu_limits_shared",
            default=None,
        )
        if swiglu_limit not in (None, 0):
            self.act_fn = SwigluStepAndMul(limit=float(swiglu_limit))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        intermediate_act = self.act_fn(gate_up)
        output, _ = self.down_proj(intermediate_act)
        return output


def _step4_moe_reduce_policy(tp_size: int, dp_size: int) -> tuple[bool, bool]:
    """Return combined-reduce and per-path-reduce settings for Step4 MoE."""
    fuse_all_reduce = tp_size > 1 and dp_size == 1
    return fuse_all_reduce, not fuse_all_reduce


class Step4Attention(nn.Module):
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
        cache_config: CacheConfig | None = None,
        rope_scaling: dict[str, Any] | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        # Step4 specific args
        sliding_window: int | None = None,
        use_head_wise_attn_gate: bool = False,
        layer_types: list | None = None,
        use_rope_layers: list | None = None,
        yarn_only_types: list | None = None,
        swa_num_attention_heads: int | None = None,
        partial_rotary_factor: float = 1.0,
        zero_centered: bool = True,
        norm_dtype: torch.dtype | None = None,
        sparse_config: dict | None = None,
        prefix_caching_enabled: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_size
        self.layer_idx = extract_layer_index(prefix)
        self.prefix = prefix
        default_layer_type = (
            "sliding_attention" if self.layer_idx % 2 == 0 else "full_attention"
        )
        layer_type = _per_layer_value(
            layer_types,
            self.layer_idx,
            name="layer_types",
            default=default_layer_type,
        )
        enable_sliding_window = layer_type == "sliding_attention"
        if yarn_only_types and layer_type not in yarn_only_types:
            rope_scaling = None

        if sliding_window is not None and enable_sliding_window:
            if swa_num_attention_heads is not None:
                num_heads = swa_num_attention_heads
                self.total_num_heads = swa_num_attention_heads
        else:
            sliding_window = None

        if isinstance(rope_theta, list):
            if not rope_theta:
                raise ValueError("Step4 rope_theta cannot be an empty list.")
            rope_theta = _per_layer_value(
                rope_theta,
                self.layer_idx,
                name="rope_theta",
                default=None,
            )

        if self.total_num_heads <= 0 or self.total_num_heads % tp_size != 0:
            raise ValueError(
                "Step4 attention heads must be positive and divisible by tensor "
                f"parallel size, got num_heads={self.total_num_heads}, "
                f"tp_size={tp_size}."
            )
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads <= 0:
            raise ValueError(
                "Step4 attention requires a positive number of KV heads, got "
                f"{self.total_num_kv_heads}."
            )
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size != 0:
                raise ValueError(
                    "Step4 KV heads must be divisible by tensor parallel size "
                    "when sharded, got "
                    f"num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
                )
        else:
            if tp_size % self.total_num_kv_heads != 0:
                raise ValueError(
                    "Step4 KV heads must divide tensor parallel size when "
                    "replicated, got "
                    f"num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
                )
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        if head_dim is None and hidden_size % self.total_num_heads != 0:
            raise ValueError(
                "Step4 hidden_size must be divisible by num_heads when head_dim "
                f"is omitted, got hidden_size={hidden_size}, "
                f"num_heads={self.total_num_heads}."
            )
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        if self.head_dim <= 0:
            raise ValueError(f"Step4 head_dim must be positive, got {self.head_dim}.")
        self.partial_rotary_factor = float(partial_rotary_factor)
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if (
            self.partial_rotary_factor <= 0.0
            or self.partial_rotary_factor > 1.0
            or self.rotary_dim <= 0
            or self.rotary_dim % 2 != 0
        ):
            raise ValueError(
                "Step4 partial_rotary_factor must produce a positive, even "
                "rotary dimension no larger than head_dim, got "
                f"head_dim={self.head_dim}, "
                f"partial_rotary_factor={self.partial_rotary_factor}, "
                f"rotary_dim={self.rotary_dim}."
            )
        if max_position is None or int(max_position) <= 0:
            raise ValueError(
                f"Step4 max_position must be a positive integer, got {max_position}."
            )
        max_position = int(max_position)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=None,
            prefix=f"{prefix}.qkv_proj",
        )
        self.params_dtype = self.qkv_proj.params_dtype
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.o_proj",
        )

        rope_parameters: dict[str, Any] = (
            dict(rope_scaling) if rope_scaling is not None else {}
        )
        rope_parameters.setdefault("rope_type", "default")
        if self.rope_theta is not None:
            rope_parameters["rope_theta"] = self.rope_theta
        rope_parameters["partial_rotary_factor"] = partial_rotary_factor

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            dtype=self.params_dtype,
        )
        # RoPE is applied eagerly from the precomputed cos/sin cache instead of
        # dispatching to the platform rotary kernel (the vllm-ascend Triton
        # rope kernel does not support this model's head_dim on Ascend 950).
        rope_cache = self.rotary_emb.cos_sin_cache
        self.rope_cos, self.rope_sin = rope_cache.chunk(2, dim=-1)
        self.rope_is_neox_style = getattr(self.rotary_emb, "is_neox_style", True)

        self.zero_centered = zero_centered
        self.q_norm = Step4RMSNorm(
            self.head_dim,
            eps=rms_norm_eps,
            zero_centered=self.zero_centered,
            dtype=norm_dtype,
        )
        self.k_norm = Step4RMSNorm(
            self.head_dim,
            eps=rms_norm_eps,
            zero_centered=self.zero_centered,
            dtype=norm_dtype,
        )
        self.use_head_wise_attn_gate = use_head_wise_attn_gate
        if use_head_wise_attn_gate:
            self.g_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.g_proj",
            )

        self.use_rope = bool(
            _per_layer_value(
                use_rope_layers,
                self.layer_idx,
                name="use_rope_layers",
                default=True,
            )
        )

        # ---- DSA sparse attention (full-attention layers only) ----
        self.use_dsa = sparse_config is not None
        if self.use_dsa:
            self.proxy_dim = int(sparse_config["proxy_dim"])
            self.num_provider_groups = int(sparse_config["num_provider_groups"])
            num_index_heads = int(sparse_config["sparse_indexer_num_heads"])
            self.index_heads_per_group = num_index_heads // self.num_provider_groups
            indexer_rope_dim = int(sparse_config["sparse_indexer_rope_dim"])
            self.sparse_indexer_q = Step4ProviderGroupLinear(
                hidden_size,
                num_index_heads * self.proxy_dim,
                self.num_provider_groups,
                self.params_dtype,
            )
            self.sparse_indexer_k = ReplicatedLinear(
                hidden_size,
                int(sparse_config["sparse_indexer_num_k_heads"]) * self.proxy_dim,
                bias=False,
                quant_config=None,
                params_dtype=self.params_dtype,
                prefix=f"{prefix}.sparse_indexer_k",
            )
            self.sparse_indexer_z = ReplicatedLinear(
                hidden_size,
                int(sparse_config["sparse_indexer_num_k_heads"]) * self.proxy_dim,
                bias=False,
                quant_config=None,
                params_dtype=self.params_dtype,
                prefix=f"{prefix}.sparse_indexer_z",
            )
            # Deployment loads this tiny fp32 matrix into a bf16 parameter and
            # widens the GEMM result afterwards; mirror that boundary.
            self.sparse_indexer_w = Step4ProviderGroupLinear(
                hidden_size,
                num_index_heads,
                self.num_provider_groups,
                self.params_dtype,
            )
            self.sparse_indexer_q_norm = Step4RMSNorm(
                self.proxy_dim,
                eps=rms_norm_eps,
                zero_centered=zero_centered,
                dtype=norm_dtype,
            )
            self.sparse_indexer_k_norm = Step4IndexerLayerNorm(
                self.proxy_dim, eps=rms_norm_eps
            )
            indexer_rope_parameters: dict[str, Any] = (
                dict(rope_scaling) if rope_scaling is not None else {}
            )
            indexer_rope_parameters.setdefault("rope_type", "default")
            if self.rope_theta is not None:
                indexer_rope_parameters["rope_theta"] = self.rope_theta
            indexer_rope_parameters["partial_rotary_factor"] = (
                indexer_rope_dim / self.proxy_dim
            )
            self.sparse_indexer_rotary_emb = get_rope(
                head_size=self.proxy_dim,
                max_position=max_position,
                rope_parameters=indexer_rope_parameters,
                dtype=self.params_dtype,
            )
            indexer_cache = self.sparse_indexer_rotary_emb.cos_sin_cache
            self.indexer_rope_cos, self.indexer_rope_sin = indexer_cache.chunk(2, dim=-1)
            self.indexer_rotary_dim = indexer_rope_dim

        extra_impl_args: dict[str, Any] = {}
        attn_backend = None
        if self.use_dsa:
            attn_backend = AscendStep4DSAAttentionBackend
            extra_impl_args = {
                "sparse_config": sparse_config,
                "prefix_caching_enabled": prefix_caching_enabled,
            }
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=None,
            prefix=f"{prefix}.attn",
            per_layer_sliding_window=sliding_window,
            attn_type=attn_type,
            attn_backend=attn_backend,
            **extra_impl_args,
        )

    def _rotate_with_tables(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
        head_size: int,
        rotary_dim: int,
    ) -> torch.Tensor:
        """Eager partial NEOX RoPE from precomputed cos/sin tables."""
        positions = positions.reshape(-1).long()
        cos = cos_table[positions].unsqueeze(1)  # [tokens, 1, rotary_dim // 2]
        sin = sin_table[positions].unsqueeze(1)
        x_shape = x.shape
        x = x.view(x_shape[0], -1, head_size)
        rot, passthrough = x.split([rotary_dim, head_size - rotary_dim], dim=-1)
        cos_u = cos.to(x.dtype)
        sin_u = sin.to(x.dtype)
        x1, x2 = rot.chunk(2, dim=-1)
        rot = torch.cat([x1 * cos_u - x2 * sin_u, x2 * cos_u + x1 * sin_u], dim=-1)
        return torch.cat([rot, passthrough], dim=-1).view(x_shape)

    def _apply_rope(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Eager partial RoPE from the precomputed cos/sin cache.

        Matches RotaryEmbedding.forward_static: cos/sin each have width
        rotary_dim // 2 (the fork does not duplicate the frequencies).
        """
        return (
            self._rotate_with_tables(
                q, positions, self.rope_cos, self.rope_sin, self.head_dim, self.rotary_dim
            ),
            self._rotate_with_tables(
                k, positions, self.rope_cos, self.rope_sin, self.head_dim, self.rotary_dim
            ),
        )

    def _project_sparse_indexer(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project and normalize the sparse indexer's q/k/z plus head weights."""
        num_tokens = hidden_states.shape[0]
        index_q = self.sparse_indexer_q(hidden_states)
        index_k, _ = self.sparse_indexer_k(hidden_states)
        index_z, _ = self.sparse_indexer_z(hidden_states)

        index_q = index_q.view(num_tokens, -1, self.proxy_dim)
        index_q = self.sparse_indexer_q_norm(index_q)
        index_k = self.sparse_indexer_k_norm(index_k)
        index_q = self._rotate_with_tables(
            index_q,
            positions,
            self.indexer_rope_cos,
            self.indexer_rope_sin,
            self.proxy_dim,
            self.indexer_rotary_dim,
        )
        index_k = self._rotate_with_tables(
            index_k,
            positions,
            self.indexer_rope_cos,
            self.indexer_rope_sin,
            self.proxy_dim,
            self.indexer_rotary_dim,
        )

        num_local_groups = len(self.sparse_indexer_q.local_groups)
        index_q = index_q.view(
            num_tokens, num_local_groups, self.index_heads_per_group, self.proxy_dim
        )
        index_k = index_k.view(num_tokens, 1, self.proxy_dim)
        index_z = index_z.view(num_tokens, 1, self.proxy_dim)

        weights = self.sparse_indexer_w(hidden_states).float()
        weights = weights.view(num_tokens, num_local_groups, self.index_heads_per_group)
        weights = weights * (float(self.index_heads_per_group) ** -0.5)
        return (
            index_q.contiguous(),
            index_k.contiguous(),
            index_z.contiguous(),
            weights.contiguous(),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Per-head QK-norm, replacing the fused CuTeDSL kernel.
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head.contiguous())
        q = q_by_head.view(q.shape)

        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head.contiguous())
        k = k_by_head.view(k.shape)
        if self.use_rope:
            q, k = self._apply_rope(positions, q, k)

        if self.use_dsa:
            index_q, index_k, index_z, weights = self._project_sparse_indexer(
                positions, hidden_states
            )
            hidden_size_attn = self.num_heads * self.head_dim
            attn_output = torch.empty(
                (q.shape[0], hidden_size_attn), dtype=q.dtype, device=q.device
            )
            attn_metadata, attn_layer, kv_cache, _ = get_attention_context(
                self.attn.layer_name
            )
            attn_layer.impl.forward(
                attn_layer,
                q.view(-1, self.num_heads, self.head_dim),
                k.view(-1, self.num_kv_heads, self.head_dim),
                v.view(-1, self.num_kv_heads, self.head_dim),
                kv_cache,
                attn_metadata,
                output=attn_output.view(-1, self.num_heads, self.head_dim),
                dsa_proxy_query=index_q,
                dsa_proxy_key=index_k,
                dsa_proxy_weights=weights,
                dsa_proxy_z=index_z,
            )
        else:
            # The Ascend attention kernel requires a contiguous value tensor; the
            # qkv split leaves v as a strided view.
            attn_output = self.attn(q, k, v.contiguous())
        if self.use_head_wise_attn_gate:
            extra_dims, _ = self.g_proj(hidden_states)
            gated = (
                attn_output.view(*attn_output.shape[:-1], self.num_heads, self.head_dim)
                * extra_dims.unsqueeze(-1).sigmoid()
            )
            attn_output = gated.view(attn_output.shape)
        output, _ = self.o_proj(attn_output)
        return output


class Step4StackedExpertWeight(nn.Module):
    """One ``[n_local_experts, out, in]`` weight plus its fp8 block scale.

    Follows the step4-hf reference layout: experts are sharded *whole*
    (EP-style contiguous expert slices with the full inner dim), never along
    the inner dim -- that is the only layout under which the fp8 block
    scale's K-block grid (``ceil(in_features / 128)``) stays aligned with
    the GEMM's K-iters. Slicing the inner dim would put a single K-iter
    across a weight-block boundary and break the block-scaling assumption.

    ``weight_scale_inv`` exists only for fp8 layers; its presence (not the
    layer index) selects the fp8 GEMM path, so a checkpoint can graduate a
    layer off the not-convert list without code changes.
    """

    def __init__(
        self,
        num_global_experts: int,
        experts_start_idx: int,
        n_local_experts: int,
        out_features: int,
        in_features: int,
        *,
        quantized: bool,
        block: int = 128,
    ) -> None:
        super().__init__()
        self.num_global_experts = num_global_experts
        self.experts_start_idx = experts_start_idx
        self.n_local_experts = n_local_experts
        self.block = block
        weight_dtype = torch.float8_e4m3fn if quantized else torch.bfloat16
        self.weight = nn.Parameter(
            torch.empty(n_local_experts, out_features, in_features, dtype=weight_dtype),
            requires_grad=False,
        )
        self.weight.weight_loader = self._weight_loader
        if quantized:
            self.weight_scale_inv = nn.Parameter(
                torch.empty(
                    n_local_experts,
                    -(-out_features // block),
                    -(-in_features // block),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            self.weight_scale_inv.weight_loader = self._weight_loader
        else:
            self.weight_scale_inv = None

    def _weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_name: str | None = None,
        *,
        shard_id: Any = None,
        expert_id: int | None = None,
    ) -> None:
        del loaded_name, shard_id
        n_local = param.shape[0]
        if loaded_weight.dim() == param.dim() - 1 and expert_id is not None:
            global_idx = int(expert_id)
            if not (
                self.experts_start_idx
                <= global_idx
                < self.experts_start_idx + n_local
            ):
                return
            piece = loaded_weight.unsqueeze(0)
            target = slice(global_idx - self.experts_start_idx, global_idx - self.experts_start_idx + 1)
        else:
            total = loaded_weight.shape[0]
            if total == 1 and n_local != 1:
                piece = loaded_weight
                target = slice(None)
            elif total == self.num_global_experts:
                if n_local == self.num_global_experts:
                    piece = loaded_weight
                    target = slice(None)
                else:
                    piece = loaded_weight[
                        self.experts_start_idx : self.experts_start_idx + n_local
                    ]
                    target = slice(None)
            else:
                raise ValueError(
                    "Step4 expert tensor has an unexpected leading dimension: "
                    f"expected {self.num_global_experts} (or a broadcastable "
                    f"1), got {total}."
                )
        with torch.no_grad():
            if target == slice(None):
                param.data.copy_(piece.to(param.dtype).expand_as(param.data))
            else:
                param.data[target].copy_(piece.to(param.dtype))

    def apply_expert(
        self, expert_idx: int, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """``y = x @ w.T`` over this expert's full inner dim (fp8 or bf16)."""
        scale = (
            None
            if self.weight_scale_inv is None
            else self.weight_scale_inv[expert_idx]
        )
        return linear_fp8_or_bf16(hidden_states, self.weight[expert_idx], scale)


class Step4Experts(nn.Module):
    """Routed experts with step4-hf arithmetic.

    Sigmoid+bias router top-k (eager, parity-checked), per-expert grouped
    fp8 block GEMMs with clamped SwiGLU between the two projections, and a
    top-k *slot*-order fp32 accumulation of the weighted contributions --
    the deployed ``ep_gather`` order, kept so the rounding sequence matches
    the reference. Experts are sharded whole across the TP group (co-located
    EP semantics): each rank contributes only its local experts' slots and
    the partial result is completed by the decoder layer's fp32 all-reduce.
    """

    def __init__(
        self,
        config: Any,
        layer_idx: int,
        *,
        bf16_expert_layers: set[int],
        tp_rank: int,
        tp_size: int,
        quant_block: int,
    ) -> None:
        super().__init__()
        self.top_k = config.moe_top_k
        self.renormalize = config.norm_expert_weight
        self.routed_scaling_factor = config.moe_router_scaling_factor
        swiglu_limits = config.swiglu_limits or []
        self.swiglu_limit = (
            float(swiglu_limits[layer_idx])
            if layer_idx < len(swiglu_limits)
            else None
        )

        n_routed = config.moe_num_experts
        if n_routed % tp_size:
            raise ValueError(
                f"Step4 has {n_routed} routed experts, not divisible by "
                f"TP size {tp_size}."
            )
        self.n_local_experts = n_routed // tp_size
        self.experts_start_idx = tp_rank * self.n_local_experts
        quantized = layer_idx not in bf16_expert_layers
        self.gate_proj = Step4StackedExpertWeight(
            n_routed,
            self.experts_start_idx,
            self.n_local_experts,
            config.moe_intermediate_size,
            config.hidden_size,
            quantized=quantized,
            block=quant_block,
        )
        self.up_proj = Step4StackedExpertWeight(
            n_routed,
            self.experts_start_idx,
            self.n_local_experts,
            config.moe_intermediate_size,
            config.hidden_size,
            quantized=quantized,
            block=quant_block,
        )
        self.down_proj = Step4StackedExpertWeight(
            n_routed,
            self.experts_start_idx,
            self.n_local_experts,
            config.hidden_size,
            config.moe_intermediate_size,
            quantized=quantized,
            block=quant_block,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        router_bias: torch.Tensor,
    ) -> torch.Tensor:
        topk_weights, topk_ids = step4_router_bias_eager(
            None,
            router_logits,
            self.top_k,
            self.renormalize,
            router_bias=router_bias,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        num_tokens, hidden_dim = hidden_states.shape
        limit = self.swiglu_limit
        slot_output = torch.zeros(
            num_tokens,
            self.top_k,
            hidden_dim,
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        for local in range(self.n_local_experts):
            global_idx = self.experts_start_idx + local
            rows, slots = (topk_ids == global_idx).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            tokens = hidden_states[rows]
            gate_out = self.gate_proj.apply_expert(local, tokens)
            up_out = self.up_proj.apply_expert(local, tokens)
            if limit is None:
                activated = torch.nn.functional.silu(gate_out.float()) * up_out.float()
                activated = activated.to(gate_out.dtype)
            else:
                activated = clamped_swiglu(gate_out, up_out, limit)
            contribution = self.down_proj.apply_expert(local, activated)
            slot_output[rows, slots] = contribution.to(torch.bfloat16)

        accumulator = torch.zeros(
            num_tokens, hidden_dim, dtype=torch.float32, device=hidden_states.device
        )
        for slot in range(self.top_k):
            accumulator += (
                slot_output[:, slot].float() * topk_weights[:, slot].unsqueeze(-1)
            )
        return accumulator.to(torch.bfloat16)


class Step4MoEBlock(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.layer_idx = extract_layer_index(prefix)

        parallel_config = vllm_config.parallel_config
        config = vllm_config.model_config.hf_config

        self.hidden_size = config.hidden_size
        self.enable_eplb = parallel_config.enable_eplb
        self.n_routed_experts = config.moe_num_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = parallel_config.eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // max(
            self.tp_size, 1
        )
        if self.enable_eplb:
            raise NotImplementedError(
                "Step4 Ascend fp8 experts do not support EPLB: whole-expert "
                "sharding has no physical/logical expert remapping."
            )
        if (
            vllm_config.compilation_config.pass_config.enable_sp
            and self.tp_size > 1
        ):
            raise NotImplementedError(
                "Step4 Ascend fp8 experts do not support sequence parallel."
            )

        if self.tp_size > config.moe_num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.moe_num_experts}."
            )

        self.gate = FP32ReplicatedLinear(
            config.hidden_size,
            config.moe_num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.use_moe_router_bias = config.use_moe_router_bias
        if not self.use_moe_router_bias:
            raise ValueError("Step4 MoE currently requires use_moe_router_bias=true.")
        self.routed_scaling_factor = config.moe_router_scaling_factor
        self.router_bias = nn.Parameter(
            torch.zeros(config.moe_num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        self.need_fp32_gate = config.need_fp32_gate
        if not self.need_fp32_gate:
            raise ValueError(
                "Step4 MoE requires need_fp32_gate=true for stable router logits."
            )

        swiglu_limits = config.swiglu_limits or []
        swiglu_limit = (
            swiglu_limits[self.layer_idx]
            if self.layer_idx < len(swiglu_limits)
            else None
        )
        if swiglu_limit not in (None, 0):
            swiglu_limit = float(swiglu_limit)
            if swiglu_limit != 7.0:
                raise ValueError(
                    "Step4 fused MoE supports only swiglu_limit=7.0, got "
                    f"{swiglu_limit}."
                )
        del swiglu_limit  # enforced in Step4Experts via config

        bf16_expert_layers, quant_block = _step4_fp8_expert_layout(config)

        self.fuse_all_reduce, reduce_results = _step4_moe_reduce_policy(
            self.tp_size,
            get_dp_group().world_size,
        )

        self.share_expert = Step4MLP(
            config=config,
            hidden_size=self.hidden_size,
            intermediate_size=config.share_expert_dim,
            hidden_act="silu",
            reduce_results=reduce_results,
            prefix=f"{prefix}.share_expert",
        )
        self.experts = Step4Experts(
            config,
            self.layer_idx,
            bf16_expert_layers=bf16_expert_layers,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            quant_block=quant_block,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_is_sequence_parallel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_is_sequence_parallel:
            raise NotImplementedError(
                "Step4 Ascend fp8 experts do not support sequence parallel."
            )
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        shared_output = self.share_expert(hidden_states)

        router_logits, _ = self.gate(hidden_states)
        routed_output = self.experts(
            hidden_states, router_logits, self.router_bias
        )

        # Kept separate so _forward_ffn can combine in fp32 and all-reduce after.
        return shared_output, routed_output



class Step4DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.fp32_residual_connection = config.fp32_residual_connection
        layer_idx = extract_layer_index(prefix)
        self.layer_idx = layer_idx
        cache_config = vllm_config.cache_config
        # Step4 uses layer_types to decide which layers are SWA. Preserve
        # the cache-config window, then clear it only on this layer's copy;
        # otherwise a window supplied via --sliding-window is lost and every
        # layer is registered with a FullAttentionSpec.
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
        if config.att_impl_type == "GQA":
            norm_dtype = get_norm_dtype(config)
            num_attention_heads = None
            num_attention_groups = None
            head_dim = None
            layer_types = _step_layer_types(config)
            layer_type = (
                layer_types[layer_idx]
                if layer_idx < len(layer_types)
                else "full_attention"
            )
            sparse_config = _get_step4_sparse_config(config, layer_type)
            prefix_caching_enabled = bool(
                getattr(cache_config, "enable_prefix_caching", False)
            )
            if (
                sparse_config is not None
                and prefix_caching_enabled
            ):
                raise ValueError(
                    "Step4 DSA on Ascend does not support prefix caching "
                    "(the indexer state of cached tokens cannot be "
                    "reconstructed). Restart with --no-enable-prefix-caching."
                )
            if (
                getattr(config, "attention_other_setting", None)
                and layer_idx < len(layer_types)
                and layer_types[layer_idx]
                == config.attention_other_setting["attention_type"]
            ):
                num_attention_heads = config.attention_other_setting[
                    "num_attention_heads"
                ]
                num_attention_groups = config.attention_other_setting[
                    "num_attention_groups"
                ]
                head_dim = config.attention_other_setting["head_dim"]
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
            self.self_attn = Step4Attention(
                hidden_size=self.hidden_size,
                num_heads=num_attention_heads
                if num_attention_heads
                else config.num_attention_heads,
                max_position=max_position,
                num_kv_heads=num_attention_groups
                if num_attention_groups
                else config.num_attention_groups,
                rope_theta=config.rope_theta,
                rms_norm_eps=config.rms_norm_eps,
                qkv_bias=getattr(config, "attention_bias", False),
                head_dim=head_dim if head_dim else getattr(config, "head_dim", None),
                cache_config=cache_config,
                rope_scaling=getattr(config, "rope_scaling", None),
                sliding_window=sliding_window,
                use_head_wise_attn_gate=getattr(
                    config, "use_head_wise_attn_gate", False
                ),
                layer_types=layer_types,
                use_rope_layers=getattr(config, "use_rope_layers", []),
                yarn_only_types=getattr(config, "yarn_only_types", []),
                swa_num_attention_heads=getattr(
                    config, "swa_num_attention_heads", None
                ),
                partial_rotary_factor=partial_rotary_factor,
                prefix=f"{prefix}.self_attn",
                zero_centered=config.zero_centered,
                norm_dtype=norm_dtype,
                sparse_config=sparse_config,
                prefix_caching_enabled=prefix_caching_enabled,
            )
        else:
            raise ValueError(
                f"Unsupported attention implementation: {config.att_impl_type}"
            )
        self.use_moe = False
        self.tp_group = get_tp_group()
        self.use_fused_all_reduce = (
            get_tensor_model_parallel_world_size() > 1
            and get_dp_group().world_size == 1
        )

        moe_layers_idx = _get_step4_moe_layer_indices(config)
        if layer_idx in moe_layers_idx:
            self.moe = Step4MoEBlock(
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
                reduce_results=True,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = Step4RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            zero_centered=config.zero_centered,
            dtype=norm_dtype,
        )
        self.post_attention_layernorm = Step4RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            zero_centered=config.zero_centered,
            dtype=norm_dtype,
        )
        self.prefix = prefix

    def add_and_maybe_inplace_all_reduce(
        self, in1: torch.Tensor, in2: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self._cast_for_residual(in1) + self._cast_for_residual(in2)
        if not self.use_fused_all_reduce:
            return hidden_states
        return self.tp_group.all_reduce(hidden_states)

    def _cast_for_param_op(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.fp32_residual_connection:
            return hidden_states
        return hidden_states.to(torch.bfloat16)

    def _cast_for_residual(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.fp32_residual_connection:
            return hidden_states
        return hidden_states.to(torch.float32)

    def _forward_ffn(
        self,
        hidden_states: torch.Tensor,
        input_is_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        if self.use_moe:
            shared_output, moe_output = self.moe(
                hidden_states, input_is_sequence_parallel=input_is_sequence_parallel
            )
            # Combine shared and routed expert outputs in fp32.
            combined = self._cast_for_residual(moe_output) + self._cast_for_residual(
                shared_output
            )
            # When fuse_all_reduce=True, the runner does NOT
            # all-reduce (reduce_results=False), so we must all-reduce
            # the combined output here. When fuse_all_reduce=False,
            # routed output is either already reduced by the combine kernel or
            # reduced by _maybe_reduce_output. The shared expert path is a
            # separate RowParallelLinear, so DP/EP paths configure it to reduce
            # internally before it is combined with routed output.
            if self.moe.fuse_all_reduce:
                if self.use_fused_all_reduce:
                    combined = self.tp_group.all_reduce(combined)
                else:
                    combined = tensor_model_parallel_all_reduce(combined)
                return combined
            return combined
        return self.mlp(hidden_states)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self._cast_for_param_op(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states = self._cast_for_residual(hidden_states)
        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self._cast_for_param_op(hidden_states)

        ffn_output = self._forward_ffn(hidden_states)
        ffn_output = self._cast_for_residual(ffn_output)
        hidden_states = ffn_output + residual
        return hidden_states


def _is_fp8_weight(name: str, tensor: torch.Tensor) -> bool:
    return name.endswith(".weight") and tensor.dtype in FP8_DTYPES


def _is_fp8_scale(name: str) -> bool:
    return name.endswith(".weight_scale_inv")


# ---------------------------------------------------------------------------
# DSA sparse-indexer modules
# ---------------------------------------------------------------------------


def _get_step4_sparse_config(config: Any, layer_type: str) -> dict | None:
    """Return the checkpoint's sparse (DSA) config for a full-attention layer."""
    section = getattr(config, "sparse_config", None)
    if not isinstance(section, dict) or not section.get("enabled", False):
        return None
    apply_to = section.get("apply_to_layer_types", ("full_attention",))
    if layer_type not in apply_to:
        return None
    return section


class Step4ProviderGroupLinear(nn.Module):
    """Column-parallel-style linear sharded by sparse-indexer provider groups.

    The sparse indexer's provider groups are the same partition as the main
    attention's KV groups: with tp_size >= groups each rank owns exactly one
    (replicated) group; with tp_size < groups each rank owns a contiguous slice
    of groups.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_groups: int,
        params_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        if output_size % num_groups != 0:
            raise ValueError(
                f"Step4 indexer output {output_size} must divide into "
                f"{num_groups} provider groups."
            )
        self.rows_per_group = output_size // num_groups
        if tp_size >= num_groups:
            if tp_size % num_groups != 0:
                raise ValueError(
                    f"Step4 indexer TP size {tp_size} must be a multiple of "
                    f"the provider group count {num_groups}."
                )
            self.local_groups = [tp_rank // (tp_size // num_groups)]
        else:
            if num_groups % tp_size != 0:
                raise ValueError(
                    f"Step4 indexer provider groups {num_groups} must be a "
                    f"multiple of TP size {tp_size}."
                )
            per_rank = num_groups // tp_size
            self.local_groups = list(range(tp_rank * per_rank, (tp_rank + 1) * per_rank))
        self.weight = nn.Parameter(
            torch.empty(
                len(self.local_groups) * self.rows_per_group,
                input_size,
                dtype=params_dtype,
            )
        )
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        shards = [
            loaded_weight.narrow(0, g * self.rows_per_group, self.rows_per_group)
            for g in self.local_groups
        ]
        param.data.copy_(torch.cat(shards, dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)


class Step4IndexerLayerNorm(nn.Module):
    """Per-head LayerNorm (weight + bias) for the sparse indexer K."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute = x.float()
        mean = compute.mean(dim=-1, keepdim=True)
        # The deployed implementation computes var = E[x^2] - E[x]^2.
        variance = (compute.pow(2).mean(dim=-1, keepdim=True) - mean * mean).clamp_min(0)
        out = (compute - mean) * torch.rsqrt(variance + self.variance_epsilon)
        out = out * self.weight.float() + self.bias.float()
        return out.to(x.dtype)


class Step4Model(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        self.vocab_size = config.vocab_size
        self.config = config
        self.fp32_residual_connection = config.fp32_residual_connection
        logger.info(
            "Step4 fp32_residual_connection: %s",
            self.fp32_residual_connection,
        )

        self.moe_num_experts = config.moe_num_experts

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
            lambda prefix: Step4DecoderLayer(
                vllm_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            norm_dtype = get_norm_dtype(config)
            self.norm = Step4RMSNorm(
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

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _cast_for_residual(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.fp32_residual_connection:
            return hidden_states
        return hidden_states.to(torch.float32)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
        hidden_states = self._cast_for_residual(hidden_states)
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states = layer(positions, hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                }
            )

        return hidden_states

    def _load_expert_tensor(
        self,
        local_name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, torch.nn.Parameter],
        loaded_params: set[str],
    ) -> bool:
        """Load one 3D packed expert tensor (weight or block scale) raw.

        The expert parameters slice their global-expert dim themselves inside
        their ``weight_loader``, so no dequantization or per-expert loop is
        needed here.
        """
        for param_prefix, weight_prefix in self.expert_params_mapping:
            if weight_prefix not in local_name:
                continue
            replaced_name = local_name.replace(weight_prefix, param_prefix)
            if is_pp_missing_parameter(replaced_name, self):
                return True
            if replaced_name not in params_dict:
                return True
            param = params_dict[replaced_name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight)
            loaded_params.add(replaced_name)
            return True
        return False

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm.model_executor.model_loader.weight_utils import (
            default_weight_loader,
            maybe_remap_kv_scale_name,
        )

        config = self.config
        if config.num_attention_groups <= 1:
            raise ValueError(
                "Step4 weight loading currently supports only GQA "
                "(num_attention_groups > 1)."
            )
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        # Routed experts live under ``moe.experts.{gate,up,down}_proj``; the
        # checkpoint stores them as direct children of ``moe``.
        self.expert_params_mapping = [
            (".moe.experts.gate_proj.", ".moe.gate_proj."),
            (".moe.experts.up_proj.", ".moe.up_proj."),
            (".moe.experts.down_proj.", ".moe.down_proj."),
        ]
        disable_moe_stacked_params = [data[1] for data in self.expert_params_mapping]

        loaded_params: set[str] = set()
        # FP8 weights arrive as separate weight / weight_scale_inv entries;
        # buffer both halves and load them raw once the pair is complete.
        pending_fp8: dict[str, dict[str, torch.Tensor]] = {}

        def _flush_fp8_pair(key: str) -> None:
            entry = pending_fp8.pop(key)
            if not self._load_expert_tensor(
                f"{key}.weight", entry["weight"], params_dict, loaded_params
            ):
                raise ValueError(
                    f"Step4 FP8 weight {key}.weight does not match any expert "
                    "parameter mapping."
                )
            if not self._load_expert_tensor(
                f"{key}.weight_scale_inv", entry["scale"], params_dict, loaded_params
            ):
                raise ValueError(
                    f"Step4 FP8 scale {key}.weight_scale_inv does not match "
                    "any expert parameter mapping."
                )

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if name.startswith("model."):
                local_name = name[len("model.") :]
                full_name = name
            else:
                local_name = name
                full_name = f"model.{name}" if name else "model"

            spec_layer = get_spec_layer_idx_from_weight_name(config, full_name)
            if spec_layer is not None:
                # skip spec decode (MTP) layers for the main model
                continue

            # Skip any layers beyond the main model's depth (e.g., MTP layers)
            if full_name.startswith("model.layers."):
                parts = full_name.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    layer_idx = int(parts[2])
                    if layer_idx >= config.num_hidden_layers:
                        continue

            # DSA sparse-indexer weights have no consumer in the dense fallback.
            if any(marker in local_name for marker in _DSA_ONLY_WEIGHT_MARKERS):
                continue

            if _is_fp8_weight(local_name, loaded_weight) or _is_fp8_scale(local_name):
                key = local_name[: -len(".weight_scale_inv")] if _is_fp8_scale(
                    local_name
                ) else local_name[: -len(".weight")]
                entry = pending_fp8.setdefault(key, {})
                entry["scale" if _is_fp8_scale(local_name) else "weight"] = (
                    loaded_weight
                )
                if "weight" in entry and "scale" in entry:
                    _flush_fp8_pair(key)
                continue

            remapped_name = maybe_remap_kv_scale_name(local_name, params_dict)
            if remapped_name is None:
                continue
            local_name = remapped_name

            loaded = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in local_name:
                    continue
                if any(
                    disable_moe_stacked_param in local_name
                    for disable_moe_stacked_param in disable_moe_stacked_params
                ):
                    continue
                replaced_name = local_name.replace(weight_name, param_name)
                if is_pp_missing_parameter(replaced_name, self):
                    continue
                if replaced_name not in params_dict:
                    continue
                param = params_dict[replaced_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(replaced_name)
                loaded = True
                break
            if loaded:
                continue

            if self._load_expert_tensor(
                local_name, loaded_weight, params_dict, loaded_params
            ):
                continue

            if is_pp_missing_parameter(local_name, self):
                continue
            if local_name not in params_dict:
                continue
            param = params_dict[local_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(local_name)

        if pending_fp8:
            raise RuntimeError(
                "Step4 FP8 weights are missing their weight_scale_inv pair: "
                f"{sorted(pending_fp8)}"
            )
        return loaded_params


class AscendStep4ForCausalLM(nn.Module, SupportsPP, MixtureOfExperts):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_regex={
            re.compile(r"^vit_large_projector\.weight$"): None,
        },
        orig_to_new_substr={".share_expert.": ".moe.share_expert."},
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        model_config = vllm_config.model_config
        valid_vocab_size = _require_resolved_valid_vocab_size(model_config)
        config = model_config.hf_config
        self.config = config
        self.fp32_residual_connection = config.fp32_residual_connection
        if config.quantization_config is not None:
            logger.info_once(
                "Step4 checkpoint declares quantization_config; the Ascend "
                "port runs routed experts directly in fp8 (128x128 block "
                "scales, dynamic activation quantization), mirroring the "
                "step4-hf reference."
            )
        self.model = Step4Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=None,
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
        example_layer: Step4MoEBlock | None = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            assert isinstance(layer, Step4DecoderLayer)
            if hasattr(layer, "moe") and isinstance(layer.moe, Step4MoEBlock):
                example_layer = layer.moe
                self.moe_layers.append(layer.moe.experts)

        _set_step4_moe_protocol_metadata(self, example_layer)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.fp32_residual_connection:
            hidden_states = hidden_states.to(torch.bfloat16)
        hidden_states = self.model.norm(hidden_states)
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        if self.num_local_physical_experts != num_local_physical_experts:
            raise ValueError(
                "Step4 EPLB cannot change the number of local physical experts: "
                f"expected={self.num_local_physical_experts}, "
                f"got={num_local_physical_experts}."
            )
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if not isinstance(layer, Step4DecoderLayer):
                continue
            moe = getattr(layer, "moe", None)
            if not isinstance(moe, Step4MoEBlock):
                continue
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = ["vision_model."]
        if self.config.tie_word_embeddings:
            skip_prefixes.append("lm_head.")
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=skip_prefixes,
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
