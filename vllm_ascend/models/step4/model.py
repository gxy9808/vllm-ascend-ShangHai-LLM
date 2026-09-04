# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Step4 model for vllm-ascend.

Ported from the Step4 vLLM adaptation (``vllm/models/step4/model.py``).
Differences against the CUDA port:

- The DSA sparse attention backends (CuTeDSL, SM90-only) are not available on
  Ascend. A checkpoint that enables its sparse section is rejected; set
  ``VLLM_STEP4_SPARSE=0`` to run the dense fallback of a DSA-capable
  checkpoint (indexer weights are then ignored and full attention is
  computed on every layer).
- The fused QKNorm+RoPE+KV-cache-write operator of the Optimus stack is
  replaced by the torch ``fused_qknorm_rope_forward_impl`` op followed by
  the stock per-backend KV cache update.
- ``valid_vocab_size`` resolution (a vLLM-core patch on the CUDA side) is
  optional here: stock ModelConfig carries no such field, so the padded
  checkpoint vocabulary is passed to LogitsProcessor via ``org_vocab_size``.
"""

import copy
import functools
import math
import typing
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import regex as re
import torch
from torch import nn
from torch.nn.parameter import Parameter

from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
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
from vllm.model_executor.layers.layernorm import RMSNorm as NaiveRMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
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
    sequence_parallel_chunk,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionType

from . import envs as step4_envs
from .kernels import (
    fused_qknorm_rope_forward_impl,
    get_step4_sparse_config,
    router_bias_func,
)
from .layernorm import OptimusRMSNorm

logger = init_logger(__name__)

_DSA_UNSUPPORTED_MSG = (
    "Step4 DSA sparse attention is not yet supported on Ascend: the CUDA "
    "port backs it with SM90 CuTeDSL kernels. Set VLLM_STEP4_SPARSE=0 to "
    "run the dense fallback of this checkpoint (full attention on every "
    "layer, indexer weights ignored), or use a native dense Step4 variant."
)


def step4_materialize_gate_input(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.clone()


def step4_materialize_gate_input_fake(tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(tensor)


direct_register_custom_op(
    op_name="step4_materialize_gate_input",
    op_func=step4_materialize_gate_input,
    fake_impl=step4_materialize_gate_input_fake,
)


_OPTIONAL_FP8_ATTN_SCALE_SUFFIXES = (
    ".attn.q_scale",
    ".attn.k_scale",
    ".attn.v_scale",
    ".attn.q_quant_scale",
    ".attn.k_quant_scale",
    ".attn.v_quant_scale",
    ".attn.prob_scale",
)

STEP4_PACKED_MODULES_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "qkvg_proj": ["q_proj", "k_proj", "v_proj", "g_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def _require_resolved_valid_vocab_size(model_config: ModelConfig) -> int:
    valid = getattr(model_config, "valid_vocab_size", None)
    if valid is not None:
        return int(valid)
    return model_config.get_vocab_size()


def _step_layer_types(config: Any) -> list[str]:
    """Per-layer attention types spanning the dense stack and the MTP layers."""
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


def _mark_optional_fp8_attention_scales_loaded(
    loaded_params: set[str],
    params_dict: dict[str, torch.nn.Parameter],
) -> None:
    # Step FP8 attention scale tensors are optional calibration metadata.
    loaded_params.update(
        name for name in params_dict if name.endswith(_OPTIONAL_FP8_ATTN_SCALE_SUFFIXES)
    )


def RMSNormFactory(
    hidden_size: int,
    eps: float = 1e-6,
    zero_centered: bool = False,
    dtype: torch.dtype | None = None,
):
    if zero_centered:
        return OptimusRMSNorm(hidden_size, eps, zero_centered, dtype=dtype)
    return NaiveRMSNorm(hidden_size, eps, dtype=dtype)


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


def pad_param(
    weight: torch.Tensor,
    name: str,
    param: torch.nn.Parameter,
    quant_config: QuantizationConfig | None = None,
) -> torch.Tensor:
    """Pad 2D weight for groupwise quantization TP sharding."""
    if weight.dim() != 2:
        return weight

    quant_method = getattr(param, "quant_method", None)
    if (
        quant_config is None
        or quant_config.get_name() != "groupwise_quant"
        or not quant_method
    ):
        return weight

    world_size = get_tensor_model_parallel_world_size()
    group_size = quant_config.group_size

    if ("down_proj.scales" in name) or ("w2_weight_scale" in name):
        group_size = 1

    ic, oc = weight.shape
    if ("down" in name) or ("w2" in name):
        ic_pad = (
            int(math.ceil(ic / group_size / world_size) * world_size * group_size) - ic
        )
        out = torch.nn.functional.pad(weight, (0, 0, 0, ic_pad), "constant", 0)
    else:
        oc_pad = (
            int(math.ceil(oc / group_size / world_size) * world_size * group_size) - oc
        )
        out = torch.nn.functional.pad(weight, (0, oc_pad, 0, 0), "constant", 0)

    logger.debug(
        "padding %s, quant_config=%s, original weight.shape=%s, padded weight.shape=%s",
        name,
        quant_config,
        tuple(weight.shape),
        tuple(out.shape),
    )
    return out


def _pad_size_for_groupwise_quant(
    size: int,
    quant_config: QuantizationConfig | None = None,
) -> int:
    """Pad `size` to a multiple of `group_size * tensor_parallel_world_size`."""
    if quant_config is None or quant_config.get_name() != "groupwise_quant":
        return size

    group_size = getattr(quant_config, "group_size", None)
    if not isinstance(group_size, int) or group_size <= 0:
        return size

    world_size = get_tensor_model_parallel_world_size()
    multiple = group_size * world_size
    return int(math.ceil(size / multiple) * multiple)


def _is_mxfp4_moe_quant_config(quant_config: QuantizationConfig | None) -> bool:
    return quant_config is not None and quant_config.get_name() == "mxfp4"


class FP32ReplicatedLinear(ReplicatedLinear):
    """Replicated gate projection computed in FP32 for router stability."""

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        router_logits = torch.nn.functional.linear(
            x.to(torch.float32), self.weight.to(torch.float32)
        )
        return router_logits, None


class Step4MLP(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        intermediate_size = _pad_size_for_groupwise_quant(
            intermediate_size, quant_config
        )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )

        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()
        self.prefix = prefix
        self.hidden_size = hidden_size
        self.limit = None
        layer_idx = extract_layer_index(prefix)
        swiglu_limit = _per_layer_value(
            getattr(config, "swiglu_limits_shared", None),
            layer_idx,
            name="swiglu_limits_shared",
            default=None,
        )
        if swiglu_limit not in (None, 0):
            self.limit = swiglu_limit
            self.act_fn = SwigluStepAndMul(limit=self.limit)

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
        quant_config: QuantizationConfig | None = None,
        rope_scaling: dict[str, Any] | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        # Step4 specific args
        sliding_window: int | None = None,
        use_head_wise_attn_gate: bool = False,
        layer_types: list = None,
        use_rope_layers: list = None,
        yarn_only_types: list = None,
        swa_num_attention_heads: int | None = None,
        partial_rotary_factor: float = 1.0,
        zero_centered: bool = True,
        vllm_config: VllmConfig | None = None,
        norm_dtype: torch.dtype | None = None,
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
            sliding_window = sliding_window
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

        self.rank = get_tensor_model_parallel_rank()
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
        linear_quant_config = (
            quant_config
            if quant_config is None or quant_config.get_name() != "fp8"
            else None
        )

        # Q/K/V and the per-head attention gate consume the same normalized
        # hidden states.  When explicitly enabled, pack them into one
        # column-parallel GEMM if KV heads are partitioned (rather than
        # replicated) across TP ranks.  The local output layout is
        # [Q, K, V, gate].
        self.fuse_qkv_gate = (
            step4_envs.enable_qkvg_proj()
            and use_head_wise_attn_gate
            and self.total_num_kv_heads >= tp_size
        )
        if self.fuse_qkv_gate:
            self.qkvg_proj = MergedColumnParallelLinear(
                hidden_size,
                [
                    self.total_num_heads * self.head_dim,
                    self.total_num_kv_heads * self.head_dim,
                    self.total_num_kv_heads * self.head_dim,
                    self.total_num_heads,
                ],
                bias=qkv_bias,
                quant_config=linear_quant_config,
                # Keep the original prefix so quantization configuration that
                # targets qkv_proj continues to apply to the fused module.
                prefix=f"{prefix}.qkv_proj",
            )
            if self.qkvg_proj.bias is not None:
                # g_proj is bias-free.  The fused allocation includes a gate
                # bias only because qkv_bias applies to the first three shards.
                with torch.no_grad():
                    self.qkvg_proj.bias[-self.num_heads :].zero_()
            self.params_dtype = self.qkvg_proj.params_dtype
        else:
            self.qkv_proj = QKVParallelLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=qkv_bias,
                quant_config=linear_quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
            self.params_dtype = self.qkv_proj.params_dtype
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=linear_quant_config,
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

        self.zero_centered = zero_centered
        self.q_norm = RMSNormFactory(
            self.head_dim,
            eps=rms_norm_eps,
            zero_centered=self.zero_centered,
            dtype=norm_dtype,
        )
        self.k_norm = RMSNormFactory(
            self.head_dim,
            eps=rms_norm_eps,
            zero_centered=self.zero_centered,
            dtype=norm_dtype,
        )
        self.use_head_wise_attn_gate = use_head_wise_attn_gate
        if use_head_wise_attn_gate and not self.fuse_qkv_gate:
            self.g_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads,
                bias=False,
                quant_config=linear_quant_config,
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

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            per_layer_sliding_window=sliding_window,
            attn_type=attn_type,
        )
        self.max_position_embeddings = max_position

        self.rotary_cache = self.rotary_emb.cos_sin_cache
        self.rope_cos, self.rope_sin = self.rotary_cache.chunk(2, dim=-1)
        # The fused QKNorm+RoPE path is the torch op registered in
        # ``.kernels``; the KV cache write stays with the attention backend.
        self.use_optimus_qknorm = self.use_rope

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        reduce_scatter_output: bool = False,
    ) -> torch.Tensor:
        if self.fuse_qkv_gate:
            qkvg, _ = self.qkvg_proj(hidden_states)
            qkv, extra_dims = qkvg.split(
                [self.q_size + 2 * self.kv_size, self.num_heads], dim=-1
            )
            qkv = qkv.contiguous()
            extra_dims = extra_dims.contiguous()
        else:
            qkv, _ = self.qkv_proj(hidden_states)
            extra_dims = None

        if self.use_optimus_qknorm:
            eps = self.q_norm.variance_epsilon
            q, k, v = fused_qknorm_rope_forward_impl(
                qkv,
                self.q_norm.weight,
                self.k_norm.weight,
                self.rope_cos,
                self.rope_sin,
                positions,
                self.head_dim,
                self.num_heads,
                self.num_kv_heads,
                self.rotary_dim // 2,
                eps,
                norm_weight_bias=1.0 if self.zero_centered else 0.0,
            )
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            # Add qk-norm inline similar to Qwen3 MOE attention
            q_by_head = q.view(
                *q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim
            )
            q_by_head = self.q_norm(q_by_head.contiguous())
            q = q_by_head.view(q.shape)

            k_by_head = k.view(
                *k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim
            )
            k_by_head = self.k_norm(k_by_head.contiguous())
            k = k_by_head.view(k.shape)
            if self.use_rope:
                q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        if extra_dims is None and self.use_head_wise_attn_gate:
            extra_dims, _ = self.g_proj(hidden_states)

        if extra_dims is not None:
            # Keep an opaque materialization op so the broadcasted gate
            # multiply consumes stable BF16 buffers under compilation.
            attn_output = torch.ops.vllm.step4_materialize_gate_input(attn_output)
            extra_dims = torch.ops.vllm.step4_materialize_gate_input(extra_dims)

        if self.use_head_wise_attn_gate:
            output = (
                attn_output.view(*attn_output.shape[:-1], self.num_heads, self.head_dim)
                * extra_dims.unsqueeze(-1).sigmoid()
            )
            attn_output = output.view(*attn_output.shape)
        if reduce_scatter_output:
            output, _ = self.o_proj(
                attn_output, reduce_scatter_results=True, reduce_scatter_dim=0
            )
        else:
            output, _ = self.o_proj(attn_output)
        return output


class FusedMoEBlock(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        from vllm.model_executor.layers.fused_moe import FusedMoEFactory

        self.tp_size = get_tensor_model_parallel_world_size()
        self.layer_idx = extract_layer_index(prefix)

        self.ep_size = get_ep_group().device_group.size()
        self.ep_rank = get_ep_group().device_group.rank()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        self.enable_eplb = parallel_config.enable_eplb
        self.n_routed_experts = config.moe_num_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = parallel_config.eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
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

        activation = "silu"
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
            activation = "swiglustep"
            logger.debug(
                "step4 layer_idx: %s, activation: %s, limit: %s",
                self.layer_idx,
                activation,
                swiglu_limit,
            )

        # The MoE runner does not forward the correction bias or routed
        # scaling factor to custom routing functions, so bind both here.
        # router_bias is loaded in place, making the captured Parameter stable.
        custom_routing_function = functools.partial(
            router_bias_func,
            router_bias=self.router_bias,
            routed_scaling_factor=config.moe_router_scaling_factor,
        )
        share_expert_dim = _pad_size_for_groupwise_quant(
            config.share_expert_dim, quant_config
        )
        moe_intermediate_size = _pad_size_for_groupwise_quant(
            config.moe_intermediate_size, quant_config
        )
        self.fuse_all_reduce, reduce_results = _step4_moe_reduce_policy(
            self.tp_size,
            get_dp_group().world_size,
        )
        effective_sequence_parallel = (
            vllm_config.compilation_config.pass_config.enable_sp and self.tp_size > 1
        )

        self.share_expert = Step4MLP(
            config=config,
            hidden_size=self.hidden_size,
            intermediate_size=share_expert_dim,
            hidden_act="silu",
            reduce_results=reduce_results,
            is_sequence_parallel=effective_sequence_parallel,
            quant_config=quant_config
            if quant_config and quant_config.get_name() != "fp8"
            else None,
            prefix=f"{prefix}.share_expert",
        )
        # Keep the shared expert outside FusedMoEFactory so Step4 can combine
        # shared and routed outputs in FP32 before the final all-reduce.
        kwargs = {"custom_routing_function": custom_routing_function}
        self.experts = FusedMoEFactory(
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            hidden_size=config.hidden_size,
            intermediate_size=moe_intermediate_size,
            reduce_results=reduce_results,
            renormalize=config.norm_expert_weight,
            quant_config=quant_config,
            activation=activation,
            prefix=f"{prefix}.experts",
            e_score_correction_bias=self.router_bias,
            routed_scaling_factor=config.moe_router_scaling_factor,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=effective_sequence_parallel,
            router_logits_dtype=torch.float32,
            **kwargs,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_is_sequence_parallel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)
        if (
            self.experts.moe_config.is_sequence_parallel
            and not input_is_sequence_parallel
        ):
            hidden_states = sequence_parallel_chunk(hidden_states)

        shared_output = self.share_expert(hidden_states)

        if self.experts.is_internal_router:
            routed_output = self.experts(
                hidden_states=hidden_states, router_logits=hidden_states
            )
        else:
            router_logits, _ = self.gate(hidden_states)
            routed_output = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
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
        quant_config = vllm_config.quant_config
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
        if get_step4_sparse_config(config) is not None:
            raise NotImplementedError(_DSA_UNSUPPORTED_MSG)
        if config.att_impl_type == "GQA":
            norm_dtype = get_norm_dtype(config)
            num_attention_heads = None
            num_attention_groups = None
            head_dim = None
            layer_types = _step_layer_types(config)
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
                quant_config=quant_config,
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
                vllm_config=vllm_config,
                norm_dtype=norm_dtype,
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
        if self.use_fused_all_reduce:
            logger.warning_once("Enable custom fused all reduce...")
        else:
            logger.warning_once("Disable custom fused all reduce...")

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
        self.use_attention_o_proj_reduce_scatter = (
            step4_envs.o_proj_reduce_scatter()
            and self.use_moe
            and self.moe.experts.moe_config.is_sequence_parallel
            and self.tp_group.world_size > 1
        )
        if self.use_attention_o_proj_reduce_scatter:
            logger.warning_once("Enable Step4 attention o_proj reduce-scatter path.")

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
        residual: torch.Tensor | None = None,
        orig_num_tokens: int | None = None,
    ) -> torch.Tensor:
        if self.use_moe:
            shared_output, moe_output = self.moe(
                hidden_states, input_is_sequence_parallel=input_is_sequence_parallel
            )
            if self.moe.experts.moe_config.is_sequence_parallel:
                if input_is_sequence_parallel:
                    assert residual is not None
                    assert orig_num_tokens is not None
                    ffn_output = self._cast_for_residual(
                        moe_output
                    ) + self._cast_for_residual(shared_output)
                    hidden_states = tensor_model_parallel_all_gather(
                        ffn_output + residual, dim=0
                    )
                    return hidden_states[:orig_num_tokens]
                ffn_output = tensor_model_parallel_all_gather(
                    self._cast_for_residual(moe_output)
                    + self._cast_for_residual(shared_output),
                    dim=0,
                )
                return ffn_output[: hidden_states.shape[0]]
            # Combine shared and routed expert outputs
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
        use_attention_o_proj_reduce_scatter = (
            self.use_attention_o_proj_reduce_scatter and hidden_states.dim() == 2
        )
        orig_num_tokens = hidden_states.shape[0]
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self._cast_for_param_op(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            reduce_scatter_output=use_attention_o_proj_reduce_scatter,
        )
        hidden_states = self._cast_for_residual(hidden_states)
        if use_attention_o_proj_reduce_scatter:
            residual = sequence_parallel_chunk(residual)
        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self._cast_for_param_op(hidden_states)

        ffn_output = self._forward_ffn(
            hidden_states,
            input_is_sequence_parallel=use_attention_o_proj_reduce_scatter,
            residual=residual if use_attention_o_proj_reduce_scatter else None,
            orig_num_tokens=(
                orig_num_tokens if use_attention_o_proj_reduce_scatter else None
            ),
        )
        if use_attention_o_proj_reduce_scatter:
            return ffn_output
        ffn_output = self._cast_for_residual(ffn_output)
        hidden_states = ffn_output + residual
        return hidden_states


@support_torch_compile
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
            lambda prefix: Step4DecoderLayer(
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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm.model_executor.layers.fused_moe import (
            fused_moe_make_expert_params_mapping,
        )
        from vllm.model_executor.model_loader.weight_utils import (
            default_weight_loader,
            maybe_remap_kv_scale_name,
        )

        config = self.config
        quant_config = self.vllm_config.quant_config
        if config.num_attention_groups <= 1:
            raise ValueError(
                "Step4 weight loading currently supports only GQA "
                "(num_attention_groups > 1)."
            )
        qkv_params_mapping = []
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkvg_proj", "q_proj", 0),
            ("qkvg_proj", "k_proj", 1),
            ("qkvg_proj", "v_proj", 2),
            ("qkvg_proj", "g_proj", 3),
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        # Quantized expert wrappers insert base_layer into parameter names.
        base_layer = (
            "base_layer." if any(".base_layer." in name for name in params_dict) else ""
        )
        is_mxfp4_moe_quant = _is_mxfp4_moe_quant_config(quant_config)

        # Old packed 3D format: .moe.gate_proj.weight [num_experts, out, in]
        expert_params_mapping = [
            (
                f".moe.experts.routed_experts.{base_layer}w13_weight",
                ".moe.gate_proj.weight",
                "w1",
            ),
            (
                f".moe.experts.routed_experts.{base_layer}w13_weight",
                ".moe.up_proj.weight",
                "w3",
            ),
            (
                f".moe.experts.routed_experts.{base_layer}w2_weight",
                ".moe.down_proj.weight",
                "w2",
            ),
            (
                f".moe.experts.routed_experts.{base_layer}w13_weight_scale_2",
                ".moe.gate_proj.weight_scale_2",
                "w1",
            ),
            (
                f".moe.experts.routed_experts.{base_layer}w13_weight_scale_2",
                ".moe.up_proj.weight_scale_2",
                "w3",
            ),
            (
                f".moe.experts.routed_experts.{base_layer}w2_weight_scale_2",
                ".moe.down_proj.weight_scale_2",
                "w2",
            ),
            (
                f".moe.experts.routed_experts.{base_layer}w13_weight_scale",
                ".moe.gate_proj.weight_scale",
                "w1",
            ),
            (
                f".moe.experts.routed_experts.{base_layer}w13_weight_scale",
                ".moe.up_proj.weight_scale",
                "w3",
            ),
            (
                f".moe.experts.routed_experts.{base_layer}w2_weight_scale",
                ".moe.down_proj.weight_scale",
                "w2",
            ),
            # Required due to the Step3 HF model's packed expert format:
            # input scales are stored as moe.{gate,up,down}_proj.input_scale
            # rather than the standard per-expert format handled generically.
            (
                f".moe.experts.routed_experts.{base_layer}w13_input_scale",
                ".moe.gate_proj.input_scale",
                "w1",
            ),
            (
                f".moe.experts.routed_experts.{base_layer}w13_input_scale",
                ".moe.up_proj.input_scale",
                "w3",
            ),
            (
                f".moe.experts.routed_experts.{base_layer}w2_input_scale",
                ".moe.down_proj.input_scale",
                "w2",
            ),
        ]
        if is_mxfp4_moe_quant:
            expert_params_mapping = [
                (".moe.experts.w13_weight_scale", ".moe.gate_proj.weight_scale", "w1"),
                (".moe.experts.w13_weight_scale", ".moe.up_proj.weight_scale", "w3"),
                (".moe.experts.w2_weight_scale", ".moe.down_proj.weight_scale", "w2"),
                (".moe.experts.w13_weight", ".moe.gate_proj.weight", "w1"),
                (".moe.experts.w13_weight", ".moe.up_proj.weight", "w3"),
                (".moe.experts.w2_weight", ".moe.down_proj.weight", "w2"),
            ]

        is_groupwise_quant = (
            quant_config is not None and quant_config.get_name() == "groupwise_quant"
        )
        if is_groupwise_quant:
            expert_params_mapping = [
                (".moe.experts.w13_weight", ".moe.gate_proj.qweight", "w1"),
                (".moe.experts.w13_weight", ".moe.up_proj.qweight", "w3"),
                (".moe.experts.w2_weight", ".moe.down_proj.qweight", "w2"),
                (".moe.experts.w13_weight_scale", ".moe.gate_proj.scales", "w1"),
                (".moe.experts.w13_weight_scale", ".moe.up_proj.scales", "w3"),
                (".moe.experts.w2_weight_scale", ".moe.down_proj.scales", "w2"),
            ]

        # New per-expert format: .moe.experts.E.gate_proj.weight_packed [out, in]
        per_expert_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.moe_num_experts,
        )

        disable_moe_stacked_params = [data[1] for data in expert_params_mapping]

        def _as_mxfp4_param_dtype(
            param: torch.nn.Parameter, weight: torch.Tensor
        ) -> torch.Tensor:
            fp8_e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
            raw_dtypes = (torch.int8,)
            if fp8_e8m0_dtype is not None:
                raw_dtypes = raw_dtypes + (fp8_e8m0_dtype,)
            if param.dtype == torch.uint8 and weight.dtype in raw_dtypes:
                return weight.contiguous().view(torch.uint8)
            return weight

        def _load_mxfp4_weight(
            param: torch.nn.Parameter,
            weight: torch.Tensor,
            name: str,
            shard_id: str,
        ) -> None:
            mxfp4_block = 32
            use_ep = self.parallel_config.enable_expert_parallel
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
            intermediate_size = self.config.moe_intermediate_size
            intermediate_size_block = intermediate_size // mxfp4_block
            per_rank_intermediate_size_block = math.ceil(
                intermediate_size_block / tp_size
            )
            per_rank_intermediate_size = per_rank_intermediate_size_block * mxfp4_block
            tp_rank_start = tp_rank * per_rank_intermediate_size
            tp_rank_end = min(
                (tp_rank + 1) * per_rank_intermediate_size, intermediate_size
            )

            if use_ep:
                ep_size = get_ep_group().world_size
                ep_rank = get_ep_group().rank
                experts_per_rank = self.config.moe_num_experts // ep_size
                expert_slice = slice(
                    ep_rank * experts_per_rank, (ep_rank + 1) * experts_per_rank
                )
            else:
                expert_slice = slice(None)

            if ".w13_weight_scale" in name:
                weight_slice = (
                    weight[expert_slice, ...]
                    if use_ep
                    else weight[:, tp_rank_start:tp_rank_end, ...]
                )
                weight_slice = _as_mxfp4_param_dtype(param, weight_slice)
                dest = param.data[: weight_slice.shape[0]]
                rows, cols = weight_slice.shape[1], weight_slice.shape[2]
                # MXFP4 backend layout conversion expects w13 as gate/up
                # row pairs, not contiguous gate and up blocks.
                if shard_id == "w1":
                    dest[:, : 2 * rows : 2, :cols].copy_(weight_slice)
                elif shard_id == "w3":
                    dest[:, 1 : 2 * rows : 2, :cols].copy_(weight_slice)
                else:
                    dest[:, :rows, :cols].copy_(weight_slice)
                return

            if ".w2_weight_scale" in name:
                start = tp_rank_start // mxfp4_block
                end = tp_rank_end // mxfp4_block
                weight_slice = (
                    weight[expert_slice, ...] if use_ep else weight[..., start:end]
                )
                weight_slice = _as_mxfp4_param_dtype(param, weight_slice)
                dest = param.data[: weight_slice.shape[0]]
                rows, cols = weight_slice.shape[1], weight_slice.shape[2]
                dest[:, :rows, :cols].copy_(weight_slice)
                return

            if ".w13_weight" in name:
                weight_slice = (
                    weight[expert_slice, ...]
                    if use_ep
                    else weight[:, tp_rank_start:tp_rank_end, ...]
                )
                weight_slice = _as_mxfp4_param_dtype(param, weight_slice)
                dest = param.data[: weight_slice.shape[0]]
                rows, cols = weight_slice.shape[1], weight_slice.shape[2]
                if shard_id == "w1":
                    dest[:, : 2 * rows : 2, :cols].copy_(weight_slice)
                elif shard_id == "w3":
                    dest[:, 1 : 2 * rows : 2, :cols].copy_(weight_slice)
                else:
                    dest[:, :rows, :cols].copy_(weight_slice)
                return

            if ".w2_weight" in name:
                start = tp_rank_start // 2
                end = tp_rank_end // 2
                weight_slice = (
                    weight[expert_slice, ...] if use_ep else weight[..., start:end]
                )
                weight_slice = _as_mxfp4_param_dtype(param, weight_slice)
                dest = param.data[: weight_slice.shape[0]]
                rows, cols = weight_slice.shape[1], weight_slice.shape[2]
                dest[:, :rows, :cols].copy_(weight_slice)

        import threading

        # for name, loaded_weight in weights:
        def _worker(name, loaded_weight):
            loaded_params: set[str] = set()
            if name.startswith("model."):
                local_name = name[len("model.") :]
                full_name = name
            else:
                local_name = name
                full_name = f"model.{name}" if name else "model"

            spec_layer = get_spec_layer_idx_from_weight_name(config, full_name)
            if spec_layer is not None:
                return loaded_params

            # Skip any layers beyond the main model's depth (e.g., MTP layers)
            if full_name.startswith("model.layers."):
                parts = full_name.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    layer_idx = int(parts[2])
                    if layer_idx >= config.num_hidden_layers:
                        return loaded_params

            remapped_name = maybe_remap_kv_scale_name(local_name, params_dict)
            if remapped_name is None:
                return loaded_params
            local_name = remapped_name

            # Per-expert MoE weights (new format from LLM Compressor):
            # .moe.experts.{E}.{gate,up,down}_proj.{weight_packed,scale,...}
            if ".moe.experts." in local_name:
                is_expert_weight = False
                for mapping in per_expert_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in local_name:
                        continue
                    is_expert_weight = True
                    name_mapped = local_name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    if name_mapped not in params_dict:
                        continue
                    param = params_dict[name_mapped]
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    loaded_weight_padded = pad_param(
                        loaded_weight,
                        name_mapped,
                        param,
                        quant_config,
                    )
                    success = weight_loader(
                        param,
                        loaded_weight_padded,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        loaded_params.add(name_mapped)
                        break
                else:
                    if (
                        not is_expert_weight
                        and not is_pp_missing_parameter(local_name, self)
                        and local_name in params_dict
                    ):
                        param = params_dict[local_name]
                        weight_loader = getattr(
                            param,
                            "weight_loader",
                            default_weight_loader,
                        )
                        loaded_weight_padded = pad_param(
                            loaded_weight,
                            local_name,
                            param,
                            quant_config,
                        )
                        weight_loader(param, loaded_weight_padded)
                        loaded_params.add(local_name)
                return loaded_params

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
                loaded_weight_padded = pad_param(
                    loaded_weight,
                    replaced_name,
                    param,
                    quant_config,
                )
                weight_loader(param, loaded_weight_padded, shard_id)
                loaded_params.add(replaced_name)
                break
            else:
                for param_name, weight_name, shard_id in expert_params_mapping:
                    if weight_name not in local_name:
                        continue
                    replaced_name = local_name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(replaced_name, self):
                        continue
                    if (
                        replaced_name.endswith(".bias")
                        or replaced_name.endswith("_bias")
                    ) and replaced_name not in params_dict:
                        continue
                    if replaced_name not in params_dict:
                        continue
                    param = params_dict[replaced_name]
                    weight_loader = param.weight_loader
                    moe_expert_num = self.moe_num_experts
                    if is_mxfp4_moe_quant:
                        _load_mxfp4_weight(
                            param, loaded_weight, replaced_name, shard_id
                        )
                        loaded_params.add(replaced_name)
                        break
                    # Per-tensor global scales (e.g. weight_global_scale)
                    # have shape [1] in compressed-tensors NVFP4 checkpoints.
                    # Expand to per-expert before the iteration loop.
                    if loaded_weight.ndim == 0:
                        loaded_weight = loaded_weight.unsqueeze(0).expand(
                            moe_expert_num
                        )
                    elif (
                        loaded_weight.shape[0] == 1
                        and loaded_weight.shape[0] != moe_expert_num
                    ):
                        loaded_weight = loaded_weight.expand(
                            moe_expert_num, *loaded_weight.shape[1:]
                        )
                    assert loaded_weight.shape[0] == moe_expert_num
                    for expert_id in range(moe_expert_num):
                        loaded_weight_expert = pad_param(
                            loaded_weight[expert_id],
                            replaced_name,
                            param,
                            quant_config,
                        )
                        weight_loader(
                            param,
                            loaded_weight_expert,
                            replaced_name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    loaded_params.add(replaced_name)
                    break
                else:
                    for (
                        param_name,
                        weight_name,
                        start_idx,
                        end_idx,
                    ) in qkv_params_mapping:
                        if weight_name not in local_name:
                            continue
                        replaced_name = local_name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(replaced_name, self):
                            continue
                        if replaced_name not in params_dict:
                            continue
                        param = params_dict[replaced_name]
                        dim = param.shape[param.output_dim]
                        begin_idx = int(start_idx * dim)
                        end_idx = int(end_idx * dim)
                        param_slice = param.narrow(
                            param.output_dim, begin_idx, end_idx - begin_idx
                        )
                        param_slice.copy_(loaded_weight)
                        loaded_params.add(replaced_name)
                        break
                    else:
                        if is_pp_missing_parameter(local_name, self):
                            return loaded_params
                        if "expert_bias" in local_name:
                            logger.warning_once("ignore expert_bias")
                            return loaded_params
                        if local_name not in params_dict:
                            return loaded_params
                        param = params_dict[local_name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        try:
                            loaded_weight_padded = pad_param(
                                loaded_weight,
                                local_name,
                                param,
                                quant_config,
                            )
                            weight_loader(param, loaded_weight_padded)
                        except Exception as e:
                            logger.error(
                                "shape: %s, param shape: %s, %s",
                                loaded_weight.shape,
                                param.shape,
                                local_name,
                            )
                            raise e

                        loaded_params.add(local_name)
            return loaded_params

        worker_num = 8
        logger.info("Loading weights by %s workers... %s", worker_num, type(weights))

        # Limited concurrency to make tqdm happy.
        throttle = threading.BoundedSemaphore(worker_num)
        futures = []
        with ThreadPoolExecutor(worker_num) as executor:
            for name, loaded_weight in weights:
                throttle.acquire()
                futures.append(executor.submit(_worker, name, loaded_weight))
                futures[-1].add_done_callback(lambda _: throttle.release())
        for future in as_completed(futures):
            loaded_params |= future.result()
        _mark_optional_fp8_attention_scales_loaded(loaded_params, params_dict)
        return loaded_params


def _reset_fused_qkv_indexer_load_state(module: nn.Module) -> None:
    # The fused qkv+indexer projection is not constructed on Ascend yet.
    del module


def _validate_fused_qkv_indexer_weights(module: nn.Module) -> set[str]:
    del module
    return set()


class Step4ForCausalLM(nn.Module, SupportsPP, MixtureOfExperts):
    # Required so quantization exclude lists match fused module prefixes.
    packed_modules_mapping = STEP4_PACKED_MODULES_MAPPING
    _enable_weights_track_by_default = True

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
        if current_platform.device_type != "npu":
            raise NotImplementedError(
                "The vllm-ascend Step4 port targets Ascend NPUs; use the CUDA "
                "vLLM Step4 implementation on other platforms."
            )
        self.vllm_config = vllm_config
        model_config = vllm_config.model_config
        valid_vocab_size = _require_resolved_valid_vocab_size(model_config)
        config = model_config.hf_config
        self.config = config
        self.fp32_residual_connection = config.fp32_residual_connection
        self.model = Step4Model(
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
            # org_vocab_size masks the padded checkpoint vocabulary, matching
            # the valid_vocab_size semantics of the CUDA port.
            self.logits_processor = LogitsProcessor(
                config.vocab_size,
                org_vocab_size=valid_vocab_size,
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
            assert isinstance(layer, Step4DecoderLayer)
            if hasattr(layer, "moe") and isinstance(layer.moe, FusedMoEBlock):
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
            if not isinstance(moe, FusedMoEBlock):
                continue
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm.model_executor.model_loader.mtp_validation import (
            is_mtp_completeness_check_enabled,
        )

        validate_completeness = is_mtp_completeness_check_enabled()
        if validate_completeness:
            _reset_fused_qkv_indexer_load_state(self.model)
        skip_prefixes = ["vision_model."]
        if self.config.tie_word_embeddings:
            skip_prefixes.append("lm_head.")
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=skip_prefixes,
        )
        loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        if validate_completeness:
            loaded_params.update(
                f"model.{name}"
                for name in _validate_fused_qkv_indexer_weights(self.model)
            )
        return loaded_params
