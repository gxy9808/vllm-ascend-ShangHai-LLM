# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step4 compute operators and sparse-config parsing for vllm-ascend.

Ported from the Step4 vLLM adaptation (``vllm/models/step4/kernels.py``). The
CUDA port backs the fused operators with Optimus/CuTeDSL kernels; this port
implements them with plain torch, following the numerics of the standalone
Step-4 minimal inference release (``step4-inference/inference/kernel.py``):

- fused QK RMSNorm + partial NeoX RoPE (main attention),
- fused indexer norm + RoPE (RMSNorm q, LayerNorm k, z passthrough),
- MoE router-bias top-k routing.

The DSA sparse attention backend itself (CuTeDSL, SM90-only in the CUDA port)
is not available on Ascend yet.
"""

from dataclasses import dataclass, fields
from typing import Any

import torch

from vllm.utils.torch_utils import direct_register_custom_op

from . import envs as step4_envs


@dataclass(frozen=True)
class Step4SparseConfig:
    enabled: bool = True
    proxy_dim: int = 256
    sparse_indexer_rope_dim: int = 32
    sparse_indexer_use_rope: bool = True
    sparse_indexer_num_heads: int = 16
    sparse_indexer_num_k_heads: int = 1
    sparse_indexer_q_norm_type: str = "rmsnorm"
    sparse_indexer_k_norm_type: str = "layernorm"
    sparse_indexer_csa_z_norm_type: str = "none"
    index_tp_size: int = 4
    topk: int = 512
    region_block_size: int = 8
    compression_method: str = "csa_block_compress"
    attention_impl: str = "sparse_gqa"
    decode_split_max: int = 16
    sparse_indexer_softmax_variant: str = "softmax"
    sparse_indexer_ssmax_s_granularity: str = "q_head"
    apply_to_layer_types: tuple[str, ...] = ("full_attention",)


_STEP4_SPARSE_SECTION_NAMES = (
    "step4_sparse_config",
    "step3p5_sparse_config",
    "sparse_config",
)
_STEP4_SPARSE_ENABLE_ENV = "VLLM_STEP4_SPARSE"
_STEP4_SPARSE_ENV = {
    "proxy_dim": "VLLM_STEP4_SPARSE_PROXY_DIM",
    "sparse_indexer_rope_dim": "VLLM_STEP4_SPARSE_INDEXER_ROPE_DIM",
    "index_tp_size": "VLLM_STEP4_DSA_INDEX_TP_SIZE",
    "topk": "VLLM_STEP4_SPARSE_TOPK",
    "region_block_size": "VLLM_STEP4_SPARSE_REGION_BLOCK_SIZE",
    "attention_impl": "VLLM_STEP4_SPARSE_ATTENTION_IMPL",
    "decode_split_max": "VLLM_STEP4_SPARSE_DECODE_SPLIT_MAX",
}

_STEP4_SPARSE_PROXY_DIM = 256
_STEP4_SPARSE_REGION_BLOCK_SIZE = 8


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _config_value(config: Any, name: str) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _get_sparse_config_section(config: Any) -> Any:
    for name in _STEP4_SPARSE_SECTION_NAMES:
        value = _config_value(config, name)
        if value is not None:
            return value
    return None


def checkpoint_has_step4_sparse_config(config: Any) -> bool:
    """Whether the checkpoint declares the Step4 sparse layout."""
    return _get_sparse_config_section(config) is not None


def get_step4_sparse_config(config: Any) -> Step4SparseConfig | None:
    defaults = Step4SparseConfig()
    section = _get_sparse_config_section(config)
    if step4_envs.sparse_enabled_is_set():
        enabled = step4_envs.sparse_enabled()
        if enabled and section is None:
            raise ValueError(
                "VLLM_STEP4_SPARSE=1 requires a checkpoint-declared "
                "Step4 sparse config section; refusing to apply DSA defaults "
                "to a native dense checkpoint."
            )
    elif section is None:
        enabled = False
    else:
        enabled = _truthy(_config_value(section, "enabled"))
        if _config_value(section, "enabled") is None:
            enabled = True
    if not enabled:
        return None
    values: dict[str, Any] = {}
    for field in fields(Step4SparseConfig):
        default = getattr(defaults, field.name)
        if field.name == "apply_to_layer_types":
            continue
        env_name = _STEP4_SPARSE_ENV.get(field.name)
        value = step4_envs.sparse_env_override(env_name)
        if value is None:
            value = _config_value(section, field.name)
        if field.name == "sparse_indexer_softmax_variant" and value is None:
            value = _config_value(section, "softmax_variant")
        if field.name == "index_tp_size" and value is None:
            value = _config_value(section, "num_provider_groups")
        if value is None or value == "":
            value = default
        elif isinstance(default, bool):
            value = _truthy(value)
        elif isinstance(default, int):
            value = int(value)
        elif isinstance(default, str):
            value = str(value)
        values[field.name] = value
    values["enabled"] = True
    apply_to_layer_types = _config_value(section, "apply_to_layer_types")
    if apply_to_layer_types is None:
        apply_to_layer_types = defaults.apply_to_layer_types
    if isinstance(apply_to_layer_types, str):
        apply_to_layer_types = tuple(
            item.strip().lower()
            for item in apply_to_layer_types.split(",")
            if item.strip()
        )
    else:
        apply_to_layer_types = tuple(
            str(item).strip().lower()
            for item in apply_to_layer_types
            if str(item).strip()
        )
    if apply_to_layer_types != ("full_attention",):
        raise ValueError(
            "Step4 DSA production integration requires "
            f"apply_to_layer_types=('full_attention',); got {apply_to_layer_types!r}."
        )
    values["apply_to_layer_types"] = apply_to_layer_types
    proxy_dim = int(values["proxy_dim"])
    if proxy_dim != _STEP4_SPARSE_PROXY_DIM:
        raise ValueError(
            f"Step4 DSA requires proxy_dim=256, got {proxy_dim}."
        )
    region_block_size = int(values["region_block_size"])
    if region_block_size != _STEP4_SPARSE_REGION_BLOCK_SIZE:
        raise ValueError(
            f"Step4 DSA requires region_block_size=8, got {region_block_size}."
        )
    index_tp_size = int(values["index_tp_size"])
    if index_tp_size <= 0:
        raise ValueError(
            f"Step4 sparse indexer index_tp_size must be positive, got {index_tp_size}."
        )
    indexer_num_heads = int(values["sparse_indexer_num_heads"])
    if indexer_num_heads <= 0 or indexer_num_heads % index_tp_size != 0:
        raise ValueError(
            "Step4 sparse_indexer_num_heads must be positive and divisible by "
            f"index_tp_size, got sparse_indexer_num_heads={indexer_num_heads}, "
            f"index_tp_size={index_tp_size}."
        )
    indexer_rope_dim = int(values["sparse_indexer_rope_dim"])
    if (
        indexer_rope_dim <= 0
        or indexer_rope_dim > proxy_dim
        or indexer_rope_dim % 2 != 0
    ):
        raise ValueError(
            "Step4 sparse indexer RoPE dimension must be positive, even, and "
            f"no larger than proxy_dim, got {indexer_rope_dim}, "
            f"proxy_dim={proxy_dim}."
        )
    return Step4SparseConfig(**values)


def _rms_norm_per_head(
    x: torch.Tensor,
    weight: torch.Tensor,
    head_dim: int,
    eps: float,
    weight_bias: float,
) -> torch.Tensor:
    """Zero-centerable RMSNorm over the last ``head_dim`` dims, fp32 compute."""
    orig_shape = x.shape
    xf = x.reshape(-1, orig_shape[-1] // head_dim, head_dim).float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(variance + eps)
    scale = weight.float() + weight_bias
    return (xf * scale).reshape(orig_shape).to(x.dtype)


def _layer_norm_per_head(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    head_dim: int,
    eps: float,
) -> torch.Tensor:
    orig_shape = x.shape
    xf = x.reshape(-1, orig_shape[-1] // head_dim, head_dim).float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = xf.var(dim=-1, unbiased=False, keepdim=True)
    xf = (xf - mean) * torch.rsqrt(var + eps)
    return (
        (xf * weight.float() + bias.float()).reshape(orig_shape).to(x.dtype)
    )


def _apply_neox_partial_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    head_dim: int,
    rotary_dim: int,
) -> torch.Tensor:
    """Rotate the leading ``2 * rotary_dim`` dims of each head, NeoX pairs."""
    tokens = x.shape[0]
    xh = x.view(tokens, num_heads, head_dim)
    pos = positions.reshape(-1).to(device=x.device, dtype=torch.long)
    c = cos.index_select(0, pos).to(torch.float32).unsqueeze(1)
    s = sin.index_select(0, pos).to(torch.float32).unsqueeze(1)
    rot = xh[..., : 2 * rotary_dim].float()
    real = rot[..., :rotary_dim]
    imag = rot[..., rotary_dim:]
    rotated = torch.cat(
        [real * c - imag * s, real * s + imag * c], dim=-1
    ).to(x.dtype)
    out = torch.cat([rotated, xh[..., 2 * rotary_dim :]], dim=-1)
    return out.reshape(tokens, num_heads * head_dim)


def _fused_qknorm_rope_forward_impl(
    qkv: torch.Tensor,
    qnorm_weight: torch.Tensor,
    knorm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pos_id: torch.Tensor,
    head_dim: int,
    num_q_head: int,
    num_kv_head: int,
    rotary_dim: int,
    eps: float = 1e-5,
    norm_weight_bias: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_size = num_q_head * head_dim
    kv_size = num_kv_head * head_dim
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    # v stays a raw column slice of the packed projection; make it explicit
    # memory since downstream NPU attention kernels require contiguous
    # values in their non-paged path (q/k are freshly written by the norms).
    v = v.contiguous()
    q = _rms_norm_per_head(q, qnorm_weight, head_dim, eps, norm_weight_bias)
    k = _rms_norm_per_head(k, knorm_weight, head_dim, eps, norm_weight_bias)
    q = _apply_neox_partial_rope(q, cos, sin, pos_id, num_q_head, head_dim, rotary_dim)
    k = _apply_neox_partial_rope(k, cos, sin, pos_id, num_kv_head, head_dim, rotary_dim)
    return q, k, v


def _fused_qknorm_rope_forward_impl_fake(
    qkv: torch.Tensor,
    qnorm_weight: torch.Tensor,
    knorm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pos_id: torch.Tensor,
    head_dim: int,
    num_q_head: int,
    num_kv_head: int,
    rotary_dim: int,
    eps: float = 1e-5,
    norm_weight_bias: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = qkv.shape[0]
    q = torch.empty((tokens, num_q_head * head_dim), device=qkv.device, dtype=qkv.dtype)
    k = torch.empty((tokens, num_kv_head * head_dim), device=qkv.device, dtype=qkv.dtype)
    v = torch.empty((tokens, num_kv_head * head_dim), device=qkv.device, dtype=qkv.dtype)
    return q, k, v


direct_register_custom_op(
    op_name="fused_qknorm_rope_forward_impl",
    op_func=_fused_qknorm_rope_forward_impl,
    mutates_args=[],
    fake_impl=_fused_qknorm_rope_forward_impl_fake,
)


def fused_qknorm_rope_forward_impl(
    qkv: torch.Tensor,
    qnorm_weight: torch.Tensor,
    knorm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pos_id: torch.Tensor,
    head_dim: int,
    num_q_head: int,
    num_kv_head: int,
    rotary_dim: int,
    eps: float = 1e-5,
    norm_weight_bias: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.fused_qknorm_rope_forward_impl(
        qkv,
        qnorm_weight,
        knorm_weight,
        cos,
        sin,
        pos_id,
        head_dim,
        num_q_head,
        num_kv_head,
        rotary_dim,
        eps,
        norm_weight_bias,
    )


def _fused_indexer_norm_rope_forward_impl(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    qnorm_weight: torch.Tensor,
    knorm_weight: torch.Tensor,
    knorm_bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pos_id: torch.Tensor,
    head_dim: int,
    num_q_head: int,
    num_k_head: int,
    rotary_dim: int,
    eps: float = 1e-6,
    q_norm_weight_bias: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = _rms_norm_per_head(index_q, qnorm_weight, head_dim, eps, q_norm_weight_bias)
    k = _layer_norm_per_head(index_k, knorm_weight, knorm_bias, head_dim, eps)
    q = _apply_neox_partial_rope(q, cos, sin, pos_id, num_q_head, head_dim, rotary_dim)
    k = _apply_neox_partial_rope(k, cos, sin, pos_id, num_k_head, head_dim, rotary_dim)
    return q, k, index_z.contiguous()


def _fused_indexer_norm_rope_forward_impl_fake(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    qnorm_weight: torch.Tensor,
    knorm_weight: torch.Tensor,
    knorm_bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pos_id: torch.Tensor,
    head_dim: int,
    num_q_head: int,
    num_k_head: int,
    rotary_dim: int,
    eps: float = 1e-6,
    q_norm_weight_bias: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(index_q),
        torch.empty_like(index_k),
        torch.empty_like(index_z),
    )


direct_register_custom_op(
    op_name="fused_indexer_norm_rope_forward_impl",
    op_func=_fused_indexer_norm_rope_forward_impl,
    mutates_args=[],
    fake_impl=_fused_indexer_norm_rope_forward_impl_fake,
)


def fused_indexer_norm_rope_forward_impl(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    qnorm_weight: torch.Tensor,
    knorm_weight: torch.Tensor,
    knorm_bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pos_id: torch.Tensor,
    head_dim: int,
    num_q_head: int,
    num_k_head: int,
    rotary_dim: int,
    eps: float = 1e-6,
    q_norm_weight_bias: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.fused_indexer_norm_rope_forward_impl(
        index_q,
        index_k,
        index_z,
        qnorm_weight,
        knorm_weight,
        knorm_bias,
        cos,
        sin,
        pos_id,
        head_dim,
        num_q_head,
        num_k_head,
        rotary_dim,
        eps,
        q_norm_weight_bias,
    )


ROUTER_RENORM_EPS = 1e-20


def router_bias_func(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    router_bias: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
    nan_row_i_out: int = 0,
    indices_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid router with a selection-only bias, in torch.

    Selection ranks ``sigmoid(logits) + bias``; the returned weight is that
    score with the bias subtracted back out, then renormalized and scaled.
    NaN gating rows get zero weights and ``nan_row_i_out`` indices, matching
    the deployed CUDA router kernel.
    """
    assert renormalize
    del hidden_states
    if indices_dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"Step4 router indices_dtype must be int32 or int64, got {indices_dtype}."
        )
    assert router_bias is not None

    gating = gating_output.float()
    bias = router_bias.to(gating.device).float()
    nan_mask = gating != gating
    has_bad = nan_mask.any(dim=-1, keepdim=True)
    scores = torch.sigmoid(gating) + bias
    scores = torch.where(nan_mask, float("-inf"), scores)

    topk_scores, topk_ids = torch.topk(scores, topk, dim=-1)
    selected_bias = bias.unsqueeze(0).expand_as(scores).gather(-1, topk_ids)
    weights = topk_scores - selected_bias

    weight_sum = weights.sum(dim=-1, keepdim=True)
    weights = weights / (weight_sum + ROUTER_RENORM_EPS)
    if routed_scaling_factor != 1.0:
        weights = weights * routed_scaling_factor

    weights = torch.where(has_bad, torch.zeros_like(weights), weights)
    ids = torch.where(
        has_bad,
        torch.full_like(topk_ids, nan_row_i_out),
        topk_ids,
    ).to(indices_dtype)
    return weights, ids
