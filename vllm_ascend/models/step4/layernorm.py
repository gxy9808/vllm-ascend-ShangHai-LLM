# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step4 normalization layers for vllm-ascend.

Ported from the Step4 vLLM adaptation. The CUDA port dispatches to the
Optimus extension and ``torch.ops._C`` kernels; on Ascend both fall back to
the fp32 torch implementations kept here. The non-zero-centered residual
path reuses stock vLLM ``fused_add_rms_norm``, which vllm-ascend already
adapts for NPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.utils.torch_utils import direct_register_custom_op


def _optimus_rms_norm_native(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    zero_centered: bool,
) -> torch.Tensor:
    compute = x.float()
    variance = compute.pow(2).mean(dim=-1, keepdim=True)
    compute = compute * torch.rsqrt(variance + variance_epsilon)
    scale = weight.float()
    if zero_centered:
        scale = scale + 1.0
    return (compute * scale).to(x.dtype)


def apply_optimus_rms_norm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    out: torch.Tensor | None = None,
    zero_centered: bool = False,
) -> torch.Tensor:
    del weight, variance_epsilon, zero_centered
    return torch.empty_like(x) if out is None else out


def apply_optimus_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    out: torch.Tensor | None = None,
    zero_centered: bool = False,
) -> torch.Tensor:
    result = _optimus_rms_norm_native(x, weight, variance_epsilon, zero_centered)
    if out is None:
        return result
    if out.shape != result.shape or out.dtype != result.dtype:
        raise ValueError(
            "OptimusRMSNorm output buffer must match the computed output "
            f"shape/dtype, got out={tuple(out.shape)}/{out.dtype}, "
            f"result={tuple(result.shape)}/{result.dtype}."
        )
    out.copy_(result)
    return out


direct_register_custom_op(
    op_name="optimus_rms_norm",
    op_func=apply_optimus_rms_norm,
    mutates_args=["out"],
    fake_impl=apply_optimus_rms_norm_fake,
)


def apply_optimus_fused_add_rms_norm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    zero_centered: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    del weight, variance_epsilon, zero_centered
    return torch.empty_like(x), torch.empty_like(residual)


def apply_optimus_fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    zero_centered: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = x.dtype
    residual_out = (
        (x.float() + residual.float()).to(residual.dtype)
        if orig_dtype == torch.float16
        else x + residual
    )
    output = _optimus_rms_norm_native(
        residual_out,
        weight,
        variance_epsilon,
        zero_centered,
    ).to(orig_dtype)
    return output, residual_out


direct_register_custom_op(
    op_name="optimus_fused_add_rms_norm",
    op_func=apply_optimus_fused_add_rms_norm,
    mutates_args=[],
    fake_impl=apply_optimus_fused_add_rms_norm_fake,
)


class OptimusRMSNorm(nn.Module):
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

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        output: torch.Tensor | None = None,
        fp16_out: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            if output is not None or fp16_out:
                raise ValueError(
                    "Residual OptimusRMSNorm does not support output buffers "
                    "or fp16_out."
                )
            if self.zero_centered:
                return torch.ops.vllm.optimus_fused_add_rms_norm(
                    x,
                    residual,
                    self.weight,
                    self.variance_epsilon,
                    zero_centered=True,
                )

            from vllm import _custom_ops as ops

            ops.fused_add_rms_norm(
                x,
                residual,
                self.weight.data,
                self.variance_epsilon,
            )
            return x, residual

        if fp16_out:
            raise ValueError("OptimusRMSNorm does not support fp16_out.")
        return torch.ops.vllm.optimus_rms_norm(
            x,
            self.weight,
            self.variance_epsilon,
            out=output,
            zero_centered=self.zero_centered,
        )


class OptimusLayerNorm(nn.Module):
    """Per-head LayerNorm for the Step4 sparse-attention indexer."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(
            x.unflatten(-1, (-1, self.hidden_size)),
            (self.hidden_size,),
            self.weight,
            self.bias,
            self.variance_epsilon,
        )
        return normalized.flatten(-2, -1)
