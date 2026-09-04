# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step4 model entry point for vllm-ascend.

Ported from the Step4 vLLM adaptation (vllm/models/step4). The DSA sparse
attention backends of the CUDA port (CuTeDSL, SM90-only) are not available on
Ascend; this package currently runs the dense fallback path with torch
implementations of every Step4-specific operator.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Step4ForCausalLM
    from .mtp import Step4MTP

__all__ = ["Step4ForCausalLM", "Step4MTP"]


def __getattr__(name: str):
    if name == "Step4ForCausalLM":
        from .model import Step4ForCausalLM

        return Step4ForCausalLM
    if name == "Step4MTP":
        from .mtp import Step4MTP

        return Step4MTP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
