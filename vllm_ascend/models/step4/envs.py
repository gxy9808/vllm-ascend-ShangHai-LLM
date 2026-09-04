# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step4 environment variables for vllm-ascend.

The upstream Step4 vLLM patch adds these to ``vllm.envs``. Stock vLLM (the
pip dependency of vllm-ascend) does not carry them, so they are registered in
the central ``vllm_ascend.envs`` module with the same names, defaults, and
semantics; this module is a typed façade over that registry.
"""

from __future__ import annotations

import os

from vllm_ascend import envs as _central_envs


def _is_set(name: str) -> bool:
    return os.getenv(name) is not None


def enable_qkvg_proj() -> bool:
    return _central_envs.VLLM_STEP4_ENABLE_QKVG_PROJ


def o_proj_reduce_scatter() -> bool:
    return _central_envs.VLLM_STEP4_O_PROJ_REDUCE_SCATTER


def sparse_enabled_is_set() -> bool:
    return _is_set("VLLM_STEP4_SPARSE")


def sparse_enabled() -> bool:
    return _central_envs.VLLM_STEP4_SPARSE


def sparse_env_override(name: str | None) -> str | None:
    """Return the raw override for a sparse-config field, or None.

    Callers coerce to the field's type; the env value is always a string.
    """
    if name is None or not _is_set(name):
        return None
    return os.getenv(name)
