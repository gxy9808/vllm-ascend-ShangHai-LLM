# SPDX-License-Identifier: Apache-2.0
"""Ascend Step-4 MTP draft model.

The upstream ``Step4MTP`` is CUDA-only (its ``__init__`` raises on
non-CUDA platforms and its predictor blocks build the CUDA
``Step4DecoderLayer``).  This fork keeps the upstream wire format and
interfaces (``forward``/``compute_logits``/``get_top_tokens``/``load_weights``
are all inherited untouched) and only re-creates the ``__init__`` paths so
the draft transformer block runs through ``AscendStep4DecoderLayer`` --
i.e. the platform Ascend attention backends plus the NPU MoE/rope fixes
already proven on the target model.  The MTP layer is a dense sliding
layer, so it lands on the platform sliding-window backend, not DSA.

Weight loading, module names and parameter names mirror the upstream
implementation exactly, so the checkpoint's ``model.layers.92.*`` MTP
weights load unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.models.step4.mtp import (
    SharedHead,
    Step4MTP,
    Step4MultiTokenPredictor,
    Step4MultiTokenPredictorLayer,
    _build_mtp_norm,
    _get_mtp_config,
    _get_mtp_vllm_config,
    _get_step4_mtp_moe_blocks,
    _require_resolved_valid_vocab_size,
    _set_step4_moe_protocol_metadata,
)
from vllm.model_executor.models.utils import maybe_prefix

from vllm_ascend.models.step4.model import AscendStep4DecoderLayer


class AscendStep4MultiTokenPredictorLayer(Step4MultiTokenPredictorLayer):
    """Upstream predictor layer with the draft block swapped to Ascend."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        # Bypass the parent __init__: it hardcodes the CUDA
        # Step4DecoderLayer.  Everything else mirrors it verbatim so
        # module/parameter names stay checkpoint-compatible.
        nn.Module.__init__(self)
        # The parent's @support_torch_compile wrapper reads this attribute
        # in __call__; the attribute itself is normally set by the
        # decorator's __init__ patch, which the bypass above skips.
        self.do_not_compile = True
        mtp_vllm_config = _get_mtp_vllm_config(vllm_config)
        config = mtp_vllm_config.model_config.hf_text_config
        self.fp32_residual_connection = config.fp32_residual_connection
        quant_config = vllm_config.quant_config
        self.enorm = _build_mtp_norm(config)
        self.hnorm = _build_mtp_norm(config)
        self.eh_proj = nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )
        self.shared_head = SharedHead(
            config=config, quant_config=quant_config, prefix=f"{prefix}.shared_head"
        )
        self.mtp_block = AscendStep4DecoderLayer(
            mtp_vllm_config,
            prefix=f"{prefix}.mtp_block",
        )


class AscendStep4MultiTokenPredictor(Step4MultiTokenPredictor):
    """Upstream predictor whose layers use the Ascend draft blocks."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = _get_mtp_config(vllm_config)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        if self.num_mtp_layers <= 0:
            raise ValueError(
                "Step4 MTP requires num_nextn_predict_layers > 0, got "
                f"{self.num_mtp_layers}."
            )
        # to map the exact layer index from weights
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): AscendStep4MultiTokenPredictorLayer(
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.layers.{idx}",
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )

        self.logits_processor = LogitsProcessor(
            config.vocab_size,
            valid_vocab_size=_require_resolved_valid_vocab_size(
                _get_mtp_vllm_config(vllm_config).model_config
            ),
        )


class AscendStep4MTP(Step4MTP):
    """Step4MTP for Ascend: same wire format, Ascend draft decoder layer."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Bypass Step4MTP.__init__ (raises on non-CUDA platforms); the body
        # below replicates it with the Ascend predictor.
        nn.Module.__init__(self)
        self.config = _get_mtp_config(vllm_config)
        self.vllm_config = vllm_config
        _require_resolved_valid_vocab_size(
            _get_mtp_vllm_config(vllm_config).model_config
        )
        self.model = AscendStep4MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        moe_blocks = _get_step4_mtp_moe_blocks(self.model)
        # Expose the actual draft topology so dense MTP forwards do not
        # inherit the target model's DP collectives.
        self.is_dense_mtp = not moe_blocks
        self.moe_layers = [moe.experts for moe in moe_blocks]
        _set_step4_moe_protocol_metadata(
            self,
            moe_blocks[0] if moe_blocks else None,
        )
