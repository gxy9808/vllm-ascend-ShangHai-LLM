# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend Step4 MTP (multi-token prediction) draft model.

Ported from the CUDA/Optimus implementation in ``vllm.models.step4.mtp``.
The draft layer reuses the dense Ascend Step4 decoder layer; FP8 attention
scale bookkeeping and CUDA-specific validation from the original
implementation are dropped because the Ascend port runs the model in bf16.
"""

import typing
from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import VllmConfig
from vllm.config.utils import replace
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.model_executor.models.utils import (
    get_spec_layer_idx_from_weight_name,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from .model import (
    _DSA_ONLY_WEIGHT_MARKERS,
    _is_fp8_scale,
    _is_fp8_weight,
    _require_resolved_valid_vocab_size,
    _set_step4_moe_protocol_metadata,
    Step4DecoderLayer,
    Step4MoEBlock,
    Step4RMSNorm,
    get_norm_dtype,
)

logger = init_logger(__name__)


def _get_mtp_vllm_config(vllm_config: VllmConfig) -> VllmConfig:
    speculative_config = vllm_config.speculative_config
    if speculative_config is None or speculative_config.draft_model_config is None:
        return vllm_config
    return replace(vllm_config, model_config=speculative_config.draft_model_config)


def _get_mtp_config(vllm_config: VllmConfig) -> PretrainedConfig:
    return _get_mtp_vllm_config(vllm_config).model_config.hf_text_config


def _build_mtp_norm(config: PretrainedConfig) -> nn.Module:
    return Step4RMSNorm(
        config.hidden_size,
        eps=config.rms_norm_eps,
        zero_centered=config.zero_centered,
        dtype=get_norm_dtype(config),
    )


class SharedHead(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Keep MTP normalization and residual precision aligned with the target
        # model configuration.
        self.fp32_residual_connection = config.fp32_residual_connection
        self.norm = _build_mtp_norm(config)
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=None,
            prefix=f"{prefix}.head",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.fp32_residual_connection:
            hidden_states = hidden_states.to(torch.bfloat16)
        return self.norm(hidden_states)


class Step4MultiTokenPredictorLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        mtp_vllm_config = _get_mtp_vllm_config(vllm_config)
        config = mtp_vllm_config.model_config.hf_text_config
        self.fp32_residual_connection = config.fp32_residual_connection
        self.enorm = _build_mtp_norm(config)
        self.hnorm = _build_mtp_norm(config)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.shared_head = SharedHead(config=config, prefix=f"{prefix}.shared_head")
        self.mtp_block = Step4DecoderLayer(
            mtp_vllm_config,
            prefix=f"{prefix}.mtp_block",
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        if self.fp32_residual_connection:
            inputs_embeds = inputs_embeds.to(torch.bfloat16)
            previous_hidden_states = previous_hidden_states.to(torch.bfloat16)
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )
        if self.fp32_residual_connection:
            hidden_states = hidden_states.to(torch.float32)

        hidden_states = self.mtp_block(positions=positions, hidden_states=hidden_states)
        return hidden_states


class Step4MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
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
                str(idx): Step4MultiTokenPredictorLayer(
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

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        logits = self.logits_processor(
            mtp_layer.shared_head.head, mtp_layer.shared_head(hidden_states)
        )
        return logits

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        return self.logits_processor.get_top_tokens(
            mtp_layer.shared_head.head,
            mtp_layer.shared_head(hidden_states),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.get_input_embeddings(input_ids)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


def _get_step4_mtp_moe_blocks(
    model: Step4MultiTokenPredictor,
) -> list[Step4MoEBlock]:
    blocks: list[Step4MoEBlock] = []
    for predictor_layer in model.layers.values():
        mtp_block = getattr(predictor_layer, "mtp_block", None)
        moe = getattr(mtp_block, "moe", None)
        if isinstance(moe, Step4MoEBlock):
            blocks.append(moe)
    return blocks


class AscendStep4MTP(nn.Module, MixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = _get_mtp_config(vllm_config)
        self.vllm_config = vllm_config
        _require_resolved_valid_vocab_size(
            _get_mtp_vllm_config(vllm_config).model_config
        )
        self.model = Step4MultiTokenPredictor(
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

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.get_input_embeddings(input_ids)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model.get_top_tokens(hidden_states, spec_step_idx)

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        if self.num_local_physical_experts != num_local_physical_experts:
            raise ValueError(
                "Step4 MTP EPLB cannot change the number of local physical "
                f"experts: expected={self.num_local_physical_experts}, "
                f"got={num_local_physical_experts}."
            )
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in _get_step4_mtp_moe_blocks(self.model):
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        expert_params_mapping = [
            (".moe.experts.gate_proj.", ".moe.gate_proj."),
            (".moe.experts.up_proj.", ".moe.up_proj."),
            (".moe.experts.down_proj.", ".moe.down_proj."),
        ]

        def _load_expert_tensor(name: str, loaded_weight: torch.Tensor) -> bool:
            for param_prefix, weight_prefix in expert_params_mapping:
                if weight_prefix not in name:
                    continue
                replaced_name = name.replace(weight_prefix, param_prefix)
                if replaced_name not in params_dict:
                    return True
                param = params_dict[replaced_name]
                param.weight_loader(param, loaded_weight)
                loaded_params.add(replaced_name)
                return True
            return False

        loaded_params: set[str] = set()
        pending_fp8: dict[str, dict[str, torch.Tensor]] = {}
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if name.startswith("layers."):
                name = f"model.{name}"
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if "embed_tokens" not in name and spec_layer is None:
                continue
            name = self._rewrite_spec_layer_name(spec_layer, name)
            # DSA sparse-indexer weights have no consumer in the dense fallback.
            if any(marker in name for marker in _DSA_ONLY_WEIGHT_MARKERS):
                continue

            if _is_fp8_weight(name, loaded_weight) or _is_fp8_scale(name):
                key = (
                    name[: -len(".weight_scale_inv")]
                    if _is_fp8_scale(name)
                    else name[: -len(".weight")]
                )
                entry = pending_fp8.setdefault(key, {})
                entry["scale" if _is_fp8_scale(name) else "weight"] = loaded_weight
                if "weight" in entry and "scale" in entry:
                    del pending_fp8[key]
                    if not _load_expert_tensor(f"{key}.weight", entry["weight"]):
                        raise ValueError(
                            f"Step4 MTP FP8 weight {key}.weight does not match "
                            "any expert parameter mapping."
                        )
                    if not _load_expert_tensor(
                        f"{key}.weight_scale_inv", entry["scale"]
                    ):
                        raise ValueError(
                            f"Step4 MTP FP8 scale {key}.weight_scale_inv does "
                            "not match any expert parameter mapping."
                        )
                continue

            loaded = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "experts" in name or "moe" in name:
                    continue
                replaced_name = name.replace(weight_name, param_name)
                if replaced_name not in params_dict:
                    continue
                name = replaced_name
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                loaded = True
                break
            if loaded:
                continue

            if _load_expert_tensor(name, loaded_weight):
                continue

            if (name.endswith(".bias") and name not in params_dict) or (
                "tok_embeddings" in name
            ):
                continue

            if spec_layer is not None and ".transformer." in name:
                name = name.replace(".transformer.", ".")
            if "shared_head" in name:
                name = name.replace("shared_head.output", "shared_head.head")
            if "embed_tokens" in name:
                assert (
                    hasattr(self.config, "num_nextn_predict_layers")
                    and self.config.num_nextn_predict_layers > 0
                )
                name = "model.embed_tokens.weight"
            if name not in params_dict:
                logger.warning_once(
                    "Skipping unexpected MTP checkpoint weight: %s", name
                )
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        if pending_fp8:
            raise RuntimeError(
                "Step4 MTP FP8 weights are missing their weight_scale_inv pair: "
                f"{sorted(pending_fp8)}"
            )
        return loaded_params

    def _rewrite_spec_layer_name(self, spec_layer: int | None, name: str) -> str:
        """
        Rewrite the weight name to match the format of the original model.
        Add .mtp_block for modules in transformer layer block for spec layer
        """
        spec_layer_weight_names = [
            "embed_tokens",
            "enorm",
            "hnorm",
            "eh_proj",
            "shared_head",
        ]
        spec_layer_weight = False
        for weight_name in spec_layer_weight_names:
            if weight_name in name:
                spec_layer_weight = True
                break
        if not spec_layer_weight:
            # treat rest weights as weights for transformer layer block
            name = name.replace(
                f"model.layers.{spec_layer}.", f"model.layers.{spec_layer}.mtp_block."
            )
        return name
