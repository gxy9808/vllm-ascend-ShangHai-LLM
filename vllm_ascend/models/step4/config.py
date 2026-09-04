# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step4 model configuration for vllm-ascend.

Ported verbatim from the Step4 vLLM patch
(``vllm/transformers_utils/configs/step4.py``): stock vLLM (the pip
dependency of vllm-ascend) does not ship it, so the classes live here and
are injected into vLLM's config registry at plugin load time.
"""

from typing import Any

from transformers.configuration_utils import PretrainedConfig


class Step4Config(PretrainedConfig):
    model_type = "step4"

    def __init__(
        self,
        hidden_size: int = 5120,
        intermediate_size: int = 13312,
        num_attention_heads: int = 40,
        num_attention_groups: int = 8,
        num_hidden_layers: int = 48,
        max_seq_len: int = 4096,
        vocab_size: int = 65536,
        tie_word_embeddings: bool = False,
        rms_norm_eps: float = 1e-5,
        use_moe: bool = False,
        moe_intermediate_size: int = 10240,
        moe_num_experts: int = 16,
        moe_top_k: int = 4,
        rope_theta: float | list[float] | None = 500000,
        rope_scaling: dict[str, Any] | None = None,
        head_dim: int | None = None,
        share_expert_dim: int | None = None,
        norm_dtype: str = "bf16",
        norm_expert_weight: bool = True,
        bos_token_id: list[int] | int | None = None,
        eos_token_id: list[int] | int | None = None,
        moe_router_activation: str = "softmax",
        moe_router_scaling_factor: float = 1.0,
        att_impl_type: str = "GQA",
        use_head_wise_attn_gate: bool = False,
        use_moe_router_bias: bool = False,
        need_fp32_gate: bool = False,
        layer_types: list[str] | None = None,
        use_rope_layers: list[bool] | None = None,
        partial_rotary_factors: list[float] | None = None,
        yarn_only_types: list[str] | None = None,
        attention_other_setting: dict[str, Any] | None = None,
        num_nextn_predict_layers: int = 0,
        swa_num_attention_heads: int | None = None,
        swiglu_limits: list[float] | None = None,
        swiglu_limits_shared: list[float] | None = None,
        zero_centered: bool = True,
        fp32_residual_connection: bool = False,
        max_position_embeddings: int | None = None,
        **kwargs,
    ) -> None:
        if "max_position_embedding" in kwargs:
            raise ValueError(
                "Found 'max_position_embedding' in config.json. "
                "Please use 'max_position_embeddings' (with 's') instead."
            )
        if num_nextn_predict_layers < 0:
            raise ValueError(
                "num_nextn_predict_layers must be non-negative, got "
                f"{num_nextn_predict_layers}."
            )
        if max_position_embeddings is not None and max_position_embeddings <= 0:
            raise ValueError(
                "max_position_embeddings must be positive, got "
                f"{max_position_embeddings}."
            )
        if max_position_embeddings is None and max_seq_len <= 0:
            raise ValueError(
                "max_seq_len must be positive when max_position_embeddings is "
                f"omitted, got {max_seq_len}."
            )

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.num_hidden_layers = num_hidden_layers
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.use_moe = use_moe
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_num_experts = moe_num_experts
        self.num_experts_per_tok = moe_top_k
        self.moe_top_k = moe_top_k

        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.head_dim = head_dim
        self.share_expert_dim = (
            moe_intermediate_size * moe_top_k
            if share_expert_dim is None
            else share_expert_dim
        )
        self.norm_dtype = norm_dtype
        self.norm_expert_weight = norm_expert_weight

        self.max_position_embeddings = (
            max_seq_len if max_position_embeddings is None else max_position_embeddings
        )
        self.moe_router_activation = moe_router_activation
        self.moe_router_scaling_factor = moe_router_scaling_factor
        self.use_moe_router_bias = use_moe_router_bias
        self.need_fp32_gate = need_fp32_gate

        self.att_impl_type = att_impl_type
        self.use_head_wise_attn_gate = use_head_wise_attn_gate
        # Step's per-layer arrays cover the dense stack plus the appended MTP
        # layers, which use absolute layer indices.
        # transformers 5.x validates len(layer_types) == num_hidden_layers, so
        # keep the MTP-inclusive copy under a Step-owned name and expose only
        # the dense slice as `layer_types`, which is what transformers and
        # vLLM's hybrid-attention detection both mean by it.
        total_num_layers = num_hidden_layers + num_nextn_predict_layers
        serialized_layer_types = kwargs.pop("layer_types_with_mtp", None)
        if serialized_layer_types is not None:
            if layer_types and list(layer_types) != list(
                serialized_layer_types[:num_hidden_layers]
            ):
                raise ValueError(
                    "layer_types must match the dense prefix of layer_types_with_mtp."
                )
            layer_types = serialized_layer_types
        if layer_types and len(layer_types) < total_num_layers:
            raise ValueError(
                "Step4 layer_types must include the dense stack and every MTP "
                f"layer: expected at least {total_num_layers} entries, got "
                f"{len(layer_types)}."
            )
        self.layer_types_with_mtp = (
            list(layer_types[:total_num_layers]) if layer_types else None
        )
        self.layer_types = (
            self.layer_types_with_mtp[:num_hidden_layers]
            if self.layer_types_with_mtp
            else None
        )
        self.use_rope_layers = use_rope_layers
        self.partial_rotary_factors = partial_rotary_factors
        self.yarn_only_types = yarn_only_types
        self.attention_other_setting = attention_other_setting
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.swa_num_attention_heads = swa_num_attention_heads
        self.swiglu_limits = swiglu_limits
        self.swiglu_limits_shared = swiglu_limits_shared
        self.zero_centered = zero_centered
        self.fp32_residual_connection = fp32_residual_connection

        resolved_bos_token_id = 1 if bos_token_id is None else bos_token_id
        resolved_eos_token_id = [2, 3] if eos_token_id is None else eos_token_id
        self.bos_token_id = resolved_bos_token_id
        self.eos_token_id = resolved_eos_token_id

        super().__init__(
            bos_token_id=resolved_bos_token_id,
            eos_token_id=resolved_eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class Step4MTPConfig(Step4Config):
    """Configuration for a standalone Step4 MTP checkpoint.

    The usual in-tree MTP path derives its draft config from ``Step4Config``
    and changes the instance model type in memory. A separately published
    draft checkpoint is parsed directly with ``model_type="step4_mtp"`` and
    therefore needs a distinct registry class. Reusing ``Step4Config`` for
    both registry keys would be unsafe because registration mutates its
    class-level ``model_type``.
    """

    model_type = "step4_mtp"
    architectures = ["Step4MTP"]

    def __init__(self, num_nextn_predict_layers: int = 1, **kwargs: Any) -> None:
        if kwargs.get("architectures") is None:
            kwargs["architectures"] = list(type(self).architectures)
        super().__init__(
            num_nextn_predict_layers=num_nextn_predict_layers,
            **kwargs,
        )


def register_step4_configs() -> None:
    """Inject the Step4 config classes into stock vLLM's config registry.

    vLLM resolves ``model_type`` through ``_CONFIG_REGISTRY`` by importing the
    class from ``vllm.transformers_utils.configs``. Stock vLLM does not know
    ``step4``, so register the Ascend-port classes under that namespace.
    """
    import vllm.transformers_utils.configs as vllm_configs
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    if getattr(vllm_configs, "Step4Config", None) is None:
        vllm_configs.Step4Config = Step4Config
        vllm_configs.Step4MTPConfig = Step4MTPConfig
        _CONFIG_REGISTRY.setdefault("step4", "Step4Config")
        _CONFIG_REGISTRY.setdefault("step4_mtp", "Step4MTPConfig")
