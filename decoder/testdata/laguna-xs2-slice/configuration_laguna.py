# Copyright 2026 Poolside and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Literal

from huggingface_hub.dataclasses import strict

from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters
from transformers.utils import auto_docstring


@auto_docstring(checkpoint="poolside/laguna-XS.2")
@strict
class LagunaConfig(PreTrainedConfig):
    r"""
    partial_rotary_factor (`float`, *optional*):
        Fraction of ``head_dim`` to rotate. Folded into each ``rope_parameters[layer_type]``
        entry by ``__post_init__``.
    num_attention_heads_per_layer (`list[int]`, *optional*):
        Per-layer override for ``num_attention_heads``. Length must equal ``num_hidden_layers``.
    mlp_layer_types (`list[str]`, *optional*):
        Per-layer MLP type — ``"dense"`` or ``"sparse"``. Length must equal
        ``num_hidden_layers``. Defaults to first layer dense, rest sparse.
    moe_routed_scaling_factor (`float`, *optional*, defaults to 1.0):
        Scalar applied to routed-expert output before combining with the shared-expert output.
    moe_apply_router_weight_on_input (`bool`, *optional*, defaults to `False`):
        Whether to apply router weights to the MoE input rather than the output. Not supported
        in transformers yet; ``True`` will raise a ``NotImplementedError`` for now.
    moe_router_logit_softcapping (`float`, *optional*, defaults to 0.0):
        Scaling factor when applying tanh softcapping on the logits of the MoE router logits.

    Example:

    ```python
    >>> from transformers import LagunaModel, LagunaConfig

    >>> configuration = LagunaConfig()
    >>> model = LagunaModel(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "laguna"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.g_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    # Qwen2Moe-inherited defaults we want to override for Laguna's typical shape.
    vocab_size: int = 100352
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 40
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    sliding_window: int | None = None
    attention_dropout: float | int = 0.0
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    num_experts_per_tok: int = 8
    num_experts: int = 256
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    layer_types: list[str] | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    # Laguna-specific attention
    head_dim: int = 128
    attention_bias: bool = False
    partial_rotary_factor: float | None = None
    num_attention_heads_per_layer: list[int] | None = None
    # Laguna-specific MoE
    mlp_layer_types: list[str] | None = None
    moe_routed_scaling_factor: float = 1.0
    moe_apply_router_weight_on_input: bool = False
    moe_router_logit_softcapping: float = 0.0

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] + ["sparse"] * (self.num_hidden_layers - 1)
        if self.num_attention_heads_per_layer is None:
            self.num_attention_heads_per_layer = [self.num_attention_heads] * self.num_hidden_layers

        default_rope_params: dict[Literal["full_attention", "sliding_attention"], dict[str, Any]] = {
            "full_attention": {"rope_type": "default", "rope_theta": 500000.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        }
        if self.rope_parameters is None:
            self.rope_parameters = default_rope_params

        self._normalize_rope_parameters()
        # Skip ``Qwen2MoeConfig.__post_init__`` — it references ``mlp_only_layers`` /
        # ``use_sliding_window`` / ``max_window_layers`` which Laguna drops above.
        super().__post_init__(**kwargs)

    def _normalize_rope_parameters(self):
        """Coerce ``rope_parameters`` to the nested ``{layer_type: {...}}`` shape.

        Accepts an already-nested dict as-is, or a flat dict that gets broadcast to every
        layer type. A top-level ``partial_rotary_factor`` is folded into each sub-dict as
        a default.
        """
        layer_types = set(self.layer_types)
        rope_params = self.rope_parameters or {}
        is_nested = isinstance(rope_params, dict) and any(k in layer_types for k in rope_params)
        if is_nested:
            nested = {lt: dict(rope_params.get(lt, {})) for lt in layer_types}
        else:
            nested = {lt: dict(rope_params) for lt in layer_types}

        if self.partial_rotary_factor is not None:
            for params in nested.values():
                params.setdefault("partial_rotary_factor", self.partial_rotary_factor)

        for params in nested.values():
            params.setdefault("rope_type", "default")

        self.rope_parameters = nested
        # Null the top-level field now that its value lives in each sub-dict — otherwise
        # ``standardize_rope_params`` would overwrite per-type values with the global one.
        self.partial_rotary_factor = None

    def convert_rope_params_to_dict(self, **kwargs):
        # No need to handle BC for new models, because they have no old-format `rope_scaling`
        return kwargs

    def _validate_yarn_rope_parameters(self, rope_parameters: dict, ignore_keys=None):
        """Override: parent reads ``self.rope_parameters["original_max_position_embeddings"]``
        for its post-hoc factor sanity-check, which works for flat rope configs but raises
        ``KeyError`` when ``self.rope_parameters`` is the Laguna/Gemma3-style per-layer-type
        map (its keys are layer types like ``"full_attention"``). Fix locally by reading
        from the per-call ``rope_parameters`` dict that ``validate_rope`` already passes in.
        """
        # Delegate to parent for the shared checks by temporarily swapping in a flat
        # ``self.rope_parameters`` that has the key the parent expects. Cheapest way to
        # share the parent's logic without reimplementing it here.
        flat = getattr(self, "rope_parameters", None)
        self.rope_parameters = rope_parameters
        try:
            super()._validate_yarn_rope_parameters(rope_parameters, ignore_keys=ignore_keys)
        finally:
            self.rope_parameters = flat

    def validate_architecture(self):
        """Part of ``@strict``-powered validation."""
        if self.moe_apply_router_weight_on_input:
            raise NotImplementedError(
                "moe_apply_router_weight_on_input=True is not yet supported in the "
                "transformers implementation of Laguna."
            )
        if (
            self.num_attention_heads_per_layer is not None
            and len(self.num_attention_heads_per_layer) != self.num_hidden_layers
        ):
            raise ValueError(
                f"num_attention_heads_per_layer length ({len(self.num_attention_heads_per_layer)}) "
                f"must equal num_hidden_layers ({self.num_hidden_layers})."
            )
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types length ({len(self.layer_types)}) "
                f"must equal num_hidden_layers ({self.num_hidden_layers})."
            )
        if len(self.mlp_layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"mlp_layer_types length ({len(self.mlp_layer_types)}) "
                f"must equal num_hidden_layers ({self.num_hidden_layers})."
            )


__all__ = ["LagunaConfig"]
