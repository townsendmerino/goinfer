package config

import (
	"fmt"
)

// ValidateQwen35Dense pins Qwen3.8 (model_type qwen3_5). It is ValidateQwen35 minus the MoE checks plus one addition:
// intermediate_size is load-bearing here, where the MoE sibling treats it as vestigial (its experts carry their own width). A
// zero there would silently build a dense FFN of width 0 rather than fail.
func (c *Config) ValidateQwen35Dense() error {
	switch {
	case c.HiddenDim == 0 || c.NumLayers == 0 || c.NumHeads == 0 || c.HeadDim == 0:
		return fmt.Errorf("decoder(qwen3_5): missing core dims (hidden=%d layers=%d heads=%d headDim=%d)",
			c.HiddenDim, c.NumLayers, c.NumHeads, c.HeadDim)
	case c.NumKVHeads == 0 || c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("decoder(qwen3_5): num_heads %d not a multiple of num_kv_heads %d (GQA)", c.NumHeads, c.NumKVHeads)
	case c.IntermediateDim <= 0:
		return fmt.Errorf("decoder(qwen3_5): intermediate_size must be >0 — it is the DENSE FFN width here, not vestigial as in the MoE sibling")
	case len(c.LayerTypes) != c.NumLayers:
		return fmt.Errorf("decoder(qwen3_5): layer_types has %d entries, want %d", len(c.LayerTypes), c.NumLayers)
	case c.LinearConvKernelDim <= 0 || c.LinearKeyHeadDim <= 0 || c.LinearValueHeadDim <= 0:
		return fmt.Errorf("decoder(qwen3_5): missing linear (DeltaNet) dims (conv=%d kHead=%d vHead=%d)",
			c.LinearConvKernelDim, c.LinearKeyHeadDim, c.LinearValueHeadDim)
	case c.LinearNumKeyHeads <= 0 || c.LinearNumValueHeads <= 0 || c.LinearNumValueHeads%c.LinearNumKeyHeads != 0:
		return fmt.Errorf("decoder(qwen3_5): linear_num_value_heads %d not a multiple of linear_num_key_heads %d (GVA)",
			c.LinearNumValueHeads, c.LinearNumKeyHeads)
	case c.NumExperts > 0:
		return fmt.Errorf("decoder(qwen3_5): num_experts=%d — this adapter is the DENSE variant; a MoE checkpoint belongs on qwen3_5_moe", c.NumExperts)
	case len(c.RopeParameters) == 0:
		return fmt.Errorf("decoder(qwen3_5): rope_parameters required")
	case c.RMSNormEps <= 0:
		return fmt.Errorf("decoder(qwen3_5): rms_norm_eps must be >0")
	}
	return nil
}

// ValidateQwen35 pins the qwen3_5_moe assumptions: a 3:1 linear/full layer mix (layer_types), the Gated DeltaNet dims (GVA:
// value-head count a multiple of the key-head count), per-attention-type RoPE in rope_parameters, QK-norm softmax layers,
// and a routed + shared MoE on every layer.
func (c *Config) ValidateQwen35() error {
	switch {
	case c.HiddenDim == 0 || c.NumLayers == 0 || c.NumHeads == 0 || c.HeadDim == 0:
		return fmt.Errorf("decoder(qwen3_5_moe): missing core dims (hidden=%d layers=%d heads=%d headDim=%d)",
			c.HiddenDim, c.NumLayers, c.NumHeads, c.HeadDim)
	case c.NumKVHeads == 0 || c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("decoder(qwen3_5_moe): num_heads %d not a multiple of num_kv_heads %d (GQA)", c.NumHeads, c.NumKVHeads)
	case len(c.LayerTypes) != c.NumLayers:
		return fmt.Errorf("decoder(qwen3_5_moe): layer_types has %d entries, want %d", len(c.LayerTypes), c.NumLayers)
	case c.LinearConvKernelDim <= 0 || c.LinearKeyHeadDim <= 0 || c.LinearValueHeadDim <= 0:
		return fmt.Errorf("decoder(qwen3_5_moe): missing linear (DeltaNet) dims (conv=%d kHead=%d vHead=%d)",
			c.LinearConvKernelDim, c.LinearKeyHeadDim, c.LinearValueHeadDim)
	case c.LinearNumKeyHeads <= 0 || c.LinearNumValueHeads <= 0 || c.LinearNumValueHeads%c.LinearNumKeyHeads != 0:
		return fmt.Errorf("decoder(qwen3_5_moe): linear_num_value_heads %d not a multiple of linear_num_key_heads %d (GVA)",
			c.LinearNumValueHeads, c.LinearNumKeyHeads)
	case c.NumExperts <= 0 || c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > c.NumExperts:
		return fmt.Errorf("decoder(qwen3_5_moe): bad MoE (experts=%d top_k=%d)", c.NumExperts, c.NumExpertsPerTok)
	case c.MoeIntermediateSize <= 0:
		return fmt.Errorf("decoder(qwen3_5_moe): moe_intermediate_size must be >0")
	case len(c.RopeParameters) == 0:
		return fmt.Errorf("decoder(qwen3_5_moe): rope_parameters required")
	case c.RMSNormEps <= 0:
		return fmt.Errorf("decoder(qwen3_5_moe): rms_norm_eps must be >0")
	}
	return nil
}
