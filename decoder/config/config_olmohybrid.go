package config

import (
	"fmt"
)

// ValidateOlmoHybrid checks Olmo Hybrid's (model_type olmo_hybrid) core dims: the same Gated-DeltaNet geometry checks as
// qwen3_5, and no MoE (it is a dense hybrid with no num_experts field). The released sizes are MHA, but any GQA-divisible
// head count passes.
func (c *Config) ValidateOlmoHybrid() error {
	switch {
	case c.HiddenDim == 0 || c.NumLayers == 0 || c.NumHeads == 0:
		return fmt.Errorf("decoder(olmo_hybrid): missing core dims (hidden=%d layers=%d heads=%d)",
			c.HiddenDim, c.NumLayers, c.NumHeads)
	case c.NumKVHeads == 0 || c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("decoder(olmo_hybrid): num_heads %d not a multiple of num_kv_heads %d (GQA)", c.NumHeads, c.NumKVHeads)
	case c.IntermediateDim <= 0:
		return fmt.Errorf("decoder(olmo_hybrid): intermediate_size must be >0 — this is a dense hybrid, not MoE")
	case len(c.LayerTypes) != c.NumLayers:
		return fmt.Errorf("decoder(olmo_hybrid): layer_types has %d entries, want %d", len(c.LayerTypes), c.NumLayers)
	case c.LinearConvKernelDim <= 0 || c.LinearKeyHeadDim <= 0 || c.LinearValueHeadDim <= 0:
		return fmt.Errorf("decoder(olmo_hybrid): missing linear (DeltaNet) dims (conv=%d kHead=%d vHead=%d)",
			c.LinearConvKernelDim, c.LinearKeyHeadDim, c.LinearValueHeadDim)
	case c.LinearNumKeyHeads <= 0 || c.LinearNumValueHeads <= 0 || c.LinearNumValueHeads%c.LinearNumKeyHeads != 0:
		return fmt.Errorf("decoder(olmo_hybrid): linear_num_value_heads %d not a multiple of linear_num_key_heads %d (GVA)",
			c.LinearNumValueHeads, c.LinearNumKeyHeads)
	case c.NumExperts > 0:
		return fmt.Errorf("decoder(olmo_hybrid): num_experts=%d — this family has no MoE variant", c.NumExperts)
	case len(c.RopeParameters) == 0:
		return fmt.Errorf("decoder(olmo_hybrid): rope_parameters required (rope_theta may be null — NoPE on the released checkpoint)")
	case c.RMSNormEps <= 0:
		return fmt.Errorf("decoder(olmo_hybrid): rms_norm_eps must be >0")
	}
	return nil
}
