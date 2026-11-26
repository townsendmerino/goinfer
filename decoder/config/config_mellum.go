package config

import (
	"fmt"
)

// ValidateMellum pins the assumptions the mellum adapter makes: a sparse MoE on
// every layer (num_experts / num_experts_per_tok / moe_intermediate_size), the
// per-attention-type rope_parameters (full_attention YaRN + sliding plain), and
// the explicit layer_types interleave. The dense FFN axes (RMSNorm, SwiGLU,
// derived head_dim) are the same as llama and resolved there.
func (c *Config) ValidateMellum() error {
	switch {
	case c.HiddenDim == 0 || c.NumLayers == 0 || c.NumHeads == 0 || c.NumKVHeads == 0:
		return fmt.Errorf("decoder(mellum): missing core dims (hidden/layers/heads)")
	case c.NumExperts <= 0:
		return fmt.Errorf("decoder(mellum): num_experts must be >0, got %d", c.NumExperts)
	case c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > c.NumExperts:
		return fmt.Errorf("decoder(mellum): num_experts_per_tok %d out of range (1..%d)", c.NumExpertsPerTok, c.NumExperts)
	case c.MoeIntermediateSize <= 0:
		return fmt.Errorf("decoder(mellum): moe_intermediate_size must be >0, got %d", c.MoeIntermediateSize)
	case len(c.RopeParameters) == 0:
		return fmt.Errorf("decoder(mellum): rope_parameters required (full_attention + sliding_attention)")
	case len(c.LayerTypes) != c.NumLayers:
		return fmt.Errorf("decoder(mellum): layer_types has %d entries, want %d", len(c.LayerTypes), c.NumLayers)
	}
	return nil
}
