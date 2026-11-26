package config

import (
	"fmt"
)

// ValidateGlm4Moe pins the GLM-4.5/4.6 (glm4_moe) assumptions: a valid DeepSeek-style
// sparse MoE (n_routed_experts / num_experts_per_tok / moe_intermediate_size) with a
// shared expert (n_shared_experts ≥ 1), and a first_k_dense_replace in [0, num_layers)
// dense prefix. The attention is qwen3-like (QK-norm, partial RoPE, no bias) and is
// validated by the generic head-dim/layer checks at resolve time.
func (c *Config) ValidateGlm4Moe() error {
	switch {
	case c.NRoutedExperts <= 0:
		return fmt.Errorf("decoder(glm4_moe): n_routed_experts must be >0, got %d", c.NRoutedExperts)
	case c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > c.NRoutedExperts:
		return fmt.Errorf("decoder(glm4_moe): num_experts_per_tok %d out of range (1..%d)", c.NumExpertsPerTok, c.NRoutedExperts)
	case c.MoeIntermediateSize <= 0:
		return fmt.Errorf("decoder(glm4_moe): moe_intermediate_size must be >0, got %d", c.MoeIntermediateSize)
	case c.NSharedExperts <= 0:
		return fmt.Errorf("decoder(glm4_moe): n_shared_experts must be >0, got %d", c.NSharedExperts)
	case c.FirstKDenseReplace < 0 || c.FirstKDenseReplace >= c.NumLayers:
		return fmt.Errorf("decoder(glm4_moe): first_k_dense_replace %d out of range (0..%d)", c.FirstKDenseReplace, c.NumLayers-1)
	}
	return nil
}
