package config

import (
	"fmt"
)

// ValidateQwen2 pins the assumptions the qwen2 forward path makes: like llama
// (dense, SwiGLU, GQA, single-base RoPE, no QK-norm, derived head_dim) but with
// q/k/v projection bias. Sliding-window attention (use_sliding_window) is a
// follow-up — reject it rather than silently run full attention.
func (c *Config) ValidateQwen2() error {
	if err := c.ValidateLlama(); err != nil {
		return err
	}
	if c.UseSlidingWindow {
		return fmt.Errorf("decoder(qwen2): use_sliding_window=true not yet supported (full-attention checkpoints only)")
	}
	return nil
}

// ValidateQwen2Moe pins the Qwen2-MoE assumptions: the qwen2 dense constraints
// (llama + q/k/v bias) plus a valid sparse MoE on every layer (num_experts /
// num_experts_per_tok / moe_intermediate_size) and an always-on shared expert
// (shared_expert_intermediate_size). decoder_sparse_step must be 1 (every layer
// sparse) — the only configuration shipped by the Qwen-MoE family.
func (c *Config) ValidateQwen2Moe() error {
	if err := c.ValidateLlama(); err != nil {
		return err
	}
	switch {
	case c.NumExperts <= 0:
		return fmt.Errorf("decoder(qwen2_moe): num_experts must be >0, got %d", c.NumExperts)
	case c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > c.NumExperts:
		return fmt.Errorf("decoder(qwen2_moe): num_experts_per_tok %d out of range (1..%d)", c.NumExpertsPerTok, c.NumExperts)
	case c.MoeIntermediateSize <= 0:
		return fmt.Errorf("decoder(qwen2_moe): moe_intermediate_size must be >0, got %d", c.MoeIntermediateSize)
	case c.SharedExpertIntermediateSize <= 0:
		return fmt.Errorf("decoder(qwen2_moe): shared_expert_intermediate_size must be >0, got %d", c.SharedExpertIntermediateSize)
	}
	return nil
}
