package config

import (
	"fmt"
)

// ValidateQwen3 pins the assumptions the qwen3 forward path makes (dense, SwiGLU, GQA, single-base RoPE); ValidateQwen3Moe
// layers the sparse-MoE checks over it.
func (c *Config) ValidateQwen3() error {
	switch {
	case c.HiddenDim == 0 || c.NumLayers == 0 || c.NumHeads == 0 || c.HeadDim == 0:
		return fmt.Errorf("decoder(qwen3): missing required dim (hidden=%d layers=%d heads=%d headDim=%d)",
			c.HiddenDim, c.NumLayers, c.NumHeads, c.HeadDim)
	case c.NumKVHeads == 0 || c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("decoder(qwen3): num_heads %d not a multiple of num_kv_heads %d (GQA)", c.NumHeads, c.NumKVHeads)
	case c.VocabSize == 0:
		return fmt.Errorf("decoder(qwen3): vocab_size is zero")
	case c.IntermediateDim == 0:
		return fmt.Errorf("decoder(qwen3): intermediate_size is zero")
	case c.HiddenAct != "" && c.HiddenAct != "silu":
		return fmt.Errorf("decoder(qwen3): hidden_act=%q unsupported (silu/SwiGLU only)", c.HiddenAct)
	case c.RMSNormEps <= 0:
		return fmt.Errorf("decoder(qwen3): rms_norm_eps must be >0, got %v", c.RMSNormEps)
	case c.RoPEGlobalBase <= 0:
		return fmt.Errorf("decoder(qwen3): rope_theta must be >0, got %v", c.RoPEGlobalBase)
	}
	return nil
}

// ValidateQwen3Moe pins the Qwen3-MoE assumptions (model_type "qwen3_moe"): the qwen3 dense constraints (QK-norm, no q/k/v
// bias, single-base RoPE) plus a valid sparse MoE on every layer (num_experts / num_experts_per_tok / moe_intermediate_size).
// Unlike qwen2_moe there is no shared expert: the released config has no shared_expert_intermediate_size.
func (c *Config) ValidateQwen3Moe() error {
	if err := c.ValidateQwen3(); err != nil {
		return err
	}
	switch {
	case c.NumExperts <= 0:
		return fmt.Errorf("decoder(qwen3_moe): num_experts must be >0, got %d", c.NumExperts)
	case c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > c.NumExperts:
		return fmt.Errorf("decoder(qwen3_moe): num_experts_per_tok %d out of range (1..%d)", c.NumExpertsPerTok, c.NumExperts)
	case c.MoeIntermediateSize <= 0:
		return fmt.Errorf("decoder(qwen3_moe): moe_intermediate_size must be >0, got %d", c.MoeIntermediateSize)
	}
	return nil
}
