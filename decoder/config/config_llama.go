package config

import (
	"fmt"
)

// ValidateLlama pins the assumptions the llama forward path makes (dense, SwiGLU, GQA, single-base RoPE, no QK-norm). It
// differs from ValidateQwen3 by allowing head_dim to be derived (EffectiveHeadDim()). RoPE scaling is handled by the adapter via
// parseRopeScaling; attention bias is rejected. Plain Llama-2/3 and Mistral checkpoints pass.
func (c *Config) ValidateLlama() error {
	switch {
	case c.HiddenDim == 0 || c.NumLayers == 0 || c.NumHeads == 0 || c.EffectiveHeadDim() == 0:
		return fmt.Errorf("decoder(llama): missing required dim (hidden=%d layers=%d heads=%d headDim=%d)",
			c.HiddenDim, c.NumLayers, c.NumHeads, c.EffectiveHeadDim())
	case c.NumKVHeads == 0 || c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("decoder(llama): num_heads %d not a multiple of num_kv_heads %d (GQA)", c.NumHeads, c.NumKVHeads)
	case c.VocabSize == 0:
		return fmt.Errorf("decoder(llama): vocab_size is zero")
	case c.IntermediateDim == 0:
		return fmt.Errorf("decoder(llama): intermediate_size is zero")
	case c.HiddenAct != "" && c.HiddenAct != "silu":
		return fmt.Errorf("decoder(llama): hidden_act=%q unsupported (silu/SwiGLU only)", c.HiddenAct)
	case c.RMSNormEps <= 0:
		return fmt.Errorf("decoder(llama): rms_norm_eps must be >0, got %v", c.RMSNormEps)
	case c.RoPEGlobalBase <= 0:
		return fmt.Errorf("decoder(llama): rope_theta must be >0, got %v", c.RoPEGlobalBase)
	case c.AttentionBias:
		return fmt.Errorf("decoder(llama): attention_bias=true (q/k/v/o bias) not yet supported")
	}
	return nil
}
