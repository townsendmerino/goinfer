package config

import (
	"fmt"
)

// ValidateGPT2 pins the assumptions the gpt2 forward path makes: LayerNorm,
// learned absolute positions (no RoPE), a non-gated GELU MLP, fused q/k/v with
// bias, and tied embeddings. GPT-2's config uses the n_embd/n_head/n_layer keys.
func (c *Config) ValidateGPT2() error {
	switch {
	case c.NEmbd == 0 || c.NLayer == 0 || c.NHead == 0:
		return fmt.Errorf("decoder(gpt2): missing required dim (n_embd=%d n_layer=%d n_head=%d)", c.NEmbd, c.NLayer, c.NHead)
	case c.NEmbd%c.NHead != 0:
		return fmt.Errorf("decoder(gpt2): n_embd %d not divisible by n_head %d", c.NEmbd, c.NHead)
	case c.VocabSize == 0:
		return fmt.Errorf("decoder(gpt2): vocab_size is zero")
	case c.NPositions == 0:
		return fmt.Errorf("decoder(gpt2): n_positions is zero (need the learned position table size)")
	case c.LayerNormEpsilon <= 0:
		return fmt.Errorf("decoder(gpt2): layer_norm_epsilon must be >0, got %v", c.LayerNormEpsilon)
	case c.ActivationFunction != "" && c.ActivationFunction != "gelu_new" && c.ActivationFunction != "gelu":
		return fmt.Errorf("decoder(gpt2): activation_function=%q unsupported (gelu_new/gelu)", c.ActivationFunction)
	}
	return nil
}
