package config

import (
	"fmt"
)

// ValidateCohere pins Cohere / Command-R (model_type "cohere", CohereForCausalLM): bias-free LayerNorm, the parallel
// attn+MLP block, gated SiLU MLP, tied 256k embeddings, GPT-J interleaved RoPE, and the logit_scale multiplier. use_qk_norm
// (Command-R+ only) is deferred: Cohere's QK-norm is LayerNorm-style, distinct from the RMSNorm QK-norm hook, so admitting
// it would run silently wrong; it is rejected loudly here rather than mis-normalized.
func (c *Config) ValidateCohere() error {
	switch {
	case c.HiddenDim == 0 || c.NumLayers == 0 || c.NumHeads == 0 || c.EffectiveHeadDim() == 0:
		return fmt.Errorf("decoder(cohere): missing required dim (hidden=%d layers=%d heads=%d headDim=%d)",
			c.HiddenDim, c.NumLayers, c.NumHeads, c.EffectiveHeadDim())
	case c.NumKVHeads == 0 || c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("decoder(cohere): num_heads %d not a multiple of num_kv_heads %d (GQA)", c.NumHeads, c.NumKVHeads)
	case c.VocabSize == 0:
		return fmt.Errorf("decoder(cohere): vocab_size is zero")
	case c.IntermediateDim == 0:
		return fmt.Errorf("decoder(cohere): intermediate_size is zero")
	case c.HiddenAct != "" && c.HiddenAct != "silu":
		return fmt.Errorf("decoder(cohere): hidden_act=%q unsupported (silu/SwiGLU only)", c.HiddenAct)
	case c.LayerNormEps <= 0:
		return fmt.Errorf("decoder(cohere): layer_norm_eps must be >0, got %v", c.LayerNormEps)
	case c.RoPEGlobalBase <= 0:
		return fmt.Errorf("decoder(cohere): rope_theta must be >0, got %v", c.RoPEGlobalBase)
	case c.LogitScale <= 0:
		return fmt.Errorf("decoder(cohere): logit_scale must be >0, got %v", c.LogitScale)
	case c.AttentionBias:
		return fmt.Errorf("decoder(cohere): attention_bias=true not supported (Cohere has no q/k/v/o bias)")
	case c.UseQKNorm:
		return fmt.Errorf("decoder(cohere): use_qk_norm=true (Command-R+) not yet supported — Cohere's QK-norm is LayerNorm-style, a Phase-2 primitive; Command-R/Aya/Aya-Expanse (use_qk_norm=false) load fine")
	case c.SlidingWindow != 0:
		return fmt.Errorf("decoder(cohere): sliding_window set — that is cohere2 (Command-R7B/Command-A), a separate Phase-2 model_type")
	}
	return nil
}

// ValidateCohere2 pins Cohere2 / Command-R7B (model_type "cohere2",
// Cohere2ForCausalLM: Command-R7B, Command-A, R7B-arabic). It is cohere1's stack
// — bias-free LayerNorm, the parallel attn+MLP block, gated SiLU, tied embeddings,
// GPT-J interleaved RoPE, logit_scale — PLUS interleaved sliding-window/full
// layers where the full (global) layers are NoPE. Unlike cohere1 there is NO
// QK-norm at all. Requires the sliding-window interleave to be expressible (either
// explicit layer_types or a sliding_window_pattern).
func (c *Config) ValidateCohere2() error {
	switch {
	case c.HiddenDim == 0 || c.NumLayers == 0 || c.NumHeads == 0 || c.EffectiveHeadDim() == 0:
		return fmt.Errorf("decoder(cohere2): missing required dim (hidden=%d layers=%d heads=%d headDim=%d)",
			c.HiddenDim, c.NumLayers, c.NumHeads, c.EffectiveHeadDim())
	case c.NumKVHeads == 0 || c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("decoder(cohere2): num_heads %d not a multiple of num_kv_heads %d (GQA)", c.NumHeads, c.NumKVHeads)
	case c.VocabSize == 0:
		return fmt.Errorf("decoder(cohere2): vocab_size is zero")
	case c.IntermediateDim == 0:
		return fmt.Errorf("decoder(cohere2): intermediate_size is zero")
	case c.HiddenAct != "" && c.HiddenAct != "silu":
		return fmt.Errorf("decoder(cohere2): hidden_act=%q unsupported (silu/SwiGLU only)", c.HiddenAct)
	case c.LayerNormEps <= 0:
		return fmt.Errorf("decoder(cohere2): layer_norm_eps must be >0, got %v", c.LayerNormEps)
	case c.RoPEGlobalBase <= 0:
		return fmt.Errorf("decoder(cohere2): rope_theta must be >0, got %v", c.RoPEGlobalBase)
	case c.LogitScale <= 0:
		return fmt.Errorf("decoder(cohere2): logit_scale must be >0, got %v", c.LogitScale)
	case c.AttentionBias:
		return fmt.Errorf("decoder(cohere2): attention_bias=true not supported (Cohere has no q/k/v/o bias)")
	case c.UseQKNorm:
		return fmt.Errorf("decoder(cohere2): use_qk_norm=true unexpected — Cohere2 has no QK-norm")
	case c.SlidingWindow <= 0:
		return fmt.Errorf("decoder(cohere2): sliding_window must be >0 (that is the cohere1↔cohere2 distinction)")
	case len(c.LayerTypes) == 0 && c.SlidingWindowPattern <= 0:
		return fmt.Errorf("decoder(cohere2): need layer_types or sliding_window_pattern to place the sliding/global interleave")
	}
	return nil
}
