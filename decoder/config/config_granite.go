package config

import (
	"fmt"
)

// ValidateGranite pins the Granite-4.0-H (granitemoehybrid) assumptions: a layer_types list covering every layer
// (mamba | attention), a valid Mamba-2 geometry, and a routed+shared MoE (num_local_experts / num_experts_per_tok /
// intermediate_size / shared_intermediate_size).
func (c *Config) ValidateGranite() error {
	switch {
	case len(c.LayerTypes) != c.NumLayers:
		return fmt.Errorf("decoder(granite): layer_types has %d entries, want %d", len(c.LayerTypes), c.NumLayers)
	case c.MambaNHeads <= 0 || c.MambaDHead <= 0 || c.MambaDState <= 0 || c.MambaDConv <= 0 || c.MambaNGroups <= 0:
		return fmt.Errorf("decoder(granite): bad mamba dims (n_heads=%d d_head=%d d_state=%d d_conv=%d n_groups=%d)", c.MambaNHeads, c.MambaDHead, c.MambaDState, c.MambaDConv, c.MambaNGroups)
	case c.MambaNHeads%c.MambaNGroups != 0:
		return fmt.Errorf("decoder(granite): mamba_n_heads %d not divisible by mamba_n_groups %d", c.MambaNHeads, c.MambaNGroups)
	case c.NumLocalExperts <= 0 || c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > c.NumLocalExperts:
		return fmt.Errorf("decoder(granite): bad MoE (experts=%d top_k=%d)", c.NumLocalExperts, c.NumExpertsPerTok)
	case c.IntermediateDim <= 0 || c.SharedIntermediateSize <= 0:
		return fmt.Errorf("decoder(granite): intermediate_size=%d shared_intermediate_size=%d must be >0", c.IntermediateDim, c.SharedIntermediateSize)
	}
	return nil
}

// ValidateGraniteDense pins the dense Granite 4.2 assumptions (model_type "granite"): a plain llama skeleton (GQA, SwiGLU,
// single-base RoPE, no bias, no QK-norm) plus Granite's scalar multipliers. residual_multiplier is the one multiplier the
// generic forward path cannot apply (granitemoehybrid's own forward does, via graniteParams; see
// graniteDenseArchitecture); every released 4.2 size ships 1.0, so anything else is rejected loudly rather than silently
// dropped.
func (c *Config) ValidateGraniteDense() error {
	switch {
	case c.HiddenDim == 0 || c.NumLayers == 0 || c.NumHeads == 0 || c.EffectiveHeadDim() == 0:
		return fmt.Errorf("decoder(granite): missing required dim (hidden=%d layers=%d heads=%d headDim=%d)",
			c.HiddenDim, c.NumLayers, c.NumHeads, c.EffectiveHeadDim())
	case c.NumKVHeads == 0 || c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("decoder(granite): num_heads %d not a multiple of num_kv_heads %d (GQA)", c.NumHeads, c.NumKVHeads)
	case c.VocabSize == 0:
		return fmt.Errorf("decoder(granite): vocab_size is zero")
	case c.IntermediateDim == 0:
		return fmt.Errorf("decoder(granite): intermediate_size is zero")
	case c.HiddenAct != "" && c.HiddenAct != "silu":
		return fmt.Errorf("decoder(granite): hidden_act=%q unsupported (silu/SwiGLU only)", c.HiddenAct)
	case c.RMSNormEps <= 0:
		return fmt.Errorf("decoder(granite): rms_norm_eps must be >0, got %v", c.RMSNormEps)
	case c.AttentionBias:
		return fmt.Errorf("decoder(granite): attention_bias=true (q/k/v/o bias) not yet supported")
	case c.ResidualMultiplier != 0 && c.ResidualMultiplier != 1:
		return fmt.Errorf("decoder(granite): residual_multiplier=%v not yet supported — the generic "+
			"forward path this family rides has no residual-scale hook (only granitemoehybrid's "+
			"own-forward does); every released 4.2 size ships 1.0", c.ResidualMultiplier)
	}
	return nil
}
