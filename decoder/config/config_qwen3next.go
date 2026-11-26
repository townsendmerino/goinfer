package config

import (
	"fmt"
)

// NormalizeQwen3NextLayerTypes synthesizes LayerTypes from FullAttentionInterval for Qwen3-Next, whose released config.json
// has no layer_types field (unlike qwen3_5_moe, which states the pattern). The formula follows transformers'
// configuration_qwen3_next.py:
//
//	"linear_attention" if (i+1)%interval else "full_attention"
//
// 0-indexed i, so layer 3 (not layer 4) is the first full-attention layer at the default interval=4. A no-op if LayerTypes
// is already populated (a transformers-instantiated config carries it directly, the same asymmetry NormalizeNemotronBlocks
// handles).
func (c *Config) NormalizeQwen3NextLayerTypes() error {
	if len(c.LayerTypes) > 0 || c.FullAttentionInterval == 0 {
		return nil
	}
	if c.FullAttentionInterval < 0 {
		return fmt.Errorf("decoder(qwen3_next): full_attention_interval must be >0, got %d", c.FullAttentionInterval)
	}
	if c.NumLayers <= 0 {
		return fmt.Errorf("decoder(qwen3_next): num_hidden_layers must be >0 before layer_types can be derived")
	}
	types := make([]string, c.NumLayers)
	for i := range types {
		if (i+1)%c.FullAttentionInterval == 0 {
			types[i] = "full_attention"
		} else {
			types[i] = "linear_attention"
		}
	}
	c.LayerTypes = types
	return nil
}

// ValidateQwen3Next pins the same shape assumptions as ValidateQwen35, which this family shares field for field, EXCEPT the
// RoPE check: the released config carries a flat rope_theta and no rope_parameters object, so requiring it (as
// ValidateQwen35 does) would reject every real Qwen3-Next checkpoint. Accepts either shape, matching qwen3NextArchitecture's
// dual-path RoPE resolution.
func (c *Config) ValidateQwen3Next() error {
	switch {
	case c.HiddenDim == 0 || c.NumLayers == 0 || c.NumHeads == 0 || c.HeadDim == 0:
		return fmt.Errorf("decoder(qwen3_next): missing core dims (hidden=%d layers=%d heads=%d headDim=%d)",
			c.HiddenDim, c.NumLayers, c.NumHeads, c.HeadDim)
	case c.NumKVHeads == 0 || c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("decoder(qwen3_next): num_heads %d not a multiple of num_kv_heads %d (GQA)", c.NumHeads, c.NumKVHeads)
	case len(c.LayerTypes) != c.NumLayers:
		return fmt.Errorf("decoder(qwen3_next): layer_types has %d entries, want %d (full_attention_interval unset or invalid?)", len(c.LayerTypes), c.NumLayers)
	case c.LinearConvKernelDim <= 0 || c.LinearKeyHeadDim <= 0 || c.LinearValueHeadDim <= 0:
		return fmt.Errorf("decoder(qwen3_next): missing linear (DeltaNet) dims (conv=%d kHead=%d vHead=%d)",
			c.LinearConvKernelDim, c.LinearKeyHeadDim, c.LinearValueHeadDim)
	case c.LinearNumKeyHeads <= 0 || c.LinearNumValueHeads <= 0 || c.LinearNumValueHeads%c.LinearNumKeyHeads != 0:
		return fmt.Errorf("decoder(qwen3_next): linear_num_value_heads %d not a multiple of linear_num_key_heads %d (GVA)",
			c.LinearNumValueHeads, c.LinearNumKeyHeads)
	case c.NumExperts <= 0 || c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > c.NumExperts:
		return fmt.Errorf("decoder(qwen3_next): bad MoE (experts=%d top_k=%d)", c.NumExperts, c.NumExpertsPerTok)
	case c.MoeIntermediateSize <= 0:
		return fmt.Errorf("decoder(qwen3_next): moe_intermediate_size must be >0")
	case len(c.RopeParameters) == 0 && c.RoPEGlobalBase <= 0:
		return fmt.Errorf("decoder(qwen3_next): need either rope_parameters or a top-level rope_theta")
	case c.RMSNormEps <= 0:
		return fmt.Errorf("decoder(qwen3_next): rms_norm_eps must be >0")
	}
	return nil
}
