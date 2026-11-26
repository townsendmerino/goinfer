package config

import (
	"fmt"
)

// ValidateFlint pins the assumptions the Flint forward is built on, matching
// transformers' FlintConfig (PR #49824) and the released Cognichip/Flint
// checkpoint. When layer_types or mlp_only_layers is omitted from config.json,
// it synthesizes the upstream defaults before validating.
func (c *Config) ValidateFlint() error {
	if len(c.LayerTypes) == 0 && c.NumLayers > 0 {
		c.LayerTypes = make([]string, c.NumLayers)
		for i := range c.NumLayers {
			if i < 2 || i%4 == 1 || i == c.NumLayers-1 {
				c.LayerTypes[i] = "full_attention"
			} else {
				c.LayerTypes[i] = "sliding_attention"
			}
		}
	}
	if c.MlpOnlyLayers == nil && len(c.MlpLayerTypes) == 0 && c.NumLayers >= 2 {
		c.MlpOnlyLayers = []int{0, 1}
	}
	switch {
	case c.HiddenDim <= 0 || c.NumLayers <= 0 || c.NumHeads <= 0 || c.NumKVHeads <= 0:
		return fmt.Errorf("decoder(flint): hidden_size/num_hidden_layers/num_attention_heads/num_key_value_heads must all be >0")
	case c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("decoder(flint): num_attention_heads %d not a multiple of num_key_value_heads %d (GQA)", c.NumHeads, c.NumKVHeads)
	case c.EffectiveHeadDim() <= 0 || c.EffectiveHeadDim()%2 != 0:
		return fmt.Errorf("decoder(flint): head_dim %d must be positive and even", c.EffectiveHeadDim())
	case c.NumExperts <= 0:
		return fmt.Errorf("decoder(flint): num_experts must be >0, got %d", c.NumExperts)
	case c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > c.NumExperts:
		return fmt.Errorf("decoder(flint): num_experts_per_tok=%d out of range for num_experts=%d", c.NumExpertsPerTok, c.NumExperts)
	case c.MoeIntermediateSize <= 0:
		return fmt.Errorf("decoder(flint): moe_intermediate_size must be >0")
	case c.SharedExpertIntermediateSize <= 0:
		return fmt.Errorf("decoder(flint): shared_expert_intermediate_size must be >0")
	case c.HiddenAct != "" && c.HiddenAct != "silu":
		return fmt.Errorf("decoder(flint): hidden_act=%q unsupported (silu/SwiGLU only)", c.HiddenAct)
	case c.RouterScoreFunction != "" && c.RouterScoreFunction != "sigmoid":
		return fmt.Errorf("decoder(flint): router_score_function=%q unsupported (sigmoid only)", c.RouterScoreFunction)
	case c.RouterExpertBias:
		return fmt.Errorf("decoder(flint): router_expert_bias=true unsupported")
	case c.AttentionSink != nil && !*c.AttentionSink:
		return fmt.Errorf("decoder(flint): attention_sink=false unsupported")
	case c.SoftmaxType != "" && c.SoftmaxType != "learnable":
		return fmt.Errorf("decoder(flint): softmax_type=%q unsupported (learnable only)", c.SoftmaxType)
	case c.RMSNormEps <= 0:
		return fmt.Errorf("decoder(flint): rms_norm_eps must be >0, got %v", c.RMSNormEps)
	}
	if len(c.LayerTypes) != c.NumLayers {
		return fmt.Errorf("decoder(flint): layer_types has %d entries, want %d", len(c.LayerTypes), c.NumLayers)
	}
	for i, t := range c.LayerTypes {
		if t != "full_attention" && t != "sliding_attention" {
			return fmt.Errorf("decoder(flint): layer_types[%d]=%q unsupported (full_attention / sliding_attention)", i, t)
		}
		if t == "sliding_attention" && c.SlidingWindow <= 0 {
			return fmt.Errorf("decoder(flint): layer_types has sliding_attention at %d but sliding_window=%d", i, c.SlidingWindow)
		}
	}
	if len(c.RopeParameters) == 0 && c.RoPEGlobalBase <= 0 {
		return fmt.Errorf("decoder(flint): need rope_parameters or rope_theta")
	}
	if len(c.NoRopeLayerIndices) > 0 {
		nopeSet := make(map[int]bool, len(c.NoRopeLayerIndices))
		for _, idx := range c.NoRopeLayerIndices {
			if idx < 0 || idx >= c.NumLayers {
				return fmt.Errorf("decoder(flint): no_rope_layer_indices contains out-of-range layer %d (num_hidden_layers=%d)", idx, c.NumLayers)
			}
			nopeSet[idx] = true
		}
		for i, t := range c.LayerTypes {
			wantNoPE := t == "full_attention"
			if nopeSet[i] != wantNoPE {
				return fmt.Errorf("decoder(flint): no_rope_layer_indices=%v disagrees with layer_types[%d]=%q (full_attention layers must match NoPE indices)",
					c.NoRopeLayerIndices, i, t)
			}
		}
	}
	firstK, err := c.LagunaFirstKDense()
	if err != nil {
		return err
	}
	if firstK > 0 && c.IntermediateDim <= 0 {
		return fmt.Errorf("decoder(flint): intermediate_size must be >0 when dense MLP prefix layers exist (firstKDense=%d)", firstK)
	}
	return nil
}
