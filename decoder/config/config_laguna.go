package config

import (
	"encoding/json"
	"fmt"
)

// LagunaGatePerHead resolves Laguna's `gating` field to the gate granularity. The field ships as "per-head", true or
// "per-element" across releases of one model_type, so this mirrors the vendor's resolution (modeling_laguna.py):
//
//	self.gating       = bool(gating)              # false only for literal `false`
//	self.gate_per_head = (gating == "per-head")   # STRING compare, so true => per-element
//
// `true` and "per-element" are the same path and only "per-head" differs. Returns (enabled, perHead, error). An absent field
// is HF's default of true (gating on, per-element); a checkpoint that means "no gating" must say `false` explicitly.
func (c *Config) LagunaGatePerHead() (enabled, perHead bool, err error) {
	if len(c.Gating) == 0 || string(c.Gating) == "null" {
		return true, false, nil // HF default: gating=True ⇒ per-element
	}
	var s string
	if err := json.Unmarshal(c.Gating, &s); err == nil {
		switch s {
		case "per-head":
			return true, true, nil
		case "per-element":
			return true, false, nil
		default:
			return false, false, fmt.Errorf("decoder(laguna): gating=%q unsupported (per-head / per-element / true / false)", s)
		}
	}
	var b bool
	if err := json.Unmarshal(c.Gating, &b); err != nil {
		return false, false, fmt.Errorf("decoder(laguna): gating must be a string or bool, got %s", string(c.Gating))
	}
	return b, false, nil // true ⇒ per-element (the vendor's string compare fails on a bool)
}

// LagunaFirstKDense turns Laguna's dense-layer declaration into goinfer's FirstKDense prefix count.
//
// There are two spellings, and mlp_layer_types is the reliable one: every released config carries it (["dense","sparse",...]),
// but some drop mlp_only_layers, and reading only that yields FirstKDense=0, which would make the loader treat a dense layer 0
// as MoE and demand expert tensors that do not exist. So mlp_layer_types wins when present. Either way the dense layers must
// form a contiguous prefix, which is all FirstKDense can express; a non-contiguous layout is rejected rather than silently
// truncated.
func (c *Config) LagunaFirstKDense() (int, error) {
	if len(c.MlpLayerTypes) > 0 {
		if n := len(c.MlpLayerTypes); n != c.NumLayers {
			return 0, fmt.Errorf("decoder(laguna): mlp_layer_types has %d entries, want %d (num_hidden_layers)", n, c.NumLayers)
		}
		k := 0
		for i, t := range c.MlpLayerTypes {
			switch t {
			case "dense":
				if i != k {
					return 0, fmt.Errorf("decoder(laguna): mlp_layer_types has a non-prefix dense layer at %d; FirstKDense cannot express it", i)
				}
				k++
			case "sparse":
			default:
				return 0, fmt.Errorf("decoder(laguna): mlp_layer_types[%d]=%q unsupported (dense / sparse)", i, t)
			}
		}
		return k, nil
	}
	if len(c.MlpOnlyLayers) == 0 {
		return 0, nil
	}
	seen := make(map[int]bool, len(c.MlpOnlyLayers))
	maxL := -1
	for _, l := range c.MlpOnlyLayers {
		if l < 0 || l >= c.NumLayers {
			return 0, fmt.Errorf("decoder(laguna): mlp_only_layers has out-of-range layer %d (num_hidden_layers=%d)", l, c.NumLayers)
		}
		seen[l] = true
		maxL = max(maxL, l)
	}
	if len(seen) != maxL+1 {
		return 0, fmt.Errorf("decoder(laguna): mlp_only_layers %v is not a contiguous prefix; FirstKDense cannot express it", c.MlpOnlyLayers)
	}
	return maxL + 1, nil
}

// ValidateLaguna pins the assumptions the Laguna forward is built on, read against the three released configs.
func (c *Config) ValidateLaguna() error {
	switch {
	case c.HiddenDim <= 0 || c.NumLayers <= 0 || c.NumHeads <= 0 || c.NumKVHeads <= 0:
		return fmt.Errorf("decoder(laguna): hidden_size/num_hidden_layers/num_attention_heads/num_key_value_heads must all be >0")
	case c.EffectiveHeadDim() <= 0:
		return fmt.Errorf("decoder(laguna): head_dim must be >0")
	case c.NumExperts <= 0:
		return fmt.Errorf("decoder(laguna): num_experts must be >0, got %d", c.NumExperts)
	case c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > c.NumExperts:
		return fmt.Errorf("decoder(laguna): num_experts_per_tok=%d out of range for num_experts=%d", c.NumExpertsPerTok, c.NumExperts)
	case c.MoeIntermediateSize <= 0:
		return fmt.Errorf("decoder(laguna): moe_intermediate_size must be >0")
	case c.SharedExpertIntermediateSize <= 0:
		return fmt.Errorf("decoder(laguna): shared_expert_intermediate_size must be >0")
	case c.HiddenAct != "" && c.HiddenAct != "silu":
		return fmt.Errorf("decoder(laguna): hidden_act=%q unsupported (silu/SwiGLU only)", c.HiddenAct)
	// The vendor raises NotImplementedError on this rather than diverge numerically;
	// match that instead of silently applying the routing weight on the output.
	case c.MoeApplyRouterWeightOnInput:
		return fmt.Errorf("decoder(laguna): moe_apply_router_weight_on_input=true is unsupported")
	}
	if n := len(c.NumAttentionHeadsPerLayer); n != 0 && n != c.NumLayers {
		return fmt.Errorf("decoder(laguna): num_attention_heads_per_layer has %d entries, want %d (num_hidden_layers)", n, c.NumLayers)
	}
	for i, h := range c.NumAttentionHeadsPerLayer {
		if h <= 0 || h%c.NumKVHeads != 0 {
			return fmt.Errorf("decoder(laguna): num_attention_heads_per_layer[%d]=%d must be >0 and a multiple of num_key_value_heads=%d (GQA groups)", i, h, c.NumKVHeads)
		}
	}
	if n := len(c.LayerTypes); n != 0 && n != c.NumLayers {
		return fmt.Errorf("decoder(laguna): layer_types has %d entries, want %d", n, c.NumLayers)
	}
	for i, t := range c.LayerTypes {
		if t != "full_attention" && t != "sliding_attention" {
			return fmt.Errorf("decoder(laguna): layer_types[%d]=%q unsupported (full_attention / sliding_attention)", i, t)
		}
		if t == "sliding_attention" && c.SlidingWindow <= 0 {
			return fmt.Errorf("decoder(laguna): layer_types has sliding_attention but sliding_window=%d", c.SlidingWindow)
		}
	}
	if _, _, err := c.LagunaGatePerHead(); err != nil {
		return err
	}
	_, err := c.LagunaFirstKDense()
	return err
}
