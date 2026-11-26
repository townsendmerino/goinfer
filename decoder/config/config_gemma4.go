package config

import (
	"encoding/json"
	"fmt"
	"strconv"
)

// gemma4BidirectionalAttention is Config.UseBidirectionalAttention's type; see that field for why a plain string is unsafe.
// It accepts a real JSON string and resolves anything else (bool, null, absent) to "".
type gemma4BidirectionalAttention string

func (b *gemma4BidirectionalAttention) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		*b = gemma4BidirectionalAttention(s)
		return nil
	}
	*b = ""
	return nil
}

// ValidateGemma4MoE pins the Gemma 4 26B-A4B MoE config, checked only when enable_moe_block is set (the dense E2B/E4B/12B
// variants never reach here): num_experts > 0, 1 <= top_k_experts <= num_experts, moe_intermediate_size > 0. The
// router.scale / per_expert_scale tensors are checked at load.
func (c *Config) ValidateGemma4MoE() error {
	switch {
	case c.NumExperts <= 0:
		return fmt.Errorf("decoder(gemma4-moe): num_experts must be >0, got %d", c.NumExperts)
	case c.TopKExperts <= 0 || c.TopKExperts > c.NumExperts:
		return fmt.Errorf("decoder(gemma4-moe): top_k_experts %d out of range (1..%d)", c.TopKExperts, c.NumExperts)
	case c.MoeIntermediateSize <= 0:
		return fmt.Errorf("decoder(gemma4-moe): moe_intermediate_size must be >0, got %d", c.MoeIntermediateSize)
	}
	return nil
}

// Gemma4PartialRotary returns the fraction of the global (full-attention) head
// RoPE rotates. Gemma 4's "proportional" RoPE keeps this in the per-attention-type
// rope_parameters.full_attention.partial_rotary_factor (0.25), NOT the top-level
// partial_rotary_factor (which is Phi's spelling and absent here). Prefers the
// top-level value when present (GGUF injects it there); else reads the nested one.
// 0 ⇒ the global layer is NoPE.
func (c *Config) Gemma4PartialRotary() float64 {
	if c.PartialRotaryFactor > 0 {
		return c.PartialRotaryFactor
	}
	if len(c.RopeParameters) == 0 {
		return 0
	}
	var rp map[string]struct {
		PartialRotaryFactor float64 `json:"partial_rotary_factor"`
	}
	if err := json.Unmarshal(c.RopeParameters, &rp); err != nil {
		return 0
	}
	return rp["full_attention"].PartialRotaryFactor
}

// Gemma4GlobalGeometry returns the full-attention layers' head_dim and KV-head count. The flat global_head_dim /
// num_global_key_value_heads win when present (every released checkpoint, and every config written before transformers
// 5.16). A config saved by 5.16 or later carries per_layer_config instead, so a re-saved checkpoint would otherwise read 0 and
// load every layer at the local geometry. goinfer has exactly two attention geometries, so the overrides must name only
// full-attention layers and agree with each other; anything else is refused rather than half-applied.
func (c *Config) Gemma4GlobalGeometry() (headDim, kvHeads int, err error) {
	headDim, kvHeads = c.GlobalHeadDim, c.NumGlobalKVHeads
	if len(c.PerLayerConfig) == 0 || string(c.PerLayerConfig) == "null" {
		return headDim, kvHeads, nil
	}
	var pl map[string]map[string]json.RawMessage
	if err := json.Unmarshal(c.PerLayerConfig, &pl); err != nil {
		return 0, 0, fmt.Errorf("decoder(gemma4): per_layer_config: %w", err)
	}
	var hd, kv int
	for k, ov := range pl {
		i, aerr := strconv.Atoi(k)
		if aerr != nil || i < 0 || i >= c.NumLayers {
			return 0, 0, fmt.Errorf("decoder(gemma4): per_layer_config key %q is not a layer index", k)
		}
		if !c.IsGlobalLayer(i) {
			return 0, 0, fmt.Errorf("decoder(gemma4): per_layer_config overrides sliding layer %d; only full-attention overrides are supported", i)
		}
		for f, raw := range ov {
			var v int
			if err := json.Unmarshal(raw, &v); err != nil {
				return 0, 0, fmt.Errorf("decoder(gemma4): per_layer_config[%d].%s: %w", i, f, err)
			}
			var dst *int
			switch f {
			case "head_dim":
				dst = &hd
			case "num_key_value_heads":
				dst = &kv
			default:
				return 0, 0, fmt.Errorf("decoder(gemma4): per_layer_config[%d] overrides %q, which is not supported", i, f)
			}
			if *dst != 0 && *dst != v {
				return 0, 0, fmt.Errorf("decoder(gemma4): per_layer_config gives full-attention layers different %s (%d and %d)", f, *dst, v)
			}
			*dst = v
		}
	}
	if headDim == 0 {
		headDim = hd
	} else if hd != 0 && hd != headDim {
		return 0, 0, fmt.Errorf("decoder(gemma4): global_head_dim %d disagrees with per_layer_config head_dim %d", headDim, hd)
	}
	if kvHeads == 0 {
		kvHeads = kv
	} else if kv != 0 && kv != kvHeads {
		return 0, 0, fmt.Errorf("decoder(gemma4): num_global_key_value_heads %d disagrees with per_layer_config num_key_value_heads %d", kvHeads, kv)
	}
	return headDim, kvHeads, nil
}

// Gemma4RopeBases returns the (local, global) RoPE base wavelengths. Gemma 4's real
// unified checkpoints keep these NESTED in rope_parameters — sliding_attention.rope_theta
// (local, 10000) and full_attention.rope_theta (global, 1e6) — with NO top-level
// rope_theta / rope_local_base_freq, whereas the tiny/text goldens and the GGUF path
// carry the flat fields. Prefer the flat values, fall back to the nested ones. Base 0
// (both absent) makes every RoPE frequency 0/NaN and the model emits garbage, so the
// nested fallback is load-bearing, not cosmetic.
func (c *Config) Gemma4RopeBases() (local, global float64) {
	local, global = c.RoPELocalBase, c.RoPEGlobalBase
	if (local > 0 && global > 0) || len(c.RopeParameters) == 0 {
		return
	}
	var rp map[string]struct {
		RopeTheta float64 `json:"rope_theta"`
	}
	if err := json.Unmarshal(c.RopeParameters, &rp); err != nil {
		return
	}
	if local == 0 {
		local = rp["sliding_attention"].RopeTheta
	}
	if global == 0 {
		global = rp["full_attention"].RopeTheta
	}
	return
}
