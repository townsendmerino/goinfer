package config

import (
	"fmt"
)

// ValidateBailingHybrid pins Bailing Hybrid's (Ling 3.0) assumptions: a valid MLA geometry (the checks of ValidateDeepseek), a
// valid KDA geometry (conv kernel, per-head dim), and a routed+shared MoE using this family's own field spellings
// (num_experts/num_shared_experts, not DeepSeek's n_routed_experts/n_shared_experts).
func (c *Config) ValidateBailingHybrid() error {
	switch {
	case c.KVLoRARank <= 0:
		return fmt.Errorf("decoder(bailing_hybrid): kv_lora_rank must be >0, got %d", c.KVLoRARank)
	case c.QKNopeHeadDim <= 0 || c.QKRopeHeadDim <= 0 || c.VHeadDim <= 0:
		return fmt.Errorf("decoder(bailing_hybrid): bad MLA head dims (qk_nope=%d qk_rope=%d v=%d)", c.QKNopeHeadDim, c.QKRopeHeadDim, c.VHeadDim)
	case c.QLoRARank < 0:
		return fmt.Errorf("decoder(bailing_hybrid): q_lora_rank must be ≥0, got %d", c.QLoRARank)
	case c.LayerGroupSize <= 0:
		return fmt.Errorf("decoder(bailing_hybrid): layer_group_size must be >0, got %d", c.LayerGroupSize)
	case c.HeadDim <= 0 || c.ShortConvKernelSize <= 0:
		return fmt.Errorf("decoder(bailing_hybrid): bad KDA dims (head_dim=%d short_conv_kernel_size=%d)", c.HeadDim, c.ShortConvKernelSize)
	case c.NumExperts <= 0:
		return fmt.Errorf("decoder(bailing_hybrid): num_experts must be >0, got %d", c.NumExperts)
	case c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > c.NumExperts:
		return fmt.Errorf("decoder(bailing_hybrid): num_experts_per_tok %d out of range (1..%d)", c.NumExpertsPerTok, c.NumExperts)
	case c.MoeIntermediateSize <= 0:
		return fmt.Errorf("decoder(bailing_hybrid): moe_intermediate_size must be >0, got %d", c.MoeIntermediateSize)
	case c.NumSharedExperts <= 0:
		return fmt.Errorf("decoder(bailing_hybrid): num_shared_experts must be >0, got %d", c.NumSharedExperts)
	case c.FirstKDenseReplace < 0 || c.FirstKDenseReplace >= c.NumLayers:
		return fmt.Errorf("decoder(bailing_hybrid): first_k_dense_replace %d out of range (0..%d)", c.FirstKDenseReplace, c.NumLayers-1)
	case c.NGroup > 1 && (c.TopkGroup <= 0 || c.TopkGroup > c.NGroup):
		return fmt.Errorf("decoder(bailing_hybrid): topk_group %d out of range (1..%d)", c.TopkGroup, c.NGroup)
	case c.NGroup > 1 && c.NumExperts%c.NGroup != 0:
		return fmt.Errorf("decoder(bailing_hybrid): num_experts %d not divisible by n_group %d", c.NumExperts, c.NGroup)
	}
	return nil
}

// NormalizeBailingLayerTypes synthesizes LayerTypes from LayerGroupSize for Bailing Hybrid (Ling 3.0), whose released
// config.json has no layer_types field. The formula follows BailingMoeV3DecoderLayer.__init__:
//
//	"attention" if (i+1)%group==0 or i >= (numLayers//group)*group else "linear_attention"
//
// 0-indexed i. The second clause is a tail-cleanup for a NumLayers that is not a multiple of LayerGroupSize, replicated
// rather than dropped because the vendor's formula is the authority. "full_attention" (this tree's spelling) stands for HF's
// "attention", so IsGlobalLayer/IsLinearLayer read it with no new predicate. A no-op if LayerTypes is already populated.
func (c *Config) NormalizeBailingLayerTypes() error {
	if len(c.LayerTypes) > 0 || c.LayerGroupSize == 0 {
		return nil
	}
	if c.LayerGroupSize < 0 {
		return fmt.Errorf("decoder(bailing_hybrid): layer_group_size must be >0, got %d", c.LayerGroupSize)
	}
	if c.NumLayers <= 0 {
		return fmt.Errorf("decoder(bailing_hybrid): num_hidden_layers must be >0 before layer_types can be derived")
	}
	tailStart := (c.NumLayers / c.LayerGroupSize) * c.LayerGroupSize
	types := make([]string, c.NumLayers)
	for i := range types {
		if (i+1)%c.LayerGroupSize == 0 || i >= tailStart {
			types[i] = "full_attention"
		} else {
			types[i] = "linear_attention"
		}
	}
	c.LayerTypes = types
	return nil
}
