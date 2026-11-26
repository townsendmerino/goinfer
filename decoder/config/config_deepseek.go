package config

import (
	"fmt"
)

// ValidateDeepseek pins the DeepSeek-V2/V3 (MLA) assumptions: a valid MLA geometry
// (kv_lora_rank + the split per-head qk_nope/qk_rope/v dims) and a DeepSeekMoE
// (n_routed_experts / num_experts_per_tok / moe_intermediate_size + n_shared_experts ≥ 1,
// a first_k_dense_replace dense prefix). q_lora_rank 0/absent is the V2-Lite direct-q
// path. Group-limited routing (n_group/topk_group) is optional (≤1 ⇒ plain top-k).
func (c *Config) ValidateDeepseek() error {
	switch {
	case c.KVLoRARank <= 0:
		return fmt.Errorf("decoder(%s): kv_lora_rank must be >0, got %d", c.ModelType, c.KVLoRARank)
	case c.QKNopeHeadDim <= 0 || c.QKRopeHeadDim <= 0 || c.VHeadDim <= 0:
		return fmt.Errorf("decoder(%s): bad MLA head dims (qk_nope=%d qk_rope=%d v=%d)", c.ModelType, c.QKNopeHeadDim, c.QKRopeHeadDim, c.VHeadDim)
	case c.QLoRARank < 0:
		return fmt.Errorf("decoder(%s): q_lora_rank must be ≥0, got %d", c.ModelType, c.QLoRARank)
	case c.NRoutedExperts <= 0:
		return fmt.Errorf("decoder(%s): n_routed_experts must be >0, got %d", c.ModelType, c.NRoutedExperts)
	case c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > c.NRoutedExperts:
		return fmt.Errorf("decoder(%s): num_experts_per_tok %d out of range (1..%d)", c.ModelType, c.NumExpertsPerTok, c.NRoutedExperts)
	case c.MoeIntermediateSize <= 0:
		return fmt.Errorf("decoder(%s): moe_intermediate_size must be >0, got %d", c.ModelType, c.MoeIntermediateSize)
	case c.NSharedExperts <= 0:
		return fmt.Errorf("decoder(%s): n_shared_experts must be >0, got %d", c.ModelType, c.NSharedExperts)
	case c.FirstKDenseReplace < 0 || c.FirstKDenseReplace >= c.NumLayers:
		return fmt.Errorf("decoder(%s): first_k_dense_replace %d out of range (0..%d)", c.ModelType, c.FirstKDenseReplace, c.NumLayers-1)
	case c.NGroup > 1 && (c.TopkGroup <= 0 || c.TopkGroup > c.NGroup):
		return fmt.Errorf("decoder(%s): topk_group %d out of range (1..%d)", c.ModelType, c.TopkGroup, c.NGroup)
	case c.NGroup > 1 && c.NRoutedExperts%c.NGroup != 0:
		return fmt.Errorf("decoder(%s): n_routed_experts %d not divisible by n_group %d", c.ModelType, c.NRoutedExperts, c.NGroup)
	}
	return nil
}
