package decoder

import (
	"fmt"
	"math"
)

// qwen2MoeArchitecture expresses Qwen-MoE / Qwen2-MoE (Qwen1.5-MoE-A2.7B,
// Qwen2-57B-A14B): qwen2's attention (q/k/v bias, no QK-norm, derived head_dim,
// single-base RoPE) with the FFN replaced on every layer by a sparse MoE PLUS an
// always-on shared expert. The router picks top-k of num_experts experts at
// moe_intermediate_size; the shared expert is a gated MLP at
// shared_expert_intermediate_size scaled by sigmoid(shared_gate·h). The tensor
// schema is qwen2MoeTensorSchema.
func qwen2MoeArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateQwen2Moe(); err != nil {
		return nil, nil, err
	}
	scaling, err := parseRopeScaling(cfg.RopeScaling)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(qwen2_moe): %w", err)
	}
	hd := cfg.headDim()
	normTopK := false // HF Qwen2MoeConfig default (norm_topk_prob false)
	if cfg.NormTopKProb != nil {
		normTopK = *cfg.NormTopKProb
	}
	return &Architecture{
		Name:            "qwen2_moe",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         hd,
		IntermediateDim: cfg.IntermediateDim,
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		RMSAddOne:       false,
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   NormPre2,
		Act:             ActSiLU,
		QKVBias:         true, // qwen2-style q/k/v bias
		QKNorm:          false,
		MoE: &MoEConfig{
			NumExperts:            cfg.NumExperts,
			TopK:                  cfg.NumExpertsPerTok,
			NormTopKProb:          normTopK,
			IntermediateDim:       cfg.MoeIntermediateSize,
			SharedIntermediateDim: cfg.SharedExpertIntermediateSize,
		},
		AttnScale:      math.Pow(float64(hd), -0.5),
		SlidingWindow:  0, // full attention
		layerIsGlobal:  nil,
		RoPELocalBase:  cfg.RoPEGlobalBase,
		RoPEGlobalBase: cfg.RoPEGlobalBase,
		RotaryDim:      cfg.rotaryDim(),
		ropeScaling:    scaling,
		EmbedScale:     0,
		TiedLMHead:     false, // finalized from lm_head.weight presence at load
	}, &qwen2MoeTensorSchema, nil
}
