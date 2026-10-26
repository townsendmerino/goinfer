package decoder

import (
	"fmt"
	"math"
)

// qwen3MoeArchitecture expresses Qwen3-MoE (model_type qwen3_moe): qwen3's dense attention (per-head q_norm/k_norm, GQA, no q/k/v bias,
// 1/√head_dim scale, single-base RoPE) with the FFN replaced on every layer by a sparse MoE: qwen2_moe's router shape (top-k of num_experts
// at moe_intermediate_size, norm_topk_prob) but with NO always-on shared expert. The GGUF loader's generic loadLayer path (gated on
// arch.QKNorm / arch.MoE / arch.MoE.SharedIntermediateDim>0) handles the family with no loader code of its own. The tensor schema is
// qwen3MoeTensorSchema.
func qwen3MoeArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := backfillFlatRope(cfg, "qwen3_moe"); err != nil {
		return nil, nil, err
	}
	if err := cfg.validateQwen3Moe(); err != nil {
		return nil, nil, err
	}
	scaling, err := parseRopeScaling(cfg.RopeScaling)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(qwen3_moe): %w", err)
	}
	normTopK := true // HF Qwen3MoeConfig default (norm_topk_prob true)
	if cfg.NormTopKProb != nil {
		normTopK = *cfg.NormTopKProb
	}
	return &Architecture{
		Name:            "qwen3_moe",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         cfg.HeadDim,
		IntermediateDim: cfg.IntermediateDim,
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		RMSAddOne:       false,
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   NormPre2,
		Act:             ActSiLU,
		QKNorm:          true,
		MoE: &MoEConfig{
			NumExperts:      cfg.NumExperts,
			TopK:            cfg.NumExpertsPerTok,
			NormTopKProb:    normTopK,
			IntermediateDim: cfg.MoeIntermediateSize,
			// No SharedIntermediateDim: qwen3_moe has no shared expert (unlike qwen2_moe).
		},
		AttnScale:      math.Pow(float64(cfg.HeadDim), -0.5),
		SlidingWindow:  0, // full attention
		layerIsGlobal:  nil,
		RoPELocalBase:  cfg.RoPEGlobalBase,
		RoPEGlobalBase: cfg.RoPEGlobalBase,
		RotaryDim:      cfg.rotaryDim(),
		ropeScaling:    scaling,
		EmbedScale:     0,
		TiedLMHead:     false, // finalized from lm_head.weight presence at load
	}, &qwen3MoeTensorSchema, nil
}
