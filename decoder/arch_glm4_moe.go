package decoder

import (
	"fmt"
	"math"
)

// glm4moeArchitecture expresses GLM-4.5/4.6 (model_type glm4_moe): qwen3-like softmax attention (per-head QK-norm, partial RoPE, NO q/k/v
// bias) over a DeepSeek-style MoE. The MoE differs from Qwen2-MoE on three axes, all carried by MoEConfig: experts are scored by per-expert
// sigmoid (not softmax), an e_score_correction_bias steers the top-k SELECTION while the weights stay the un-biased sigmoid scores, the
// routed weights are scaled by routed_scaling_factor, and the always-on shared expert is added UNGATED. first_k_dense_replace layers at the
// top are plain dense MLPs (no router), handled by the loader populating Experts only on i ≥ FirstKDense. The num_nextn_predict_layers MTP
// head is dropped (only num_hidden_layers load). The tensor schema is glm4moeTensorSchema. See
// docs/completed/task-model-families-glm-granite.md.
func glm4moeArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateGlm4Moe(); err != nil {
		return nil, nil, err
	}
	// GLM nests rope_theta + partial_rotary_factor inside rope_parameters
	// (rope_type "default"); there is no top-level rope_theta.
	spec, partialRotary, err := parseRopeFlat(cfg.RopeParameters)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(glm4_moe): %w", err)
	}
	hd := cfg.headDim()
	rotaryDim := 0
	if partialRotary > 0 && partialRotary < 1 {
		rotaryDim = int(partialRotary * float64(hd))
	}
	normTopK := true // HF Glm4MoeConfig default
	if cfg.NormTopKProb != nil {
		normTopK = *cfg.NormTopKProb
	}
	return &Architecture{
		Name:            "glm4_moe",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         hd,
		IntermediateDim: cfg.IntermediateDim, // dense width — used by the first_k_dense_replace prefix layers
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		RMSAddOne:       false,
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   NormPre2,
		Act:             ActSiLU,
		QKVBias:         cfg.AttentionBias, // GLM-4.5/4.6 carry q/k/v bias (o_proj biasless); the tiny synthetic turns it off
		QKNorm:          cfg.UseQKNorm,     // use_qk_norm: true on the tiny synthetic, FALSE on the real GLM-4.5-Air/4.6
		FirstKDense:     cfg.FirstKDenseReplace,
		MoE: &MoEConfig{
			NumExperts:            cfg.NRoutedExperts,
			TopK:                  cfg.NumExpertsPerTok,
			NormTopKProb:          normTopK,
			IntermediateDim:       cfg.MoeIntermediateSize,
			SharedIntermediateDim: cfg.NSharedExperts * cfg.MoeIntermediateSize, // shared expert(s) at moe_intermediate_size each
			RouterSigmoid:         true,
			RoutedScale:           cfg.RoutedScalingFactor,
			SharedUngated:         true,
		},
		AttnScale:      math.Pow(float64(hd), -0.5),
		SlidingWindow:  0, // full attention
		layerIsGlobal:  nil,
		RoPELocalBase:  spec.base,
		RoPEGlobalBase: spec.base, // single base (rope_parameters.rope_theta)
		ropeScaling:    spec.scaling,
		RotaryDim:      rotaryDim, // partial_rotary_factor · head_dim
		EmbedScale:     0,
		TiedLMHead:     false, // finalized from lm_head.weight presence at load
	}, &glm4moeTensorSchema, nil
}
