package decoder

import (
	"fmt"
	"math"
)

// qwen35Architecture expresses Qwen3.5/3.6-MoE (model_type qwen3_5_moe): a 3:1 hybrid where most layers are Gated DeltaNet (linear attention
// with a recurrent matrix state, its own forward path) and the rest are QK-norm softmax attention with partial RoPE, over a routed + shared
// MoE on every layer. The descriptor marks the per-layer kind (layerIsLinear) and carries the DeltaNet geometry; the softmax layers reuse the
// uniform attention fields. See docs/completed/qwen3_5_moe.md.
func qwen35Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateQwen35(); err != nil {
		return nil, nil, err
	}
	spec, partialRotary, err := parseRopeFlat(cfg.RopeParameters)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(qwen3_5_moe): %w", err)
	}
	rotaryDim := 0
	if partialRotary > 0 && partialRotary < 1 {
		rotaryDim = int(partialRotary * float64(cfg.HeadDim))
	}
	// Qwen3_5MoeTopKRouter ALWAYS renormalizes the top-k probabilities
	// (router_top_value /= sum), regardless of the (absent) norm_topk_prob field.
	normTopK := true
	if cfg.NormTopKProb != nil {
		normTopK = *cfg.NormTopKProb
	}
	arch := &Architecture{
		Name:            "qwen3_5_moe",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         cfg.HeadDim,
		IntermediateDim: cfg.IntermediateDim, // dense width (vestigial; experts use the MoE width)
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		RMSAddOne:       true, // Qwen3_5MoeRMSNorm: output × (1 + weight), weight zero-init
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   NormPre2,
		Act:             ActSiLU,
		MoE: &MoEConfig{
			NumExperts:            cfg.NumExperts,
			TopK:                  cfg.NumExpertsPerTok,
			NormTopKProb:          normTopK,
			IntermediateDim:       cfg.MoeIntermediateSize,
			SharedIntermediateDim: cfg.SharedExpertIntermediateSize,
		},
		QKNorm:           true, // softmax layers have per-head q_norm/k_norm
		AttnScale:        math.Pow(float64(cfg.HeadDim), -0.5),
		layerIsGlobal:    cfg.IsGlobalLayer, // softmax layers are full attention
		layerIsLinear:    cfg.IsLinearLayer, // Gated DeltaNet layers
		RoPELocalBase:    spec.base,
		RoPEGlobalBase:   spec.base, // single base; only the softmax layers use RoPE
		ropeScaling:      spec.scaling,
		ropeScalingLocal: spec.scaling,
		RotaryDim:        rotaryDim, // partial_rotary_factor · head_dim
		EmbedScale:       0,
		TiedLMHead:       false, // finalized from lm_head.weight presence at load
		qwen35: &qwen35Params{
			ConvKernel:    cfg.LinearConvKernelDim,
			KeyHeadDim:    cfg.LinearKeyHeadDim,
			ValueHeadDim:  cfg.LinearValueHeadDim,
			NumKeyHeads:   cfg.LinearNumKeyHeads,
			NumValueHeads: cfg.LinearNumValueHeads,
			AttnGate:      true, // qwen3_5_moe's own double-width [query‖gate] q_proj
		},
	}
	arch.MRopeSection, arch.MRopeInterleaved = parseMRopeFlat(cfg.RopeParameters)
	return arch, &qwen35TensorSchema, nil
}
