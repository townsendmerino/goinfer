package decoder

import (
	"fmt"
	"math"
)

// mellumArchitecture expresses JetBrains Mellum2 (a 12B-A2.5B code model): the
// llama skeleton (RMS no-offset, Pre2, SwiGLU, derived head_dim, no QK-norm, no
// bias, untied head) combining two axes we already had separately — a sparse MoE
// FFN on EVERY layer (64 experts, top-8, the narrower moe_intermediate_size) and
// a 3:1 sliding/full attention interleave (layer_types) — plus the one new piece:
// per-attention-type RoPE from rope_parameters, with YaRN (and its attention_factor
// mscale) on the full layers and plain RoPE on the sliding layers, both at theta
// 500000.
func mellumArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateMellum(); err != nil {
		return nil, nil, err
	}
	full, sliding, err := parseRopeParameters(cfg.RopeParameters)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(mellum): %w", err)
	}
	if full == nil || sliding == nil {
		return nil, nil, fmt.Errorf("decoder(mellum): rope_parameters needs full_attention + sliding_attention")
	}
	hd := cfg.headDim()
	normTopK := true
	if cfg.NormTopKProb != nil {
		normTopK = *cfg.NormTopKProb
	}
	return &Architecture{
		Name:            "mellum",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         hd,
		IntermediateDim: cfg.IntermediateDim, // dense width (vestigial; experts use the MoE width)
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		RMSAddOne:       false,
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   NormPre2,
		Act:             ActSiLU,
		MoE: &MoEConfig{
			NumExperts:      cfg.NumExperts,
			TopK:            cfg.NumExpertsPerTok,
			NormTopKProb:    normTopK,
			IntermediateDim: cfg.MoeIntermediateSize,
		},
		QKNorm:           true, // Mellum has q_norm/k_norm per head (like Qwen3)
		AttnScale:        math.Pow(float64(hd), -0.5),
		SlidingWindow:    cfg.SlidingWindow,
		layerIsGlobal:    cfg.IsGlobalLayer, // from layer_types (3:1 sliding/full)
		RoPEGlobalBase:   full.base,         // full_attention layers
		RoPELocalBase:    sliding.base,      // sliding_attention layers (same theta, plain)
		ropeScaling:      full.scaling,      // YaRN on full layers
		ropeScalingLocal: sliding.scaling,   // nil (plain) on sliding layers
		RotaryDim:        cfg.rotaryDim(),
		EmbedScale:       0,
		TiedLMHead:       false, // finalized from lm_head.weight presence at load
	}, &mellumTensorSchema, nil
}
