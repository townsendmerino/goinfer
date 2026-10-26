package decoder

import (
	"fmt"
	"math"
)

// mixtralArchitecture expresses Mixtral: the llama
// descriptor (RMS no-offset, Pre2, SwiGLU experts, single-base RoPE, no QK-norm,
// no bias, untied head) with the dense FFN replaced by a sparse mixture of
// experts — a router picks top-k of NumExperts experts per token. Recent HF
// Mixtral uses full attention (the config's sliding_window is vestigial), so
// this does too. The tensor schema is mixtralTensorSchema.
func mixtralArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateMixtral(); err != nil {
		return nil, nil, err
	}
	scaling, err := parseRopeScaling(cfg.RopeScaling)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(mixtral): %w", err)
	}
	hd := cfg.headDim()
	normTopK := true // HF MixtralConfig default
	if cfg.NormTopKProb != nil {
		normTopK = *cfg.NormTopKProb
	}
	return &Architecture{
		Name:            "mixtral",
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
		MoE: &MoEConfig{
			NumExperts:      cfg.NumLocalExperts,
			TopK:            cfg.NumExpertsPerTok,
			NormTopKProb:    normTopK,
			IntermediateDim: cfg.IntermediateDim, // Mixtral experts use the dense width
		},
		QKNorm:         false,
		AttnScale:      math.Pow(float64(hd), -0.5),
		SlidingWindow:  0, // full attention (HF Mixtral ignores config sliding_window)
		layerIsGlobal:  nil,
		RoPELocalBase:  cfg.RoPEGlobalBase,
		RoPEGlobalBase: cfg.RoPEGlobalBase,
		RotaryDim:      cfg.rotaryDim(),
		ropeScaling:    scaling,
		EmbedScale:     0,
		TiedLMHead:     false, // finalized from lm_head.weight presence at load
	}, &mixtralTensorSchema, nil
}
