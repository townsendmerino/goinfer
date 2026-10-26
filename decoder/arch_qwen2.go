package decoder

import (
	"fmt"
	"math"
)

// qwen2Architecture expresses Qwen2/Qwen2.5 dense: identical to the llama
// descriptor (RMS no-offset, Pre2, SwiGLU, single-base RoPE, no QK-norm, derived
// head_dim, tied-or-untied head) plus QKVBias — Qwen2 carries an additive bias
// on the q/k/v projections (o_proj stays biasless). The tensor schema is
// qwen2TensorSchema.
func qwen2Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := backfillFlatRope(cfg, "qwen2"); err != nil {
		return nil, nil, err
	}
	if err := cfg.validateQwen2(); err != nil {
		return nil, nil, err
	}
	scaling, err := parseRopeScaling(cfg.RopeScaling) // linear + llama3 (Qwen2.5 leaves it null)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(qwen2): %w", err)
	}
	hd := cfg.headDim()
	return &Architecture{
		Name:            "qwen2",
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
		QKVBias:         true, // the one knob that differs from llama
		QKNorm:          false,
		AttnScale:       math.Pow(float64(hd), -0.5),
		SlidingWindow:   0,   // validateQwen2 rejects use_sliding_window=true
		layerIsGlobal:   nil, // all-global
		RoPELocalBase:   cfg.RoPEGlobalBase,
		RoPEGlobalBase:  cfg.RoPEGlobalBase,
		RotaryDim:       cfg.rotaryDim(),
		ropeScaling:     scaling,
		EmbedScale:      0,
		TiedLMHead:      false, // finalized from lm_head.weight presence at load
	}, &qwen2TensorSchema, nil
}
