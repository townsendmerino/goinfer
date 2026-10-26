package decoder

import (
	"fmt"
	"math"
)

// mistralArchitecture expresses Mistral dense: the llama descriptor (RMS
// no-offset, Pre2, SwiGLU, single-base RoPE, no QK-norm, no bias, derived
// head_dim) with sliding-window attention on EVERY layer (Gemma's window
// machinery, but all-local rather than 5:1). A checkpoint with sliding_window
// null/0 (Mistral-v0.2+) falls back to full attention. The tensor schema is
// llamaTensorSchema (Mistral and Llama share tensor names).
func mistralArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := backfillFlatRope(cfg, "mistral"); err != nil {
		return nil, nil, err
	}
	if err := cfg.validateLlama(); err != nil {
		return nil, nil, err
	}
	scaling, err := parseRopeScaling(cfg.RopeScaling)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(mistral): %w", err)
	}
	hd := cfg.headDim()
	// All layers local when a window is set; else all global (full attention).
	var layerIsGlobal func(int) bool
	if cfg.SlidingWindow > 0 {
		layerIsGlobal = func(int) bool { return false }
	}
	return &Architecture{
		Name:            "mistral",
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
		QKNorm:          false,
		AttnScale:       math.Pow(float64(hd), -0.5),
		SlidingWindow:   cfg.SlidingWindow, // 0 ⇒ full attention
		layerIsGlobal:   layerIsGlobal,     // all-local when windowed
		RoPELocalBase:   cfg.RoPEGlobalBase,
		RoPEGlobalBase:  cfg.RoPEGlobalBase,
		RotaryDim:       cfg.rotaryDim(),
		ropeScaling:     scaling,
		EmbedScale:      0,
		TiedLMHead:      false, // finalized from lm_head.weight presence at load
	}, &llamaTensorSchema, nil
}
