package decoder

import "math"

// qwen3Architecture expresses Qwen3 dense: RMSNorm
// without the (1+w) offset, the Pre2 norm placement (no post-sublayer norms),
// SwiGLU, QK-norm (Qwen3 keeps it), 1/√head_dim attention scale, single-base
// RoPE, no embedding scale, and a separate lm_head (untied). No QKV bias
// (Qwen3 dropped Qwen2's). The tensor schema is qwen3TensorSchema.
func qwen3Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := backfillFlatRope(cfg, "qwen3"); err != nil {
		return nil, nil, err
	}
	if err := cfg.validateQwen3(); err != nil {
		return nil, nil, err
	}
	return &Architecture{
		Name:            "qwen3",
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
		AttnScale:       math.Pow(float64(cfg.HeadDim), -0.5),
		SlidingWindow:   0,                  // Qwen3 dense: full attention
		layerIsGlobal:   nil,                // all-global
		RoPELocalBase:   cfg.RoPEGlobalBase, // single base (rope_theta)
		RoPEGlobalBase:  cfg.RoPEGlobalBase,
		EmbedScale:      0,     // none
		TiedLMHead:      false, // finalized from lm_head.weight presence at load
	}, &qwen3TensorSchema, nil
}
