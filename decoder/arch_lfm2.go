package decoder

import "math"

// lfm2FFNDim is HF's Lfm2MLP width: intermediate_size, or with block_auto_adjust_ff_dim int(2*intermediate_size/3) and,
// only when block_ffn_dim_multiplier is set, int(multiplier*width) rounded up to block_multiple_of (HF's rounding sits
// inside the multiplier branch).
func lfm2FFNDim(cfg *Config) int {
	d := cfg.IntermediateDim
	if !cfg.BlockAutoAdjustFFDim {
		return d
	}
	d = int(2 * float64(d) / 3)
	if cfg.BlockFFNDimMultiplier != nil {
		d = int(*cfg.BlockFFNDimMultiplier * float64(d))
		if m := cfg.BlockMultipleOf; m > 0 {
			d = m * ((d + m - 1) / m)
		}
	}
	return d
}

// lfm2Architecture expresses LFM2 / LFM2.5 (model_type lfm2): a gated-short-convolution + softmax-attention hybrid. Every layer has a SwiGLU
// FFN; layer_types decides whether its mixer is a conv block or GQA attention with per-head RMSNorm on Q and K. EXPERIMENTAL tier: validated
// against the HF reference on a real checkpoint, not against a full-model T3.
//
// Checked against the released LFM2.5-2.6B rather than inherited from the scoping brief: QK-norm is RMSNorm, not LayerNorm (the reference
// uses Lfm2RMSNorm(head_dim) per head and the checkpoint carries q_layernorm.weight with no bias tensor), which is the difference between
// reusing the existing QK-norm path and writing a bias-carrying LayerNorm variant; and intermediate_size is stated, not computed from
// block_multiple_of, so the block_ffn_dim_multiplier / block_multiple_of machinery is inert here and is not read.
// LFM2-VL's text_config sets block_auto_adjust_ff_dim, in which case the width is computed (lfm2FFNDim: 12288 -> 8192).
func lfm2Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateLFM2(); err != nil {
		return nil, nil, err
	}
	cfg.IntermediateDim = lfm2FFNDim(cfg)
	hd := cfg.HeadDim
	if hd == 0 {
		hd = cfg.HiddenDim / cfg.NumHeads
	}
	// rope_parameters is nested on every released LFM2.5 checkpoint (rope_type "default", no
	// scaling), but accept the flat form too — the same shape granite/deepseek/phi3 use, and
	// the tiny fixture is built by a different transformers version than the release.
	base, scaling, err := ropeBaseFlatOrNested(cfg, "lfm2")
	if err != nil {
		return nil, nil, err
	}
	types := cfg.LayerTypes
	return &Architecture{
		Name:            "lfm2",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         hd,
		IntermediateDim: cfg.IntermediateDim,
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		NormEps:         cfg.NormEps,                 // "norm_eps", NOT rms_norm_eps -- see Config.NormEps
		AttnScale:       math.Pow(float64(hd), -0.5), // Lfm2Attention: self.scaling = head_dim**-0.5
		NormPlacement:   NormPre2,                    // operator_norm before the mixer, ffn_norm before the FFN
		Act:             ActSiLU,                     // SwiGLU (block_use_swiglu true)
		QKVBias:         false,
		QKNorm:          true, // per-head RMSNorm over head_dim — see the note above
		RoPELocalBase:   base,
		RoPEGlobalBase:  base,
		ropeScaling:     scaling,
		RotaryDim:       0, // full rotary
		TiedLMHead:      true,
		// conv_dim is optional and absent from Lfm2Config-written checkpoints; hidden_size is
		// what upstream's Lfm2ShortConv actually uses. validateLFM2 has already refused any
		// non-zero conv_dim that disagrees with hidden_size, so this is a default, not a guess.
		lfm2:        &lfm2Params{ConvDim: cfg.HiddenDim, ConvLCache: cfg.ConvLCache},
		layerIsConv: func(i int) bool { return i < len(types) && types[i] == "conv" },
	}, &lfm2TensorSchema, nil
}
