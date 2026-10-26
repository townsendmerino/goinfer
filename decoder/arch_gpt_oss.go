package decoder

import (
	"fmt"
	"math"
)

// gptOssArchitecture expresses the gpt-oss sparse-MoE family (model_type gpt_oss, GGUF arch gpt-oss). It marks the family with a gptoss
// escape hatch: its forward (forward_gptoss.go) adds a learned per-head attention SINK to each softmax and uses a clamped interleaved-SwiGLU
// expert with per-expert biases, while the layer skeleton stays the shared pre-norm path. Attention alternates sliding (even layers, window
// sliding_window) and full (odd); q/k/v/o all carry biases; no QK-norm; YaRN RoPE (one table for both attention types). The router carries a
// per-expert logit bias (LayerWeights.RouterBias). A resident backend admits the family only by declaring FeatAttnSink and FeatOutBias
// (features.go).
func gptOssArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateGptOss(); err != nil {
		return nil, nil, err
	}
	scaling, err := parseRopeScaling(cfg.RopeScaling)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(gpt-oss): %w", err)
	}
	hd := cfg.headDim()
	nExperts := cfg.NumLocalExperts
	if nExperts == 0 {
		nExperts = cfg.NumExperts
	}
	expInter := cfg.MoeIntermediateSize
	if expInter == 0 {
		expInter = cfg.IntermediateDim
	}
	return &Architecture{
		Name:             "gpt-oss",
		HiddenDim:        cfg.HiddenDim,
		NumLayers:        cfg.NumLayers,
		NumHeads:         cfg.NumHeads,
		NumKVHeads:       cfg.NumKVHeads,
		HeadDim:          hd,
		IntermediateDim:  expInter, // gpt-oss has no dense layers; experts use this width
		VocabSize:        cfg.VocabSize,
		Norm:             NormRMS,
		RMSAddOne:        false,
		NormEps:          cfg.RMSNormEps,
		NormPlacement:    NormPre2,
		Act:              ActSiLU, // the gpt-oss forward uses its own clamped activation; this is only for the matrix/logs
		QKVBias:          true,    // q/k/v projection biases
		OutBias:          true,    // attn_output bias
		QKNorm:           false,
		AttnScale:        math.Pow(float64(hd), -0.5),
		SlidingWindow:    cfg.SlidingWindow,
		layerIsGlobal:    func(i int) bool { return i%2 == 1 }, // even = sliding, odd = full
		RoPEGlobalBase:   cfg.RoPEGlobalBase,
		RoPELocalBase:    cfg.RoPEGlobalBase, // one RoPE for both attention types
		ropeScaling:      scaling,
		ropeScalingLocal: scaling, // same YaRN table on sliding + full layers
		RotaryDim:        cfg.rotaryDim(),
		MoE: &MoEConfig{
			NumExperts:      nExperts,
			TopK:            cfg.NumExpertsPerTok,
			NormTopKProb:    true, // softmax over the top-k logits == softmax-all + renormalize
			IntermediateDim: expInter,
		},
		gptoss:     &gptOssParams{SwigluAlpha: 1.702, SwigluLimit: 7.0}, // gpt-oss constants (not in metadata)
		EmbedScale: 0,
		TiedLMHead: false, // finalized from output.weight presence at load
	}, nil, nil
}
