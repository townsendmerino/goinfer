package decoder

import "math"

// gpt2Act maps GPT-2's activation_function to the ActKind that implements it: "gelu_new" (and the empty default, GPT-2's own) is the TANH
// approximation, "gelu" the exact erf function. They differ by up to 4.73e-4, small enough to pass unnoticed and still wrong, so a
// checkpoint declaring the exact function must not get the approximation.
func gpt2Act(name string) ActKind {
	if name == "gelu" {
		return ActGelu
	}
	return ActGeluTanh
}

// gpt2Architecture expresses GPT-2: the GPT-2/NeoX class that breaks the Llama mold on several axes: LayerNorm (mean-centered, with bias)
// instead of RMSNorm, learned absolute position embeddings instead of RoPE, a non-gated GELU MLP (up→gelu→down), fused q/k/v with bias, an
// attention output bias, and tied embeddings. The Conv1D weight layout and fused projections need a dedicated loader (buildGPT2Weights), so
// this returns gpt2TensorSchema as a marker; its field names are unused.
func gpt2Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateGPT2(); err != nil {
		return nil, nil, err
	}
	hd := cfg.NEmbd / cfg.NHead
	inter := cfg.NInner
	if inter == 0 {
		inter = 4 * cfg.NEmbd // GPT-2 default feed-forward width
	}
	// Canonicalize the standard Config dims from GPT-2's n_* keys so Model.Config()
	// (used by the demos for the load banner) reports them uniformly. cfg is a
	// freshly-parsed, per-Load pointer, so this mutation is local.
	cfg.HiddenDim, cfg.NumLayers, cfg.NumHeads = cfg.NEmbd, cfg.NLayer, cfg.NHead
	cfg.NumKVHeads, cfg.HeadDim, cfg.IntermediateDim = cfg.NHead, hd, inter
	cfg.MaxPositions = cfg.NPositions
	return &Architecture{
		Name:            "gpt2",
		HiddenDim:       cfg.NEmbd,
		NumLayers:       cfg.NLayer,
		NumHeads:        cfg.NHead,
		NumKVHeads:      cfg.NHead, // no GQA
		HeadDim:         hd,
		IntermediateDim: inter,
		VocabSize:       cfg.VocabSize,
		MaxPositions:    cfg.NPositions,
		Norm:            NormLayer,
		NormEps:         cfg.LayerNormEpsilon,
		NormPlacement:   NormPre2, // ln_1 pre-attn, ln_2 pre-MLP
		Act:             gpt2Act(cfg.ActivationFunction),
		NonGatedMLP:     true,
		QKVBias:         true,
		OutBias:         true,
		QKNorm:          false,
		LearnedPosEmbed: true,
		AttnScale:       math.Pow(float64(hd), -0.5),
		EmbedScale:      0,
		TiedLMHead:      true, // GPT-2 ties wte as the LM head
	}, &gpt2TensorSchema, nil
}
