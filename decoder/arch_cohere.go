package decoder

import "math"

// cohereArchitecture expresses Cohere / Command-R (model_type "cohere":
// Command-R, Command-R+, Aya, Aya-Expanse). Two things break the Llama mold:
//
//   - Norm is bias-free LayerNorm (mean-subtract + variance), NOT RMSNorm —
//     NormLayer with a nil bias (the generic loader leaves PreAttnNormBias nil).
//   - The block is PARALLEL: one shared input_layernorm feeds both attention and
//     the MLP, whose outputs sum into a single residual add (NormParallel). There
//     is no post_attention_layernorm — the schema's PreMLPNorm is empty.
//
// Everything else rides shipped features: gated SiLU MLP, GQA (Command-R v01 is
// MHA, kv==heads), tied 256k embeddings, full-dim RoPE. RoPE is GPT-J interleaved
// (ropeInterleave). logit_scale MULTIPLIES the logits in HF; goinfer's LogitScale
// divides, so we store its reciprocal and reuse the Granite logit-scale kernel.
// use_qk_norm (Command-R+) and sliding_window (that's cohere2) are rejected in
// validateCohere, so reaching here implies a plain cohere1 checkpoint.
func cohereArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	// Older CohereLabs configs carry rope_theta top-level; transformers ≥5 nests it
	// under rope_parameters. Backfill the flat field so both load identically.
	if err := backfillFlatRope(cfg, "cohere"); err != nil {
		return nil, nil, err
	}
	if err := cfg.validateCohere(); err != nil {
		return nil, nil, err
	}
	hd := cfg.headDim()
	return &Architecture{
		Name:            "cohere",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         hd,
		IntermediateDim: cfg.IntermediateDim,
		VocabSize:       cfg.VocabSize,
		Norm:            NormLayer, // bias-free LayerNorm (mean-subtract), NOT RMSNorm
		NormEps:         cfg.LayerNormEps,
		NormPlacement:   NormParallel, // one input norm → attn + mlp → single residual add
		Act:             ActSiLU,
		QKNorm:          false, // cohere1 use_qk_norm is rejected in validateCohere
		AttnScale:       math.Pow(float64(hd), -0.5),
		SlidingWindow:   0, // cohere1: full attention (sliding_window ⇒ cohere2, rejected)
		layerIsGlobal:   nil,
		RoPELocalBase:   cfg.RoPEGlobalBase, // single base (rope_theta)
		RoPEGlobalBase:  cfg.RoPEGlobalBase,
		RotaryDim:       cfg.rotaryDim(), // 0 = full head_dim
		ropeInterleave:  true,            // GPT-J pairwise rotation (rope_gptj), not NeoX
		EmbedScale:      0,
		LogitScale:      1.0 / cfg.LogitScale, // reciprocal: HF multiplies, goinfer divides
		TiedLMHead:      true,                 // Cohere always ties (no lm_head.weight in checkpoint)
	}, &cohereTensorSchema, nil
}
