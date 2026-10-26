package decoder

import "math"

// cohere2Architecture expresses Cohere2 / Command-R7B (model_type "cohere2":
// Command-R7B, Command-A, R7B-arabic). It is cohere1's stack (bias-free LayerNorm,
// parallel block, gated SiLU, tied embeddings, GPT-J interleaved RoPE, reciprocal
// logit_scale) with two additions and one subtraction:
//
//   - Interleaved SLIDING-WINDOW / full attention: every sliding_window_pattern-th
//     layer is global (full attention), the rest are windowed at sliding_window.
//   - The GLOBAL layers are NoPE (no positional encoding); only the sliding layers
//     carry RoPE. layerNoPE == the global predicate, so isNoPELayer skips RoPE on
//     exactly the full-attention layers (the per-layer NoPE primitive Llama 4 wants).
//   - NO QK-norm at all (cohere2 dropped cohere1's config-gated q/k norm).
//
// Reuses cohereTensorSchema (identical tensor names — one shared input_layernorm
// per layer, no biases, tied head).
func cohere2Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := backfillFlatRope(cfg, "cohere2"); err != nil {
		return nil, nil, err
	}
	if err := cfg.validateCohere2(); err != nil {
		return nil, nil, err
	}
	hd := cfg.headDim()
	// Snapshot the classifier inputs so the closure doesn't retain cfg. Global =
	// full-attention layer = NoPE; local = sliding + RoPE (Config.IsGlobalLayer is
	// the authoritative layer_types-then-pattern rule).
	pattern := cfg.SlidingWindowPattern
	layerTypes := append([]string(nil), cfg.LayerTypes...)
	isGlobal := func(i int) bool {
		if i >= 0 && i < len(layerTypes) {
			return layerTypes[i] == "full_attention"
		}
		if pattern <= 0 {
			return true
		}
		return (i+1)%pattern == 0
	}
	return &Architecture{
		Name:            "cohere2",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         hd,
		IntermediateDim: cfg.IntermediateDim,
		VocabSize:       cfg.VocabSize,
		Norm:            NormLayer,
		NormEps:         cfg.LayerNormEps,
		NormPlacement:   NormParallel,
		Act:             ActSiLU,
		QKNorm:          false, // cohere2 has no QK-norm
		AttnScale:       math.Pow(float64(hd), -0.5),
		SlidingWindow:   cfg.SlidingWindow, // sliding layers window; global layers full
		layerIsGlobal:   isGlobal,
		layerNoPE:       isGlobal, // global (full) layers are NoPE; sliding layers keep RoPE
		RoPELocalBase:   cfg.RoPEGlobalBase,
		RoPEGlobalBase:  cfg.RoPEGlobalBase,
		RotaryDim:       cfg.rotaryDim(),
		ropeInterleave:  true, // GPT-J pairwise on the sliding (RoPE) layers
		EmbedScale:      0,
		LogitScale:      1.0 / cfg.LogitScale,
		TiedLMHead:      true,
	}, &cohereTensorSchema, nil
}
