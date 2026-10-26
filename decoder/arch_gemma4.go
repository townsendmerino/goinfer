package decoder

import "math"

// gemma4Architecture expresses Gemma 4 dense (HF model_type
// "gemma4_unified_text"; GGUF arch "gemma4"). It is Gemma 3's stack — RMSNorm
// (1+w), Sandwich4 norm placement, GeGLU (gelu_tanh), QK-norm, query_pre_attn
// scaling, √hidden embed scale, tied head, no soft-capping, dual-base RoPE with a
// 5:1 local:global interleave — PLUS Gemma 4's per-layer attention deltas
// (gemma4Params): global/full layers use a wider head (global_head_dim), a single
// KV head (num_global_key_value_heads), shared K=V (attention_k_eq_v), and
// partial-rotary RoPE (partial_rotary_factor). The 12B dense is PLE-free; the
// E-models (E2B/E4B) add PLE (per-layer embeddings) — there is NO AltUp/Laurel
// (those are Gemma 3n). E2B also has cross-layer KV sharing + variable FFN +
// final-logit softcap 30.
//
// The forward pass (the Gemma 4 program walk) consumes the per-layer deltas; E2B/E4B and
// the 12B dense (K=V) variants are parity-gated against the HF bf16 oracle.
func gemma4Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	globalHeadDim, globalKVHeads, err := cfg.gemma4GlobalGeometry()
	if err != nil {
		return nil, nil, err
	}
	globalRotary := 0
	if prf := cfg.gemma4PartialRotary(); prf > 0 && globalHeadDim > 0 {
		globalRotary = int(prf * float64(globalHeadDim))
	}
	// The real unified 26B keeps the RoPE bases nested in rope_parameters (no top-level
	// rope_theta / rope_local_base_freq); resolve them so the frequencies aren't 0/NaN.
	localBase, globalBase := cfg.gemma4RopeBases()
	// Gemma 4 26B-A4B: enable_moe_block turns on the parallel dense+MoE FFN sub-block. The dense E2B/E4B/12B variants leave it false, so MoE
	// stays nil and their forward is unchanged. The router semantics (softmax-all → top-k → UNCONDITIONAL renorm, a weightless-norm/
	// learned-scale pre-projection, a per-expert scale) are pinned in docs/completed/task-gemma4-moe.md Phase 1a. The gemma4 forward reads
	// these; the generic moeMLP/swiGLUExpert are NOT reused (experts are gelu-tanh, not SiLU).
	var moe *MoEConfig
	if cfg.EnableMoeBlock {
		if err := cfg.validateGemma4MoE(); err != nil {
			return nil, nil, err
		}
		moe = &MoEConfig{
			NumExperts:      cfg.NumExperts,
			TopK:            cfg.TopKExperts,
			NormTopKProb:    true,
			IntermediateDim: cfg.MoeIntermediateSize,
			RouterPreNorm:   true,
			PerExpertScale:  true,
		}
	}
	return &Architecture{
		Name:            "gemma4",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         cfg.HeadDim,
		IntermediateDim: cfg.IntermediateDim,
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		// HF Gemma4RMSNorm is plain `x*weight` (weight init 1), not Gemma 3's `x*(1+weight)` (init 0), so the spec is no offset. Unverified
		// against the real GGUF: if llama.cpp's gemma4 convert pre-subtracts 1 (as some Gemma converts do), flip this to true (modeling_gemma4.py).
		RMSAddOne:     false,
		NormEps:       cfg.RMSNormEps,
		NormPlacement: NormSandwich4,
		Act:           ActGeluTanh,
		QKNorm:        true,
		// Gemma 4 text attention uses scale 1.0 (modeling_gemma4.py): the learned q/k-norm weights absorb the scaling, unlike Gemma 3's explicit
		// query_pre_attn_scalar^-0.5. A new scale-less v_norm normalizes V.
		AttnScale:      1.0,
		SlidingWindow:  cfg.SlidingWindow,
		layerIsGlobal:  cfg.IsGlobalLayer,
		RoPELocalBase:  localBase,
		RoPEGlobalBase: globalBase,
		EmbedScale:     math.Sqrt(float64(cfg.HiddenDim)),
		TiedLMHead:     true,
		// Gemma 4 re-added Gemma 2's final-logit softcap (30 in the GGUF); Gemma 3
		// had dropped it. Attn softcap stays 0.
		FinalLogitSoftcap: cfg.FinalLogitSoftcap,
		MoE:               moe, // nil for the dense E2B/E4B/12B variants (byte-unchanged)
		gemma4: &gemma4Params{
			GlobalHeadDim:           globalHeadDim,
			NumGlobalKVHeads:        globalKVHeads,
			GlobalRotaryDim:         globalRotary,
			KVShared:                cfg.AttentionKEqV,
			SharedKVLayers:          cfg.SharedKVLayers,
			FFNPerLayer:             cfg.FFNPerLayer,
			HiddenSizePerLayerInput: cfg.HiddenSizePerLayerInput,
			VocabSizePerLayerInput:  cfg.VocabSizePerLayerInput,
			PadTokenID:              cfg.PadTokenID,
		},
	}, &gemma3TensorSchema, nil // a dedicated gemma4 tensor schema lands with the loader work
}
