package decoder

import (
	"fmt"
	"math"
)

// nopePredicate turns HF's position_embedding_type into the per-layer NoPE predicate.
// "nope" ⇒ every attention layer skips RoPE; absent or "rope" ⇒ nil (every layer ropes),
// which is both HF's default and the pre-existing behaviour.
func nopePredicate(kind string) func(int) bool {
	if kind == "nope" {
		return func(int) bool { return true }
	}
	return nil
}

// graniteArchitecture expresses Granite-4.0-H (model_type granitemoehybrid): a
// Mamba-2 + softmax-attention hybrid (per-layer kind from layer_types) with a routed
// + shared MoE on EVERY layer, plus four Granite scalar multipliers. The mamba
// layers run the selective-scan mixer (mamba2.go) with recurrent state in the hybrid
// cache; the attention layers are plain GQA + RoPE (no QK-norm, no bias) scaled by
// attention_multiplier instead of 1/√d. The MoE is GraniteMoe-style (fused gate+up
// experts + an ungated shared expert) but its softmax-top-k routing is identical to
// the Mixtral path, so moeMLP serves it unchanged once the fused tensors are split
// at load. Dedicated loader (buildGraniteWeights); its program is Architecture.graniteLayerOps.
func graniteArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateGranite(); err != nil {
		return nil, nil, err
	}
	// RoPE: the tiny fixture (built by instantiating GraniteMoeHybridConfig on a current
	// transformers) carries rope_parameters; the RELEASED granite-4.0-h checkpoints were
	// saved by transformers 4.56 and carry a top-level rope_theta + rope_scaling instead.
	// Accept both — the same shape deepseekArchitecture and phi3Architecture already use.
	// Reading only rope_parameters made goinfer reject every published Granite-4.0-H.
	base, scaling, err := ropeBaseFlatOrNested(cfg, "granite")
	if err != nil {
		return nil, nil, err
	}
	if cfg.NumHeads <= 0 { // hd = hidden/heads below — a hostile config.json has no validateGGUFDims gate (M16)
		return nil, nil, fmt.Errorf("decoder(granite): num_attention_heads must be > 0, got %d", cfg.NumHeads)
	}
	hd := cfg.HiddenDim / cfg.NumHeads
	types := cfg.LayerTypes
	one := func(v float64) float32 {
		if v == 0 {
			return 1
		}
		return float32(v)
	}
	attnMul := cfg.AttentionMultiplier
	if attnMul == 0 {
		attnMul = math.Pow(float64(hd), -0.5) // HF default when unset
	}
	logitScale := cfg.LogitsScaling
	if logitScale == 0 {
		logitScale = 1
	}
	return &Architecture{
		Name:            "granitemoehybrid",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         hd,
		IntermediateDim: cfg.IntermediateDim, // expert FFN width (GraniteMoe uses intermediate_size)
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		RMSAddOne:       false,
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   NormPre2,
		Act:             ActSiLU,
		QKVBias:         false,
		QKNorm:          false,
		MoE: &MoEConfig{
			NumExperts:            cfg.NumLocalExperts,
			TopK:                  cfg.NumExpertsPerTok,
			NormTopKProb:          true, // top-k then softmax == softmax then top-k+renorm
			IntermediateDim:       cfg.IntermediateDim,
			SharedIntermediateDim: cfg.SharedIntermediateSize,
			SharedUngated:         true, // GraniteMoe shared_mlp has no sigmoid gate
		},
		AttnScale:      attnMul, // Granite attention_multiplier (not 1/√d)
		RoPELocalBase:  base,
		RoPEGlobalBase: base,
		ropeScaling:    scaling,
		RotaryDim:      0, // full rotary
		// position_embedding_type "nope" ⇒ NO positional encoding on the attention layers
		// (HF builds no rotary_emb at all and passes position_embeddings=None). It is what
		// granite-4.0-h-tiny ships, and absent/"rope" keeps the roped path — so the flag is
		// read rather than assumed. Roping a NoPE checkpoint still generates text; it just
		// generates the wrong text, which is why this is config-driven and gated by the
		// real-checkpoint oracle rather than by the tiny fixture (which is roped).
		layerNoPE:    nopePredicate(cfg.PositionEmbeddingType),
		EmbedScale:   0, // embedding_multiplier is the program's embed scale (Architecture.embedScale), not the Gemma sqrt path
		TiedLMHead:   false,
		LogitScale:   logitScale,
		layerIsMamba: func(i int) bool { return i < len(types) && types[i] == "mamba" },
		granite: &graniteParams{
			NHeads: cfg.MambaNHeads, HeadDim: cfg.MambaDHead, DState: cfg.MambaDState,
			NGroups: cfg.MambaNGroups, DConv: cfg.MambaDConv,
			EmbMul: one(cfg.EmbeddingMultiplier), ResidMul: one(cfg.ResidualMultiplier),
		},
	}, &graniteTensorSchema, nil
}
