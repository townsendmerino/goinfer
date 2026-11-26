package ggufconfig

import (
	"github.com/townsendmerino/aikit/embed"
)

// ggufLlama4Config builds a Llama 4 text-decoder Config from the llama4.* metadata. The
// GGUF carries the dims/expert counts but OMITS the iRoPE scalars (no_rope_layers,
// use_qk_norm, attn_temperature, floor_scale, attn_scale) — llama.cpp applies them by arch
// convention — so they are injected here: NoPE every 4th layer (the interval-4 pattern),
// QK-norm + attention-temperature on (8192, 0.1). interleave_moe_layer_step (in the GGUF)
// drives the dense/MoE split (Scout=1 ⇒ all MoE). The llama3 long-context rope scaling is
// NOT applied (negligible below original_max_position_embeddings; the GGUF stores it as a
// precomputed rope_freqs.weight goinfer doesn't consume) — fine for short-prompt gates.
func ggufLlama4Config(g *embed.GGUFFile) (*Config, error) {
	u := ggufUintReader(g, "llama4.")
	gf := ggufFloatReader(g, "llama4.")
	nLayers, err := ggufLayerCount(u("block_count"))
	if err != nil {
		return nil, err
	}
	cfg := &Config{
		ModelType:              "llama4_text",
		MaxPositions:           u("context_length"),
		HiddenDim:              u("embedding_length"),
		NumLayers:              nLayers,
		NumHeads:               u("attention.head_count"),
		NumKVHeads:             u("attention.head_count_kv"),
		HeadDim:                u("attention.key_length"),
		IntermediateDim:        u("expert_feed_forward_length"), // routed/shared expert width (intermediate_size)
		IntermediateSizeMLP:    u("feed_forward_length"),        // dense-layer width (intermediate_size_mlp)
		NumLocalExperts:        u("expert_count"),
		NumExpertsPerTok:       u("expert_used_count"),
		InterleaveMoeLayerStep: u("interleave_moe_layer_step"),
		VocabSize:              ggufVocabSize(g),
		RMSNormEps:             gf("attention.layer_norm_rms_epsilon"),
		RoPEGlobalBase:         gf("rope.freq_base"),
		HiddenAct:              "silu",
		// Injected iRoPE scalars (absent from the GGUF; llama.cpp arch defaults).
		UseQKNorm:             true,
		AttnTemperatureTuning: true,
		FloorScale:            8192,
		AttnScaleL4:           0.1,
	}
	// no_rope_layers: NoPE on every 4th layer ((i+1)%4==0), else RoPE — the released
	// Scout/Maverick pattern.
	cfg.NoRopeLayers = make([]int, nLayers)
	for i := range cfg.NoRopeLayers {
		if (i+1)%4 == 0 {
			cfg.NoRopeLayers[i] = 0
		} else {
			cfg.NoRopeLayers[i] = 1
		}
	}
	ggufEOS(g, cfg)
	return cfg, nil
}
