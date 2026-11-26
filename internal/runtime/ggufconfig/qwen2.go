package ggufconfig

import (
	"github.com/townsendmerino/aikit/embed"
)

// ggufQwen2Config builds a Qwen2/Qwen2.5 Config from the qwen2.* metadata. It is
// the llama config plus the qwen2 model type — the only architectural difference
// is the additive q/k/v projection bias, carried by the qwen2 descriptor
// (QKVBias) and loaded by buildWeightsFromGGUF. RoPE base defaults to llama.cpp's
// qwen2 default (1e6) when absent.
func ggufQwen2Config(g *embed.GGUFFile) (*Config, error) {
	u := ggufUintReader(g, "qwen2.")
	cfg := &Config{
		ModelType:       "qwen2",
		MaxPositions:    u("context_length"),
		HiddenDim:       u("embedding_length"),
		NumLayers:       u("block_count"),
		NumHeads:        u("attention.head_count"),
		NumKVHeads:      u("attention.head_count_kv"),
		IntermediateDim: u("feed_forward_length"),
		HiddenAct:       "silu",
		VocabSize:       ggufVocabSize(g),
	}
	if eps, ok := g.Float("qwen2.attention.layer_norm_rms_epsilon"); ok {
		cfg.RMSNormEps = eps
	}
	if base, ok := g.Float("qwen2.rope.freq_base"); ok {
		cfg.RoPEGlobalBase = base
	} else {
		cfg.RoPEGlobalBase = 1000000 // llama.cpp qwen2 default
	}
	ggufEOS(g, cfg)
	return cfg, nil
}
