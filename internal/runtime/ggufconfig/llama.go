package ggufconfig

import (
	"github.com/townsendmerino/aikit/embed"
)

func ggufLlamaConfig(g *embed.GGUFFile) (*Config, error) {
	u := func(k string) int {
		v, _ := g.Uint(k)
		return int(v)
	}
	cfg := &Config{
		ModelType:       "llama",
		MaxPositions:    u("llama.context_length"),
		HiddenDim:       u("llama.embedding_length"),
		NumLayers:       u("llama.block_count"),
		NumHeads:        u("llama.attention.head_count"),
		NumKVHeads:      u("llama.attention.head_count_kv"),
		IntermediateDim: u("llama.feed_forward_length"),
		HiddenAct:       "silu",
		VocabSize:       ggufVocabSize(g),
	}
	if eps, ok := g.Float("llama.attention.layer_norm_rms_epsilon"); ok {
		cfg.RMSNormEps = eps
	}
	if base, ok := g.Float("llama.rope.freq_base"); ok {
		cfg.RoPEGlobalBase = base
	} else {
		cfg.RoPEGlobalBase = 10000 // llama.cpp default
	}
	ggufEOS(g, cfg)
	return cfg, nil
}
