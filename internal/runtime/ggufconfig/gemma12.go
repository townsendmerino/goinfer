package ggufconfig

import (
	"github.com/townsendmerino/aikit/embed"
)

// ggufGemma12Config builds a Gemma 1 (GGUF arch "gemma") or Gemma 2 ("gemma2") Config from llama.cpp's metadata. The
// GGUF stores the dims, the RMS epsilon, the Gemma 2 sliding window and both softcaps, and llama.cpp's converter adds
// 1 to every norm weight (the loader's vnorm subtracts it back, as for gemma3). What it does not store, mirrored from
// llama.cpp's own loader (llama-model.cpp) rather than guessed:
//   - Gemma 2's query_pre_attn_scalar: llama.cpp sets the attention scale to 1/sqrt(hidden/heads) on the 27B (46
//     layers) and 1/sqrt(head_dim) otherwise, which is the HF configs' 144 and 256;
//   - the sliding pattern: every other layer from 0 (set_swa_pattern(2));
//   - the RoPE base when rope.freq_base is absent: 10000.
func ggufGemma12Config(g *embed.GGUFFile, arch string) (*Config, error) {
	u := func(k string) int {
		v, _ := g.Uint(arch + "." + k)
		return int(v)
	}
	cfg := &Config{
		ModelType:        arch,
		MaxPositions:     u("context_length"),
		HiddenDim:        u("embedding_length"),
		NumLayers:        u("block_count"),
		NumHeads:         u("attention.head_count"),
		NumKVHeads:       u("attention.head_count_kv"),
		HeadDim:          u("attention.key_length"),
		IntermediateDim:  u("feed_forward_length"),
		HiddenActivation: "gelu_pytorch_tanh",
		VocabSize:        ggufVocabSize(g),
	}
	if cfg.NumKVHeads == 0 {
		cfg.NumKVHeads = cfg.NumHeads
	}
	if eps, ok := g.Float(arch + ".attention.layer_norm_rms_epsilon"); ok {
		cfg.RMSNormEps = eps
	}
	if base, ok := g.Float(arch + ".rope.freq_base"); ok {
		cfg.RoPEGlobalBase = base
	}
	if arch == "gemma2" {
		cfg.SlidingWindow = u("attention.sliding_window")
		cfg.SlidingWindowPattern = 2
		if v, ok := g.Float("gemma2.attn_logit_softcapping"); ok {
			cfg.AttnLogitSoftcap = v
		}
		if v, ok := g.Float("gemma2.final_logit_softcapping"); ok {
			cfg.FinalLogitSoftcap = v
		}
		if cfg.NumLayers == 46 && cfg.NumHeads > 0 {
			cfg.QueryPreAttnScalar = float64(cfg.HiddenDim / cfg.NumHeads)
		} else {
			cfg.QueryPreAttnScalar = float64(cfg.HeadDim)
		}
	}
	ggufEOS(g, cfg)
	return cfg, nil
}
