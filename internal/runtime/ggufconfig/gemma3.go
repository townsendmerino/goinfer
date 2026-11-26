package ggufconfig

import (
	"github.com/townsendmerino/aikit/embed"
)

// ggufGemmaConfig builds a Gemma 3 (text) Config from the gemma3.* metadata. The
// gemma3 descriptor already supplies the architecture knobs (sandwich norms,
// GeGLU, QK-norm, embed scale √hidden, dual RoPE bases, tied head); this maps the
// dims + the few values the GGUF doesn't carry explicitly. NEOX rope (no q/k
// permute). The GGUF omits the local RoPE base, the sliding/global pattern, and
// query_pre_attn_scalar, so they default to gemma3's fixed values (10000, a 5:1
// pattern, and head_dim). Gemma 3 has no logit/attn soft-capping (that's Gemma 2),
// so those stay zero — ValidateAssumptions rejects a Gemma 2 GGUF here.
func ggufGemmaConfig(g *embed.GGUFFile) (*Config, error) {
	u := ggufUintReader(g, "gemma3.")
	cfg := &Config{
		ModelType:        "gemma3",
		MaxPositions:     u("context_length"),
		HiddenDim:        u("embedding_length"),
		NumLayers:        u("block_count"),
		NumHeads:         u("attention.head_count"),
		NumKVHeads:       u("attention.head_count_kv"),
		HeadDim:          u("attention.key_length"),
		IntermediateDim:  u("feed_forward_length"),
		SlidingWindow:    u("attention.sliding_window"),
		HiddenActivation: "gelu_pytorch_tanh",
		VocabSize:        ggufVocabSize(g),
	}
	if eps, ok := g.Float("gemma3.attention.layer_norm_rms_epsilon"); ok {
		cfg.RMSNormEps = eps
	}
	if base, ok := g.Float("gemma3.rope.freq_base"); ok {
		cfg.RoPEGlobalBase = base
	} else {
		cfg.RoPEGlobalBase = 1000000 // gemma3 rope_theta (global layers)
	}
	cfg.RoPELocalBase = 10000 // gemma3 rope_local_base_freq (sliding layers); not in GGUF metadata
	// Sliding/global pattern: gemma3 is 5 local : 1 global. The GGUF carries no
	// layer_types, so IsGlobalLayer falls back to this period ((i+1)%6==0 global).
	if p := u("attention.sliding_window_pattern"); p > 0 {
		cfg.SlidingWindowPattern = p
	} else {
		cfg.SlidingWindowPattern = 6
	}
	// query_pre_attn_scalar == head_dim for gemma3 (the attention scale is its
	// inverse sqrt); the GGUF doesn't store it.
	cfg.QueryPreAttnScalar = float64(cfg.HeadDim)
	ggufEOS(g, cfg)
	return cfg, nil
}
