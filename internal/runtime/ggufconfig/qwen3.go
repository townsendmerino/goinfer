package ggufconfig

import (
	"github.com/townsendmerino/aikit/embed"
)

// ggufQwen3Config builds a Qwen3 dense Config from the qwen3.* metadata. Versus
// qwen2 it drops the q/k/v bias and adds QK-norm (per-head RMSNorm over head_dim
// before RoPE) — both carried by the qwen3 descriptor, so the GGUF loader only
// needs the metadata mapping (head_dim is explicit via attention.key_length, not
// derived). NEOX rope (no q/k permute, via ggufQKPermuted). RoPE base defaults to
// llama.cpp's qwen3 default (1e6) when absent.
func ggufQwen3Config(g *embed.GGUFFile) (*Config, error) {
	u := ggufUintReader(g, "qwen3.")
	cfg := &Config{
		ModelType:       "qwen3",
		MaxPositions:    u("context_length"),
		HiddenDim:       u("embedding_length"),
		NumLayers:       u("block_count"),
		NumHeads:        u("attention.head_count"),
		NumKVHeads:      u("attention.head_count_kv"),
		HeadDim:         u("attention.key_length"),
		IntermediateDim: u("feed_forward_length"),
		HiddenAct:       "silu",
		VocabSize:       ggufVocabSize(g),
	}
	if eps, ok := g.Float("qwen3.attention.layer_norm_rms_epsilon"); ok {
		cfg.RMSNormEps = eps
	}
	if base, ok := g.Float("qwen3.rope.freq_base"); ok {
		cfg.RoPEGlobalBase = base
	} else {
		cfg.RoPEGlobalBase = 1000000 // llama.cpp qwen3 default
	}
	ggufEOS(g, cfg)
	return cfg, nil
}

// ggufQwen3MoeConfig builds a Qwen3-MoE Config from the qwen3moe.* metadata: architecture string "qwen3moe"; the plain
// {arch}.attention.head_count/head_count_kv/key_length/layer_norm_rms_epsilon, {arch}.expert_count/expert_used_count/
// expert_feed_forward_length and {arch}.rope.freq_base keys; no sliding-window or YaRN keys (a single global RoPE, unlike
// Mellum's per-layer-type split). The tensor set carries attn_q_norm/attn_k_norm and ffn_gate_inp/ffn_{gate,up,down}_exps
// but no ffn_*_shexp (no shared expert, matching the real config's absent shared_expert_intermediate_size).
// norm_topk_prob is not GGUF metadata and is hardcoded true to match HF Qwen3MoeConfig's default, the same convention
// ggufMellumConfig uses for its own missing metadata.
func ggufQwen3MoeConfig(g *embed.GGUFFile) (*Config, error) {
	u := ggufUintReader(g, "qwen3moe.")
	gf := ggufFloatReader(g, "qwen3moe.")
	normTopK := true
	cfg := &Config{
		ModelType:           "qwen3_moe",
		MaxPositions:        u("context_length"),
		HiddenDim:           u("embedding_length"),
		NumLayers:           u("block_count"),
		NumHeads:            u("attention.head_count"),
		NumKVHeads:          u("attention.head_count_kv"),
		HeadDim:             u("attention.key_length"),
		IntermediateDim:     u("feed_forward_length"), // dense (vestigial): decoder_sparse_step=1 makes every layer MoE
		MoeIntermediateSize: u("expert_feed_forward_length"),
		NumExperts:          u("expert_count"),
		NumExpertsPerTok:    u("expert_used_count"),
		NormTopKProb:        &normTopK,
		HiddenAct:           "silu",
		VocabSize:           ggufVocabSize(g),
		RMSNormEps:          gf("attention.layer_norm_rms_epsilon"),
		RoPEGlobalBase:      gf("rope.freq_base"),
	}
	ggufEOS(g, cfg)
	return cfg, nil
}
