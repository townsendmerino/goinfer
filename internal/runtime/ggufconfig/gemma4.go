package ggufconfig

import (
	"github.com/townsendmerino/aikit/embed"
)

// ggufGemma4Config builds a Gemma 4 (E-model) Config from the gemma4.* metadata:
// the variable per-layer FFN width + head_dim, the cross-layer KV-sharing count,
// the PLE (per-layer-embedding) dims, the explicit sliding/global bool pattern,
// dual-base RoPE, and the re-added final-logit softcap (30). The proportional
// (partial-rotary) factor for the global layers isn't in the GGUF metadata, so
// it's set from the known Gemma 4 value (0.25). Defaults mirror ggufGemmaConfig.
func ggufGemma4Config(g *embed.GGUFFile) (*Config, error) {
	u := ggufUintReader(g, "gemma4.")
	cfg := &Config{
		ModelType:               "gemma4",
		MaxPositions:            u("context_length"),
		HiddenDim:               u("embedding_length"),
		NumLayers:               u("block_count"),
		NumHeads:                u("attention.head_count"),
		NumKVHeads:              u("attention.head_count_kv"),
		HeadDim:                 u("attention.key_length_swa"), // sliding/local head_dim (256)
		GlobalHeadDim:           u("attention.key_length"),     // global/full head_dim (512)
		SlidingWindow:           u("attention.sliding_window"),
		SharedKVLayers:          u("attention.shared_kv_layers"),
		HiddenSizePerLayerInput: u("embedding_length_per_layer_input"),
		HiddenActivation:        "gelu_pytorch_tanh",
		VocabSize:               ggufVocabSize(g),
		PartialRotaryFactor:     0.25, // proportional RoPE on the global layers; not in GGUF metadata
	}
	cfg.VocabSizePerLayerInput = cfg.VocabSize // PLE table shares the main vocab
	// Variable per-layer FFN width (e.g. 6144 then 12288); fall back to a scalar.
	if a, ok := g.Metadata["gemma4.feed_forward_length"]; ok {
		cfg.FFNPerLayer = ggufIntArray(a)
	}
	if len(cfg.FFNPerLayer) > 0 {
		cfg.IntermediateDim = cfg.FFNPerLayer[0]
	} else {
		cfg.IntermediateDim = u("feed_forward_length")
	}
	// Explicit per-layer attention type (true = sliding, false = full/global).
	if a, ok := g.Metadata["gemma4.attention.sliding_window_pattern"]; ok {
		for _, sliding := range ggufBoolArray(a) {
			if sliding {
				cfg.LayerTypes = append(cfg.LayerTypes, "sliding_attention")
			} else {
				cfg.LayerTypes = append(cfg.LayerTypes, "full_attention")
			}
		}
	}
	// head_count_kv may be a per-layer array (12B: 8 on sliding, 1 on global) or a
	// scalar (E2B/E4B: 1). For the array, pull the sliding + global values; the
	// scalar literal above already covered the uniform case.
	if arr := ggufIntArray(g.Metadata["gemma4.attention.head_count_kv"]); len(arr) > 0 {
		for i, lt := range cfg.LayerTypes {
			if i >= len(arr) {
				break
			}
			if lt == "full_attention" {
				cfg.NumGlobalKVHeads = arr[i]
			} else {
				cfg.NumKVHeads = arr[i]
			}
		}
	}
	if eps, ok := g.Float("gemma4.attention.layer_norm_rms_epsilon"); ok {
		cfg.RMSNormEps = eps
	}
	cfg.RoPEGlobalBase = 1000000
	if base, ok := g.Float("gemma4.rope.freq_base"); ok {
		cfg.RoPEGlobalBase = base
	}
	cfg.RoPELocalBase = 10000
	if base, ok := g.Float("gemma4.rope.freq_base_swa"); ok {
		cfg.RoPELocalBase = base
	}
	if sc, ok := g.Float("gemma4.final_logit_softcapping"); ok {
		cfg.FinalLogitSoftcap = sc
	}
	// 26B-A4B's parallel dense+MoE FFN sub-block (dense E2B/E4B/12B GGUFs carry no
	// gemma4.expert_count key at all, so EnableMoeBlock stays false and their
	// forward is byte-unchanged — same gate the safetensors path uses).
	if ec := u("expert_count"); ec > 0 {
		cfg.EnableMoeBlock = true
		cfg.NumExperts = ec
		cfg.TopKExperts = u("expert_used_count")
		cfg.MoeIntermediateSize = u("expert_feed_forward_length")
	}
	ggufEOS(g, cfg)
	return cfg, nil
}
