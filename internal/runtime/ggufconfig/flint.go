package ggufconfig

import (
	"fmt"

	"github.com/townsendmerino/aikit/embed"
)

// ggufFlintConfig builds a Flint (Cognichip) Config from the flint.* metadata
// emitted by llama.cpp PR #21882. Per-layer full vs sliding attention is encoded
// in flint.rope.scaling.factors (0.0 -> full_attention / NoPE, 1.0 ->
// sliding_attention / RoPE); when absent, ValidateFlint synthesizes the upstream
// default schedule.
func ggufFlintConfig(g *embed.GGUFFile) (*Config, error) {
	u := ggufUintReader(g, "flint.")
	gf := ggufFloatReader(g, "flint.")
	headDim := u("attention.key_length")
	numLayers := u("block_count")
	if headDim <= 0 || numLayers <= 0 {
		return nil, fmt.Errorf("decoder(gguf-flint): missing attention.key_length / block_count")
	}
	if _, err := ggufLayerCount(numLayers); err != nil {
		return nil, err
	}
	if gfn := u("expert_gating_func"); gfn != 0 && gfn != 2 {
		return nil, fmt.Errorf("decoder(gguf-flint): expert_gating_func=%d unsupported (2 = sigmoid)", gfn)
	}
	normTopK := true
	if b, ok := g.Metadata["flint.expert_weights_norm"].(bool); ok {
		normTopK = b
	} else if n, ok := g.Uint("flint.expert_weights_norm"); ok {
		normTopK = n != 0
	}
	scale := gf("expert_weights_scale")
	if scale == 0 {
		scale = 1
	}
	base := gf("rope.freq_base")
	if base == 0 {
		base = 10000
	}
	cfg := &Config{
		ModelType:                    "flint",
		MaxPositions:                 u("context_length"),
		HiddenDim:                    u("embedding_length"),
		NumLayers:                    numLayers,
		NumHeads:                     u("attention.head_count"),
		NumKVHeads:                   u("attention.head_count_kv"),
		HeadDim:                      headDim,
		IntermediateDim:              u("feed_forward_length"),
		MoeIntermediateSize:          u("expert_feed_forward_length"),
		SharedExpertIntermediateSize: u("expert_shared_feed_forward_length"),
		NumExperts:                   u("expert_count"),
		NumExpertsPerTok:             u("expert_used_count"),
		SlidingWindow:                u("attention.sliding_window"),
		RoutedScalingFactor:          scale,
		NormTopKProb:                 &normTopK,
		HiddenAct:                    "silu",
		VocabSize:                    ggufVocabSize(g),
		RMSNormEps:                   gf("attention.layer_norm_rms_epsilon"),
		RoPEGlobalBase:               base,
	}
	if k, ok := g.Uint("flint.leading_dense_block_count"); ok {
		if k > uint64(numLayers) {
			return nil, fmt.Errorf("decoder(gguf-flint): leading_dense_block_count %d exceeds block_count %d", k, numLayers)
		}
		cfg.MlpOnlyLayers = make([]int, int(k))
		for i := range cfg.MlpOnlyLayers {
			cfg.MlpOnlyLayers[i] = i
		}
	}
	if arr, ok := g.Metadata["flint.attention.sliding_window_pattern"].([]any); ok && len(arr) > 0 {
		if len(arr) != numLayers {
			return nil, fmt.Errorf("decoder(gguf-flint): attention.sliding_window_pattern has %d entries, want %d", len(arr), numLayers)
		}
		cfg.LayerTypes = make([]string, numLayers)
		for i, v := range arr {
			if b, _ := v.(bool); b {
				cfg.LayerTypes[i] = "sliding_attention"
			} else {
				cfg.LayerTypes[i] = "full_attention"
			}
		}
	} else if arr, ok := g.Metadata["flint.rope.scaling.factors"].([]any); ok && len(arr) > 0 {
		if len(arr) != numLayers {
			return nil, fmt.Errorf("decoder(gguf-flint): rope.scaling.factors has %d entries, want %d", len(arr), numLayers)
		}
		cfg.LayerTypes = make([]string, numLayers)
		for i, v := range arr {
			var f float64
			if f32, ok := v.(float32); ok {
				f = float64(f32)
			} else if f64, ok := v.(float64); ok {
				f = f64
			}
			if f == 0 {
				cfg.LayerTypes[i] = "full_attention"
			} else {
				cfg.LayerTypes[i] = "sliding_attention"
			}
		}
	}
	if arr, ok := g.Metadata["flint.attention.rope_pattern"].([]any); ok && len(arr) == numLayers {
		for i, v := range arr {
			if b, _ := v.(bool); !b {
				cfg.NoRopeLayerIndices = append(cfg.NoRopeLayerIndices, i)
			}
		}
	}
	ggufEOS(g, cfg)
	return cfg, nil
}
