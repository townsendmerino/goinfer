package ggufconfig

import (
	"encoding/json"
	"fmt"

	"github.com/townsendmerino/aikit/embed"
)

// ggufGptOssConfig builds a gpt-oss Config from the gpt-oss.* metadata. The SwiGLU
// alpha/limit are gpt-oss constants (not in the GGUF), set by gptOssArchitecture.
// YaRN is synthesized from the rope.scaling.* keys with truncate=false (gpt-oss's
// setting — floor/ceiling the correction range would shift inv_freq by ~2e-4).
func ggufGptOssConfig(g *embed.GGUFFile) (*Config, error) {
	u := func(k string) int {
		v, _ := g.Uint("gpt-oss." + k)
		return int(v)
	}
	gf := func(k string) float64 {
		if v, ok := g.Float("gpt-oss." + k); ok {
			return v
		}
		v, _ := g.Uint("gpt-oss." + k)
		return float64(v)
	}
	cfg := &Config{
		ModelType:           "gpt_oss",
		MaxPositions:        u("context_length"),
		HiddenDim:           u("embedding_length"),
		NumLayers:           u("block_count"),
		NumHeads:            u("attention.head_count"),
		NumKVHeads:          u("attention.head_count_kv"),
		HeadDim:             u("attention.key_length"),
		MoeIntermediateSize: u("expert_feed_forward_length"),
		NumLocalExperts:     u("expert_count"),
		NumExpertsPerTok:    u("expert_used_count"),
		SlidingWindow:       u("attention.sliding_window"),
		HiddenAct:           "silu",
		VocabSize:           ggufVocabSize(g),
		RMSNormEps:          gf("attention.layer_norm_rms_epsilon"),
		RoPEGlobalBase:      gf("rope.freq_base"),
	}
	factor := gf("rope.scaling.factor")
	if factor == 0 {
		factor = 1
	}
	origCtx := gf("rope.scaling.original_context_length")
	betaFast := gf("rope.scaling.yarn_beta_fast")
	betaSlow := gf("rope.scaling.yarn_beta_slow")
	if scaleType, _ := g.Str("gpt-oss.rope.scaling.type"); scaleType == "yarn" {
		cfg.RopeScaling = json.RawMessage(fmt.Sprintf(
			`{"rope_type":"yarn","factor":%g,"beta_fast":%g,"beta_slow":%g,"original_max_position_embeddings":%g,"truncate":false}`,
			factor, betaFast, betaSlow, origCtx))
	}
	ggufEOS(g, cfg)
	return cfg, nil
}
