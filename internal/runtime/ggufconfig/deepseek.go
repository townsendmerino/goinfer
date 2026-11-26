package ggufconfig

import (
	"encoding/json"
	"fmt"

	"github.com/townsendmerino/aikit/embed"
)

// ggufDeepseekConfig builds a DeepSeek-V2/V3 (MLA) Config from the deepseek2.* metadata
// (llama.cpp maps BOTH V2 and V3 to arch "deepseek2"; the routing flavor is
// expert_gating_func: 1/absent = softmax = V2, 2 = sigmoid noaux_tc = V3). The MLA head
// split comes from attention.key_length (= qk_head_dim = nope+rope) minus
// rope.dimension_count (= qk_rope), with value_length the (different) v_head_dim and
// kv_lora_rank the cached-latent width; q_lora_rank is present only on the q-LoRA models.
// The yarn block is synthesized into rope_scaling so deepseekArchitecture runs the same
// descriptor build as safetensors (yarn_log_multiplier = 0.1·mscale; DeepSeek's
// mscale == mscale_all_dim ⇒ the cos/sin attention_factor ratio is 1.0).
func ggufDeepseekConfig(g *embed.GGUFFile) (*Config, error) {
	u := ggufUintReader(g, "deepseek2.")
	gf := ggufFloatReader(g, "deepseek2.")
	keyLen := u("attention.key_length")  // qk_nope_head_dim + qk_rope_head_dim
	ropeDim := u("rope.dimension_count") // qk_rope_head_dim
	sigmoid := u("expert_gating_func") == 2
	modelType := "deepseek_v2"
	scoring := "softmax"
	if sigmoid {
		modelType, scoring = "deepseek_v3", "sigmoid"
	}
	// expert_weights_norm: absent ⇒ V2 false / V3 true (matches the released configs).
	normTopK := sigmoid
	if b, ok := g.Metadata["deepseek2.expert_weights_norm"].(bool); ok {
		normTopK = b
	}
	scale := gf("expert_weights_scale")
	if scale == 0 {
		scale = 1
	}
	cfg := &Config{
		ModelType:           modelType,
		MaxPositions:        u("context_length"),
		ScoringFunc:         scoring,
		HiddenDim:           u("embedding_length"),
		NumLayers:           u("block_count"),
		NumHeads:            u("attention.head_count"),
		NumKVHeads:          u("attention.head_count_kv"),
		QLoRARank:           u("attention.q_lora_rank"), // absent ⇒ 0 (direct q_proj)
		KVLoRARank:          u("attention.kv_lora_rank"),
		QKRopeHeadDim:       ropeDim,
		QKNopeHeadDim:       keyLen - ropeDim,
		VHeadDim:            u("attention.value_length"),
		IntermediateDim:     u("feed_forward_length"),        // dense prefix width
		MoeIntermediateSize: u("expert_feed_forward_length"), // expert FFN width
		NRoutedExperts:      u("expert_count"),
		NumExpertsPerTok:    u("expert_used_count"),
		NSharedExperts:      u("expert_shared_count"),
		FirstKDenseReplace:  u("leading_dense_block_count"),
		RoutedScalingFactor: scale,
		NGroup:              u("expert_group_count"),      // absent ⇒ 0 (no group limiting)
		TopkGroup:           u("expert_group_used_count"), // absent ⇒ 0
		NormTopKProb:        &normTopK,
		HiddenAct:           "silu",
		VocabSize:           ggufVocabSize(g),
		RMSNormEps:          gf("attention.layer_norm_rms_epsilon"),
		RoPEGlobalBase:      gf("rope.freq_base"),
	}
	// YaRN (V2-Lite, V3): synthesize the rope_scaling object. llama.cpp stores
	// yarn_log_multiplier = 0.1·mscale; DeepSeek sets mscale == mscale_all_dim, so emit
	// both (deepseekArchitecture's ratio collapses to 1.0). The inv-freq ramp uses
	// factor + original_context_length + the default beta_fast/beta_slow.
	if t, _ := g.Str("deepseek2.rope.scaling.type"); t == "yarn" {
		mscale := gf("rope.scaling.yarn_log_multiplier") * 10 // 0.1·mscale ⇒ mscale
		cfg.RopeScaling = json.RawMessage(fmt.Sprintf(
			`{"type":"yarn","factor":%g,"original_max_position_embeddings":%g,"mscale":%g,"mscale_all_dim":%g,"beta_fast":32,"beta_slow":1}`,
			gf("rope.scaling.factor"), gf("rope.scaling.original_context_length"), mscale, mscale))
	}
	ggufEOS(g, cfg)
	return cfg, nil
}
