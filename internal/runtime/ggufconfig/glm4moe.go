package ggufconfig

import (
	"encoding/json"
	"fmt"

	"github.com/townsendmerino/aikit/embed"
)

// ggufGlm4MoeConfig builds a GLM-4.5/4.6 Config from the glm4moe.* metadata: dims,
// the DeepSeek-style MoE counts (expert_count / expert_used_count /
// expert_shared_count / expert_feed_forward_length), the leading_dense_block_count
// dense prefix, routed_scaling_factor (expert_weights_scale), and a synthesized
// rope_parameters JSON (partial_rotary_factor = rope.dimension_count / head_dim) so
// glm4moeArchitecture runs the identical descriptor build as the safetensors path.
func ggufGlm4MoeConfig(g *embed.GGUFFile) (*Config, error) {
	u := ggufUintReader(g, "glm4moe.")
	gf := ggufFloatReader(g, "glm4moe.")
	normTopK := true
	if b, ok := g.Metadata["glm4moe.expert_weights_norm"].(bool); ok {
		normTopK = b
	}
	headDim := u("attention.key_length")
	scale := gf("expert_weights_scale")
	if scale == 0 {
		scale = 1
	}
	// block_count includes the trailing NextN/MTP block(s) goinfer drops.
	numLayers := u("block_count") - u("nextn_predict_layers")
	// GLM-4.5/4.6 carry q/k/v bias but NO QK-norm; the tiny synthetic is the
	// reverse. Neither has a metadata flag, so detect both from blk.0's tensors.
	_, attnBias := g.Dims("blk.0.attn_q.bias")
	_, qkNorm := g.Dims("blk.0.attn_q_norm.weight")
	cfg := &Config{
		ModelType:           "glm4_moe",
		MaxPositions:        u("context_length"),
		HiddenDim:           u("embedding_length"),
		NumLayers:           numLayers,
		AttentionBias:       attnBias,
		UseQKNorm:           qkNorm,
		NumHeads:            u("attention.head_count"),
		NumKVHeads:          u("attention.head_count_kv"),
		HeadDim:             headDim,
		IntermediateDim:     u("feed_forward_length"),        // dense prefix width
		MoeIntermediateSize: u("expert_feed_forward_length"), // expert FFN width
		NRoutedExperts:      u("expert_count"),
		NumExpertsPerTok:    u("expert_used_count"),
		NSharedExperts:      u("expert_shared_count"),
		FirstKDenseReplace:  u("leading_dense_block_count"),
		RoutedScalingFactor: scale,
		NormTopKProb:        &normTopK,
		HiddenAct:           "silu",
		VocabSize:           ggufVocabSize(g),
		RMSNormEps:          gf("attention.layer_norm_rms_epsilon"),
	}
	rot := u("rope.dimension_count")
	partial := 0.0
	if headDim > 0 {
		partial = float64(rot) / float64(headDim)
	}
	cfg.RopeParameters = json.RawMessage(fmt.Sprintf(
		`{"rope_type":"default","rope_theta":%g,"partial_rotary_factor":%g}`,
		gf("rope.freq_base"), partial))
	ggufEOS(g, cfg)
	return cfg, nil
}
