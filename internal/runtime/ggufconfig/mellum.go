package ggufconfig

import (
	"encoding/json"
	"fmt"

	"github.com/townsendmerino/aikit/embed"
)

// ggufMellumConfig builds a Mellum2 Config from the mellum.* metadata: dims,
// the MoE counts (incl. the narrower expert_feed_forward_length), the sliding/
// full layer pattern, and the YaRN rope scaling — synthesized into the same
// shapes the mellum adapter consumes (LayerTypes + a rope_parameters JSON), so
// resolveArchitecture runs the identical descriptor build as the safetensors path.
func ggufMellumConfig(g *embed.GGUFFile) (*Config, error) {
	u := ggufUintReader(g, "mellum.")
	gf := ggufFloatReader(g, "mellum.")
	normTopK := true
	// block_count includes the trailing NextN/MTP block(s) of the -MTP files (nextn_predict_layers), which goinfer does not
	// run: counted as a trunk layer, the head would be decoded as a 29th layer and the output silently wrong.
	blocks := u("block_count")
	numLayers := blocks - u("nextn_predict_layers")
	if numLayers <= 0 || numLayers > MaxLayers {
		return nil, fmt.Errorf("decoder(gguf-mellum): bad block_count=%d (nextn=%d)", blocks, u("nextn_predict_layers"))
	}
	cfg := &Config{
		ModelType:           "mellum",
		MaxPositions:        u("context_length"),
		HiddenDim:           u("embedding_length"),
		NumLayers:           numLayers,
		NumHeads:            u("attention.head_count"),
		NumKVHeads:          u("attention.head_count_kv"),
		HeadDim:             u("attention.key_length"),
		IntermediateDim:     u("feed_forward_length"),        // dense (vestigial)
		MoeIntermediateSize: u("expert_feed_forward_length"), // expert FFN width
		NumExperts:          u("expert_count"),
		NumExpertsPerTok:    u("expert_used_count"),
		SlidingWindow:       u("attention.sliding_window"),
		NormTopKProb:        &normTopK,
		HiddenAct:           "silu",
		VocabSize:           ggufVocabSize(g),
		RMSNormEps:          gf("attention.layer_norm_rms_epsilon"),
	}
	// Per-layer attention type: sliding_window_pattern[i] true ⇒ sliding (local),
	// false ⇒ full (global).
	// It covers every block, the NextN ones included; the trunk takes the first numLayers.
	pat, ok := g.Metadata["mellum.attention.sliding_window_pattern"].([]any)
	if !ok || (len(pat) != blocks && len(pat) != numLayers) {
		return nil, fmt.Errorf("decoder(gguf-mellum): sliding_window_pattern missing or wrong length")
	}
	for _, p := range pat[:numLayers] {
		if b, _ := p.(bool); b {
			cfg.LayerTypes = append(cfg.LayerTypes, "sliding_attention")
		} else {
			cfg.LayerTypes = append(cfg.LayerTypes, "full_attention")
		}
	}
	// Synthesize rope_parameters: YaRN on the full layers (freq_base + scaling),
	// plain RoPE on the sliding layers (freq_base_swa). llama.cpp truncates the
	// attention_factor to f32; the mscale tolerance absorbs that.
	base := gf("rope.freq_base")
	baseSwa := gf("rope.freq_base_swa")
	if baseSwa == 0 {
		baseSwa = base
	}
	// attention_factor: the same trap ggufLagunaConfig documents. llama.cpp writes rope.scaling.yarn_attn_factor = 1.0 as its
	// "unset" sentinel and 0 is metadata-absent; passing either through as an explicit attention_factor would override
	// YaRN's own computed mscale (get_mscale(factor) = 0.1*ln(factor)+1) with a no-op or zero. Omit the field at the
	// sentinel and absent cases so the decoder computes it.
	attnFactor := ""
	if af := gf("rope.scaling.yarn_attn_factor"); af != 0 && af != 1 {
		attnFactor = fmt.Sprintf(`,"attention_factor":%g`, af)
	}
	cfg.RopeParameters = json.RawMessage(fmt.Sprintf(
		`{"full_attention":{"rope_type":"yarn","rope_theta":%g,"factor":%g,"original_max_position_embeddings":%g,"beta_fast":%g,"beta_slow":%g%s},`+
			`"sliding_attention":{"rope_type":"default","rope_theta":%g}}`,
		base, gf("rope.scaling.factor"), gf("rope.scaling.original_context_length"),
		gf("rope.scaling.yarn_beta_fast"), gf("rope.scaling.yarn_beta_slow"),
		attnFactor, baseSwa))
	ggufEOS(g, cfg)
	return cfg, nil
}
