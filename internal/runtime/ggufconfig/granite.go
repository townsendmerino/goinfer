package ggufconfig

import (
	"encoding/json"
	"fmt"

	"github.com/townsendmerino/aikit/embed"
)

// ggufGraniteDenseConfig builds a dense Granite 4.2 Config from the granite.* metadata. The architecture string is
// "granite", distinct from the hybrid's "granitehybrid", and llama.cpp bakes Granite's scalar multipliers directly into
// metadata (attention.scale, the resolved attention_multiplier rather than 1/sqrt(d); embedding_scale; logit_scale;
// residual_scale), so no multiplier-versus-default resolution is needed as on the safetensors path. The tensor set is
// exactly llama's: the metadata namespace differs, so ggufLlamaConfig cannot be reused, but tensor loading is identical
// once resolveArchitecture returns llamaTensorSchema.
func ggufGraniteDenseConfig(g *embed.GGUFFile) (*Config, error) {
	u := ggufUintReader(g, "granite.")
	f := func(k string) float64 {
		v, _ := g.Float("granite." + k)
		return v
	}
	cfg := &Config{
		ModelType:           "granite",
		MaxPositions:        u("context_length"),
		HiddenDim:           u("embedding_length"),
		NumLayers:           u("block_count"),
		NumHeads:            u("attention.head_count"),
		NumKVHeads:          u("attention.head_count_kv"),
		IntermediateDim:     u("feed_forward_length"),
		HiddenAct:           "silu",
		VocabSize:           ggufVocabSize(g),
		RMSNormEps:          f("attention.layer_norm_rms_epsilon"),
		RoPEGlobalBase:      f("rope.freq_base"),
		AttentionMultiplier: f("attention.scale"),
		EmbeddingMultiplier: f("embedding_scale"),
		LogitsScaling:       f("logit_scale"),
		ResidualMultiplier:  f("residual_scale"),
	}
	ggufEOS(g, cfg)
	return cfg, nil
}

// ggufGraniteConfig builds a Granite-4.0-H Config from the granitehybrid.* metadata.
// Per-layer kind comes from the attention.head_count_kv array (>0 ⇒ attention layer,
// 0 ⇒ mamba). The Mamba-2 geometry is the ssm.* keys (time_step_rank IS the head
// count; head_dim = inner_size / time_step_rank). The four Granite scalars are
// embedding_scale / attention.scale / residual_scale / logit_scale. rope_parameters
// is synthesized so graniteArchitecture runs the same descriptor build as safetensors.
func ggufGraniteConfig(g *embed.GGUFFile) (*Config, error) {
	u := ggufUintReader(g, "granitehybrid.")
	gf := ggufFloatReader(g, "granitehybrid.")
	nLayers, err := ggufLayerCount(u("block_count"))
	if err != nil {
		return nil, err
	}
	// Per-layer kind + the (uniform) attention KV-head count from the head_count_kv
	// array: nonzero entries are the attention layers.
	kvArr, _ := g.Metadata["granitehybrid.attention.head_count_kv"].([]any)
	layerTypes := make([]string, nLayers)
	kvHeads := 0
	for i := range nLayers {
		kv := 0
		if i < len(kvArr) {
			switch v := kvArr[i].(type) {
			case uint32:
				kv = int(v)
			case int32:
				kv = int(v)
			case uint64:
				kv = int(v)
			case int64:
				kv = int(v)
			}
		}
		if kv > 0 {
			layerTypes[i] = "attention"
			kvHeads = kv
		} else {
			layerTypes[i] = "mamba"
		}
	}
	nHeadsMamba := u("ssm.time_step_rank")
	innerSize := u("ssm.inner_size")
	cfg := &Config{
		ModelType:              "granitemoehybrid",
		MaxPositions:           u("context_length"),
		HiddenDim:              u("embedding_length"),
		NumLayers:              nLayers,
		NumHeads:               u("attention.head_count"),
		NumKVHeads:             kvHeads,
		IntermediateDim:        u("feed_forward_length"),               // expert FFN width
		SharedIntermediateSize: u("expert_shared_feed_forward_length"), // shared expert width
		NumLocalExperts:        u("expert_count"),
		NumExpertsPerTok:       u("expert_used_count"),
		MambaNHeads:            nHeadsMamba,
		MambaDHead:             innerSize / nHeadsMamba,
		MambaDState:            u("ssm.state_size"),
		MambaDConv:             u("ssm.conv_kernel"),
		MambaNGroups:           u("ssm.group_count"),
		EmbeddingMultiplier:    gf("embedding_scale"),
		AttentionMultiplier:    gf("attention.scale"),
		ResidualMultiplier:     gf("residual_scale"),
		LogitsScaling:          gf("logit_scale"),
		LayerTypes:             layerTypes,
		HiddenAct:              "silu",
		VocabSize:              ggufVocabSize(g),
		RMSNormEps:             gf("attention.layer_norm_rms_epsilon"),
	}
	cfg.RopeParameters = json.RawMessage(fmt.Sprintf(
		`{"rope_type":"default","rope_theta":%g}`, gf("rope.freq_base")))
	// NoPE. The released granite-4.0-h models set position_embedding_type "nope", and the converter carries that across as
	// rope.scaling.finetuned; the rope.dimension_count and rope.freq_base keys are written regardless and are vestigial here.
	// Only an explicitly present key flips the behaviour; absent leaves the roped path, so an older GGUF cannot silently lose
	// its RoPE.
	if finetuned, ok := g.Metadata["granitehybrid.rope.scaling.finetuned"].(bool); ok && !finetuned {
		cfg.PositionEmbeddingType = "nope"
	}
	ggufEOS(g, cfg)
	return cfg, nil
}
