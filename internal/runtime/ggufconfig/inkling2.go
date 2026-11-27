package ggufconfig

import (
	"fmt"

	"github.com/townsendmerino/aikit/embed"

	"github.com/townsendmerino/goinfer/decoder/config"
)

// ggufInkling2Config builds an Inkling 2 Config from llama.cpp's inkling2.* / inkling2_loop.* metadata (conversion/inkling2.py).
// The vocabulary is the unpadded one (vocab_size_unpadded, the tokenizer's size); the stored embedding and head keep
// vocab_size rows, of which the loader reads the first VocabSize.
func ggufInkling2Config(g *embed.GGUFFile, arch string) (*Config, error) {
	u := ggufUintReader(g, arch+".")
	gf := ggufFloatReader(g, arch+".")
	numLayers := u("block_count")
	if _, err := ggufLayerCount(numLayers); err != nil {
		return nil, err
	}
	heads := ggufIntArray(g.Metadata[arch+".attention.head_count"])
	kvHeads := ggufIntArray(g.Metadata[arch+".attention.head_count_kv"])
	swa := ggufBoolArray(g.Metadata[arch+".attention.sliding_window_pattern"])
	if len(heads) != numLayers || len(kvHeads) != numLayers || len(swa) != numLayers {
		return nil, fmt.Errorf("decoder(gguf-%s): head counts (%d, %d) and the sliding-window pattern (%d) need %d entries", arch, len(heads), len(kvHeads), len(swa), numLayers)
	}
	for l := range numLayers {
		if heads[l] != heads[0] || kvHeads[l] != kvHeads[0] {
			return nil, fmt.Errorf("decoder(gguf-%s): per-layer head counts differ at layer %d; only uniform counts are supported", arch, l)
		}
	}
	if u("attention.key_length_swa") != u("attention.key_length") {
		return nil, fmt.Errorf("decoder(gguf-%s): the sliding layers' head width differs from the full layers'", arch)
	}
	unpadded := u("vocab_size_unpadded")
	padded := u("vocab_size")
	if unpadded <= 0 {
		unpadded = ggufVocabSize(g)
	}
	if padded <= 0 {
		padded = ggufVocabSize(g)
	}
	scale := gf("expert_weights_scale")
	if scale == 0 {
		scale = 1
	}
	normTopK := true
	ink := &config.Inkling2Config{
		SWA:            swa,
		DRel:           u("attention.relative_length"),
		RelExtent:      u("attention.relative_extent"),
		ConvK:          u("shortconv.l_cache"),
		BlockSize:      u("attn_res.block_size"),
		QKClip:         gf("attention.qk_clip"),
		RelClip:        gf("attention.relative_clip"),
		LogFloor:       u("attention.log_scale_floor"),
		LogAlpha:       gf("attention.log_scale_alpha"),
		SituBeta:       gf("activation.situ_beta"),
		SituLinearBeta: gf("activation.situ_linear_beta"),
		PaddedVocab:    padded,
	}
	if arch == "inkling2_loop" {
		ink.LoopStart, ink.LoopEnd, ink.LoopPasses, ink.GateRank = u("loop.start"), u("loop.end"), u("loop.count"), u("loop.gate_rank")
		if b, ok := g.Metadata[arch+".loop.varying"].(bool); ok && b {
			return nil, fmt.Errorf("decoder(gguf-%s): a varying loop is not supported", arch)
		}
	}
	cfg := &Config{
		ModelType:              arch,
		MaxPositions:           u("context_length"),
		HiddenDim:              u("embedding_length"),
		NumLayers:              numLayers,
		NumHeads:               heads[0],
		NumKVHeads:             kvHeads[0],
		HeadDim:                u("attention.key_length"),
		IntermediateDim:        u("feed_forward_length"),
		MoeIntermediateSize:    u("expert_feed_forward_length"),
		NumExperts:             u("expert_count"),
		NumExpertsPerTok:       u("expert_used_count"),
		SlidingWindow:          u("attention.sliding_window"),
		MoeRoutedScalingFactor: scale,
		NormTopKProb:           &normTopK,
		HiddenAct:              "situ",
		VocabSize:              unpadded,
		RMSNormEps:             gf("attention.layer_norm_rms_epsilon"),
		FinalLogitSoftcap:      gf("final_logit_softcapping"),
		Inkling2:               ink,
	}
	return cfg, nil
}
