package ggufconfig

import (
	"github.com/townsendmerino/aikit/embed"
)

// ggufPhi3Config builds a Phi-3/Phi-4 (phi3) Config from the phi3.* metadata. The fused
// attn_qkv + ffn_up (gate‖up) tensors are split by buildWeightsFromGGUF's phi3 branch.
// rope.freq_base is omitted when default (10000); rope.dimension_count < head_dim ⇒ partial
// rotary. LongRoPE (128k variants) is deferred — those store a rope.scaling block that this
// minimal reader doesn't translate (and phi3Architecture would reject).
func ggufPhi3Config(g *embed.GGUFFile) (*Config, error) {
	u := ggufUintReader(g, "phi3.")
	gf := ggufFloatReader(g, "phi3.")
	hidden, heads := u("embedding_length"), u("attention.head_count")
	headDim := hidden / heads
	base := gf("rope.freq_base")
	if base == 0 {
		base = 10000 // llama.cpp omits the key at the default
	}
	cfg := &Config{
		ModelType:       "phi3",
		MaxPositions:    u("context_length"),
		HiddenDim:       hidden,
		NumLayers:       u("block_count"),
		NumHeads:        heads,
		NumKVHeads:      u("attention.head_count_kv"),
		IntermediateDim: u("feed_forward_length"),
		VocabSize:       ggufVocabSize(g),
		RMSNormEps:      gf("attention.layer_norm_rms_epsilon"),
		RoPEGlobalBase:  base,
		HiddenAct:       "silu",
	}
	if rd := u("rope.dimension_count"); rd > 0 && rd < headDim {
		cfg.PartialRotaryFactor = float64(rd) / float64(headDim)
	}
	ggufEOS(g, cfg)
	return cfg, nil
}
