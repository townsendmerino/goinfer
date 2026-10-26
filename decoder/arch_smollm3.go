package decoder

import (
	"fmt"
	"math"
)

// smollm3Architecture expresses SmolLM3-3B (model_type smollm3): a plain llama-shaped dense GQA model (llamaTensorSchema reused verbatim)
// with per-layer NoPE on every 4th layer via `no_rope_layers`, through the same Config field and convention llama4_text uses and the same
// layerNoPE hook cohere2Architecture populates.
//
// THE FIELD NAME IS THE OPPOSITE OF ITS VALUES: a `no_rope_layers` entry of 1 means the layer HAS rope, 0 means NoPE (modeling_smollm3.py:
// `self.use_rope = config.no_rope_layers[layer_idx]`). The released list is [1,1,1,0] repeating, and configuration_smollm3.py generates it
// as `(layer_idx+1) % no_rope_layer_interval != 0` when a checkpoint omits it. Getting the polarity backwards silently inverts which layers
// use RoPE: correct shapes, plausible logits, wrong model, no crash. `layer_types` is not authoritative here (every entry reads
// "full_attention" whichever layers are NoPE); `no_rope_layers` is, unlike Gemma/cohere2 where `layer_types` carries the split.
func smollm3Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := backfillFlatRope(cfg, "smollm3"); err != nil {
		return nil, nil, err
	}
	if err := cfg.validateLlama(); err != nil {
		return nil, nil, err
	}
	scaling, err := parseRopeScaling(cfg.RopeScaling)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(smollm3): %w", err)
	}
	hd := cfg.headDim()
	noRope := cfg.NoRopeLayers
	if len(noRope) == 0 {
		interval := cfg.NoRopeLayerInterval
		if interval <= 0 {
			interval = 4 // HF's own default (configuration_smollm3.py)
		}
		noRope = make([]int, cfg.NumLayers)
		for i := range noRope {
			if (i+1)%interval != 0 {
				noRope[i] = 1 // has rope
			}
		}
	}
	layerNoPE := func(i int) bool { return i >= 0 && i < len(noRope) && noRope[i] == 0 }
	return &Architecture{
		Name:            "smollm3",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         hd,
		IntermediateDim: cfg.IntermediateDim,
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		RMSAddOne:       false,
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   NormPre2,
		Act:             ActSiLU,
		QKNorm:          false,
		AttnScale:       math.Pow(float64(hd), -0.5),
		SlidingWindow:   0, // full attention on the RoPE'd layers; confirmed null on the real release
		layerIsGlobal:   nil,
		layerNoPE:       layerNoPE,
		RoPELocalBase:   cfg.RoPEGlobalBase,
		RoPEGlobalBase:  cfg.RoPEGlobalBase,
		RotaryDim:       cfg.rotaryDim(),
		ropeScaling:     scaling,
		EmbedScale:      0,
		TiedLMHead:      false, // finalized from lm_head.weight presence at load
	}, &llamaTensorSchema, nil
}
