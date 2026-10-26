package decoder

import (
	"fmt"
	"math"
)

// llamaArchitecture expresses Llama-2/3 dense: like Qwen3 (RMSNorm no-offset, Pre2 placement, SwiGLU, 1/√head_dim scale, single-base RoPE,
// no embed scale) but WITHOUT QK-norm: Llama's attention applies RoPE to raw q/k. head_dim is derived (headDim()) since many Llama configs
// omit it. The LM head is tied on the small text models and untied on 8B+, finalized from lm_head.weight presence at load. validateLlama
// rejects attention bias, so reaching here implies a plain checkpoint; linear and llama3 rope scaling are parsed below.
func llamaArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := backfillFlatRope(cfg, "llama"); err != nil {
		return nil, nil, err
	}
	if err := cfg.validateLlama(); err != nil {
		return nil, nil, err
	}
	scaling, err := parseRopeScaling(cfg.RopeScaling) // G4: linear + llama3; unsupported types error here
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(llama): %w", err)
	}
	hd := cfg.headDim()
	return &Architecture{
		Name:            "llama",
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
		QKNorm:          false, // the one knob that differs from Qwen3
		AttnScale:       math.Pow(float64(hd), -0.5),
		SlidingWindow:   0,                  // Llama dense: full attention
		layerIsGlobal:   nil,                // all-global
		RoPELocalBase:   cfg.RoPEGlobalBase, // single base (rope_theta)
		RoPEGlobalBase:  cfg.RoPEGlobalBase,
		RotaryDim:       cfg.rotaryDim(), // 0 = full head_dim (Llama); partial for Phi
		ropeScaling:     scaling,         // llama3 (3.1+/3.2) / linear; nil = none
		EmbedScale:      0,               // none
		TiedLMHead:      false,           // finalized from lm_head.weight presence at load
	}, &llamaTensorSchema, nil
}
