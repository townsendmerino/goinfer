package decoder

import (
	"fmt"
	"math"
)

// phi3Architecture expresses Phi-3 / Phi-4 (model_type phi3): the llama skeleton —
// RMSNorm no-offset, Pre2 norm placement, SwiGLU, 1/√head_dim attention scale,
// single-base RoPE, no QK-norm, no bias, untied head — with two FUSED tensors that
// buildPhi3Weights splits at load (qkv_proj → q/k/v, gate_up_proj → gate/up) so the
// generic forward runs unchanged. Partial rotary (partial_rotary_factor) is supported via
// RotaryDim; LongRoPE (the 128k variants' su/longrope scaling) is deferred — a checkpoint
// carrying it fails loudly at parseRopeScaling. Phi-4 (14B) and Phi-3-mini-4k use full
// rotary + no scaling. The tensor schema is the phi3TensorSchema marker.
func phi3Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	hd := cfg.headDim()
	// RoPE: transformers ≥5.11 saves the flat rope_parameters {rope_theta,
	// partial_rotary_factor, rope_type}; the released Phi-4/Phi-3 configs use a
	// top-level rope_theta (+ rope_scaling for the 128k longrope variants). Accept both.
	var base float64
	var scaling *ropeScaling
	rotaryDim := cfg.rotaryDim()
	if len(cfg.RopeParameters) > 0 {
		spec, partial, err := parseRopeFlat(cfg.RopeParameters)
		if err != nil {
			return nil, nil, fmt.Errorf("decoder(phi3): %w", err)
		}
		base, scaling = spec.base, spec.scaling
		if partial > 0 && partial < 1 {
			rotaryDim = int(partial * float64(hd))
		}
	} else {
		base = cfg.RoPEGlobalBase
		sc, err := parseRopeScaling(cfg.RopeScaling) // nil for Phi-4/4k; longrope errors (deferred)
		if err != nil {
			return nil, nil, fmt.Errorf("decoder(phi3): %w", err)
		}
		scaling = sc
	}
	if base <= 0 {
		return nil, nil, fmt.Errorf("decoder(phi3): rope_theta must be >0, got %v", base)
	}
	// Sliding-window attention. Phi-3-mini-4k's config.json declares sliding_window: 2047
	// (Phi-4 omits it, and the released GGUF conversions drop the key entirely ⇒ 0 = full
	// attention, which is what that file honestly declares). All layers are local when a
	// window is set, mirroring mistralArchitecture. layerIsGlobal MUST be set alongside
	// SlidingWindow: isGlobalLayer defaults to TRUE (global) when the hook is nil, which
	// would leave the window silently inert.
	var layerIsGlobal func(int) bool
	if cfg.SlidingWindow > 0 {
		layerIsGlobal = func(int) bool { return false }
	}
	return &Architecture{
		Name:            "phi3",
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
		QKVBias:         false,
		QKNorm:          false,
		AttnScale:       math.Pow(float64(hd), -0.5),
		RoPELocalBase:   base,
		RoPEGlobalBase:  base,
		RotaryDim:       rotaryDim,         // partial_rotary_factor · head_dim; 0 = full
		SlidingWindow:   cfg.SlidingWindow, // 0 ⇒ full attention (Phi-4, and the GGUFs)
		layerIsGlobal:   layerIsGlobal,     // all-local when windowed
		ropeScaling:     scaling,
		EmbedScale:      0,
		TiedLMHead:      false, // finalized from lm_head.weight presence at load
	}, &phi3TensorSchema, nil
}
