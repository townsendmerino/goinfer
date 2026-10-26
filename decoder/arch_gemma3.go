package decoder

import (
	"fmt"
	"math"
)

// gemma3Architecture expresses Gemma 3 as a descriptor: RMSNorm with (1+w),
// the 4-norm sandwich, GeGLU, QK-norm, the query_pre_attn_scalar attention
// scale, dual-base RoPE (local/global per layer), √hidden embedding scale, and
// a tied LM head with no soft-capping. ValidateAssumptions pins the bits this
// forward pass can't vary (Gemma-2 soft-capping, unsupported activation).
func gemma3Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	// transformers ≥5.10 carries the dual RoPE bases in rope_parameters
	// ({full_attention, sliding_attention}.rope_theta) instead of the flat
	// rope_theta / rope_local_base_freq that gemma-3-270m uses (and the VL configs
	// nest it under text_config, flattened by loadConfig). Backfill the flat fields
	// from rope_parameters when absent so the adapter + ValidateAssumptions see
	// populated bases either format.
	if cfg.RoPEGlobalBase == 0 && len(cfg.RopeParameters) > 0 {
		full, sliding, err := parseRopeParameters(cfg.RopeParameters)
		if err != nil {
			return nil, nil, fmt.Errorf("decoder(gemma3): rope_parameters: %w", err)
		}
		if full != nil {
			cfg.RoPEGlobalBase = full.base
		}
		if sliding != nil {
			cfg.RoPELocalBase = sliding.base
		}
	}
	// Gemma3TextConfig class defaults. A VL checkpoint's nested text_config (e.g.
	// gemma-3-4b-it) carries only the dims that differ from the transformers
	// Gemma3TextConfig defaults — hidden/intermediate/layers/sliding_window — and
	// OMITS the rest, which HF fills from the class. Mirror that here: backfill
	// each scalar only when absent, so a flat config that sets them explicitly
	// (gemma-3-270m sets heads=4) is untouched, while the minimal VL text_config
	// loads with the correct 4B/12B/27B head + rope geometry. (Sizes that differ
	// from the defaults, like 270m's 4 heads, always set them in their config.)
	defScalarI := func(p *int, def int) {
		if *p == 0 {
			*p = def
		}
	}
	defScalarF := func(p *float64, def float64) {
		if *p == 0 {
			*p = def
		}
	}
	defScalarI(&cfg.NumHeads, 8)
	defScalarI(&cfg.NumKVHeads, 4)
	defScalarI(&cfg.HeadDim, 256)
	defScalarI(&cfg.VocabSize, 262208)
	defScalarI(&cfg.SlidingWindowPattern, 6)
	defScalarF(&cfg.QueryPreAttnScalar, 256)
	defScalarF(&cfg.RMSNormEps, 1e-6)
	defScalarF(&cfg.RoPEGlobalBase, 1_000_000)
	defScalarF(&cfg.RoPELocalBase, 10_000)
	if err := cfg.ValidateAssumptions(); err != nil {
		return nil, nil, err
	}
	return &Architecture{
		Name:              "gemma3",
		HiddenDim:         cfg.HiddenDim,
		NumLayers:         cfg.NumLayers,
		NumHeads:          cfg.NumHeads,
		NumKVHeads:        cfg.NumKVHeads,
		HeadDim:           cfg.HeadDim,
		IntermediateDim:   cfg.IntermediateDim,
		VocabSize:         cfg.VocabSize,
		Norm:              NormRMS,
		RMSAddOne:         true,
		NormEps:           cfg.RMSNormEps,
		NormPlacement:     NormSandwich4,
		Act:               ActGeluTanh,
		QKNorm:            true,
		AttnScale:         math.Pow(cfg.QueryPreAttnScalar, -0.5),
		SlidingWindow:     cfg.SlidingWindow,
		layerIsGlobal:     cfg.IsGlobalLayer,
		RoPELocalBase:     cfg.RoPELocalBase,
		RoPEGlobalBase:    cfg.RoPEGlobalBase,
		EmbedScale:        math.Sqrt(float64(cfg.HiddenDim)),
		TiedLMHead:        true,
		FinalLogitSoftcap: cfg.FinalLogitSoftcap, // 0 (ValidateAssumptions rejects nonzero)
		AttnLogitSoftcap:  cfg.AttnLogitSoftcap,
	}, &gemma3TensorSchema, nil
}
