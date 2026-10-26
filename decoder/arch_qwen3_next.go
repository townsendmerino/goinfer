package decoder

import (
	"fmt"
	"math"
)

// qwen3NextArchitecture expresses Qwen3-Next (model_type qwen3_next): the same Gated-DeltaNet/softmax/MoE hybrid as qwen3_5_moe (DeltaNet
// geometry, MoE dims, QK-norm and the Gemma-style (1+w) RMSNorm all match modular_qwen3_next.py), with two config-SHAPE deltas, not just
// value changes:
//
//  1. No layer_types field: the per-layer pattern is COMPUTED from full_attention_interval (normalizeQwen3NextLayerTypes, called below,
//     before validateQwen35 can require LayerTypes to be populated).
//  2. partial_rotary_factor is a TOP-LEVEL field, not nested inside rope_parameters, and the released config has no rope_parameters object
//     at all (top-level rope_theta + rope_scaling). So this mirrors deepseekArchitecture's dual-path RoPE resolution (nested if present, flat
//     fields otherwise) rather than qwen35Architecture's nested-only parseRopeFlat, which hard-errors on an empty rope_parameters
//     (parseRopeSpec refuses it).
func qwen3NextArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.normalizeQwen3NextLayerTypes(); err != nil {
		return nil, nil, err
	}
	if err := cfg.validateQwen3Next(); err != nil {
		return nil, nil, err
	}
	var base float64
	var scaling *ropeScaling
	var partialRotary float64
	if len(cfg.RopeParameters) > 0 {
		spec, pr, err := parseRopeFlat(cfg.RopeParameters)
		if err != nil {
			return nil, nil, fmt.Errorf("decoder(qwen3_next): %w", err)
		}
		base, scaling, partialRotary = spec.base, spec.scaling, pr
	} else {
		base = cfg.RoPEGlobalBase
		sc, err := parseRopeScaling(cfg.RopeScaling)
		if err != nil {
			return nil, nil, fmt.Errorf("decoder(qwen3_next): %w", err)
		}
		scaling = sc
		partialRotary = cfg.PartialRotaryFactor
	}
	rotaryDim := 0
	if partialRotary > 0 && partialRotary < 1 {
		rotaryDim = int(partialRotary * float64(cfg.HeadDim))
	}
	normTopK := true
	if cfg.NormTopKProb != nil {
		normTopK = *cfg.NormTopKProb
	}
	return &Architecture{
		Name:            "qwen3_next",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         cfg.HeadDim,
		IntermediateDim: cfg.IntermediateDim,
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		RMSAddOne:       true, // Qwen3NextRMSNorm(Gemma3RMSNorm): pass — inherits Gemma's (1+w), verified against modular_qwen3_next.py
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   NormPre2,
		Act:             ActSiLU,
		MoE: &MoEConfig{
			NumExperts:            cfg.NumExperts,
			TopK:                  cfg.NumExpertsPerTok,
			NormTopKProb:          normTopK,
			IntermediateDim:       cfg.MoeIntermediateSize,
			SharedIntermediateDim: cfg.SharedExpertIntermediateSize,
		},
		QKNorm:           true,
		AttnScale:        math.Pow(float64(cfg.HeadDim), -0.5),
		layerIsGlobal:    cfg.IsGlobalLayer,
		layerIsLinear:    cfg.IsLinearLayer,
		RoPELocalBase:    base,
		RoPEGlobalBase:   base,
		ropeScaling:      scaling,
		ropeScalingLocal: scaling,
		RotaryDim:        rotaryDim,
		EmbedScale:       0,
		TiedLMHead:       false,
		qwen35: &qwen35Params{
			ConvKernel:        cfg.LinearConvKernelDim,
			KeyHeadDim:        cfg.LinearKeyHeadDim,
			ValueHeadDim:      cfg.LinearValueHeadDim,
			NumKeyHeads:       cfg.LinearNumKeyHeads,
			NumValueHeads:     cfg.LinearNumValueHeads,
			AttnGate:          true, // qwen3_next's own double-width [query‖gate] q_proj, inherited from qwen3.5
			FusedDeltaNetProj: true, // in_proj_qkvz/in_proj_ba, not four separate tensors — see qwen35Params doc
		},
	}, &qwen35TensorSchema, nil
}
