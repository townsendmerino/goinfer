package decoder

import (
	"fmt"
	"math"
)

// flintArchitecture expresses Flint (RecursiveSI/flint; model_type "flint", GGUF
// general.architecture "flint"): a hybrid sliding-window + NoPE full-attention
// MoE decoder (53B total / ~3B active on the released 49-layer checkpoint).
//
// Every Flint mechanism composes shipped generic-path primitives:
//
//   - Attention: GQA (32 query heads, 4 KV heads, head_dim 128 on the release),
//     per-head QK RMSNorm (self_attn.q_norm / self_attn.k_norm; GGUF
//     blk.N.attn_q_norm.weight / attn_k_norm.weight), no QKV/O bias, and a learned
//     per-query-head attention sink logit (self_attn.sinks, or
//     self_attn.softmax_offset; GGUF blk.N.attn_sinks.weight) in the softmax
//     denominator on every layer (AttnSinks: true -> LayerWeights.AttnSinks).
//   - Hybrid attention schedule + NoPE: layer_types alternates sliding_attention
//     (window 513 on the release, standard NeoX split-half RoPE at theta 10000)
//     and full_attention (global causal, NoPE — no positional encoding at all,
//     matching FlintModel.forward's layer_type == "full_attention" ->
//     cos=ones, sin=zeros identity rotation). layerNoPE == layerIsGlobal.
//   - Dense prefix + MoE: layers in mlp_only_layers ([0, 1] on the release)
//     are plain SwiGLU MLPs at intermediate_size (FirstKDense); layers >=
//     FirstKDense are sigmoid-routed top-k MoE (256 experts, top-6,
//     norm_topk_prob=true, no expert bias) at moe_intermediate_size plus an
//     always-on ungated SwiGLU shared expert at shared_expert_intermediate_size
//     (SharedUngated: true).
func flintArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := backfillFlatRope(cfg, "flint"); err != nil {
		return nil, nil, err
	}
	if err := cfg.ValidateFlint(); err != nil {
		return nil, nil, err
	}
	firstKDense, err := cfg.LagunaFirstKDense()
	if err != nil {
		return nil, nil, err
	}
	scaling, err := parseRopeScaling(cfg.RopeScaling)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(flint): %w", err)
	}
	hd := cfg.EffectiveHeadDim()
	normTopK := true
	if cfg.NormTopKProb != nil {
		normTopK = *cfg.NormTopKProb
	}
	routedScale := cfg.RoutedScalingFactor
	if routedScale == 0 && cfg.MoeRoutedScalingFactor > 0 {
		routedScale = cfg.MoeRoutedScalingFactor
	}
	if routedScale == 0 {
		routedScale = 1.0
	}
	layerTypes := append([]string(nil), cfg.LayerTypes...)
	isGlobal := func(i int) bool {
		if i >= 0 && i < len(layerTypes) {
			return layerTypes[i] == "full_attention"
		}
		return true
	}
	return &Architecture{
		Name:            "flint",
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
		QKVBias:         cfg.AttentionBias,
		QKNorm:          true,
		AttnSinks:       true,
		FirstKDense:     firstKDense,
		MoE: &MoEConfig{
			NumExperts:            cfg.NumExperts,
			TopK:                  cfg.NumExpertsPerTok,
			NormTopKProb:          normTopK,
			IntermediateDim:       cfg.MoeIntermediateSize,
			SharedIntermediateDim: cfg.SharedExpertIntermediateSize,
			RouterSigmoid:         true,
			RoutedScale:           routedScale,
			SharedUngated:         true,
		},
		AttnScale:        math.Pow(float64(hd), -0.5),
		SlidingWindow:    cfg.SlidingWindow,
		layerIsGlobal:    isGlobal,
		layerNoPE:        isGlobal, // full_attention layers are NoPE; sliding_attention layers apply RoPE
		RoPELocalBase:    cfg.RoPEGlobalBase,
		RoPEGlobalBase:   cfg.RoPEGlobalBase,
		RotaryDim:        cfg.RotaryDim(),
		ropeScaling:      scaling,
		ropeScalingLocal: scaling,
		EmbedScale:       0,
		TiedLMHead:       false, // finalized from lm_head.weight presence at load
	}, &flintTensorSchema, nil
}
