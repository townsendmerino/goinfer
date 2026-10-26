package decoder

import (
	"fmt"
	"math"
)

// lagunaArchitecture expresses Laguna (poolside; model_type "laguna") across all
// three released generations — Laguna-XS-2.1, Laguna-XS.2, Laguna-M.1 — from ONE
// adapter. The vendor's modeling_laguna.py is byte-identical between generations
// (only import paths differ), so every generational difference is config, and this
// reads them all rather than branching on a version.
//
// The vendor's own summary is that Laguna attention "is identical to Qwen2MoE
// attention except": no QKV bias, an explicit head_dim, per-layer sliding window,
// and output gating before o_proj. That last one is the only genuinely new
// primitive; everything else composes shipped parts:
//
//   - Router: sigmoid scoring + e_score_correction_bias steering the SELECTION only
//   - norm_topk_prob + routed scaling — goinfer's DeepSeek/GLM MoE path exactly.
//   - Shared expert: added with NO outer sigmoid gate ⇒ SharedUngated. (It IS a
//     gated SwiGLU internally; SharedUngated refers to Qwen2-MoE's extra
//     sigmoid(shared_gate·h) scaling of the shared branch, which Laguna lacks.)
//   - Dense prefix: mlp_only_layers is a contiguous prefix ⇒ FirstKDense.
//   - RoPE keyed by layer type: full_attention is YaRN at theta 500000 with partial
//     rotary 0.5; sliding_attention is plain at theta 10000, full rotary. That is
//     RoPEGlobalBase/RoPELocalBase + ropeScaling/ropeScalingLocal (Mellum already
//     splits base and scaling this way) plus RotaryDimLocal for the width.
//   - M.1 drops layer_types, sliding_window, and num_attention_heads_per_layer
//     entirely: it is all-full-attention with uniform heads, so the local/sliding
//     half of each of those pairs simply never engages.
//
// Attention sinks are NOT enabled in any released checkpoint
// (swa_attention_sink_enabled absent), so that branch of the vendor code has no
// counterpart here. See docs/completed/task-laguna.md for the config verification.
func lagunaArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateLaguna(); err != nil {
		return nil, nil, err
	}
	gateOn, gatePerHead, err := cfg.lagunaGatePerHead()
	if err != nil {
		return nil, nil, err
	}
	firstKDense, err := cfg.lagunaFirstKDense()
	if err != nil {
		return nil, nil, err
	}
	full, sliding, err := parseRopeParameters(cfg.RopeParameters)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(laguna): %w", err)
	}
	if full == nil {
		return nil, nil, fmt.Errorf("decoder(laguna): rope_parameters.full_attention is required")
	}
	hd := cfg.headDim()

	// partial_rotary_factor precedence. It appears BOTH top-level and inside each
	// rope_parameters[layer_type], and the vendor documents that HF's
	// standardize_rope_params "unconditionally overwrites
	// rope_parameters['partial_rotary_factor'] with self.partial_rotary_factor" —
	// working around it by aligning the top-level field to the SWA value on a cloned
	// config. The per-layer-type value is therefore the authoritative one; the
	// top-level field is a fallback for checkpoints that carry only it.
	partialOf := func(spec *ropeLayerSpec) float64 {
		if spec != nil && spec.partial > 0 {
			return spec.partial
		}
		return cfg.PartialRotaryFactor
	}
	rotaryOf := func(p float64) int {
		if p > 0 && p < 1 {
			return int(p * float64(hd))
		}
		return 0 // 0 ⇒ full HeadDim
	}

	localBase, localScaling := full.base, full.scaling // M.1: no sliding layers; keep the tables equal
	rotaryDimLocal := 0
	if sliding != nil {
		localBase, localScaling = sliding.base, sliding.scaling
		if rl := rotaryOf(partialOf(sliding)); rl != rotaryOf(partialOf(full)) {
			rotaryDimLocal = rl
			if rotaryDimLocal == 0 {
				rotaryDimLocal = hd // sliding rotates FULL width while global is partial — must be explicit, since 0 means "same as RotaryDim"
			}
		}
	}
	normTopK := true // HF LagunaConfig default
	if cfg.NormTopKProb != nil {
		normTopK = *cfg.NormTopKProb
	}
	var lp *lagunaParams
	if gateOn || len(cfg.NumAttentionHeadsPerLayer) > 0 {
		lp = &lagunaParams{HeadsPerLayer: cfg.NumAttentionHeadsPerLayer, GatePerHead: gatePerHead}
	}
	return &Architecture{
		Name:            "laguna",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         hd,
		IntermediateDim: cfg.IntermediateDim, // dense width, used by the mlp_only_layers prefix
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		RMSAddOne:       false,
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   NormPre2,
		Act:             ActSiLU,
		QKVBias:         false, // "no QKV bias" is in the vendor's own list of departures from Qwen2MoE
		// QK-norm is UNCONDITIONAL in all three released modules — q_norm/k_norm are
		// constructed with no config flag guarding them, and the real XS.2 checkpoint
		// ships model.layers.N.self_attn.{q,k}_norm.weight of shape [head_dim] on every
		// layer. The vendor's prose ("identical to Qwen2MoE attention except …") omits
		// it, so the config is silent and only the checkpoint says so.
		QKNorm:      true,
		FirstKDense: firstKDense,
		MoE: &MoEConfig{
			NumExperts:            cfg.NumExperts,
			TopK:                  cfg.NumExpertsPerTok,
			NormTopKProb:          normTopK,
			IntermediateDim:       cfg.MoeIntermediateSize,
			SharedIntermediateDim: cfg.SharedExpertIntermediateSize,
			RouterSigmoid:         true,
			RoutedScale:           cfg.MoeRoutedScalingFactor,
			SharedUngated:         true,
		},
		AttnScale:        math.Pow(float64(hd), -0.5),
		SlidingWindow:    cfg.SlidingWindow,
		layerIsGlobal:    cfg.IsGlobalLayer, // from layer_types; absent ⇒ all-global (M.1)
		RoPEGlobalBase:   full.base,
		RoPELocalBase:    localBase,
		ropeScaling:      full.scaling,
		ropeScalingLocal: localScaling,
		RotaryDim:        rotaryOf(partialOf(full)),
		RotaryDimLocal:   rotaryDimLocal,
		EmbedScale:       0,
		TiedLMHead:       false, // finalized from lm_head.weight presence at load
		laguna:           lp,
	}, &lagunaTensorSchema, nil
}
