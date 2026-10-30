package decoder

import (
	"encoding/json"
	"fmt"
	"math"
)

// deepseekArchitecture expresses DeepSeek-V2/V3 (model_type deepseek_v2 / deepseek_v3):
// Multi-head Latent Attention over a DeepSeekMoE FFN. MLA is the new coverage axis —
// latent-KV attention: K/V compress to a shared low-rank latent (kv_lora_rank, the only
// thing cached), per-head K/V are reconstructed via kv_b_proj, and a separate
// qk_rope_head_dim key (shared across heads) carries decoupled RoPE; queries optionally
// route through a q_lora_rank bottleneck. The per-head dims are asymmetric
// (qk_head_dim = qk_nope+qk_rope ≠ v_head_dim), so it runs forward_deepseek.go rather than
// the generic attention. The MoE reuses moeMLP; routing flavor is config-driven —
// V3 scores experts with sigmoid + an e_score_correction_bias and limits selection to
// topk_group of n_group groups (noaux_tc), while V2/V2-Lite use softmax + plain greedy
// top-k. first_k_dense_replace dense-MLP prefix, ungated shared expert (the GLM path).
func deepseekArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateDeepseek(); err != nil {
		return nil, nil, err
	}
	// RoPE: the tiny synthetic carries rope_parameters {rope_theta, rope_type}; the real
	// V2-Lite/V3 carry a top-level rope_theta + a rope_scaling yarn object. Accept both.
	var base float64
	var scaling *ropeScaling
	if len(cfg.RopeParameters) > 0 {
		spec, _, err := parseRopeFlat(cfg.RopeParameters)
		if err != nil {
			return nil, nil, fmt.Errorf("decoder(%s): %w", cfg.ModelType, err)
		}
		base, scaling = spec.base, spec.scaling
	} else {
		base = cfg.RoPEGlobalBase
		sc, err := parseRopeScaling(cfg.RopeScaling)
		if err != nil {
			return nil, nil, fmt.Errorf("decoder(%s): %w", cfg.ModelType, err)
		}
		scaling = sc
	}
	// DeepSeek's YaRN attention_factor (the cos/sin mscale) is NOT the generic
	// 0.1·ln(factor)+1 — transformers computes it as the RATIO
	// get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim). When the config's
	// mscale == mscale_all_dim (V2-Lite: both 0.707) the ratio is exactly 1.0, so cos/sin
	// are unscaled. (The attention softmax scale stays qk_head_dim^-0.5 — transformers
	// 5.12 does NOT fold mscale² into it, unlike the old standalone modeling_deepseek.)
	// Override the generic yarn mscale parseRopeScaling computed.
	if scaling != nil && scaling.kind == ropeScaleYarn {
		raw := cfg.RopeScaling
		if len(cfg.RopeParameters) > 0 {
			raw = cfg.RopeParameters
		}
		var y struct {
			Factor       float64  `json:"factor"`
			Mscale       *float64 `json:"mscale"`
			MscaleAllDim *float64 `json:"mscale_all_dim"`
		}
		_ = json.Unmarshal(raw, &y)
		if y.Mscale != nil || y.MscaleAllDim != nil {
			m, mAll := 1.0, 1.0
			if y.Mscale != nil {
				m = *y.Mscale
			}
			if y.MscaleAllDim != nil {
				mAll = *y.MscaleAllDim
			}
			scaling.mscale = yarnGetMscale(y.Factor, m) / yarnGetMscale(y.Factor, mAll)
		}
	}

	// Routing flavor. V3 (noaux_tc) scores with sigmoid + e_score_correction_bias and
	// limits to topk_group groups; V2 scores with softmax. Honor explicit scoring_func/
	// topk_method when present, else default by model_type. norm_topk_prob defaults true
	// (HF DeepseekV3Config) but V2-Lite sets it false. Kimi K2 is V3-style (sigmoid +
	// noaux_tc) — its config sets scoring_func="sigmoid", but default it to sigmoid too so
	// a variant that omits the field never silently falls to the softmax (V2) branch.
	sigmoid := cfg.ModelType == "deepseek_v3" || cfg.ModelType == "kimi_k2"
	if cfg.ScoringFunc != "" {
		sigmoid = cfg.ScoringFunc == "sigmoid"
	}
	normTopK := true
	if cfg.NormTopKProb != nil {
		normTopK = *cfg.NormTopKProb
	}
	scale := cfg.RoutedScalingFactor // 0/1 ⇒ no-op (V2-Lite is 1.0)

	qk := cfg.QKNopeHeadDim + cfg.QKRopeHeadDim
	ropeInterleave := true // DeepseekV3Config default
	if cfg.RopeInterleave != nil {
		ropeInterleave = *cfg.RopeInterleave
	}
	return &Architecture{
		Name:            cfg.ModelType,
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumHeads, // MLA reconstructs per-head K/V; the cache is the shared latent
		HeadDim:         qk,           // q·k dot-product width (≠ v_head_dim); forward_deepseek uses mlaParams
		IntermediateDim: cfg.IntermediateDim,
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		RMSAddOne:       false,
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   NormPre2,
		Act:             ActSiLU,
		QKVBias:         false,
		QKNorm:          false,
		FirstKDense:     cfg.FirstKDenseReplace,
		MoE: &MoEConfig{
			NumExperts:            cfg.NRoutedExperts,
			TopK:                  cfg.NumExpertsPerTok,
			NormTopKProb:          normTopK,
			IntermediateDim:       cfg.MoeIntermediateSize,
			SharedIntermediateDim: cfg.NSharedExperts * cfg.MoeIntermediateSize,
			RouterSigmoid:         sigmoid,
			RoutedScale:           scale,
			SharedUngated:         true,
			NGroup:                cfg.NGroup,
			TopkGroup:             cfg.TopkGroup,
		},
		// Plain qk_head_dim^-0.5: transformers (5.12) does not fold YaRN's mscale_all_dim² into the attention scale, unlike the old standalone
		// modeling_deepseek. The real-model gates agree: the deepseek_v2/deepseek_v3 parity records (testdata/parity) validate against HF bf16
		// oracles (DeepSeek-V2-Lite, Moonlight-16B-A3B) at cosine_min 0.999+ with this scale, which a ~2x softmax error from a missing fold would
		// not.
		AttnScale:      math.Pow(float64(qk), -0.5),
		RoPELocalBase:  base,
		RoPEGlobalBase: base,
		ropeScaling:    scaling,
		RotaryDim:      cfg.QKRopeHeadDim, // RoPE rotates only the rope-carrying dims
		EmbedScale:     0,
		TiedLMHead:     false, // finalized from lm_head.weight presence at load
		mla: &mlaParams{
			QLoRARank: cfg.QLoRARank, KVLoRARank: cfg.KVLoRARank,
			QKNopeHeadDim: cfg.QKNopeHeadDim, QKRopeHeadDim: cfg.QKRopeHeadDim,
			VHeadDim: cfg.VHeadDim, ropeInterleave: ropeInterleave,
		},
	}, &deepseekTensorSchema, nil
}
