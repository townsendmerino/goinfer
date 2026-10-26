package decoder

import (
	"encoding/json"
	"fmt"
	"math"
)

// olmoHybridArchitecture expresses Olmo Hybrid (Ai2, model_type olmo_hybrid): qwen3_5's Gated DeltaNet on 3 of every 4 layers and olmo3's
// own full-attention shape (whole-vector QK-norm) on the rest, but not simply composed side by side: the full-attention layers use olmo3's
// NormPostOnly while the DeltaNet layers use plain NormPre2, two placements in ONE model, which is what NormPlacementLinear exists for
// (docs/completed/task-families-2026-09.md, G2 section). Every departure from a straight qwen3_5-DeltaNet + olmo3-attention composition is a
// parameterization of shared code, not new math, checked against the real modeling_olmo_hybrid.py and a real checkpoint's header:
//   - linear_allow_neg_eigval (default true) doubles the write-gate beta to [0,2) after the sigmoid: NegEigval.
//   - q_proj/k_proj/v_proj are three fully separate tensors (SeparateQKVProj), and the depthwise causal conv is also three separate tensors
//     (q_conv1d/k_conv1d/v_conv1d), split at the q/k/v channel boundaries (SeparateConv). Only the published file settled this: the
//     modeling code shows one combined conv1d, and a locally re-saved checkpoint round-tripped through transformers' conversion mapping
//     produced yet another shape.
//   - The DeltaNet output gated-RMSNorm is named o_norm and its projection o_proj (qwen3_5: norm, out_proj): DeltaNetNormSuffix/
//     DeltaNetOutProjSuffix. Its epsilon is hardcoded 1e-5 in HF's source (FLA's FusedRMSNormGated default) whatever the config says, not the
//     release's rms_norm_eps 1e-6 used elsewhere: ONormEps. Reusing NormEps there is off by 10x on the one norm most sensitive to it.
//   - rope_parameters is {"rope_theta": null} on the release: NO RoPE on any layer ("Released ckpt don't use any ROPE", OlmoHybridModel.__init__),
//     handled by the layerNoPE hook, unconditional here since Olmo 3's nested per-attention-type rope_parameters is not used by this family;
//     there is no partial_rotary_factor either.
//   - MHA, not GQA, on the released size (validated, not assumed).
//
// No MoE variant exists; the FFN is a plain dense SwiGLU on every layer, with the same tensor names as olmo3/llama.
func olmoHybridArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateOlmoHybrid(); err != nil {
		return nil, nil, err
	}
	// rope_parameters on this family is always the FLAT {"rope_theta": ...} shape (never Olmo 3's
	// nested per-attention-type one — there is no sliding_attention layer kind here at all to
	// split on), so a plain pointer-field unmarshal distinguishes "absent/null" (NoPE) from a real
	// value cleanly, mirroring Ministral 3's own llama_4_scaling_beta read.
	var ropeFlat struct {
		RopeTheta *float64 `json:"rope_theta"`
	}
	_ = json.Unmarshal(cfg.RopeParameters, &ropeFlat)
	noRope := ropeFlat.RopeTheta == nil
	var base float64
	var scaling *ropeScaling
	if !noRope {
		base = *ropeFlat.RopeTheta
		if base <= 0 {
			return nil, nil, fmt.Errorf("decoder(olmo_hybrid): rope_theta must be >0, got %v", base)
		}
		sc, err := parseRopeScaling(cfg.RopeScaling)
		if err != nil {
			return nil, nil, fmt.Errorf("decoder(olmo_hybrid): %w", err)
		}
		scaling = sc
	}
	hd := cfg.headDim()
	linearPlacement := NormPre2
	return &Architecture{
		Name:                "olmo_hybrid",
		HiddenDim:           cfg.HiddenDim,
		NumLayers:           cfg.NumLayers,
		NumHeads:            cfg.NumHeads,
		NumKVHeads:          cfg.NumKVHeads,
		HeadDim:             hd,
		IntermediateDim:     cfg.IntermediateDim,
		VocabSize:           cfg.VocabSize,
		Norm:                NormRMS,
		RMSAddOne:           false,
		NormEps:             cfg.RMSNormEps,
		NormPlacement:       NormPostOnly, // full-attention layers (olmo3's own scheme)
		NormPlacementLinear: &linearPlacement,
		Act:                 ActSiLU,
		QKNorm:              true,
		QKNormWhole:         true,
		AttnScale:           math.Pow(float64(hd), -0.5),
		layerIsGlobal:       cfg.IsGlobalLayer,
		layerIsLinear:       cfg.IsLinearLayer,
		layerNoPE:           func(i int) bool { return noRope },
		NoPositionEncoding:  noRope,
		RoPELocalBase:       base,
		RoPEGlobalBase:      base,
		ropeScaling:         scaling,
		ropeScalingLocal:    scaling,
		EmbedScale:          0,
		TiedLMHead:          false, // finalized from lm_head.weight presence at load
		qwen35: &qwen35Params{
			ConvKernel:            cfg.LinearConvKernelDim,
			KeyHeadDim:            cfg.LinearKeyHeadDim,
			ValueHeadDim:          cfg.LinearValueHeadDim,
			NumKeyHeads:           cfg.LinearNumKeyHeads,
			NumValueHeads:         cfg.LinearNumValueHeads,
			AttnGate:              false, // Olmo Hybrid's full-attention layer is olmo3's plain scheme — see AttnGate's own comment
			SeparateQKVProj:       true,
			SeparateConv:          true,
			NegEigval:             cfg.LinearAllowNegEigval,
			DeltaNetNormSuffix:    "linear_attn.o_norm.weight",
			DeltaNetOutProjSuffix: "linear_attn.o_proj.weight",
			ONormEps:              1e-5,
			PlainFullAttn:         true,
		},
	}, &olmoHybridTensorSchema, nil
}
