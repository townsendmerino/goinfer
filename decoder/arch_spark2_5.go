package decoder

import (
	"fmt"
	"math"
)

// spark25Architecture expresses Spark-X2.5 (model_type spark2_5; XHToken/Spark-X2.5-{1.7B,4B}). Two claims of the survey-level scoping
// (docs/tasks/task-spark-x2-5.md) did not survive reading the real configuration_spark.py / modeling_spark.py (trust_remote_code):
//
//   - It is not a parallel block (Command-R): the decoder layer is the standard sequential residual, the same shape as Llama/Qwen
//     (NormPre2).
//   - It is not a non-gated GELU MLP: Spark2_5MLP is gated (down(gelu(gate(x))*up(x))); the only departure from Llama's MLP is the
//     activation, exact erf GELU (HF's "gelu"), inside that gated shape (gegluExact in mlp.go).
//
// What is real, confirmed against source:
//   - Fused QKV: one q_k_v_proj Linear(hidden, qDim+2*kvDim), split Q‖K‖V by output rows in that order (buildSpark25Weights, modeled on
//     buildPhi3Weights's split).
//   - A head-wise SIGMOID attention-output gate (g_proj: Linear(hidden, numHeads)) applied to attn_output before out_proj: Laguna's g_proj
//     structure with a different activation, through AttnGate/GateSigmoid (arch.go) in the generic (non-MLA) attention path.
//   - Per-layer-type RoPE: full_attention layers partial_rotary_factor 0.25 (rotary_dim = head_dim/4) at theta 5e6, sliding_attention
//     layers full rotation at theta 1e4: Laguna's RotaryDim/RotaryDimLocal/RoPEGlobalBase/RoPELocalBase mechanism, reused verbatim.
//   - 1:3 sliding:full interleave (sliding_window 512), read from layer_types via cfg.IsGlobalLayer.
//   - No QK-norm, embed scale, logit softcap or attention sink; rotate_half is the standard NeoX half-split (ropeInterleave false).
//   - inv_freq is recomputed per forward in the reference (not a cached persistent=False buffer), so there is no internlm2-class
//     fast-init corruption risk on this family.
func spark25Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	hd := cfg.headDim()
	if hd != 256 {
		// Not a hard architectural requirement — but every released Spark-X2.5 config sets it,
		// and a silently-different value would mean this doc comment's verified claims about the
		// checkpoint no longer describe the actual model. Loosen if a released config disagrees.
	}
	if !cfg.HeadwiseAttnOutputGate {
		return nil, nil, fmt.Errorf("decoder(spark2_5): headwise_attn_output_gate=false is not implemented (every released config sets it true)")
	}
	if cfg.AttentionBias {
		return nil, nil, fmt.Errorf("decoder(spark2_5): attention_bias=true is not implemented (buildSpark25Weights loads no bias tensors; every released config sets it false)")
	}
	switch cfg.GateAttnActMode {
	case "", "sigmoid":
	default:
		return nil, nil, fmt.Errorf("decoder(spark2_5): gate_attn_act_mode %q unsupported (have: sigmoid — modeling_spark.py also defines \"silu\" but no released checkpoint uses it)", cfg.GateAttnActMode)
	}
	full, sliding, err := parseRopeParameters(cfg.RopeParameters)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(spark2_5): %w", err)
	}
	if full == nil {
		return nil, nil, fmt.Errorf("decoder(spark2_5): rope_parameters.full_attention is required")
	}
	if sliding == nil {
		return nil, nil, fmt.Errorf("decoder(spark2_5): rope_parameters.sliding_attention is required")
	}

	rotaryOf := func(p float64) int {
		if p > 0 && p < 1 {
			return int(p * float64(hd))
		}
		return 0 // 0 ⇒ full HeadDim
	}
	rotaryGlobal := rotaryOf(full.partial)
	rotaryLocal := rotaryOf(sliding.partial)
	if rotaryLocal == rotaryGlobal {
		rotaryLocal = 0 // 0 ⇒ "same as RotaryDim", the shared-width case
	} else if rotaryLocal == 0 {
		rotaryLocal = hd // sliding rotates FULL width while global is partial — must be explicit (see lagunaArchitecture's identical guard)
	}

	return &Architecture{
		Name:            "spark2_5",
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
		NormPlacement:   NormPre2, // sequential, NOT NormParallel — see the doc comment's first correction
		Act:             ActGelu,  // exact erf GELU, inside a GATED (SwiGLU-shaped) MLP — see the doc comment's second correction
		QKVBias:         false,    // guarded above: attention_bias=true rejected, buildSpark25Weights loads no bias tensors
		OutBias:         false,
		QKNorm:          false,
		AttnGate:        GateSigmoid,
		AttnScale:       math.Pow(float64(hd), -0.5),
		SlidingWindow:   cfg.SlidingWindow,
		layerIsGlobal:   cfg.IsGlobalLayer,
		RoPEGlobalBase:  full.base,
		RoPELocalBase:   sliding.base,
		RotaryDim:       rotaryGlobal,
		RotaryDimLocal:  rotaryLocal,
		EmbedScale:      0,
		TiedLMHead:      false, // finalized from lm_head.weight presence at load, same convention as phi3Architecture
	}, &spark25TensorSchema, nil
}
