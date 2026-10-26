package decoder

import (
	"fmt"
	"math"
)

// glmOcrArchitecture expresses the GLM-OCR TEXT decoder (zai-org/GLM-OCR; model_type
// "glm_ocr_text", reached as "glm_ocr" through the VL wrapper's flattened text_config — loadConfig
// merges text_config into the Config and lets the top-level keys win). Built from existing parts
// and read against the generated modeling_glm_ocr.py and the checkpoint's own tensor shapes
// (docs/tasks/task-glm-ocr-2026-10.md, O0):
//
//   - GQA, no bias on q/k/v/o, and a head_dim the config states EXPLICITLY (128 with hidden 1536 and
//     16 heads: hidden/heads would be 96 — the adapter reads head_dim, never derives it past a
//     missing key). AttnScale is head_dim^-0.5 for the same reason;
//   - plain RMSNorm (weight*x, eps 1e-5, RMSAddOne false) in the Sandwich4 placement: input norm,
//     attention, a norm on the attention OUTPUT, residual add, pre-MLP norm, MLP, a norm on the MLP
//     OUTPUT, residual add. The TENSOR NAMES are not Gemma's (see glmOcrTensorSchema);
//   - fused mlp.gate_up_proj, gate rows first, split at load (buildGlmOcrWeights);
//   - rotary theta 1e4, full width (partial_rotary_factor 1.0), PAIRWISE rotation (ropeInterleave)
//     with m-RoPE sections [16 24 24] laid out as contiguous t/h/w blocks (MRopeInterleaved false):
//     applyMRoPEPairwise. For text tokens every position component is equal, so the whole stack
//     reduces to plain pairwise RoPE;
//   - an UNTIED lm_head; the multi-token-prediction layer (layers.<NumLayers>) is never loaded.
//
// Every Architecture field is set below on purpose, including the ones left at a legal zero:
// validateResolved exists because an omitted field in a hand-built struct literal loads clean and
// computes something else.
func glmOcrArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	hd := cfg.headDim()
	if err := cfg.validateGlmOcr(hd); err != nil {
		return nil, nil, err
	}
	spec, partial, err := parseRopeFlat(cfg.RopeParameters)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(glm_ocr): %w", err)
	}
	if spec.scaling != nil {
		return nil, nil, fmt.Errorf("decoder(glm_ocr): rope_parameters.rope_type scaling is not supported (default only)")
	}
	if partial != 0 && partial != 1 {
		return nil, nil, fmt.Errorf("decoder(glm_ocr): partial_rotary_factor=%v unsupported (1.0 only: the m-RoPE sections are checked against the full head_dim/2)", partial)
	}
	section, interleaved := parseMRopeFlat(cfg.RopeParameters)
	if section == nil {
		return nil, nil, fmt.Errorf("decoder(glm_ocr): rope_parameters.mrope_section (3 entries) is required")
	}
	if interleaved {
		return nil, nil, fmt.Errorf("decoder(glm_ocr): mrope_interleaved=true unsupported (GLM-4V's m-RoPE is the contiguous layout)")
	}
	if sum := section[0] + section[1] + section[2]; sum != hd/2 {
		return nil, nil, fmt.Errorf("decoder(glm_ocr): mrope_section %v sums to %d, want head_dim/2 = %d", section, sum, hd/2)
	}
	cfg.MRopeSection = section
	return &Architecture{
		Name:             "glm_ocr",
		HiddenDim:        cfg.HiddenDim,
		NumLayers:        cfg.NumLayers, // num_hidden_layers excludes the MTP layer at index NumLayers
		NumHeads:         cfg.NumHeads,
		NumKVHeads:       cfg.NumKVHeads,
		HeadDim:          hd,
		IntermediateDim:  cfg.IntermediateDim,
		VocabSize:        cfg.VocabSize,
		MaxPositions:     0, // RoPE family: no learned position table
		Norm:             NormRMS,
		RMSAddOne:        false, // GlmOcrRMSNorm is weight * x (no 1+w)
		NormEps:          cfg.RMSNormEps,
		NormPlacement:    NormSandwich4,
		Act:              ActSiLU,
		NonGatedMLP:      false,
		MoE:              nil,
		FirstKDense:      0,
		QKVBias:          false, // GlmOcrTextAttention: bias=False on q/k/v/o
		OutBias:          false,
		QKNorm:           false, // the text attention has no q/k norm (the tower does)
		QKNormWhole:      false,
		LearnedPosEmbed:  false,
		AttnScale:        math.Pow(float64(hd), -0.5), // head_dim^-0.5 with head_dim 128, not hidden/heads
		AttnTempBeta:     0,
		SlidingWindow:    0, // full attention on every layer
		layerIsGlobal:    nil,
		layerNoPE:        nil,
		RoPELocalBase:    spec.base,
		RoPEGlobalBase:   spec.base,
		RotaryDim:        hd, // partial_rotary_factor 1.0: all 128 dims rotate
		MRopeSection:     section,
		MRopeInterleaved: false, // contiguous t/h/w blocks (chunk[i%3] over split(mrope_section))
		ropeScaling:      nil,
		ropeInterleave:   true, // rotate_half_llm: pair (2d, 2d+1) shares frequency d
		EmbedScale:       0,
		TiedLMHead:       false, // finalized from lm_head.weight presence at load; the checkpoint ships one
	}, &glmOcrTensorSchema, nil
}

// validateGlmOcr rejects a config the glm_ocr descriptor does not implement, instead of loading it
// and computing something else. hd is the resolved head_dim.
func (c *Config) validateGlmOcr(hd int) error {
	switch {
	case c.HiddenDim == 0 || c.NumLayers == 0 || c.NumHeads == 0 || hd == 0:
		return fmt.Errorf("decoder(glm_ocr): missing required dim (hidden=%d layers=%d heads=%d headDim=%d)",
			c.HiddenDim, c.NumLayers, c.NumHeads, hd)
	case c.NumKVHeads == 0 || c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("decoder(glm_ocr): num_heads %d not a multiple of num_kv_heads %d (GQA)", c.NumHeads, c.NumKVHeads)
	case hd%2 != 0:
		return fmt.Errorf("decoder(glm_ocr): head_dim %d is odd (pairwise rotation needs pairs)", hd)
	case c.VocabSize == 0:
		return fmt.Errorf("decoder(glm_ocr): vocab_size is zero")
	case c.IntermediateDim == 0:
		return fmt.Errorf("decoder(glm_ocr): intermediate_size is zero")
	case c.HiddenAct != "" && c.HiddenAct != "silu":
		return fmt.Errorf("decoder(glm_ocr): hidden_act=%q unsupported (silu/SwiGLU only)", c.HiddenAct)
	case c.RMSNormEps <= 0:
		return fmt.Errorf("decoder(glm_ocr): rms_norm_eps must be >0, got %v", c.RMSNormEps)
	case c.AttentionBias:
		return fmt.Errorf("decoder(glm_ocr): attention_bias=true unsupported (GlmOcrTextAttention is bias-free)")
	case len(c.RopeParameters) == 0:
		return fmt.Errorf("decoder(glm_ocr): rope_parameters {rope_theta, mrope_section} is required")
	}
	return nil
}
