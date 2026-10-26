package decoder

import (
	"fmt"
	"math"
)

// graniteDenseArchitecture expresses dense Granite 4.2 (model_type "granite", GraniteForCausalLM): a plain llama skeleton (tensor names
// byte-identical to llama's, so llamaTensorSchema is reused) plus Granite's four scalar multipliers, three of them generic on Architecture
// (embedding_multiplier → EmbedScale, attention_multiplier → AttnScale in place of 1/√d, logits_scaling → LogitScale). residual_multiplier
// is the exception: granitemoehybrid's own forward applies it via graniteParams.ResidMul, and the generic uniform-layer forward this family
// rides has no such hook. All three released sizes ship residual_multiplier 1.0 (identity), so validateGraniteDense rejects anything else
// rather than silently dropping it, the discipline validateLlama applies to attention bias.
func graniteDenseArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.validateGraniteDense(); err != nil {
		return nil, nil, err
	}
	// RoPE: the released checkpoints carry both a nested rope_parameters and a redundant
	// top-level rope_theta (transformers 4.57.1); accept either, same as granitemoehybrid.
	base, scaling, err := ropeBaseFlatOrNested(cfg, "granite")
	if err != nil {
		return nil, nil, err
	}
	if base <= 0 {
		return nil, nil, fmt.Errorf("decoder(granite): rope_theta must be >0, got %v", base)
	}
	hd := cfg.headDim()
	attnMul := cfg.AttentionMultiplier
	if attnMul == 0 {
		attnMul = math.Pow(float64(hd), -0.5) // HF default when unset
	}
	logitScale := cfg.LogitsScaling
	if logitScale == 0 {
		logitScale = 1
	}
	embedScale := cfg.EmbeddingMultiplier
	if embedScale == 0 {
		embedScale = 1
	}
	return &Architecture{
		Name:            "granite",
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
		AttnScale:       attnMul, // Granite attention_multiplier (not 1/√d)
		RoPELocalBase:   base,
		RoPEGlobalBase:  base,
		ropeScaling:     scaling,
		RotaryDim:       0, // full rotary
		EmbedScale:      embedScale,
		LogitScale:      logitScale,
		TiedLMHead:      false, // finalized from lm_head.weight presence at load; all three released sizes are untied
	}, &llamaTensorSchema, nil
}
