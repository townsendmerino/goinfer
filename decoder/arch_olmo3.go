package decoder

import (
	"fmt"
	"math"
)

// olmo3Architecture expresses Olmo 3 (model_type olmo3): a softmax-GQA/MHA model with two departures from every other family, both checked
// against the real modeling_olmo3.py:
//
//  1. NormPostOnly: there is NO pre-norm at all. Olmo3DecoderLayer reads the raw residual stream into both self_attn and mlp and normalizes
//     each sublayer's OUTPUT (post_attention_layernorm / post_feedforward_layernorm) before the residual add. Not NormSandwich4 (which
//     normalizes both input and output): a new placement, not a variant.
//  2. QKNormWhole: QK-norm is one RMSNorm over the FULL projected q/k vector (num_heads*head_dim elements), not per head as Qwen3/Gemma3/
//     Mellum do. It reuses rmsNorm with rows/dim swapped (QKNormWhole's comment on Architecture), not new math.
//
// Otherwise plain: MHA on the real release, sliding window on 3 of every 4 layers (layer_types, via Config.IsGlobalLayer as cohere2 does),
// untied head. Tensor names are llama-shaped except the post-only norms and the whole-vector QK-norm weight width (olmo3TensorSchema).
//
// RoPE differs by layer type, and the two tables must not be collapsed: full-attention layers get YaRN (with its mscale), sliding layers
// plain unscaled RoPE, at the same theta, which is the local/global split Mellum implements (RoPELocalBase/RoPEGlobalBase + ropeScaling/
// ropeScalingLocal, dispatched on layerIsGlobal). Olmo3RotaryEmbedding.forward takes an explicit layer_type and returns a different
// (cos, sin) per layer type; an earlier reading of the source as one shared table was wrong. The flat top-level form (the real release's
// on-disk form) is expanded by PretrainedConfig into the same nested split, so both branches agree. Evidence and the retraction:
// docs/code-notes/decoder.md#olmo3Architecture.
func olmo3Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	base := cfg.RoPEGlobalBase
	localBase := base
	var scaling, scalingLocal *ropeScaling
	if len(cfg.RopeParameters) > 0 {
		// NESTED {"full_attention": {...yarn...}, "sliding_attention": {"rope_type": "default", "rope_theta": <same theta>}}: full and sliding are
		// both real and different (see the doc comment). Theta is the same on both layer types, so only the scaling differs; sliding gets none.
		full, sliding, err := parseRopeParameters(cfg.RopeParameters)
		if err != nil {
			return nil, nil, fmt.Errorf("decoder(olmo3): %w", err)
		}
		if full != nil {
			base = full.base
			scaling = full.scaling
		}
		localBase = base
		if sliding != nil {
			localBase = sliding.base
			scalingLocal = sliding.scaling
		}
		cfg.RoPEGlobalBase = base // so validateLlama's rope_theta>0 check sees a populated base
	} else {
		sc, err := parseRopeScaling(cfg.RopeScaling)
		if err != nil {
			return nil, nil, fmt.Errorf("decoder(olmo3): %w", err)
		}
		// The flat top-level form is what PretrainedConfig expands into the SAME nested shape
		// above at construction time (standardize_rope_params) — so a real release loaded this
		// way gets identical full/sliding behavior to the nested branch: YaRN on full-attention
		// layers, plain RoPE at the same theta on sliding-attention layers.
		scaling = sc
	}
	if err := cfg.validateLlama(); err != nil {
		return nil, nil, err
	}
	hd := cfg.headDim()
	layerTypes := append([]string(nil), cfg.LayerTypes...) // snapshot so the closure doesn't retain cfg
	pattern := cfg.SlidingWindowPattern
	isGlobal := func(i int) bool {
		if i >= 0 && i < len(layerTypes) {
			return layerTypes[i] == "full_attention"
		}
		if pattern <= 0 {
			return true
		}
		return (i+1)%pattern == 0
	}
	return &Architecture{
		Name:             "olmo3",
		HiddenDim:        cfg.HiddenDim,
		NumLayers:        cfg.NumLayers,
		NumHeads:         cfg.NumHeads,
		NumKVHeads:       cfg.NumKVHeads,
		HeadDim:          hd,
		IntermediateDim:  cfg.IntermediateDim,
		VocabSize:        cfg.VocabSize,
		Norm:             NormRMS,
		RMSAddOne:        false,
		NormEps:          cfg.RMSNormEps,
		NormPlacement:    NormPostOnly,
		Act:              ActSiLU,
		QKNorm:           true,
		QKNormWhole:      true,
		AttnScale:        math.Pow(float64(hd), -0.5),
		SlidingWindow:    cfg.SlidingWindow,
		layerIsGlobal:    isGlobal,
		RoPELocalBase:    localBase, // sliding-attention layers: same theta, no YaRN scaling
		RoPEGlobalBase:   base,      // full-attention layers: YaRN-scaled
		RotaryDim:        cfg.rotaryDim(),
		ropeScaling:      scaling,
		ropeScalingLocal: scalingLocal,
		EmbedScale:       0,
		TiedLMHead:       false, // finalized from lm_head.weight presence at load
	}, &olmo3TensorSchema, nil
}
