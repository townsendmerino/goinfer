package decoder

import (
	"fmt"
	"math"
)

// nemotronhArchitecture expresses Nemotron-H (model_type nemotron_h): a SINGLE-OP-
// per-block hybrid — each layer is exactly one of mamba / attention / mlp
// (layers_block_type), pre-norm + op + residual, NOT the mixer+FFN block every other
// family uses. The mamba layers reuse the Mamba-2 mixer (mamba2.go); the attention
// layers are NoPE GQA (no RoPE — the SSM layers carry position); the mlp layers are
// non-gated relu². Plain RMSNorm, no multipliers. Dedicated loader + forward
// (buildNemotronWeights; its program is Architecture.nemotronLayerOps).
func nemotronhArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	// Released checkpoints spell the layer sequence as hybrid_override_pattern; only
	// transformers-instantiated configs carry layers_block_type. Normalize to the latter
	// before anything reads it.
	if err := cfg.normalizeNemotronBlocks(); err != nil {
		return nil, nil, err
	}
	// Nemotron-H's config has no num_hidden_layers — the layer count IS the length
	// of layers_block_type.
	if cfg.NumLayers == 0 {
		cfg.NumLayers = len(cfg.LayersBlockType)
	}
	if err := cfg.validateNemotron(); err != nil {
		return nil, nil, err
	}
	if cfg.AttentionHeadDim > 0 {
		cfg.HeadDim = cfg.AttentionHeadDim // NVIDIA's key; without it the width would fall back to hidden/heads, which is not this family's
	}
	hd := cfg.headDim()
	kind := make([]uint8, cfg.NumLayers)
	hasMoE := false
	for i, t := range cfg.LayersBlockType {
		switch t {
		case "attention", "full_attention":
			// "full_attention" is transformers' own canonicalized spelling (configuration_nemotron_h.py's remap_legacy_layer_types/_pattern_to_list): a
			// checkpoint saved through a current transformers carries it, not "attention", even when "attention" was passed in (layers_block_type came
			// back ["linear_attention","moe","full_attention",...] from ["mamba","moe","attention",...]). The real NVIDIA release ships
			// hybrid_override_pattern (parsed independently by normalizeNemotronBlocks) and is unaffected, but any checkpoint re-saved through
			// transformers hits this, so it is a real alias, not a fixture-only workaround.
			kind[i] = nemoAttn
		case "mlp":
			kind[i] = nemoMLP
		case "moe":
			kind[i] = nemoMoE
			hasMoE = true
		case "mamba", "linear_attention": // "linear_attention" is the same canonicalization, for Mamba-2 blocks
			kind[i] = nemoMamba
		default:
			return nil, nil, fmt.Errorf("decoder(nemotron_h): layers_block_type[%d] = %q unrecognized (want mamba/linear_attention, attention/full_attention, mlp, or moe)", i, t)
		}
	}
	// Nemotron 3 Nano's MoE FFN (model_type still "nemotron_h": only the pattern's "E" blocks distinguish it; plain Nemotron-H never sets hasMoE).
	// Routing follows NVIDIA's NemotronHTopkRouter: sigmoid scores + e_score_correction_bias + group-limited top-k (n_group/topk_group) +
	// routed_scaling_factor on the selected weights + an UNGATED additive shared expert; sigmoid is unconditional for this family (no
	// scoring_func key), not config-driven as DeepSeek's is. The expert FFN is non-gated relu² (up_proj/down_proj only), which routeExperts'
	// selection is agnostic to but moeMLP's SwiGLU-only expert evaluator cannot run: see nemotronMoE in forward_nemotron.go, a local function,
	// not a change to the shared moeMLP path.
	var moe *MoEConfig
	if hasMoE {
		normTopK := true
		if cfg.NormTopKProb != nil {
			normTopK = *cfg.NormTopKProb
		}
		moe = &MoEConfig{
			NumExperts:            cfg.NRoutedExperts,
			TopK:                  cfg.NumExpertsPerTok,
			NormTopKProb:          normTopK,
			IntermediateDim:       cfg.MoeIntermediateSize,
			SharedIntermediateDim: cfg.MoeSharedExpertIntermediateSize, // NOT NSharedExperts*MoeIntermediateSize — verified not derivable that way for this family
			RouterSigmoid:         true,                                // unconditional for nemotron_h's router; no scoring_func key exists to check
			RoutedScale:           cfg.RoutedScalingFactor,
			SharedUngated:         true,
			NGroup:                cfg.NGroup,
			TopkGroup:             cfg.TopkGroup,
		}
	}
	return &Architecture{
		Name:            "nemotron_h",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         hd,
		IntermediateDim: cfg.IntermediateDim, // mlp-block width
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		RMSAddOne:       false,
		NormEps:         cfg.LayerNormEpsilon,
		Act:             ActReLU2,
		NonGatedMLP:     true,
		QKVBias:         false,
		QKNorm:          false,
		AttnScale:       math.Pow(float64(hd), -0.5), // NoPE GQA, standard scale
		RotaryDim:       0,                           // no RoPE
		EmbedScale:      0,
		TiedLMHead:      false,
		MoE:             moe, // nil for plain Nemotron-H; set only when the pattern has an "moe" block
		nemotron: &nemotronParams{
			NHeads: cfg.MambaNumHeads, HeadDim: cfg.MambaHeadDim, DState: cfg.SSMStateSize,
			NGroups: cfg.NGroups, DConv: cfg.ConvKernel, blockKind: kind,
		},
	}, &nemotronTensorSchema, nil
}
