package decoder

import (
	"fmt"
	"math"
)

// bailingHybridArchitecture expresses Bailing Hybrid (inclusionAI, model_type "bailing_hybrid"; Ling 3.0): DeepSeek-style Multi-head
// Latent Attention (MLA) alternating with Kimi Delta Attention (KDA) linear-attention layers, every LayerGroupSize-th layer being MLA, over a
// DeepSeekMoE FFN. Checked against the real modeling_bailing_moe_v3.py and a real checkpoint's safetensors header:
//
//   - layer_types is NOT a config.json field for this family: the pattern is computed from layer_group_size (normalizeBailingLayerTypes,
//     including a tail-cleanup clause).
//   - MLA is reused via mlaAttentionWith UNCHANGED, parameterized for two departures: both mixers are named self.attention (mlaParams.AttnPrefix)
//     and the output projection is self.dense (mlaParams.DenseSuffix). An optional per-head sigmoid output gate
//     (gated_attention_proj_granularity_type) rides the mechanism Laguna's attention-output gate ships, activation aside.
//   - MoE reuses moeMLP/routeExperts UNCHANGED (sigmoid + expert_bias + group-limited top-k + routed_scaling_factor + an ungated shared expert
//     is DeepSeek-V3's noaux_tc shape), but this family spells its expert counts num_experts/num_shared_experts and its bias buffer
//     expert_bias.
//   - KDA (kda.go) is the one new primitive: a delta-rule recurrence like Gated DeltaNet with a PER-CHANNEL decay (from fla-org/
//     flash-linear-attention's source, not HF's opaque Triton-kernel call). Only the no_kda_lora + kda_safe_gate variant Ling-3.0-tiny's
//     config selects is implemented; validateBailingHybrid refuses anything else rather than silently mis-running it.
//
// No YaRN/rope_scaling override is wired (the mscale/mscale_all_dim ratio deepseekArchitecture applies): the released config carries
// rope_scaling: null, so a Bailing checkpoint that sets YaRN would need it added.
func bailingHybridArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if err := cfg.normalizeBailingLayerTypes(); err != nil {
		return nil, nil, err
	}
	if err := cfg.validateBailingHybrid(); err != nil {
		return nil, nil, err
	}
	if !cfg.NoKDALora {
		return nil, nil, fmt.Errorf("decoder(bailing_hybrid): no_kda_lora=false (the LoRA'd a_proj/b_proj gate split) is not implemented — only Ling-3.0-tiny's own no_kda_lora=true variant is")
	}
	if !cfg.KDASafeGate {
		return nil, nil, fmt.Errorf("decoder(bailing_hybrid): kda_safe_gate=false (the plain unbounded decay gate) is not implemented — only Ling-3.0-tiny's own kda_safe_gate=true variant is")
	}
	base := cfg.RoPEGlobalBase
	scaling, err := parseRopeScaling(cfg.RopeScaling)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(bailing_hybrid): %w", err)
	}
	normTopK := true
	if cfg.NormTopKProb != nil {
		normTopK = *cfg.NormTopKProb
	}
	qk := cfg.QKNopeHeadDim + cfg.QKRopeHeadDim
	ropeInterleave := true
	if cfg.RopeInterleave != nil {
		ropeInterleave = *cfg.RopeInterleave
	}
	return &Architecture{
		Name:            "bailing_hybrid",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumHeads, // MLA reconstructs per-head K/V; KDA has no GVA either (verified)
		HeadDim:         cfg.HeadDim,  // KDA's per-head width; MLA reads its own qkHeadDim/VHeadDim via mlaParams
		IntermediateDim: cfg.IntermediateDim,
		VocabSize:       cfg.VocabSize,
		Norm:            NormRMS,
		RMSAddOne:       false,
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   NormPre2, // uniform for BOTH mixer kinds — verified against the real decoder layer forward
		Act:             ActSiLU,
		QKVBias:         false,
		QKNorm:          false,
		FirstKDense:     cfg.FirstKDenseReplace,
		MoE: &MoEConfig{
			NumExperts:            cfg.NumExperts,
			TopK:                  cfg.NumExpertsPerTok,
			NormTopKProb:          normTopK,
			IntermediateDim:       cfg.MoeIntermediateSize,
			SharedIntermediateDim: cfg.NumSharedExperts * cfg.MoeSharedExpertIntermediateSize,
			RouterSigmoid:         true, // unconditional for this family's router; no scoring_func key exists to check
			RoutedScale:           cfg.RoutedScalingFactor,
			SharedUngated:         true,
			NGroup:                cfg.NGroup,
			TopkGroup:             cfg.TopkGroup,
		},
		AttnScale:      math.Pow(float64(qk), -0.5), // plain qk_head_dim^-0.5, the MLA layers' own scale
		RoPELocalBase:  base,
		RoPEGlobalBase: base,
		ropeScaling:    scaling,
		RotaryDim:      cfg.QKRopeHeadDim,
		EmbedScale:     0,
		TiedLMHead:     false, // finalized from lm_head.weight presence at load; tie_word_embeddings false on the release
		layerIsLinear:  cfg.IsLinearLayer,
		mla: &mlaParams{
			QLoRARank: cfg.QLoRARank, KVLoRARank: cfg.KVLoRARank,
			QKNopeHeadDim: cfg.QKNopeHeadDim, QKRopeHeadDim: cfg.QKRopeHeadDim,
			VHeadDim: cfg.VHeadDim, ropeInterleave: ropeInterleave,
			AttnPrefix: "attention", DenseSuffix: "dense.weight",
			GateGranularity: cfg.GatedAttentionProjGranularity,
		},
		kda: &kdaParams{
			HeadDim: cfg.HeadDim, NumHeads: cfg.NumHeads, ConvKernel: cfg.ShortConvKernelSize,
			NoLora: cfg.NoKDALora, SafeGate: cfg.KDASafeGate, LowerBound: cfg.KDALowerBound,
		},
	}, &bailingHybridTensorSchema, nil
}
