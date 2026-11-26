package families

import (
	"fmt"
	"math"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"github.com/townsendmerino/goinfer/internal/runtime/rope"
)

// qwen35Architecture expresses Qwen3.5/3.6-MoE (model_type qwen3_5_moe): a 3:1 hybrid where most layers are Gated DeltaNet (linear attention
// with a recurrent matrix state, its own forward path) and the rest are QK-norm softmax attention with partial RoPE, over a routed + shared
// MoE on every layer. The descriptor marks the per-layer kind (layerIsLinear) and carries the DeltaNet geometry; the softmax layers reuse the
// uniform attention fields.
func qwen35Architecture(cfg *Config) (*darch.Architecture, *archparts.TensorSchema, error) {
	if err := cfg.ValidateQwen35(); err != nil {
		return nil, nil, err
	}
	spec, partialRotary, err := rope.ParseFlat(cfg.RopeParameters)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(qwen3_5_moe): %w", err)
	}
	rotaryDim := 0
	if partialRotary > 0 && partialRotary < 1 {
		rotaryDim = int(partialRotary * float64(cfg.HeadDim))
	}
	// Qwen3_5MoeTopKRouter ALWAYS renormalizes the top-k probabilities
	// (router_top_value /= sum), regardless of the (absent) norm_topk_prob field.
	normTopK := true
	if cfg.NormTopKProb != nil {
		normTopK = *cfg.NormTopKProb
	}
	arch := &darch.Architecture{
		Name:            "qwen3_5_moe",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         cfg.HeadDim,
		IntermediateDim: cfg.IntermediateDim, // dense width (vestigial; experts use the MoE width)
		VocabSize:       cfg.VocabSize,
		Norm:            darch.NormRMS,
		RMSAddOne:       true, // Qwen3_5MoeRMSNorm: output × (1 + weight), weight zero-init
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   darch.NormPre2,
		Act:             darch.ActSiLU,
		MoE: &MoEConfig{
			NumExperts:            cfg.NumExperts,
			TopK:                  cfg.NumExpertsPerTok,
			NormTopKProb:          normTopK,
			IntermediateDim:       cfg.MoeIntermediateSize,
			SharedIntermediateDim: cfg.SharedExpertIntermediateSize,
		},
		QKNorm:           true, // softmax layers have per-head q_norm/k_norm
		AttnScale:        math.Pow(float64(cfg.HeadDim), -0.5),
		LayerIsGlobal:    cfg.IsGlobalLayer, // softmax layers are full attention
		LayerIsLinear:    cfg.IsLinearLayer, // Gated DeltaNet layers
		RoPELocalBase:    spec.Base,
		RoPEGlobalBase:   spec.Base, // single base; only the softmax layers use RoPE
		RoPEScaling:      spec.Scaling,
		RoPEScalingLocal: spec.Scaling,
		RotaryDim:        rotaryDim, // partial_rotary_factor · head_dim
		EmbedScale:       0,
		TiedLMHead:       false, // finalized from lm_head.weight presence at load
		Qwen35: &archparts.Qwen35Params{
			ConvKernel:    cfg.LinearConvKernelDim,
			KeyHeadDim:    cfg.LinearKeyHeadDim,
			ValueHeadDim:  cfg.LinearValueHeadDim,
			NumKeyHeads:   cfg.LinearNumKeyHeads,
			NumValueHeads: cfg.LinearNumValueHeads,
			AttnGate:      true, // qwen3_5_moe's own double-width [query‖gate] q_proj
		},
	}
	arch.MRopeSection, arch.MRopeInterleaved = rope.ParseMRoPEFlat(cfg.RopeParameters)
	return arch, &Qwen35TensorSchema, nil
}
