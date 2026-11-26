package decoder

import (
	"fmt"
	"math"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"github.com/townsendmerino/goinfer/internal/runtime/rope"
)

// llama4Architecture expresses the Llama 4 (Scout/Maverick) TEXT decoder (model_type llama4_text): the iRoPE stack. Most layers are RoPE GQA
// with a parameter-free L2 (RMS-over-head-dim) QK-norm; every no_rope_layers==0 layer is NoPE (no RoPE) with attention-temperature tuning (q
// scaled by log1p(floor((pos+1)/floor_scale))·attn_scale+1 for length generalization). The FFN interleaves dense (intermediate_size_mlp) and
// MoE (moe_layers) blocks; the MoE is top-1 SIGMOID routing (no group, no norm, no scale) plus an always-on UNGATED shared expert, both at
// intermediate_size. Separate q/k/v/o (no fusion, no bias). Interleaved (complex) RoPE over the full head_dim. The vision tower
// (early-fusion multimodal) and any MTP heads are dropped: text decoder only. Dedicated loader (buildLlama4Weights) + forward
// (Architecture.llama4LayerOps).
func llama4Architecture(cfg *Config) (*darch.Architecture, *archparts.TensorSchema, error) {
	if cfg.NumLocalExperts <= 0 || cfg.NumExpertsPerTok <= 0 {
		return nil, nil, fmt.Errorf("decoder(llama4): bad MoE (num_local_experts=%d num_experts_per_tok=%d)", cfg.NumLocalExperts, cfg.NumExpertsPerTok)
	}
	// RoPE: safetensors carries the flat rope_parameters; the GGUF path sets a
	// top-level rope_theta (+ optional rope_scaling). Accept both.
	var base float64
	var ropeSc *rope.Scaling
	if len(cfg.RopeParameters) > 0 {
		spec, _, err := rope.ParseFlat(cfg.RopeParameters)
		if err != nil {
			return nil, nil, fmt.Errorf("decoder(llama4): %w", err)
		}
		base, ropeSc = spec.Base, spec.Scaling
	} else {
		base = cfg.RoPEGlobalBase
		sc, err := rope.ParseScaling(cfg.RopeScaling)
		if err != nil {
			return nil, nil, fmt.Errorf("decoder(llama4): %w", err)
		}
		ropeSc = sc
	}
	hd := cfg.HeadDim
	if hd == 0 {
		if cfg.NumHeads <= 0 { // avoid a divide-by-zero on a hostile config
			return nil, nil, fmt.Errorf("decoder(llama4): num_attention_heads must be > 0 when head_dim is unset, got %d", cfg.NumHeads)
		}
		hd = cfg.HiddenDim / cfg.NumHeads
	}
	denseInter := cfg.IntermediateSizeMLP
	if denseInter == 0 {
		denseInter = cfg.IntermediateDim
	}
	// Per-layer kind. no_rope_layers[i]==1 ⇒ RoPE (the field's sense is "uses rope");
	// absent ⇒ all RoPE. moe_layers lists the MoE indices; absent ⇒ derive from
	// interleave_moe_layer_step (layers interleave_step-1, 2·step-1, … are MoE), else all MoE.
	useRope := make([]bool, cfg.NumLayers)
	for i := range useRope {
		useRope[i] = true
		if i < len(cfg.NoRopeLayers) {
			useRope[i] = cfg.NoRopeLayers[i] != 0
		}
	}
	isMoE := make([]bool, cfg.NumLayers)
	switch {
	case len(cfg.MoeLayers) > 0:
		for _, l := range cfg.MoeLayers {
			if l >= 0 && l < cfg.NumLayers {
				isMoE[l] = true
			}
		}
	case cfg.InterleaveMoeLayerStep > 0:
		for l := cfg.InterleaveMoeLayerStep - 1; l < cfg.NumLayers; l += cfg.InterleaveMoeLayerStep {
			isMoE[l] = true
		}
	default:
		for l := range isMoE {
			isMoE[l] = true
		}
	}
	floor := cfg.FloorScale
	if floor == 0 {
		floor = 8192
	}
	attnScale := cfg.AttnScaleL4
	if attnScale == 0 {
		attnScale = 0.1
	}
	return &darch.Architecture{
		Name:            "llama4_text",
		HiddenDim:       cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         hd,
		IntermediateDim: denseInter, // dense layers (intermediate_size_mlp); experts use MoE.IntermediateDim
		VocabSize:       cfg.VocabSize,
		Norm:            darch.NormRMS,
		RMSAddOne:       false,
		NormEps:         cfg.RMSNormEps,
		NormPlacement:   darch.NormPre2,
		Act:             darch.ActSiLU,
		QKVBias:         false,
		QKNorm:          false, // Llama 4's QK-norm is parameter-free L2, applied by its program's qkL2 norm (not the weighted path)
		MoE: &MoEConfig{
			NumExperts:            cfg.NumLocalExperts,
			TopK:                  cfg.NumExpertsPerTok,
			NormTopKProb:          false,               // top-1 by raw logit; weight = sigmoid(logit), no renorm
			IntermediateDim:       cfg.IntermediateDim, // routed expert width (intermediate_size)
			SharedIntermediateDim: cfg.IntermediateDim, // shared expert uses the same width
			RouterSigmoid:         true,
			RoutedScale:           1,
			SharedUngated:         true,
		},
		AttnScale:      math.Pow(float64(hd), -0.5),
		RoPELocalBase:  base,
		RoPEGlobalBase: base,
		RoPEScaling:    ropeSc,
		RotaryDim:      0, // full head_dim
		EmbedScale:     0,
		TiedLMHead:     false, // finalized from lm_head.weight presence at load
		Llama4: &archparts.Llama4Params{
			// attention_chunk_size bounds the RoPE layers' attention to chunks; dropping it attends full-causal past it.
			ChunkSize: cfg.AttentionChunkSize,
			UseRope:   useRope, IsMoE: isMoE, UseQKNorm: cfg.UseQKNorm,
			AttnTemp: cfg.AttnTemperatureTuning, FloorScale: floor, AttnScale: attnScale,
		},
	}, &llama4TensorSchema, nil
}
