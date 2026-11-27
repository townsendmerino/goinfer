package families

import (
	"math"
	"slices"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
)

// inkling2Architecture expresses Inkling 2 (thinkingmachines; inkling2 and inkling2_loop, read from llama.cpp's GGUF). It
// has no RoPE (a relative-position bias carries positions), attention-residual blocks instead of one residual stream, and on
// inkling2_loop a core of layers visited LoopPasses times. NumLayers counts those visits; each is one opInkling2 over its
// physical layer's weights (Architecture.Inkling2).
func inkling2Architecture(cfg *Config) (*darch.Architecture, *archparts.TensorSchema, error) {
	if err := cfg.ValidateInkling2(); err != nil {
		return nil, nil, err
	}
	c := cfg.Inkling2
	p := &archparts.Inkling2Params{
		PhysLayers: cfg.NumLayers, SWA: slices.Clone(c.SWA), DRel: c.DRel, RelExtent: c.RelExtent, ConvK: c.ConvK,
		BlockSize: c.BlockSize, QKClip: c.QKClip, RelClip: c.RelClip, LogFloor: c.LogFloor, LogAlpha: c.LogAlpha,
		SituBeta: c.SituBeta, SituLinearBeta: c.SituLinearBeta, PaddedVocab: c.PaddedVocab,
		LoopStart: c.LoopStart, LoopEnd: c.LoopEnd, LoopPasses: c.LoopPasses, GateRank: c.GateRank,
	}
	if cfg.ModelType == "inkling2" {
		p.LoopPasses = 0
	}
	if err := p.Validate(); err != nil {
		return nil, nil, err
	}
	visits := p.Visits()
	hd := cfg.EffectiveHeadDim()
	routedScale := cfg.MoeRoutedScalingFactor
	if routedScale == 0 {
		routedScale = 1
	}
	return &darch.Architecture{
		Name:               cfg.ModelType,
		HiddenDim:          cfg.HiddenDim,
		NumLayers:          len(visits),
		NumHeads:           cfg.NumHeads,
		NumKVHeads:         cfg.NumKVHeads,
		HeadDim:            hd,
		IntermediateDim:    cfg.MoeIntermediateSize,
		VocabSize:          cfg.VocabSize,
		Norm:               darch.NormRMS,
		NormEps:            cfg.RMSNormEps,
		NormPlacement:      darch.NormPre2,
		Act:                darch.ActSitu,
		NoPositionEncoding: true,
		MoE: &MoEConfig{
			NumExperts:      cfg.NumExperts,
			TopK:            cfg.NumExpertsPerTok,
			NormTopKProb:    true,
			IntermediateDim: cfg.MoeIntermediateSize,
			RouterSigmoid:   true,
			RoutedScale:     routedScale,
		},
		AttnScale:         math.Pow(float64(hd), -0.5),
		SlidingWindow:     cfg.SlidingWindow,
		LayerIsGlobal:     func(i int) bool { return i >= 0 && i < len(visits) && !p.SWA[visits[i].Phys] },
		FinalLogitSoftcap: cfg.FinalLogitSoftcap,
		Inkling2:          p,
	}, &archparts.TensorSchema{}, nil
}
