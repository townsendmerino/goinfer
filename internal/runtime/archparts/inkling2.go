package archparts

import (
	"fmt"

	"github.com/townsendmerino/aikit/linalg"
)

// Inkling2Params is the geometry of an Inkling 2 decoder (inkling2, and inkling2_loop when LoopPasses > 1). The
// Architecture's NumLayers counts layer VISITS: a loop model visits its core layers [LoopStart, LoopEnd) LoopPasses
// times, each visit with its own KV and conv state and the physical layer's weights (VisitPhys).
type Inkling2Params struct {
	PhysLayers int    // physical layers in the checkpoint
	SWA        []bool // per physical layer: sliding-window attention (else full attention)

	DRel, RelExtent int     // relative-state width per head; the full-attention layers' bias extent (sliding layers use the window)
	ConvK           int     // short-convolution kernel width
	BlockSize       int     // layers per attention-residual block
	QKClip, RelClip float64 // L2 clip thresholds of the query/key heads and the relative-state heads
	LogFloor        int     // log scaling: 1 + LogAlpha·ln(max(1, (pos+1)/LogFloor)) on the full layers' queries and relative states
	LogAlpha        float64
	SituBeta        float64 // expert gate softcap (SITU's 4 in 4·tanh(g/4))
	SituLinearBeta  float64 // expert up softcap
	PaddedVocab     int     // rows of the stored embedding and head; the logits keep VocabSize of them

	// The loop (inkling2_loop). LoopPasses <= 1 means no loop.
	LoopStart, LoopEnd, LoopPasses, GateRank int
}

// Visit is one layer visit of the forward: its physical layer, its pass through the loop core (0 outside it), and
// whether it starts an attention-residual block.
type Visit struct {
	Phys, Pass int
	Boundary   bool
}

// Visits lists the forward's layer visits in order: the prelude, the core once per pass, then the rest.
func (p *Inkling2Params) Visits() []Visit {
	var vs []Visit
	add := func(l, pass int) { vs = append(vs, Visit{Phys: l, Pass: pass, Boundary: l%p.BlockSize == 0}) }
	if p.LoopPasses <= 1 {
		for l := range p.PhysLayers {
			add(l, 0)
		}
		return vs
	}
	for l := range p.LoopStart {
		add(l, 0)
	}
	for pass := range p.LoopPasses {
		for l := p.LoopStart; l < p.LoopEnd; l++ {
			add(l, pass)
		}
	}
	for l := p.LoopEnd; l < p.PhysLayers; l++ {
		add(l, 0)
	}
	return vs
}

// Validate checks the geometry the forward assumes.
func (p *Inkling2Params) Validate() error {
	switch {
	case p.PhysLayers <= 0 || len(p.SWA) != p.PhysLayers:
		return fmt.Errorf("inkling2: %d physical layers with %d sliding-window flags", p.PhysLayers, len(p.SWA))
	case p.DRel <= 0 || p.RelExtent <= 0 || p.ConvK < 2 || p.BlockSize <= 0 || p.LogFloor <= 0:
		return fmt.Errorf("inkling2: invalid d_rel %d, rel_extent %d, conv kernel %d, block size %d or log floor %d", p.DRel, p.RelExtent, p.ConvK, p.BlockSize, p.LogFloor)
	case !(p.QKClip > 0) || !(p.RelClip > 0) || !(p.SituBeta > 0) || !(p.SituLinearBeta > 0) || p.LogAlpha < 0:
		return fmt.Errorf("inkling2: clip, softcap and log-scale parameters must be positive")
	}
	if p.LoopPasses > 1 {
		if p.LoopStart <= 0 || p.LoopStart >= p.LoopEnd || p.LoopEnd > p.PhysLayers || p.LoopStart%p.BlockSize != 0 || p.LoopEnd%p.BlockSize != 0 || p.GateRank <= 0 {
			return fmt.Errorf("inkling2: invalid loop core [%d, %d) with block size %d and gate rank %d over %d layers", p.LoopStart, p.LoopEnd, p.BlockSize, p.GateRank, p.PhysLayers)
		}
	}
	return nil
}

// Inkling2Controller is one cross-pass message controller: gate = Down·[unitRMS(x); unitRMS(m)], then
// x' = (1 - tanh(Forget·gate))·x + tanh(Gain·gate)·m.
type Inkling2Controller struct {
	Down   []float32 // [GateRank, 2*hidden]
	Gain   []float32 // [hidden, GateRank]
	Forget []float32 // [hidden, GateRank]
}

// Inkling2LayerWeights holds a physical Inkling 2 layer's tensors that LayerWeights has no field for. The projections
// (QProj, KProj, VProj, OProj, GProj), the attention input norm (PreAttnNorm, the query input norm on a full layer), the FFN
// input norm (PreMLPNorm), the router and the experts live on LayerWeights.
type Inkling2LayerWeights struct {
	QClip    []float32        // [heads*headDim] learned scale before the query clip
	SinkProj linalg.WeightMat // [heads, hidden] per-token sink logit
	Rel      linalg.WeightMat // [heads*DRel, hidden] relative states
	RelClip  []float32        // [heads*DRel] learned scale before the relative-state clip
	RelBias  []float32        // [extent, DRel] bias profiles by backward distance

	KConv, VConv     []float32 // [kvDim, ConvK] static taps; tap ConvK-1 weights the current token
	AttnConv, FFConv []float32 // [hidden, ConvK]

	// Full-attention layers: separate K and V input norms, and per-token taps per KV head.
	KInpNorm, VInpNorm   []float32
	KConvProj, VConvProj linalg.WeightMat // [kvHeads*ConvK, hidden]; row h*ConvK+lag, lag 0 the current token

	AttnResQuery []float32 // [1 or 3, hidden]; nil on layer 0
	FFNResQuery  []float32 // [hidden]

	// Loop core layers.
	MsgQuery []float32             // [4, hidden]: the q, k, v and FFN message queries
	Ctrl     [4]Inkling2Controller // q, k, v, FFN
}

// Inkling2HeadWeights holds the model-level tensors: the embedding norm and the final mix's query.
type Inkling2HeadWeights struct {
	EmbedNorm      []float32 // [hidden]
	OutputResQuery []float32 // [hidden]
}
