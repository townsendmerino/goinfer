package archparts

import (
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/internal/runtime/mamba2"
)

type Mamba2Params struct {
	NHeads  int
	HeadDim int // P
	DState  int // N
	NGroups int
	DConv   int // K (causal conv kernel)
	Hidden  int
	// NormGroups is the group count for the GATED RMSNorm specifically — families
	// differ here: Granite (GraniteMoeHybridRMSNormGated) normalizes over the FULL
	// dInner (NormGroups 1), Nemotron-H (Zamba2RMSNormGated) normalizes per group
	// (NormGroups = NGroups). Equal when NGroups==1. 0 ⇒ treated as 1.
	NormGroups int
}

// Mixer is p's geometry as internal/mamba2 takes it.
func (p Mamba2Params) Mixer() mamba2.Params {
	return mamba2.Params{NHeads: p.NHeads, HeadDim: p.HeadDim, DState: p.DState, NGroups: p.NGroups, DConv: p.DConv, NormGroups: p.NormGroups}
}

func (p Mamba2Params) DInner() int { return p.Mixer().DInner() }

func (p Mamba2Params) ProjDim() int { return 2*p.DInner() + 2*p.NGroups*p.DState + p.NHeads }

// Mamba2Weights holds one mixer's tensors, f32 (parity-first, like the qwen35 DeltaNet/softmax weights), row-major [out, in]
// for the projections. On a quantized Nemotron-H load the two projections are int8 instead (inQ, outQ; quantizeProjections),
// with inProj and outProj nil.
type Mamba2Weights struct {
	InProj  []float32 // [projDim, hidden]
	ConvW   []float32 // [convDim, K] (HF conv1d.weight [convDim,1,K] flattened)
	ConvB   []float32 // [convDim]
	ALog    []float32 // [nHeads] (A = -exp(aLog))
	D       []float32 // [nHeads] skip
	DtBias  []float32 // [nHeads]
	NormW   []float32 // [dInner] gated RMSNorm weight
	OutProj []float32 // [hidden, dInner]
	// inQ and outQ are in_proj and out_proj as per-row int8, weight-only (f32 activations): linalg.QuantizeInt8 of the f32
	// weights, the codes and scales the GPU backends upload.
	InQ, OutQ linalg.WeightMat
}

// Mixer is w's tensors between the projections, as internal/mamba2 takes them.
func (w *Mamba2Weights) Mixer() mamba2.Weights {
	return mamba2.Weights{ConvW: w.ConvW, ConvB: w.ConvB, ALog: w.ALog, D: w.D, DtBias: w.DtBias, NormW: w.NormW}
}

// QuantizeProjections replaces the f32 projections with their int8 form.
func (w *Mamba2Weights) QuantizeProjections(projDim, hidden, dInner int) {
	w.InQ = linalg.QuantizeInt8(w.InProj, projDim, hidden, false)
	w.OutQ = linalg.QuantizeInt8(w.OutProj, hidden, dInner, false)
	w.InProj, w.OutProj = nil, nil
}
