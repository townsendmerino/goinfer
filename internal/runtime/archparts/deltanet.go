package archparts

import (
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/internal/runtime/hybridattn"
)

// DeltaNetWeights holds one Gated DeltaNet layer's parameters, row-major, as HF
// stores them. All projections are bias-free; the depthwise conv is bias-free too.
type DeltaNetWeights struct {
	// The three dominant projections (inProjQKV, inProjZ, outProj) are quantizable WeightMats: they dominate the bytes streamed per
	// token and decode is memory-bandwidth-bound, so holding them as f32 would stream them at f32 under any quant. WeightMat keeps f32
	// when no quant is asked for (the tiny goldens still match HF exactly) and carries int8/int4 otherwise.
	InProjQKV linalg.WeightMat // [convDim, hidden]   → [q;k;v]
	InProjZ   linalg.WeightMat // [valueDim, hidden]  → output gate z
	// A and B stay f32: they are small against the projections and feed the write/decay gates, where the recurrence is most
	// sensitive to precision. Quantizing them would buy a fraction of a percent of the bytes for real numerical risk.
	InProjB []float32        // [numV, hidden]      → write-gate logits β
	InProjA []float32        // [numV, hidden]      → decay-gate input a
	ConvW   []float32        // [convDim, K]        depthwise causal conv ([convDim,1,K] flattened)
	DtBias  []float32        // [numV]
	NegExpA []float32        // [numV]              −exp(A_log), precomputed (see negExpAFromLog)
	NormW   []float32        // [headVDim]          gated RMSNorm weight
	OutProj linalg.WeightMat // [hidden, valueDim]
}

func (p Qwen35Params) DeltaNetParams() hybridattn.DeltaNetParams {
	return hybridattn.DeltaNetParams{
		ConvKernel:    p.ConvKernel,
		KeyHeadDim:    p.KeyHeadDim,
		ValueHeadDim:  p.ValueHeadDim,
		NumKeyHeads:   p.NumKeyHeads,
		NumValueHeads: p.NumValueHeads,
		NegEigval:     p.NegEigval,
		ONormEps:      p.ONormEps,
	}
}
