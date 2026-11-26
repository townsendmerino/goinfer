package archparts

import (
	"github.com/townsendmerino/aikit/linalg"
)

// ShortConvWeights is one LFM2 gated short-convolution mixer.
//
// The block, from transformers' Lfm2ShortConv:
//
//	B, C, x = split(inProj(h), 3)      each [convDim]
//	Bx      = B * x
//	conv    = depthwise_causal_conv(Bx, convW)     kernel K, NO activation
//	y       = C * conv
//	out     = outProj(y)
//
// TWO THINGS HERE ARE EASY TO GET PLAUSIBLY WRONG, so both are stated rather than left
// to the reader:
//
//   - THERE IS NO ACTIVATION on the conv output. Mamba-2's structurally-identical conv
//     applies SiLU and DeltaNet's does too, so the third copy of this loop in this tree is
//     the odd one out. Upstream passes activation=None; adding SiLU would still generate
//     fluent text, just not this model's text.
//   - THE SPLIT ORDER IS B, C, x — the gate that multiplies BEFORE the conv is first and
//     the one that multiplies AFTER is second. Swapping B and C is a silent transposition
//     of two gates that are both elementwise and both [convDim], so nothing errors.
type ShortConvWeights struct {
	InProj  linalg.WeightMat // [3*convDim, hidden] — B|C|x stacked in that order; bf16 or f32 (islandMat)
	ConvW   []float32        // [convDim, K]        — depthwise ([convDim,1,K] flattened)
	OutProj linalg.WeightMat // [hidden, convDim]; bf16 or f32 (islandMat)
}
