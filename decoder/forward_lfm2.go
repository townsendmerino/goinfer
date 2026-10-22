package decoder

// LFM2 / LFM2.5: every layer is Pre2, operator_norm → mixer → residual, then ffn_norm → SwiGLU → residual. The mixer is a gated
// short convolution (opShortConv) on most layers and GQA softmax attention with a per-head RMS QK-norm on the others
// (layer_types); its program is Architecture.layerOps.

// shortConvStep is the LFM2 gated short-convolution mixer for one token.
//
// Transcribed from transformers' Lfm2ShortConv, not inferred:
//
//	B, C, x = split(in_proj(n), 3)                each [convDim]
//	Bx      = B * x
//	conv[c] = Σ_j w[c][j] · window(c, j)          depthwise, K taps, NO activation
//	y       = C * conv
//	out     = out_proj(y)
//
// THE MISSING ACTIVATION IS THE TRAP. Mamba-2's conv (mamba2.go) and DeltaNet's
// (deltanet.go) are the same loop with SiLU on the output; upstream passes activation=None
// here. A SiLU added "for consistency" would still produce fluent text — just not this
// model's — and no test that only checks shapes would notice.
//
// The window holds the K-1 PRIOR Bx vectors; this token supplies the K-th tap directly. That
// split is why push() stores K-1 and not K: storing K would double-count the current token.
func shortConvStep(n []float32, w *shortConvWeights, g lfm2Params, hidden int, st *shortConvState) []float32 {
	cd, K := g.ConvDim, g.ConvLCache

	// in_proj is [3*convDim, hidden]; B|C|x are consecutive blocks in that order.
	bcx := make([]float32, 3*cd)
	w.inProj.MatmulBT(n, bcx, 1)
	B, C, x := bcx[:cd], bcx[cd:2*cd], bcx[2*cd:]

	bx := make([]float32, cd)
	for c := range cd {
		bx[c] = B[c] * x[c]
	}

	// Depthwise causal conv. Tap j=K-1 is the current token; j<K-1 read the window,
	// which is short (zero-padded) for the first K-1 tokens of a sequence.
	conv := make([]float32, cd)
	win := st.convWin
	for c := range cd {
		s := w.convW[c*K+(K-1)] * bx[c]
		for j := 0; j < K-1; j++ {
			if idx := len(win) - (K - 1) + j; idx >= 0 {
				s += w.convW[c*K+j] * win[idx][c]
			}
		}
		conv[c] = s
	}
	st.push(bx, K)

	y := make([]float32, cd)
	for c := range cd {
		y[c] = C[c] * conv[c]
	}
	out := make([]float32, hidden)
	w.outProj.MatmulBT(y, out, 1)
	return out
}
