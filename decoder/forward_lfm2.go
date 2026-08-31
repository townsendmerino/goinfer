package decoder

// LFM2 / LFM2.5 forward — one token per call. The caller (forward) applies the final norm
// and the tied LM head.
//
// Every layer is Pre2: operator_norm → mixer → residual, then ffn_norm → SwiGLU → residual.
// The mixer is a gated short convolution on 22 of 30 layers and GQA softmax attention on the
// other 8 (layer_types). Both halves write into the same hidden vector, so the only thing the
// layer kind changes is which mixer runs and which cache slot it touches.

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
	bcx := matvec(w.inProj, 3*cd, hidden, n)
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
	return matvec(w.outProj, hidden, cd, y)
}

// lfm2Attention is GQA + RoPE with per-head RMSNorm on Q and K.
//
// The QK-norm is RMSNorm over head_dim, applied per head BEFORE RoPE — the ordering HF uses
// and the one the existing hardcoded QK-norm path already implements, which is why declaring
// QKNorm was enough and no new primitive was needed. (The original scoping brief said
// LayerNorm; the released checkpoint carries q_layernorm.weight and no bias tensor anywhere,
// and the reference builds Lfm2RMSNorm(head_dim). See lfm2Architecture.)
func (m *Model) lfm2Attention(n []float32, lw *LayerWeights, arch *Architecture, cache *KVCache, layer, pos int) []float32 {
	nH, nKV, hd := arch.NumHeads, arch.NumKVHeads, arch.HeadDim
	hidden := arch.HiddenDim
	q := make([]float32, nH*hd)
	k := make([]float32, nKV*hd)
	v := make([]float32, nKV*hd)
	matmul(m.be, &lw.QProj, n, q, 1)
	matmul(m.be, &lw.KProj, n, k, 1)
	matmul(m.be, &lw.VProj, n, v, 1)

	if arch.QKNorm {
		rmsNorm(q, lw.QNorm, nH, hd, arch.NormEps, arch.RMSAddOne)
		rmsNorm(k, lw.KNorm, nKV, hd, arch.NormEps, arch.RMSAddOne)
	}

	invFreq := arch.ropeInvFreq(layer)
	ms := arch.ropeMscale(layer)
	applyRoPE(q, nH, hd, pos, invFreq, ms)
	applyRoPE(k, nKV, hd, pos, invFreq, ms)

	cache.Append(layer, k, v)
	ctx := make([]float32, nH*hd)
	nKeys := len(cache.Keys(layer)) / (nKV * hd)
	// Full causal attention on every attention layer: LFM2 has no sliding window — the conv
	// layers ARE its locality mechanism, so the 8 attention layers are all global.
	attendQuery(q, ctx, cache.scr.scoresBuf(nKeys), cache, layer, pos, true, arch)

	out := make([]float32, hidden)
	matmul(m.be, &lw.OProj, ctx, out, 1)
	return out
}

func (m *Model) runLayersLFM2(id int, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	if cache.scr == nil { // a cache built via NewKVCache directly (tests) skips runLayers' setup
		cache.scr = newDecodeScratch(arch)
	}
	g := arch.lfm2
	hidden := arch.HiddenDim
	eps := arch.NormEps

	// manualPos: the conv layers never Append, so position cannot be read off the KV length.
	pos := cache.Pos()

	h := make([]float32, hidden)
	m.w.Embed.Row(id, h) // no embedding scale

	for l := 0; l < arch.NumLayers; l++ {
		lw := &m.w.Layers[l]

		// Mixer sub-block. operator_norm is the pre-mixer norm for BOTH kinds — LFM2 does
		// not have separate conv/attention norm names, which is why lfm2TensorSchema maps
		// it to PreAttnNorm and both branches read the same field.
		n := append([]float32(nil), h...)
		rmsNorm(n, lw.PreAttnNorm, 1, hidden, eps, arch.RMSAddOne)
		var mix []float32
		if arch.isConvLayer(l) {
			mix = shortConvStep(n, lw.shortConv, *g, hidden, cache.conv[l])
		} else {
			mix = m.lfm2Attention(n, lw, arch, cache, l, pos)
		}
		for i := range h {
			h[i] += mix[i]
		}

		// FFN sub-block. ffn_norm is the pre-FFN norm; the FFN is SwiGLU (w1 gate, w3 up,
		// w2 down) and identical on conv and attention layers.
		n2 := append([]float32(nil), h...)
		rmsNorm(n2, lw.PreMLPNorm, 1, hidden, eps, arch.RMSAddOne)
		ffn := make([]float32, hidden)
		if err := gatedMLP(n2, ffn, lw, arch, m.be, cache.scr, nil); err != nil {
			return nil, err
		}
		for i := range h {
			h[i] += ffn[i]
		}
	}

	cache.Advance()
	return h, nil
}
