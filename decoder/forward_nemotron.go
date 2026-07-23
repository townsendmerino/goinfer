package decoder

// Nemotron-H (nemotron_h) forward path — a SINGLE-OP-per-block hybrid: each layer is
// exactly one of {Mamba-2 mixer, NoPE GQA attention, non-gated relu² MLP}, applied
// pre-norm with a residual add — NOT the mixer+FFN block every other family uses.
// The mamba layers reuse mamba2Step (recurrent state in the cache); the attention
// layers are plain causal GQA with NO RoPE (the SSM layers carry position); the mlp
// layers are up → relu² → down. Plain RMSNorm, no multipliers. Parity-first f32, one
// token per call (prefill drives it sequentially so the Mamba-2 recurrence sees every
// token), mirroring runLayersGranite.
func (m *Model) runLayersNemotron(id int, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	if cache.scr == nil { // a cache built via NewKVCache directly (tests) skips runLayers' setup
		cache.scr = newDecodeScratch(arch)
	}
	np := arch.nemotron
	hidden := arch.HiddenDim
	eps := arch.NormEps
	pos := cache.Pos()
	mp := mamba2Params{
		NHeads: np.NHeads, HeadDim: np.HeadDim, DState: np.DState,
		NGroups: np.NGroups, DConv: np.DConv, Hidden: hidden, NormGroups: np.NGroups,
	}

	h := make([]float32, hidden)
	m.w.Embed.Row(id, h)

	for l := 0; l < arch.NumLayers; l++ {
		lw := &m.w.Layers[l]
		n := append([]float32(nil), h...)
		rmsNorm(n, lw.PreAttnNorm, 1, hidden, eps, arch.RMSAddOne)
		var op []float32
		switch np.blockKind[l] {
		case nemoMamba:
			op = mamba2Step(n, lw.mamba, mp, eps, cache.mamba[l])
		case nemoAttn:
			op = m.nemotronAttention(n, lw, arch, cache, l, pos)
		case nemoMLP:
			op = m.nemotronMLP(n, lw, hidden)
		}
		for i := range h {
			h[i] += op[i]
		}
	}
	cache.Advance() // manualPos: only attention layers Append, so step pos once per token
	return h, nil
}

// nemotronAttention is plain causal GQA with NO positional embedding (NoPE) — no
// RoPE, no QK-norm, no bias — scaled by 1/√head_dim (arch.AttnScale).
func (m *Model) nemotronAttention(n []float32, lw *LayerWeights, arch *Architecture, cache *KVCache, layer, pos int) []float32 {
	nH, nKV, hd := arch.NumHeads, arch.NumKVHeads, arch.HeadDim
	hidden := arch.HiddenDim
	q := make([]float32, nH*hd)
	k := make([]float32, nKV*hd)
	v := make([]float32, nKV*hd)
	matmul(m.be, &lw.QProj, n, q, 1)
	matmul(m.be, &lw.KProj, n, k, 1)
	matmul(m.be, &lw.VProj, n, v, 1)

	cache.Append(layer, k, v)
	ctx := make([]float32, nH*hd)
	nKeys := len(cache.Keys(layer)) / (nKV * hd)
	attendQuery(q, ctx, cache.scr.scoresBuf(nKeys), cache, layer, pos, true /*full attention*/, arch)

	out := make([]float32, hidden)
	matmul(m.be, &lw.OProj, ctx, out, 1)
	return out
}

// nemotronMLP is the non-gated relu² FFN: down(relu²(up(x))).
func (m *Model) nemotronMLP(n []float32, lw *LayerWeights, hidden int) []float32 {
	inter := lw.UpProj.Rows()
	up := make([]float32, inter)
	matmul(m.be, &lw.UpProj, n, up, 1)
	for i := range up {
		up[i] = relu2(up[i])
	}
	out := make([]float32, hidden)
	matmul(m.be, &lw.DownProj, up, out, 1)
	return out
}
