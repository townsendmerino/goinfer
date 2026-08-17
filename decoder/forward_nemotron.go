package decoder

// Nemotron-H (nemotron_h) forward path — a SINGLE-OP-per-block hybrid: each layer is
// exactly one of {Mamba-2 mixer, NoPE GQA attention, non-gated relu² MLP, non-gated
// relu² MoE FFN}, applied pre-norm with a residual add — NOT the mixer+FFN block
// every other family uses. The mamba layers reuse mamba2Step (recurrent state in the
// cache); the attention layers are plain causal GQA with NO RoPE (the SSM layers
// carry position); the mlp layers are up → relu² → down. The moe layers (Nemotron 3
// Nano only — plain Nemotron-H never sets arch.MoE) are DeepSeek-V3-style sigmoid +
// e_score_correction_bias + group-limited top-k routing (routeExperts, decoder/mlp.go
// — the exact same routing primitive DeepSeek/GLM/Mellum/Qwen3.5 share) over
// non-gated relu² experts (nemotronMoE/nemotronExpertFFN below) — NOT moeMLP, which
// hard-requires gated SwiGLU experts every other MoE family in this tree has. Plain
// RMSNorm, no multipliers. Parity-first f32, one token per call (prefill drives it
// sequentially so the Mamba-2 recurrence sees every token), mirroring runLayersGranite.
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
		case nemoMoE:
			op = m.nemotronMoE(n, lw, arch, hidden)
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

// nemotronMoE is Nemotron 3 Nano's MoE FFN: route with routeExperts (sigmoid +
// e_score_correction_bias + group-limited top-k — the same primitive DeepSeek/GLM
// share), evaluate the selected experts as non-gated relu² FFNs (NOT moeMLP's gated
// SwiGLU — this family's experts have only up_proj/down_proj, no gate_proj,
// confirmed against the real safetensors index), weighted-sum them, then add the
// always-on shared expert UNGATED (routing.SharedUngated is unconditionally true for
// this family — verified against NemotronHTopkRouter's forward, which does a plain
// `hidden_states = hidden_states + self.shared_experts(residuals)`, no sigmoid gate).
func (m *Model) nemotronMoE(n []float32, lw *LayerWeights, arch *Architecture, hidden int) []float32 {
	moe := arch.MoE
	logits := make([]float32, moe.NumExperts)
	matmul(m.be, &lw.Router, n, logits, 1)
	idx, wts := routeExperts(logits, lw.RouterBias, moe.TopK, moe.RouterSigmoid, moe.NormTopKProb, moe.RoutedScale, moe.NGroup, moe.TopkGroup)

	out := make([]float32, hidden)
	for j, e := range idx {
		expOut := m.nemotronExpertFFN(&lw.Experts[e], n, moe.IntermediateDim, hidden)
		w := wts[j]
		for i := range out {
			out[i] += w * expOut[i]
		}
	}
	if moe.SharedIntermediateDim > 0 {
		shOut := m.nemotronExpertFFN(&lw.SharedExpert, n, moe.SharedIntermediateDim, hidden)
		for i := range out {
			out[i] += shOut[i]
		}
	}
	return out
}

// nemotronExpertFFN is one MoE expert's non-gated relu² FFN: down(relu²(up(x))) —
// the same shape as nemotronMLP's plain dense block, just parameterized by width
// (routed experts and the shared expert differ) and evaluated per-expert. Only
// expertWeights.Up/.Down are populated for this family; .Gate is left unset.
func (m *Model) nemotronExpertFFN(ex *expertWeights, n []float32, inter, hidden int) []float32 {
	up := make([]float32, inter)
	matmul(m.be, &ex.Up, n, up, 1)
	for i := range up {
		up[i] = relu2(up[i])
	}
	out := make([]float32, hidden)
	matmul(m.be, &ex.Down, up, out, 1)
	return out
}
