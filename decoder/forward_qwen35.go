package decoder

// Qwen3.5/3.6-MoE (qwen3_5_moe) forward path — the hybrid: most layers are Gated
// DeltaNet (linear attention, recurrent state in the cache), the rest gated
// softmax attention (KV cache), every layer a routed+shared MoE. Parity-first
// f32, allocate-per-call, mirroring runLayersGemma4 (perf is a later track). One
// token per call; the caller (forward) applies the final norm + LM head, and
// prefill drives this sequentially so the DeltaNet recurrence sees every token.
// See docs/qwen3_5_moe.md.
func (m *Model) runLayersQwen35(id int, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	if cache.scr == nil { // a cache built via NewKVCache directly (tests) skips runLayers' setup
		cache.scr = newDecodeScratch(arch)
	}
	g := arch.qwen35
	hidden := arch.HiddenDim
	eps := arch.NormEps
	pos := cache.Pos() // this token's absolute position (stable; Advance() at the end)

	h := make([]float32, hidden)
	m.w.Embed.Row(id, h) // no embedding scale for qwen3_5_moe

	for l := 0; l < arch.NumLayers; l++ {
		lw := &m.w.Layers[l]

		// Attention sub-block (Pre2: norm → mix → residual).
		n := append([]float32(nil), h...)
		rmsNorm(n, lw.PreAttnNorm, 1, hidden, eps, arch.RMSAddOne)
		var attn []float32
		if arch.isLinearLayer(l) {
			attn = gatedDeltaNetStep(n, lw.delta, *g, hidden, eps, cache.delta[l])
		} else {
			attn = m.qwen35Attention(n, lw, arch, cache, l, pos)
		}
		for i := range h {
			h[i] += attn[i]
		}

		// FFN sub-block (Pre2). post_attention_layernorm is the pre-MLP norm.
		//
		// The DENSE branch is Qwen3.8 (model_type qwen3_5): the same Gated-DeltaNet/softmax
		// 3:1 hybrid as its MoE siblings, with a plain SwiGLU where they have a router. It is
		// the ONLY structural difference in this forward — the DeltaNet step, the gated
		// attention, the hybrid cache and the sequential prefill are all untouched — so it
		// branches here rather than getting a forward of its own.
		n2 := append([]float32(nil), h...)
		rmsNorm(n2, lw.PreMLPNorm, 1, hidden, eps, arch.RMSAddOne)
		var ffn []float32
		var err error
		if arch.MoE == nil {
			ffn = make([]float32, hidden)
			err = gatedMLP(n2, ffn, lw, arch, m.be, cache.scr, nil) // scr.gate/up are sized from IntermediateDim, which the dense validator requires
		} else {
			ffn, err = moeMLP(n2, lw, arch, m.be, m.pager)
		}
		if err != nil {
			return nil, err
		}
		for i := range h {
			h[i] += ffn[i]
		}
		cache.captureResidual(l, h)
	}
	cache.Advance() // manualPos: one stored position per token
	return h, nil
}

// qwen35Attention is the gated softmax attention of a full-attention layer:
// double-width q_proj (query ‖ gate per head), per-head QK-norm, partial RoPE,
// causal GQA attention over the KV cache, then × sigmoid(gate) before o_proj.
func (m *Model) qwen35Attention(n []float32, lw *LayerWeights, arch *Architecture, cache *KVCache, layer, pos int) []float32 {
	a := lw.qattn
	nH, nKV, hd := arch.NumHeads, arch.NumKVHeads, arch.HeadDim
	hidden, eps := arch.HiddenDim, arch.NormEps

	// q_proj emits [query ‖ gate] per head; split them.
	qg := matvec(a.qProj, nH*hd*2, hidden, n)
	q := make([]float32, nH*hd)
	gate := make([]float32, nH*hd)
	for hh := range nH {
		copy(q[hh*hd:hh*hd+hd], qg[hh*2*hd:hh*2*hd+hd])
		copy(gate[hh*hd:hh*hd+hd], qg[hh*2*hd+hd:hh*2*hd+2*hd])
	}
	k := matvec(a.kProj, nKV*hd, hidden, n)
	v := matvec(a.vProj, nKV*hd, hidden, n)

	// QK-norm (query half only) then partial RoPE.
	rmsNorm(q, a.qNorm, nH, hd, eps, arch.RMSAddOne)
	rmsNorm(k, a.kNorm, nKV, hd, eps, arch.RMSAddOne)
	invFreq := arch.ropeInvFreq(layer)
	ms := arch.ropeMscale(layer)
	applyRoPE(q, nH, hd, pos, invFreq, ms)
	applyRoPE(k, nKV, hd, pos, invFreq, ms)

	cache.Append(layer, k, v)
	ctx := make([]float32, nH*hd)
	nKeys := len(cache.Keys(layer)) / (nKV * hd)
	attendQuery(q, ctx, cache.scr.scoresBuf(nKeys), cache, layer, pos, true /*full attention*/, arch)

	// Output gate, then o_proj.
	for i := range ctx {
		ctx[i] *= sigmoidf(gate[i])
	}
	return matvec(a.oProj, hidden, nH*hd, ctx)
}
