package decoder

// runLayersQwen35 is the Qwen3.5/3.6-MoE (qwen3_5_moe) forward, a hybrid: most layers are Gated DeltaNet (linear attention,
// recurrent state in the cache), the rest gated softmax attention (KV cache), every layer a routed+shared MoE. Parity-first f32,
// allocate-per-call, mirroring runLayersGemma4. One token per call; the caller (forward) applies the final norm and LM head, and
// the per-token prefill drives this sequentially so the DeltaNet recurrence sees every token (runLayersQwen35N is the batched
// twin). See docs/completed/qwen3_5_moe.md.
func (m *Model) runLayersQwen35(id int, cache *KVCache) ([]float32, error) {
	h := make([]float32, m.w.arch.HiddenDim)
	m.w.Embed.Row(id, h) // no embedding scale for qwen3_5_moe
	return m.runLayersQwen35FromEmbed(h, cache)
}

// runLayersQwen35FromEmbed is runLayersQwen35's body for a position whose residual-stream embedding is supplied directly: the
// image-splice seam, since an image row is a tower feature, not a table lookup. Same shape as runLayersGemma4FromEmbed. It
// consumes h (the layers mutate it in place) and returns it. One token per call, like its wrapper (runLayersQwen35N is the batched
// twin); the GDN layers take no position ids, so an image row needs nothing a text row does not, except the m-RoPE positions the
// full-attention layers read from cache.mropePos.
func (m *Model) runLayersQwen35FromEmbed(h []float32, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	if cache.scr == nil { // a cache built via NewKVCache directly (tests) skips runLayers' setup
		cache.scr = newDecodeScratch(arch)
	}
	g := arch.qwen35
	hidden := arch.HiddenDim
	eps := arch.NormEps
	pos := cache.Pos() // this token's absolute position (stable; Advance() at the end)

	for l := 0; l < arch.NumLayers; l++ {
		lw := &m.w.Layers[l]

		// Resolved per layer, not assumed Pre2: Olmo Hybrid's full-attention layers use NormPostOnly (NormPlacementLinear is nil for every
		// other qwen35-shaped family, so postOnly is always false there).
		postOnly := arch.normPlacementAt(l) == NormPostOnly

		// Attention sub-block (Pre2: norm → mix → residual; postOnly: mix reads the
		// raw residual directly, its OUTPUT is normalized before the add instead).
		var n []float32
		if !postOnly {
			rmsNormInto(cache.scr.norm, h, lw.PreAttnNorm, 1, hidden, eps, arch.RMSAddOne)
			n = cache.scr.norm
		} else {
			n = h
		}
		var attn []float32
		if arch.isLinearLayer(l) {
			attn = gatedDeltaNetStep(m.be, n, lw.delta, *g, hidden, eps, cache.delta[l])
		} else if g.PlainFullAttn {
			// Olmo Hybrid: a PLAIN olmo3-shaped self-attention (whole-vector QK-norm,
			// generic RoPE/NoPE, single-width q/k/v/o) — reuse the shared
			// causalAttention rather than qwen3.5's own double-width gated one.
			attn = make([]float32, hidden)
			if err := causalAttention(l, n, attn, lw, arch, cache, m.be, nil); err != nil {
				return nil, err
			}
		} else {
			attn = m.qwen35Attention(n, lw, arch, cache, l, pos)
		}
		if postOnly {
			normalize(arch, attn, lw.PostAttnNorm, nil, hidden)
		}
		addResidual(h, attn)

		// FFN sub-block (Pre2); post_attention_layernorm is the pre-MLP norm. The dense branch is Qwen3.8 (model_type qwen3_5): the same
		// Gated-DeltaNet/softmax hybrid as its MoE siblings, with a plain SwiGLU where they have a router. That is the only structural
		// difference in this forward (the DeltaNet step, the gated attention, the hybrid cache and the prefill are untouched), so it
		// branches here rather than getting a forward of its own.
		var n2 []float32
		if !postOnly {
			rmsNormInto(cache.scr.norm, h, lw.PreMLPNorm, 1, hidden, eps, arch.RMSAddOne)
			n2 = cache.scr.norm
		} else {
			n2 = h
		}
		var ffn []float32
		var err error
		if arch.MoE == nil {
			ffn = make([]float32, hidden)
			err = gatedMLP(n2, ffn, lw, arch, m.be, cache.scr, nil) // scr.gate/up are sized from IntermediateDim, which the dense validator requires
		} else {
			ffn, err = moeMLP(n2, lw, arch, m.be, cache.scr, m.pager)
		}
		if err != nil {
			return nil, err
		}
		if postOnly {
			normalize(arch, ffn, lw.PostMLPNorm, nil, hidden)
		}
		addResidual(h, ffn)
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
	eps := arch.NormEps

	// q_proj emits [query ‖ gate] per head; split them.
	cache.scr.hq.prepare(m.be, &a.qProj, n, 1) // n quantized once for q, k and v (R-13)
	qg := matvecWMPre(m.be, &a.qProj, &cache.scr.hq, n)
	q := make([]float32, nH*hd)
	gate := make([]float32, nH*hd)
	splitQGate(qg, q, gate, nH, hd)
	k := matvecWMPre(m.be, &a.kProj, &cache.scr.hq, n)
	v := matvecWMPre(m.be, &a.vProj, &cache.scr.hq, n)

	// QK-norm (query half only) then partial RoPE.
	rmsNorm(q, a.qNorm, nH, hd, eps, arch.RMSAddOne)
	rmsNorm(k, a.kNorm, nKV, hd, eps, arch.RMSAddOne)
	invFreq := arch.ropeInvFreq(layer)
	ms := arch.ropeMscale(layer)
	// ropeAt is applyRoPE unless the cache carries m-RoPE positions (GenerateQwenVL sets them for an
	// image turn), so the text path is unchanged. The interleave flag stays false: this family's
	// rotary has always been the half-split form, whatever arch.ropeInterleave says.
	ropeAt(q, nH, hd, pos, invFreq, ms, arch.MRopeSection, cache.mropePos, cache.mropeDelta, false, arch.MRopeInterleaved)
	ropeAt(k, nKV, hd, pos, invFreq, ms, arch.MRopeSection, cache.mropePos, cache.mropeDelta, false, arch.MRopeInterleaved)

	cache.Append(layer, k, v)
	ctx := cache.scr.ctxBuf(nH * hd) // R-17: from the scratch; attendQuery clears it
	nKeys := len(cache.Keys(layer)) / (nKV * hd)
	attendQuery(q, ctx, cache.scr.scoresBuf(nKeys), cache, layer, pos, true /*full attention*/, arch)

	// Output gate, then o_proj.
	qGateContext(ctx, gate)
	return matvecWM(m.be, &a.oProj, ctx)
}

// splitQGate splits one position's double-width q_proj output into q and gate. The layout is [query ‖ gate] PER HEAD,
// interleaved, not two concatenated blocks. Metal's delta_qsplit mirrors it (metal/qgate_kernels_test.go).
func splitQGate(qg, q, gate []float32, nH, hd int) {
	for hh := range nH {
		copy(q[hh*hd:hh*hd+hd], qg[hh*2*hd:hh*2*hd+hd])
		copy(gate[hh*hd:hh*hd+hd], qg[hh*2*hd+hd:hh*2*hd+2*hd])
	}
}

// qGateContext is the output gate: ctx *= sigmoid(gate), elementwise and in place, before o_proj. Metal's
// delta_attn_gate mirrors it.
func qGateContext(ctx, gate []float32) {
	for i := range ctx {
		ctx[i] *= sigmoidf(gate[i])
	}
}
