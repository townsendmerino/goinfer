package decoder

// The Qwen3.5/3.6 family (qwen3_5_moe, qwen3_5, qwen3_next, olmo_hybrid), a hybrid: most layers are Gated DeltaNet (linear
// attention, recurrent state in the cache), the rest gated softmax attention (KV cache), each followed by a routed+shared MoE or
// a dense SwiGLU. Its per-token program is the generic layer with Architecture.mixerOps' mixers; runLayersQwen35N is the batched
// twin. See docs/completed/qwen3_5_moe.md.

// qwen35AttentionWith is the gated softmax attention of a full-attention layer (opGatedAttn):
// double-width q_proj (query ‖ gate per head), per-head QK-norm, partial RoPE,
// causal GQA attention over the KV cache, then × sigmoid(gate) before o_proj.
func (m *Model) qwen35AttentionWith(p *gatedAttnParams, n []float32, lw *LayerWeights, arch *Architecture, cache *KVCache, layer, pos int) []float32 {
	a := lw.qattn
	nH, nKV, hd := arch.NumHeads, arch.NumKVHeads, arch.HeadDim

	// q_proj emits [query ‖ gate] per head; split them.
	cache.scr.hq.prepare(m.be, &a.qProj, n, 1) // n quantized once for q, k and v (R-13)
	qg := matvecWMPre(m.be, &a.qProj, &cache.scr.hq, n)
	q := make([]float32, nH*hd)
	gate := make([]float32, nH*hd)
	splitQGate(qg, q, gate, nH, hd)
	k := matvecWMPre(m.be, &a.kProj, &cache.scr.hq, n)
	v := matvecWMPre(m.be, &a.vProj, &cache.scr.hq, n)

	// QK-norm (query half only) then partial RoPE; ropeAt is applyRoPE unless the cache carries m-RoPE positions (an image turn).
	applyQKNorm(p.QKNorm, q, k, a.qNorm, a.kNorm, nH, nKV, hd)
	applyRope(&p.Rope, q, k, nH, nKV, hd, pos, cache.mropePos, cache.mropeDelta)

	cache.Append(layer, k, v)
	ctx := cache.scr.ctxBuf(nH * hd) // R-17: from the scratch; attendQueryP clears it
	nKeys := len(cache.Keys(layer)) / (nKV * hd)
	attendQueryP(p.Attn, q, ctx, cache.scr.scoresBuf(nKeys), cache, layer, pos, arch)

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
