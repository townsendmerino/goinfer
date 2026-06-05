package decoder

import "math"

// canBatchN reports whether the batched M=K path applies: the dense gated-MLP
// families (Qwen / Llama / Gemma) with K>1. MoE, GPT-2 (non-gated + learned
// positions), and K≤1 take the sequential fallback.
func (m *Model) canBatchN(K int) bool {
	a := m.w.arch
	return K > 1 && m.w.Embed.rows != 0 && a.MoE == nil && !a.NonGatedMLP && !a.LearnedPosEmbed
}

// forwardLayersN runs the embedding + all transformer layers + final norm over
// the K tokens in ids — appended as the next K positions of cache — and returns
// the [K, HiddenDim] post-final-norm hidden states (the LM head is the caller's).
// The weight matmuls run at M=K, so each weight streams from memory once and is
// reused across the K rows (aikit's column-blocked W8A8 kernel); attention stays
// per-position and causal. Bit-identical to K sequential forwards. Assumes
// canBatchN(len(ids)) — callers check.
func (m *Model) forwardLayersN(ids []int, cache *KVCache) ([]float32, error) {
	K := len(ids)
	arch := m.w.arch
	be := m.be
	hidden, nH, nKV, hd := arch.HiddenDim, arch.NumHeads, arch.NumKVHeads, arch.HeadDim
	qDim, kvDim, inter := nH*hd, nKV*hd, arch.IntermediateDim
	startPos := cache.Pos()
	sandwich := arch.NormPlacement == NormSandwich4

	h := make([]float32, K*hidden)
	for i, id := range ids {
		m.w.Embed.embedRow(id, h[i*hidden:i*hidden+hidden])
	}
	if arch.EmbedScale != 0 && arch.EmbedScale != 1 {
		s := float32(arch.EmbedScale)
		for i := range h {
			h[i] *= s
		}
	}
	norm := make([]float32, K*hidden)
	q := make([]float32, K*qDim)
	k := make([]float32, K*kvDim)
	v := make([]float32, K*kvDim)
	ctx := make([]float32, K*qDim)
	att := make([]float32, K*hidden)
	gate := make([]float32, K*inter)
	up := make([]float32, K*inter)
	mlpOut := make([]float32, K*hidden)
	var scores []float32

	row := func(b []float32, i, w int) []float32 { return b[i*w : i*w+w] }

	for l := 0; l < arch.NumLayers; l++ {
		lw := &m.w.Layers[l]
		global := arch.isGlobalLayer(l)

		copy(norm, h)
		for i := 0; i < K; i++ {
			normalize(arch, row(norm, i, hidden), lw.PreAttnNorm, lw.PreAttnNormBias, hidden)
		}
		lw.QProj.matmul(be, norm, q, K)
		lw.KProj.matmul(be, norm, k, K)
		lw.VProj.matmul(be, norm, v, K)
		if arch.QKVBias {
			for i := 0; i < K; i++ {
				addBias(row(q, i, qDim), lw.QBias)
				addBias(row(k, i, kvDim), lw.KBias)
				addBias(row(v, i, kvDim), lw.VBias)
			}
		}
		for i := 0; i < K; i++ {
			pos := startPos + i
			qi, ki, vi := row(q, i, qDim), row(k, i, kvDim), row(v, i, kvDim)
			if arch.QKNorm {
				rmsNorm(qi, lw.QNorm, nH, hd, arch.NormEps, arch.RMSAddOne)
				rmsNorm(ki, lw.KNorm, nKV, hd, arch.NormEps, arch.RMSAddOne)
			}
			invFreq := arch.ropeInvFreq(l)
			ms := arch.ropeMscale(l)
			applyRoPE(qi, nH, hd, pos, invFreq, ms)
			applyRoPE(ki, nKV, hd, pos, invFreq, ms)
			cache.Append(l, ki, vi)
			if n := len(cache.Keys(l)) / kvDim; cap(scores) < n {
				scores = make([]float32, n)
			}
			attendQuery(qi, row(ctx, i, qDim), scores, cache, l, pos, global, arch)
		}
		lw.OProj.matmul(be, ctx, att, K)
		if arch.OutBias {
			for i := 0; i < K; i++ {
				addBias(row(att, i, hidden), lw.OBias)
			}
		}
		if sandwich {
			for i := 0; i < K; i++ {
				normalize(arch, row(att, i, hidden), lw.PostAttnNorm, nil, hidden)
			}
		}
		for j := range h {
			h[j] += att[j]
		}

		copy(norm, h)
		for i := 0; i < K; i++ {
			normalize(arch, row(norm, i, hidden), lw.PreMLPNorm, lw.PreMLPNormBias, hidden)
		}
		lw.GateProj.matmul(be, norm, gate, K)
		lw.UpProj.matmul(be, norm, up, K)
		switch arch.Act {
		case ActGeluTanh:
			for j := range gate {
				gate[j] = geluTanh(gate[j]) * up[j]
			}
		case ActSiLU:
			for j := range gate {
				gate[j] = silu(gate[j]) * up[j]
			}
		default:
			return nil, errNotImplemented
		}
		lw.DownProj.matmul(be, gate, mlpOut, K)
		if sandwich {
			for i := 0; i < K; i++ {
				normalize(arch, row(mlpOut, i, hidden), lw.PostMLPNorm, nil, hidden)
			}
		}
		for j := range h {
			h[j] += mlpOut[j]
		}
	}

	for i := 0; i < K; i++ {
		normalize(arch, row(h, i, hidden), m.w.FinalNorm, m.w.FinalNormBias, hidden)
	}
	return h, nil
}

// lmHeadN projects M post-final-norm hidden rows (h is [M, HiddenDim]) to logits
// [M, VocabSize] (+ final-logit softcap), at M=K so the head weights stream once.
func (m *Model) lmHeadN(h []float32, M int) []float32 {
	arch := m.w.arch
	logits := make([]float32, M*arch.VocabSize)
	if arch.TiedLMHead {
		m.w.Embed.matmul(m.be, h, logits, M)
	} else {
		m.w.LMHead.matmul(m.be, h, logits, M)
	}
	if arch.FinalLogitSoftcap > 0 {
		sc := float32(arch.FinalLogitSoftcap)
		for j, val := range logits {
			logits[j] = sc * float32(math.Tanh(float64(val/sc)))
		}
	}
	return logits
}

// forwardN runs a batched forward over ids and returns the logits at every
// position ([K][VocabSize]) — used by the speculative verifier. Bit-identical to
// K sequential forwards. Falls back to sequential for the non-batched archs.
func (m *Model) forwardN(ids []int, cache *KVCache) ([][]float32, error) {
	K := len(ids)
	if K == 0 {
		return nil, nil
	}
	if !m.canBatchN(K) {
		out := make([][]float32, K)
		for i, id := range ids {
			l, err := m.forward(id, cache)
			if err != nil {
				return nil, err
			}
			out[i] = append([]float32(nil), l...) // forward reuses scr.logits — copy
		}
		return out, nil
	}
	h, err := m.forwardLayersN(ids, cache)
	if err != nil {
		return nil, err
	}
	vocab := m.w.arch.VocabSize
	logits := m.lmHeadN(h, K)
	out := make([][]float32, K)
	for i := 0; i < K; i++ {
		out[i] = logits[i*vocab : i*vocab+vocab]
	}
	return out, nil
}

// prefillLogits processes the whole prompt and returns the logits at its LAST
// position (the seed for the first generated token). On the batched archs it
// runs the layers at M=len(prompt) in one pass — each weight streamed once,
// reused across all positions (~1.7–2× faster prompt prefill / time-to-first-
// token than sequential M=1) — and runs the LM head on the last position ONLY
// (the others' logits aren't needed). Falls back to sequential runLayers +
// forward otherwise. Bit-identical to the sequential prefill (the seed token is
// unchanged). The cache is filled with the whole prompt either way.
func (m *Model) prefillLogits(prompt []int, cache *KVCache) ([]float32, error) {
	if !m.canBatchN(len(prompt)) {
		for _, id := range prompt[:len(prompt)-1] {
			if _, err := m.runLayers(id, cache); err != nil {
				return nil, err
			}
		}
		return m.forward(prompt[len(prompt)-1], cache)
	}
	h, err := m.forwardLayersN(prompt, cache)
	if err != nil {
		return nil, err
	}
	hidden := m.w.arch.HiddenDim
	last := h[(len(prompt)-1)*hidden:] // [HiddenDim] — LM head on the last row only
	return m.lmHeadN(last, 1), nil
}
