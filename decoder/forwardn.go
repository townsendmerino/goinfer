package decoder

import "math"

// forwardN runs a batched forward over the K tokens in ids — appended as the
// next K positions of cache — and returns the logits at every position
// ([K][VocabSize]). The weight matmuls run at M=K (each weight matrix streamed
// from memory ONCE and reused across all K positions, the speculative-verify
// efficiency win that batch=1 decode can't get); attention stays per-position
// and causal. Output is numerically identical to K sequential forward calls, so
// it does not move parity.
//
// Only the dense gated-MLP families (Qwen / Llama / Gemma — the speculative
// target case) take the batched path; MoE, GPT-2 (non-gated + learned positions),
// and K≤1 fall back to sequential single-token forwards (still correct).
func (m *Model) forwardN(ids []int, cache *KVCache) ([][]float32, error) {
	K := len(ids)
	if K == 0 {
		return nil, nil
	}
	arch := m.w.arch
	if m.w.Embed.rows == 0 {
		return nil, errNotImplemented
	}
	if K == 1 || arch.MoE != nil || arch.NonGatedMLP || arch.LearnedPosEmbed {
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

	be := m.be
	hidden, nH, nKV, hd := arch.HiddenDim, arch.NumHeads, arch.NumKVHeads, arch.HeadDim
	qDim, kvDim, inter := nH*hd, nKV*hd, arch.IntermediateDim
	startPos := cache.Pos() // the K tokens occupy positions startPos .. startPos+K-1
	sandwich := arch.NormPlacement == NormSandwich4

	// [K, *] batch buffers (allocated per call — forwardN runs once per speculative
	// round of K tokens, not per token, so this is off the hot path).
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
	var scores []float32 // grown to the largest nKeys

	row := func(b []float32, i, w int) []float32 { return b[i*w : i*w+w] }

	for l := 0; l < arch.NumLayers; l++ {
		lw := &m.w.Layers[l]
		global := arch.isGlobalLayer(l)

		copy(norm, h)
		for i := 0; i < K; i++ {
			normalize(arch, row(norm, i, hidden), lw.PreAttnNorm, lw.PreAttnNormBias, hidden)
		}
		// Projections at M=K (weights streamed once, reused across the K rows).
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
		// Per-position attention. Append in position order so each query attends
		// causally to the earlier tokens of this batch (and all prior cache), never
		// to later ones.
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
	vocab := arch.VocabSize
	logits := make([]float32, K*vocab)
	if arch.TiedLMHead {
		m.w.Embed.matmul(be, h, logits, K)
	} else {
		m.w.LMHead.matmul(be, h, logits, K)
	}
	if arch.FinalLogitSoftcap > 0 {
		sc := float32(arch.FinalLogitSoftcap)
		for j, val := range logits {
			logits[j] = sc * float32(math.Tanh(float64(val/sc)))
		}
	}
	out := make([][]float32, K)
	for i := 0; i < K; i++ {
		out[i] = logits[i*vocab : i*vocab+vocab]
	}
	return out, nil
}
