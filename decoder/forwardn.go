package decoder

import (
	"math"

	"github.com/townsendmerino/aikit/linalg"
)

// canBatchN reports whether the batched M=K path applies: the dense gated-MLP
// families (Qwen / Llama / Gemma) with K>1. MoE, GPT-2 (non-gated + learned
// positions), and K≤1 take the sequential fallback.
func (m *Model) canBatchN(K int) bool {
	a := m.w.arch
	// Gemma 4 has its own sequential forward (per-layer head_dim, KV-sharing, PLE);
	// the batched M=K path here is the uniform-shape gemma3-family one, so exclude it.
	return K > 1 && m.w.Embed.rows != 0 && a.MoE == nil && !a.NonGatedMLP && !a.LearnedPosEmbed && a.gemma4 == nil
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
	// Batched per-head attention scratch (reused across layers; nKeys = startPos+K
	// is the same for every layer in this sweep). See attendBatchedHeads.
	maxKeys := startPos + K
	aqh := make([]float32, K*hd)
	akh := make([]float32, maxKeys*hd)
	avt := make([]float32, maxKeys*hd)
	ascores := make([]float32, K*maxKeys)
	ach := make([]float32, K*hd)
	// Batch the qkv and gate/up projections (shared activation) so a GPU backend
	// runs each group as one submit (BatchTiled) instead of per-matmul syncs.
	var ws linalg.Workspace
	var qkvOps [3]linalg.W8A8Op
	var guOps [2]linalg.W8A8Op

	row := func(b []float32, i, w int) []float32 { return b[i*w : i*w+w] }

	for l := 0; l < arch.NumLayers; l++ {
		lw := &m.w.Layers[l]
		global := arch.isGlobalLayer(l)

		copy(norm, h)
		for i := range K {
			normalize(arch, row(norm, i, hidden), lw.PreAttnNorm, lw.PreAttnNormBias, hidden)
		}
		if lw.QProj.isW8A8() && lw.KProj.isW8A8() && lw.VProj.isW8A8() {
			qkvOps[0] = linalg.W8A8Op{BQ: lw.QProj.q8, Scales: lw.QProj.scales, Dst: q, N: lw.QProj.rows}
			qkvOps[1] = linalg.W8A8Op{BQ: lw.KProj.q8, Scales: lw.KProj.scales, Dst: k, N: lw.KProj.rows}
			qkvOps[2] = linalg.W8A8Op{BQ: lw.VProj.q8, Scales: lw.VProj.scales, Dst: v, N: lw.VProj.rows}
			matmulW8A8Batch(be, &ws, norm, K, lw.QProj.cols, qkvOps[:])
		} else {
			lw.QProj.matmul(be, norm, q, K)
			lw.KProj.matmul(be, norm, k, K)
			lw.VProj.matmul(be, norm, v, K)
		}
		if arch.QKVBias {
			for i := range K {
				addBias(row(q, i, qDim), lw.QBias)
				addBias(row(k, i, kvDim), lw.KBias)
				addBias(row(v, i, kvDim), lw.VBias)
			}
		}
		invFreq := arch.ropeInvFreq(l)
		ms := arch.ropeMscale(l)
		for i := range K {
			pos := startPos + i
			qi, ki, vi := row(q, i, qDim), row(k, i, kvDim), row(v, i, kvDim)
			if arch.QKNorm {
				rmsNorm(qi, lw.QNorm, nH, hd, arch.NormEps, arch.RMSAddOne)
				rmsNorm(ki, lw.KNorm, nKV, hd, arch.NormEps, arch.RMSAddOne)
			}
			applyRoPE(qi, nH, hd, pos, invFreq, ms)
			applyRoPE(ki, nKV, hd, pos, invFreq, ms)
			cache.Append(l, ki, vi)
		}
		// QKᵀ and scores·V for all K positions, per head, on the SIMD A·Bᵀ kernel
		// (the L² terms) instead of the scalar per-position attendQuery.
		attendBatchedHeads(q, ctx, cache, l, startPos, K, global, arch, aqh, akh, avt, ascores, ach)
		lw.OProj.matmul(be, ctx, att, K)
		if arch.OutBias {
			for i := range K {
				addBias(row(att, i, hidden), lw.OBias)
			}
		}
		if sandwich {
			for i := range K {
				normalize(arch, row(att, i, hidden), lw.PostAttnNorm, nil, hidden)
			}
		}
		for j := range h {
			h[j] += att[j]
		}

		copy(norm, h)
		for i := range K {
			normalize(arch, row(norm, i, hidden), lw.PreMLPNorm, lw.PreMLPNormBias, hidden)
		}
		if lw.GateProj.isW8A8() && lw.UpProj.isW8A8() {
			guOps[0] = linalg.W8A8Op{BQ: lw.GateProj.q8, Scales: lw.GateProj.scales, Dst: gate, N: lw.GateProj.rows}
			guOps[1] = linalg.W8A8Op{BQ: lw.UpProj.q8, Scales: lw.UpProj.scales, Dst: up, N: lw.UpProj.rows}
			matmulW8A8Batch(be, &ws, norm, K, lw.GateProj.cols, guOps[:])
		} else {
			lw.GateProj.matmul(be, norm, gate, K)
			lw.UpProj.matmul(be, norm, up, K)
		}
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
			for i := range K {
				normalize(arch, row(mlpOut, i, hidden), lw.PostMLPNorm, nil, hidden)
			}
		}
		for j := range h {
			h[j] += mlpOut[j]
		}
	}

	for i := range K {
		normalize(arch, row(h, i, hidden), m.w.FinalNorm, m.w.FinalNormBias, hidden)
	}
	return h, nil
}

// attendBatchedHeads computes grouped-query causal attention for K query
// positions at once, per head, via the SIMD A·Bᵀ matmul (linalg.MatmulBT)
// instead of the scalar per-position attendQuery. The two O(L²) terms — QKᵀ and
// scores·V — move off the scalar triple-loops onto the vector kernel, which an
// end-to-end prefill profile showed were ~half the forward's CPU time.
//
// Per KV head it gathers K_head [nKeys,hd] and V_headᵀ [hd,nKeys] once (reused
// across the GQA group). Per query head: scores[K,nKeys] = Q_head·K_headᵀ; a
// scaled, causal/window-masked softmax per row (row i attends to
// [WindowStart(startPos+i), startPos+i], masked entries zeroed so they drop out
// of the next matmul); then ctx_head[K,hd] = scores·V_head, expressed as
// MatmulBT(scores, V_headᵀ); scattered into ctx[K,qDim].
//
// NOT bit-identical to attendQuery: QKᵀ moves from float64 to f32 accumulation
// and the matmul reassociates the reduction. Parity is argmax-exact + cosine —
// the same standard the GPU residency attention already meets. The softmax exp
// stays per-row in float64. Scratch slices (qh:[K*hd], kh:[maxKeys*hd],
// vt:[maxKeys*hd], scores:[K*maxKeys], ch:[K*hd]) are caller-owned, reused across
// layers.
func attendBatchedHeads(q, ctx []float32, cache *KVCache, layer, startPos, K int, global bool, arch *Architecture, qh, kh, vt, scores, ch []float32) {
	nH, nKV, hd := arch.NumHeads, arch.NumKVHeads, arch.HeadDim
	kvDim, qDim := nKV*hd, nH*hd
	group := nH / nKV
	scale := arch.AttnScale
	keys, vals := cache.Keys(layer), cache.Vals(layer)
	nKeys := len(keys) / kvDim

	for kvh := 0; kvh < nKV; kvh++ {
		// Gather this KV head's keys [nKeys,hd] and values transposed [hd,nKeys]
		// once; every query head in the GQA group reuses them. The V transpose is
		// folded into the gather (free) so scores·V is MatmulBT(scores, V_headᵀ).
		for s := 0; s < nKeys; s++ {
			base := s*kvDim + kvh*hd
			copy(kh[s*hd:s*hd+hd], keys[base:base+hd])
			vrow := vals[base : base+hd]
			for d := 0; d < hd; d++ {
				vt[d*nKeys+s] = vrow[d]
			}
		}
		for g := 0; g < group; g++ {
			qhead := kvh*group + g
			for i := 0; i < K; i++ { // gather Q_head [K,hd]
				base := i*qDim + qhead*hd
				copy(qh[i*hd:i*hd+hd], q[base:base+hd])
			}
			// QKᵀ: scores[K,nKeys] = Q_head[K,hd] · K_head[nKeys,hd]ᵀ
			linalg.MatmulBT(qh[:K*hd], kh[:nKeys*hd], scores[:K*nKeys], K, hd, nKeys)
			// Scaled, masked softmax per query row; zero the out-of-range entries
			// so they contribute nothing to the scores·V matmul below.
			for i := 0; i < K; i++ {
				pos := startPos + i
				start := cache.WindowStart(pos, global)
				rowS := scores[i*nKeys : i*nKeys+nKeys]
				maxS := math.Inf(-1)
				for s := start; s <= pos; s++ {
					sc := float64(rowS[s]) * scale
					rowS[s] = float32(sc)
					if sc > maxS {
						maxS = sc
					}
				}
				var sum float64
				for s := start; s <= pos; s++ {
					e := math.Exp(float64(rowS[s]) - maxS)
					rowS[s] = float32(e)
					sum += e
				}
				inv := 1.0 / sum
				for s := 0; s < start; s++ {
					rowS[s] = 0
				}
				for s := start; s <= pos; s++ {
					rowS[s] = float32(float64(rowS[s]) * inv)
				}
				for s := pos + 1; s < nKeys; s++ {
					rowS[s] = 0
				}
			}
			// scores·V: ctx_head[K,hd] = scores[K,nKeys] · V_head[nKeys,hd]
			//                          = MatmulBT(scores, V_headᵀ[hd,nKeys])
			linalg.MatmulBT(scores[:K*nKeys], vt[:hd*nKeys], ch[:K*hd], K, nKeys, hd)
			for i := 0; i < K; i++ { // scatter ctx_head into ctx[K,qDim]
				base := i*qDim + qhead*hd
				copy(ctx[base:base+hd], ch[i*hd:i*hd+hd])
			}
		}
	}
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
	for i := range K {
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
