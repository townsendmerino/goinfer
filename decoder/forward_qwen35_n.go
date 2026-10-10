package decoder

import (
	"context"
)

// The batched Qwen3.5 forward: runLayersQwen35FromEmbed over K rows at once, for PromptHidden (Route B's hidden state,
// docs/tasks/task-constrained-confidence.md). The per-token forward reads every weight once per token, which on a CPU is decode
// speed and memory-bandwidth bound. Here every projection is one M=K matmul, so a weight is read once per prompt:
//
//   - DeltaNet layers: in_proj_qkv, in_proj_z and out_proj are batched. The conv window, the gates, the recurrence and the gated
//     norm stay sequential per token, through deltaNetCore, the same code the per-token step runs.
//   - Full-attention layers: q/k/v/o are batched and attention is attendBatchedHeads over the cache, with f64 accumulation
//     (useAcc64), which its own comment records as bit-identical to the sequential attendQuery.
//   - FFN: a dense SwiGLU is batched; an MoE runs moeMLP per row, as the generic batched path does.
//
// It is not claimed bit-identical to the per-token forward: a batched f32 or quantized matmul may reduce in a different order from
// its matvec. TestPromptHidden_batchedMatchesSequential bounds the difference.
//
// qwen35BatchN reports whether it applies. Olmo Hybrid (PlainFullAttn, per-layer norm placement), a capture request and a dense
// layer pager take the per-token path.
func (m *Model) qwen35BatchN(K int, cache *KVCache) bool {
	return cache.mropePos == nil && m.qwen35BatchNAnyPos(K, cache)
}

// qwen35BatchNAnyPos is qwen35BatchN without its m-RoPE exclusion: the image prefill (prefillLogitsQwen35VL) runs this forward
// over a cache that carries m-RoPE positions, which runLayersQwen35N rotates the full-attention layers' q and k by. qwen35BatchN
// keeps the exclusion for its other callers (prefix sharing documents "no m-RoPE" as a precondition).
func (m *Model) qwen35BatchNAnyPos(K int, cache *KVCache) bool {
	a := m.w.arch
	return K > 1 && a.qwen35 != nil && !a.qwen35.PlainFullAttn && a.NormPlacementLinear == nil &&
		a.NormPlacement == NormPre2 && m.layerPager == nil && cache.captureLayers == nil && cache.treeMask == nil &&
		!cache.localAny && m.w.Embed.Rows() != 0 &&
		(a.MoE != nil || a.Act == ActSiLU)
}

// runLayersQwen35N runs K embedded rows h [K*HiddenDim] through every layer, advances cache by K, and returns the
// post-final-norm hidden states [K*HiddenDim]. h is consumed. The caller checks qwen35BatchN first.
func (m *Model) runLayersQwen35N(reqCtx context.Context, h []float32, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	be := m.be
	g := arch.qwen35
	hidden, eps := arch.HiddenDim, arch.NormEps
	nH, nKV, hd := arch.NumHeads, arch.NumKVHeads, arch.HeadDim
	qDim, kvDim := nH*hd, nKV*hd
	K := len(h) / hidden
	startPos := cache.Pos()
	nk, nv := g.NumKeyHeads, g.NumValueHeads
	keyDim, valueDim := g.KeyHeadDim*nk, g.ValueHeadDim*nv
	convDim := 2*keyDim + valueDim

	row := func(b []float32, i, w int) []float32 { return b[i*w : i*w+w] }
	norm := make([]float32, K*hidden)
	mix := make([]float32, K*hidden) // the attention or DeltaNet output
	ffn := make([]float32, K*hidden)
	var dnZ, dnCore []float32
	var qg, q, gate, k, v, ctx []float32
	var gt, up []float32
	var attnPool []headWorkerScratch
	var alk, alv []float32

	var hq w4a8Act // the K-row normed block, quantized once per projection group (R-13); reused per layer
	for l := 0; l < arch.NumLayers; l++ {
		if err := reqCtx.Err(); err != nil {
			return nil, err
		}
		lw := &m.w.Layers[l]
		for i := range K {
			rmsNormInto(row(norm, i, hidden), row(h, i, hidden), lw.PreAttnNorm, 1, hidden, eps, arch.RMSAddOne)
		}

		if arch.isLinearLayer(l) {
			d := lw.delta
			st := cache.delta[l]
			// A fresh mixed buffer per layer: deltaNetCore keeps each row in the conv window.
			mixed := make([]float32, K*convDim)
			if dnZ == nil {
				dnZ, dnCore = make([]float32, K*valueDim), make([]float32, K*valueDim)
			}
			hq.prepare(be, &d.inProjQKV, norm, K)
			matmulPre(be, &d.inProjQKV, &hq, norm, mixed, K)
			matmulPre(be, &d.inProjZ, &hq, norm, dnZ, K)
			if deltaNetCoreNOK(K) {
				// The gate projections per row, then the conv, the recurrence (heads in parallel) and the gated norm over all K rows at once;
				// bit-identical to the per-token loop below (deltanet_n.go).
				bts, ats := make([]float32, K*nv), make([]float32, K*nv)
				fanOut(K, deltaNetWorkers(K), func(lo, hi int) {
					for i := lo; i < hi; i++ {
						n := row(norm, i, hidden)
						copy(bts[i*nv:(i+1)*nv], matvec(d.inProjB, nv, hidden, n))
						copy(ats[i*nv:(i+1)*nv], matvec(d.inProjA, nv, hidden, n))
					}
				})
				deltaNetCoreN(dnCore, mixed, bts, ats, dnZ, K, d, *g, eps, st)
			} else {
				for i := range K {
					n := row(norm, i, hidden)
					bt := matvec(d.inProjB, nv, hidden, n)
					at := matvec(d.inProjA, nv, hidden, n)
					deltaNetCore(row(dnCore, i, valueDim), row(mixed, i, convDim), bt, at, row(dnZ, i, valueDim), d, *g, eps, st)
				}
			}
			matmul(be, &d.outProj, dnCore, mix, K)
		} else {
			a := lw.qattn
			if qg == nil {
				qg, q, gate = make([]float32, K*2*qDim), make([]float32, K*qDim), make([]float32, K*qDim)
				k, v, ctx = make([]float32, K*kvDim), make([]float32, K*kvDim), make([]float32, K*qDim)
				attnPool = newHeadWorkerPoolK(m.knobs, prefillAttnWorkersK(m.knobs, K, startPos+K, hd, nH), K, startPos+K, hd, false)
				if cache.quant == kvI8 {
					alk, alv = make([]float32, (startPos+K)*kvDim), make([]float32, (startPos+K)*kvDim)
				}
			}
			hq.prepare(be, &a.qProj, norm, K)
			matmulPre(be, &a.qProj, &hq, norm, qg, K)
			matmulPre(be, &a.kProj, &hq, norm, k, K)
			matmulPre(be, &a.vProj, &hq, norm, v, K)
			invFreq := arch.ropeInvFreq(l)
			ms := arch.ropeMscale(l)
			for i := range K {
				pos := startPos + i
				qgi, qi, gi := row(qg, i, 2*qDim), row(q, i, qDim), row(gate, i, qDim)
				splitQGate(qgi, qi, gi, nH, hd) // q_proj emits [query ‖ gate] per head
				ki, vi := row(k, i, kvDim), row(v, i, kvDim)
				rmsNorm(qi, a.qNorm, nH, hd, eps, arch.RMSAddOne)
				rmsNorm(ki, a.kNorm, nKV, hd, eps, arch.RMSAddOne)
				ropeAt(qi, nH, hd, pos, invFreq, ms, arch.MRopeSection, cache.mropePos, cache.mropeDelta, false, arch.MRopeInterleaved)
				ropeAt(ki, nKV, hd, pos, invFreq, ms, arch.MRopeSection, cache.mropePos, cache.mropeDelta, false, arch.MRopeInterleaved)
				cache.Append(l, ki, vi)
			}
			if cache.quant == kvI8 {
				nKeys := cache.dequantGlobalLayer(l, kvDim, alk, alv)
				attendBatchedHeads(q, ctx, alk[:nKeys*kvDim], alv[:nKeys*kvDim], 0, cache, l, startPos, K, true, arch, true, attnPool)
			} else {
				attendBatchedHeads(q, ctx, cache.Keys(l), cache.Vals(l), 0, cache, l, startPos, K, true, arch, true, attnPool)
			}
			qGateContext(ctx, gate)
			matmul(be, &a.oProj, ctx, mix, K)
		}
		addResidual(h, mix)

		for i := range K {
			rmsNormInto(row(norm, i, hidden), row(h, i, hidden), lw.PreMLPNorm, 1, hidden, eps, arch.RMSAddOne)
		}
		if arch.MoE == nil {
			inter := arch.IntermediateDim
			if gt == nil {
				gt, up = make([]float32, K*inter), make([]float32, K*inter)
			}
			hq.prepare(be, &lw.GateProj, norm, K)
			matmulPre(be, &lw.GateProj, &hq, norm, gt, K)
			matmulPre(be, &lw.UpProj, &hq, norm, up, K)
			if len(gt) < activationFanoutThreshold {
				swiglu(gt, up)
			} else {
				parallelElementwise(len(gt), func(lo, hi int) {
					for j := lo; j < hi; j++ {
						gt[j] = silu(gt[j]) * up[j]
					}
				})
			}
			matmul(be, &lw.DownProj, gt, ffn, K)
		} else {
			for i := range K {
				ff, err := moeMLP(row(norm, i, hidden), lw, arch, be, nil, m.pager)
				if err != nil {
					return nil, err
				}
				copy(row(ffn, i, hidden), ff)
			}
		}
		addResidual(h, ffn)
	}
	cache.advanceTo(startPos + K)
	for i := range K {
		normalize(arch, row(h, i, hidden), m.w.FinalNorm, m.w.FinalNormBias, hidden)
	}
	return h, nil
}
