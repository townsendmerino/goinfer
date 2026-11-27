package cpuengine

import (
	darch "github.com/townsendmerino/goinfer/decoder/arch"
)

import "github.com/townsendmerino/goinfer/internal/runtime/hybridattn"

// qwen35PassScratch is the batched walker's buffers for qwen3.5's two mixers, started on the pass's first such layer. The
// DeltaNet's mixed buffer is not here: deltaNetCore keeps each row in the conv window, so every layer takes a fresh one.
type qwen35PassScratch struct {
	dnZ, dnCore, bts, ats []float32 // [K*valueDim], [K*valueDim], [K*nv], [K*nv]
	qg, gate              []float32 // [K*2*qDim], [K*qDim]
}

// q35 returns the pass's qwen3.5 buffers, starting them on first use.
func (ps *PrefillPassScratch) q35() *qwen35PassScratch {
	if ps.qwen35 == nil {
		ps.qwen35 = &qwen35PassScratch{}
	}
	return ps.qwen35
}

// deltaNetBatch is opDeltaNet over K rows of x ([K*hidden]) into out: in_proj_qkv and in_proj_z at M=K, the gate projections
// per row, then the conv, the recurrence and the gated norm over st, and out_proj at M=K. The core runs over all K rows at once
// when deltaNetCoreNOK, else row by row through deltaNetCore; the two are bit-identical (deltanet_n.go).
func (m *Engine) deltaNetBatch(p deltaNetOpParams, x, out []float32, lw *darch.LayerWeights, st *DeltaState, K int, hq *W4A8Act, s *qwen35PassScratch) {
	arch, be := m.W.Arch, m.Be
	g := *arch.Qwen35
	g.ONormEps, g.NegEigval = p.ONormEps, p.NegEigval
	hidden := arch.HiddenDim
	nv := g.NumValueHeads
	keyDim, valueDim := g.KeyHeadDim*g.NumKeyHeads, g.ValueHeadDim*nv
	convDim := 2*keyDim + valueDim
	row := func(b []float32, i, w int) []float32 { return b[i*w : i*w+w] }

	d := lw.Delta
	if activeHooks().WalkBatchPlant == PlantBatchDropConvWindow {
		st.ConvWin = nil
	}
	mixed := make([]float32, K*convDim)
	s.dnZ, s.dnCore = sizedF32(s.dnZ, K*valueDim), sizedF32(s.dnCore, K*valueDim)
	hq.Prepare(be, &d.InProjQKV, x, K)
	MatmulPre(be, &d.InProjQKV, hq, x, mixed, K)
	MatmulPre(be, &d.InProjZ, hq, x, s.dnZ, K)
	if deltaNetCoreNOK(K) {
		s.bts, s.ats = sizedF32(s.bts, K*nv), sizedF32(s.ats, K*nv)
		bts, ats := s.bts, s.ats
		fanOut(K, deltaNetWorkers(K), func(lo, hi int) {
			for i := lo; i < hi; i++ {
				n := row(x, i, hidden)
				copy(bts[i*nv:(i+1)*nv], hybridattn.Matvec(d.InProjB, nv, hidden, n))
				copy(ats[i*nv:(i+1)*nv], hybridattn.Matvec(d.InProjA, nv, hidden, n))
			}
		})
		DeltaNetCoreN(s.dnCore, mixed, bts, ats, s.dnZ, K, d, g, p.Eps, st)
	} else {
		for i := range K {
			n := row(x, i, hidden)
			bt := hybridattn.Matvec(d.InProjB, nv, hidden, n)
			at := hybridattn.Matvec(d.InProjA, nv, hidden, n)
			DeltaNetCore(row(s.dnCore, i, valueDim), row(mixed, i, convDim), bt, at, row(s.dnZ, i, valueDim), d, g, p.Eps, st)
		}
	}
	Matmul(be, &d.OutProj, s.dnCore, out, K)
}

// gatedAttnBatch is opGatedAttn over K rows of x into out: the double-width q_proj and k/v at M=K, then per row the [query ‖
// gate] split, the QK-norm, the rotation at the row's position and the K/V append; attention for all K rows over the cache with
// f64 accumulation (AttendBatchedHeads, bit-identical to the per-token attendQueryP); the sigmoid gate; o_proj at M=K.
// q, k, v and ctx are the walker's [K*qDim], [K*kvDim], [K*kvDim] and [K*qDim] buffers; alk/alv its dequantised-keys buffers
// for a widened cache.
func (m *Engine) gatedAttnBatch(p *gatedAttnParams, x, out []float32, lw *darch.LayerWeights, cache *Cache, l, startPos, K int,
	q, k, v, ctx, alk, alv []float32, pool []HeadWorkerScratch, hq *W4A8Act, s *qwen35PassScratch) {
	arch, be := m.W.Arch, m.Be
	nH, nKV, hd := arch.NumHeads, arch.NumKVHeads, arch.HeadDim
	qDim, kvDim := nH*hd, nKV*hd
	row := func(b []float32, i, w int) []float32 { return b[i*w : i*w+w] }

	a := lw.QAttn
	s.qg, s.gate = sizedF32(s.qg, K*2*qDim), sizedF32(s.gate, K*qDim)
	hq.Prepare(be, &a.QProj, x, K)
	MatmulPre(be, &a.QProj, hq, x, s.qg, K)
	MatmulPre(be, &a.KProj, hq, x, k, K)
	MatmulPre(be, &a.VProj, hq, x, v, K)
	for i := range K {
		qi, gi := row(q, i, qDim), row(s.gate, i, qDim)
		hybridattn.SplitQGate(row(s.qg, i, 2*qDim), qi, gi, nH, hd) // q_proj emits [query ‖ gate] per head
		ki, vi := row(k, i, kvDim), row(v, i, kvDim)
		applyQKNorm(p.QKNorm, qi, ki, a.QNorm, a.KNorm, nH, nKV, hd)
		applyRope(&p.Rope, qi, ki, nH, nKV, hd, startPos+i, cache.MRopePos, cache.MRopeDelta)
		cache.Append(l, ki, vi)
	}
	if cache.Quant.widened() {
		nKeys := cache.DequantGlobalLayer(l, kvDim, alk, alv)
		AttendBatchedHeads(q, ctx, alk[:nKeys*kvDim], alv[:nKeys*kvDim], 0, cache, l, startPos, K, true, arch, true, pool)
	} else {
		AttendBatchedHeads(q, ctx, cache.Keys(l), cache.Vals(l), 0, cache, l, startPos, K, true, arch, true, pool)
	}
	if activeHooks().WalkBatchPlant != PlantBatchNoAttnGate || l != firstNonLinearLayer(arch) {
		hybridattn.QGateContext(ctx, s.gate)
	}
	Matmul(be, &a.OProj, ctx, out, K)
}

// firstNonLinearLayer is the first layer of a that is not a linear (DeltaNet) layer, or -1; PlantBatchNoAttnGate's target.
func firstNonLinearLayer(a *darch.Architecture) int {
	for l := range a.NumLayers {
		if !a.IsLinearLayer(l) {
			return l
		}
	}
	return -1
}

// sizedF32 returns b[:n], allocating when b holds less. The contents are not cleared: every caller overwrites them.
func sizedF32(b []float32, n int) []float32 {
	if cap(b) < n {
		return make([]float32, n)
	}
	return b[:n]
}
