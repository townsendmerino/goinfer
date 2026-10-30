package decoder

// walkBatchPlantForTest plants one defect in the batched walker's qwen3.5 or Gemma 4 operations, so the bounds that cover them
// can show they go red (walkbatch_plants_test.go). Zero in production.
var walkBatchPlantForTest walkBatchPlant

type walkBatchPlant uint8

const (
	plantBatchNone            walkBatchPlant = iota
	plantBatchNoAttnGate                     // gatedAttnBatch skips the sigmoid gate on the first gated-attention layer
	plantBatchDropConvWindow                 // deltaNetBatch starts every pass from an empty conv window
	plantBatchNoBidirectional                // gemma4BatchRange ends a span row's range at its own position
	plantBatchImagePLEToken                  // newGemma4Pass gives an image row its token id's PLE, not the pad id's
)

// qwen35PassScratch is the batched walker's buffers for qwen3.5's two mixers, started on the pass's first such layer. The
// DeltaNet's mixed buffer is not here: deltaNetCore keeps each row in the conv window, so every layer takes a fresh one.
type qwen35PassScratch struct {
	dnZ, dnCore, bts, ats []float32 // [K*valueDim], [K*valueDim], [K*nv], [K*nv]
	qg, gate              []float32 // [K*2*qDim], [K*qDim]
}

// q35 returns the pass's qwen3.5 buffers, starting them on first use.
func (ps *prefillPassScratch) q35() *qwen35PassScratch {
	if ps.qwen35 == nil {
		ps.qwen35 = &qwen35PassScratch{}
	}
	return ps.qwen35
}

// deltaNetBatch is opDeltaNet over K rows of x ([K*hidden]) into out: in_proj_qkv and in_proj_z at M=K, the gate projections
// per row, then the conv, the recurrence and the gated norm over st, and out_proj at M=K. The core runs over all K rows at once
// when deltaNetCoreNOK, else row by row through deltaNetCore; the two are bit-identical (deltanet_n.go).
func (m *Model) deltaNetBatch(p deltaNetOpParams, x, out []float32, lw *LayerWeights, st *deltaState, K int, hq *w4a8Act, s *qwen35PassScratch) {
	arch, be := m.w.arch, m.be
	g := *arch.qwen35
	g.ONormEps, g.NegEigval = p.ONormEps, p.NegEigval
	hidden := arch.HiddenDim
	nv := g.NumValueHeads
	keyDim, valueDim := g.KeyHeadDim*g.NumKeyHeads, g.ValueHeadDim*nv
	convDim := 2*keyDim + valueDim
	row := func(b []float32, i, w int) []float32 { return b[i*w : i*w+w] }

	d := lw.delta
	if walkBatchPlantForTest == plantBatchDropConvWindow {
		st.convWin = nil
	}
	mixed := make([]float32, K*convDim)
	s.dnZ, s.dnCore = sizedF32(s.dnZ, K*valueDim), sizedF32(s.dnCore, K*valueDim)
	hq.prepare(be, &d.inProjQKV, x, K)
	matmulPre(be, &d.inProjQKV, hq, x, mixed, K)
	matmulPre(be, &d.inProjZ, hq, x, s.dnZ, K)
	if deltaNetCoreNOK(K) {
		s.bts, s.ats = sizedF32(s.bts, K*nv), sizedF32(s.ats, K*nv)
		bts, ats := s.bts, s.ats
		fanOut(K, deltaNetWorkers(K), func(lo, hi int) {
			for i := lo; i < hi; i++ {
				n := row(x, i, hidden)
				copy(bts[i*nv:(i+1)*nv], matvec(d.inProjB, nv, hidden, n))
				copy(ats[i*nv:(i+1)*nv], matvec(d.inProjA, nv, hidden, n))
			}
		})
		deltaNetCoreN(s.dnCore, mixed, bts, ats, s.dnZ, K, d, g, p.Eps, st)
	} else {
		for i := range K {
			n := row(x, i, hidden)
			bt := matvec(d.inProjB, nv, hidden, n)
			at := matvec(d.inProjA, nv, hidden, n)
			deltaNetCore(row(s.dnCore, i, valueDim), row(mixed, i, convDim), bt, at, row(s.dnZ, i, valueDim), d, g, p.Eps, st)
		}
	}
	matmul(be, &d.outProj, s.dnCore, out, K)
}

// gatedAttnBatch is opGatedAttn over K rows of x into out: the double-width q_proj and k/v at M=K, then per row the [query ‖
// gate] split, the QK-norm, the rotation at the row's position and the K/V append; attention for all K rows over the cache with
// f64 accumulation (attendBatchedHeads, bit-identical to the per-token attendQueryP); the sigmoid gate; o_proj at M=K.
// q, k, v and ctx are the walker's [K*qDim], [K*kvDim], [K*kvDim] and [K*qDim] buffers; alk/alv its dequantised-keys buffers
// for a widened cache.
func (m *Model) gatedAttnBatch(p *gatedAttnParams, x, out []float32, lw *LayerWeights, cache *KVCache, l, startPos, K int,
	q, k, v, ctx, alk, alv []float32, pool []headWorkerScratch, hq *w4a8Act, s *qwen35PassScratch) {
	arch, be := m.w.arch, m.be
	nH, nKV, hd := arch.NumHeads, arch.NumKVHeads, arch.HeadDim
	qDim, kvDim := nH*hd, nKV*hd
	row := func(b []float32, i, w int) []float32 { return b[i*w : i*w+w] }

	a := lw.qattn
	s.qg, s.gate = sizedF32(s.qg, K*2*qDim), sizedF32(s.gate, K*qDim)
	hq.prepare(be, &a.qProj, x, K)
	matmulPre(be, &a.qProj, hq, x, s.qg, K)
	matmulPre(be, &a.kProj, hq, x, k, K)
	matmulPre(be, &a.vProj, hq, x, v, K)
	for i := range K {
		qi, gi := row(q, i, qDim), row(s.gate, i, qDim)
		splitQGate(row(s.qg, i, 2*qDim), qi, gi, nH, hd) // q_proj emits [query ‖ gate] per head
		ki, vi := row(k, i, kvDim), row(v, i, kvDim)
		applyQKNorm(p.QKNorm, qi, ki, a.qNorm, a.kNorm, nH, nKV, hd)
		applyRope(&p.Rope, qi, ki, nH, nKV, hd, startPos+i, cache.mropePos, cache.mropeDelta)
		cache.Append(l, ki, vi)
	}
	if cache.quant.widened() {
		nKeys := cache.dequantGlobalLayer(l, kvDim, alk, alv)
		attendBatchedHeads(q, ctx, alk[:nKeys*kvDim], alv[:nKeys*kvDim], 0, cache, l, startPos, K, true, arch, true, pool)
	} else {
		attendBatchedHeads(q, ctx, cache.Keys(l), cache.Vals(l), 0, cache, l, startPos, K, true, arch, true, pool)
	}
	if walkBatchPlantForTest != plantBatchNoAttnGate || l != firstNonLinearLayer(arch) {
		qGateContext(ctx, s.gate)
	}
	matmul(be, &a.oProj, ctx, out, K)
}

// firstNonLinearLayer is the first layer of a that is not a linear (DeltaNet) layer, or -1; plantBatchNoAttnGate's target.
func firstNonLinearLayer(a *Architecture) int {
	for l := range a.NumLayers {
		if !a.isLinearLayer(l) {
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
