//go:build darwin

package metal

// D-P01's batched expert GEMM (docs/audit-metal-2026-09-30.md; docs/tasks/task-m26-mac-2026-10.md, "D-P01"): a layer-major
// paged Gemma 4 prefill runs one slot group's routed experts expert by expert, every (row, expert) pair the group routes to
// that expert in one pass over its weights, instead of row by row. Bit-identical to the per-row phase 2
// (encodeG4Phase2Paged) by construction:
//   - gate|up and down run moe_batch_gemv, which computes sa_rows_acc's per-row sums for up to B activations at once: each
//     (activation, weight row) accumulator sees exactly sa_rows_acc's operations in its order, and the weights are read once
//     for all of them;
//   - SwiGLU and its int8 quantisation run mc3_swiglu_quant_rows, swiglu_quant's body per pair;
//   - the down projection stores each pair's raw sum, and moe_batch_combine adds a row's k experts in route order with the
//     per-row path's own epilogue, fma(wgt*acc, asc, out) from zero. The floating-point order is the per-row one.
//
// g4ExpertBatchOn picks it in g4LayerMajorRows; on by default.
// History: docs/code-notes/metal.md#g4ExpertBatchOn.
var g4ExpertBatchOn = true

// moeBatchB is the most pairs one moe_batch_gemv threadgroup takes: B activations of K halves in threadgroup memory,
// 22.5 KB at M26's hidden size of 2816 for gate|up.
const moeBatchB = 4

// moeBatchR is the rows a simdgroup moe_batch_gemv takes for a projection of n output rows: 4 or 2 where full
// threadgroups divide n (sa_rows_acc's condition), else 0 (the batched path declines).
func moeBatchR(n int) int {
	for _, R := range []int{4, 2} {
		if n%(8*R) == 0 {
			return R
		}
	}
	return 0
}

// g4Batch is one layer-major call's scratch for the batched phase 2, grown to the largest group it meets: pair-major
// activations, scales, router weights, gate|up outputs, SwiGLU outputs, raw down sums and each row's combined experts.
type g4Batch struct {
	rows, pairs                               int
	aq, asc, wgt, gu, dq, dsc, dacc, x2       Buffer
	ent                                       Buffer
	entCap                                    int
	uK, uN, uTilesGU, uTilesDown, uOne, uZero Buffer
}

func (r *resident) g4BatchEnsure(b *g4Batch, n, entries int, made func(Buffer) Buffer) {
	g, d := r.g4moe, r.d
	P := n * g.topK
	if n > b.rows {
		b.rows = n
		b.aq, b.asc = made(byteBuf(d, n*r.H)), made(d.NewBufferLen(n))
		b.x2 = made(d.NewBufferLen(n * r.H))
	}
	if P > b.pairs {
		b.pairs = P
		b.wgt, b.dsc = made(d.NewBufferLen(P)), made(d.NewBufferLen(P))
		b.gu = made(d.NewBufferLen(P * 2 * g.moeInter))
		b.dq = made(byteBuf(d, P*g.moeInter))
		b.dacc = made(d.NewBufferLen(P * r.H))
	}
	if entries > b.entCap {
		b.entCap = entries
		b.ent = made(NewBufferUint32s(d, make([]uint32, entries*(2+moeBatchB))))
	}
	if b.uK == (Buffer{}) {
		b.uK, b.uOne = made(NewBufferU32(d, uint32(g.topK))), made(NewBufferU32(d, 1))
		b.uZero = made(NewBufferU32(d, 0))
		b.uN = made(NewBufferU32(d, 0))
		b.uTilesGU, b.uTilesDown = made(NewBufferU32(d, 0)), made(NewBufferU32(d, 0))
	}
}

// g4BatchEntries lays out a group's moe_batch_gemv entries: for each pool slot the group uses, in first-use order, its
// pairs (row*k + j, rows counted from the group's first) in order, at most moeBatchB to an entry.
func g4BatchEntries(slots [][]uint32, topK int) []uint32 {
	var order []uint32
	pairs := map[uint32][]uint32{}
	for i, si := range slots {
		for j := range topK {
			s := si[j]
			if _, ok := pairs[s]; !ok {
				order = append(order, s)
			}
			pairs[s] = append(pairs[s], uint32(i*topK+j))
		}
	}
	var out []uint32
	for _, s := range order {
		for ps := pairs[s]; len(ps) > 0; {
			c := min(len(ps), moeBatchB)
			e := make([]uint32, 2+moeBatchB)
			e[0], e[1] = s, uint32(c)
			copy(e[2:], ps[:c])
			out = append(out, e...)
			ps = ps[c:]
		}
	}
	return out
}

// encodeG4Phase2Batch encodes a slot group's phase 2 (rows grp, their slot tables already filled) through the batched
// kernels, then each row's join from its own combined experts. The rows' phase 1 is complete (the host has read their
// routes), so their activations, scales and router weights are packed here, pair-major.
func (r *resident) encodeG4Phase2Batch(e *Encoder, pool *expertPool, b *g4Batch, grp []*g4Row,
	join func(e *Encoder, rw *g4Row, x2 Buffer), made func(Buffer) Buffer) {
	g := r.g4moe
	n, k, H, I := len(grp), g.topK, r.H, g.moeInter
	slots := make([][]uint32, n)
	for i, rw := range grp {
		slots[i] = rw.slotIdx.U32s()[:k]
	}
	ent := g4BatchEntries(slots, k)
	nEnt := len(ent) / (2 + moeBatchB)
	r.g4BatchGroups++
	for x := range nEnt {
		r.g4BatchMaxCnt = max(r.g4BatchMaxCnt, int(ent[x*(2+moeBatchB)+1]))
	}
	r.g4BatchEnsure(b, n, nEnt, made)
	aq, asc, wgt := b.aq.Int8s(), b.asc.Floats(), b.wgt.Floats()
	for i, rw := range grp {
		copy(aq[i*H:(i+1)*H], rw.mq.Int8s()[:H])
		asc[i] = rw.mSc.Floats()[0]
		copy(wgt[i*k:(i+1)*k], rw.rWgt.Floats()[:k])
	}
	copy(b.ent.U32s(), ent)
	b.uN.SetU32(uint32(n))
	guR, downR := moeBatchR(2*I), moeBatchR(H)
	tilesGU, tilesDown := 2*I/(8*guR), H/(8*downR)
	b.uTilesGU.SetU32(uint32(tilesGU))
	b.uTilesDown.SetU32(uint32(tilesDown))
	e.DispatchTG(r.pMoeBatch[guR], nEnt*tilesGU*256, 256, moeBatchB*H*2,
		pool.guW, pool.guS, b.aq, b.asc, b.gu, r.uH, b.ent, g.uMoeGU, b.uTilesGU, b.uK, b.uOne)
	e.Dispatch(r.pSwRows, n*k*256, 256, b.gu, b.gu, b.dq, b.dsc, g.uMoeInter, r.uAct)
	e.DispatchTG(r.pMoeBatch[downR], nEnt*tilesDown*256, 256, moeBatchB*I*2,
		pool.dW, pool.dS, b.dq, b.dsc, b.dacc, g.uMoeInter, b.ent, r.uH, b.uTilesDown, b.uOne, b.uZero)
	e.Dispatch(r.pMoeCombine, n*H, 256, b.dacc, b.wgt, b.dsc, b.x2, r.uH, b.uK, b.uN)
	for i, rw := range grp {
		join(e, rw, b.x2.At(4*i*H))
	}
}

// g4ExpertBatchOK reports whether the batched phase 2 can take this resident's shapes: both projections in full
// threadgroups, and the gate|up activations of a full entry within threadgroup memory.
func (r *resident) g4ExpertBatchOK() bool {
	g := r.g4moe
	return g4ExpertBatchOn && g != nil && r.pMoeCombine != (Pipeline{}) && moeBatchR(2*g.moeInter) > 0 &&
		moeBatchR(r.H) > 0 && moeBatchB*max(r.H, g.moeInter)*2 <= 32*1024
}
