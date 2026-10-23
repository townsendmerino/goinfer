//go:build darwin

package metal

// KS-D9's batched phase 2 for the generic paged MoE prefill (docs/tasks/task-kernel-survey-2026-10.md, "KS-D9"): D-P01's
// batched expert GEMM (prefill_g4batch.go) on moeLayerMajorRows' slot groups. Bit-identical to the per-row phase 2
// (encodeMoEExpertsPaged) by construction: moe_batch_gemv is sa_rows_acc per pair, as gemv_w4a8_moe_rows is per row;
// mc3_swiglu_quant_rows is swiglu_quant per pair; moe_batch_combine_acc adds a row's k down sums into its residual in route
// order with gemv_w4a8_moe_wacc's epilogue. The shared expert then runs per row, as before.

// moeExpertBatchOn picks the batched phase 2 in moeLayerMajorRows; on by default. Tests turn it off for the per-row arm.
var moeExpertBatchOn = true

// moeExpertBatchOK reports whether the batched phase 2 can take this resident's routed experts: int4 and unbiased (not the
// native int8 lane, not gpt-oss), both projections in full threadgroups, and a full entry's gate|up activations within
// threadgroup memory.
func (r *resident) moeExpertBatchOK() bool {
	mo := r.moe
	return moeExpertBatchOn && mo != nil && !mo.w8 && !mo.isGptOss && r.pMoeCombineAcc != (Pipeline{}) &&
		moeBatchR(2*mo.inter) > 0 && moeBatchR(r.H) > 0 && moeBatchB*max(r.H, mo.inter)*2 <= 32*1024
}

// moeBatchRows is one slot group's rows for encodeMoEPhase2Batch: each row's residual, post-attention int8 activation and
// scale, router weights and slot table, and how to make it the resident's current row (for its shared expert).
type moeBatchRows struct {
	n                         int
	x, mq, mSc, rWgt, slotIdx func(i int) Buffer
	use                       func(i int)
}

// moeBatch is one moeLayerMajorRows call's scratch for the batched phase 2, grown to the largest group it meets: pair-major
// activations and scales, router weights, gate|up outputs, SwiGLU outputs, raw down sums, the rows' packed residuals and the
// moe_batch_gemv entries.
type moeBatch struct {
	rows, pairs, entCap                       int
	aq, asc, xs, wgt, gu, dq, dsc, dacc, ent  Buffer
	uK, uN, uTilesGU, uTilesDown, uOne, uZero Buffer
}

func (r *resident) moeBatchEnsure(b *moeBatch, n, entries int, made func(Buffer) Buffer) {
	mo, d := r.moe, r.d
	P := n * mo.k
	if n > b.rows {
		b.rows = n
		b.aq, b.asc, b.xs = made(byteBuf(d, n*r.H)), made(d.NewBufferLen(n)), made(d.NewBufferLen(n*r.H))
	}
	if P > b.pairs {
		b.pairs = P
		b.wgt, b.dsc = made(d.NewBufferLen(P)), made(d.NewBufferLen(P))
		b.gu = made(d.NewBufferLen(P * 2 * mo.inter))
		b.dq = made(byteBuf(d, P*mo.inter))
		b.dacc = made(d.NewBufferLen(P * r.H))
	}
	if entries > b.entCap {
		b.entCap = entries
		b.ent = made(NewBufferUint32s(d, make([]uint32, entries*(2+moeBatchB))))
	}
	if b.uK == (Buffer{}) {
		b.uK, b.uOne, b.uZero = made(NewBufferU32(d, uint32(mo.k))), made(NewBufferU32(d, 1)), made(NewBufferU32(d, 0))
		b.uN, b.uTilesGU, b.uTilesDown = made(NewBufferU32(d, 0)), made(NewBufferU32(d, 0)), made(NewBufferU32(d, 0))
	}
}

// encodeMoEPhase2Batch encodes one slot group's phase 2 through the batched kernels, then each row's shared expert. The rows'
// phase 1 is complete and their slot tables filled, so their residuals, activations, scales and router weights are packed on
// the host; the combined residuals are copied back to each row after its shared expert.
func (r *resident) encodeMoEPhase2Batch(e *Encoder, L *residLayer, pool *expertPool, b *moeBatch, rs moeBatchRows,
	made func(Buffer) Buffer) {
	mo := r.moe
	n, k, H, I := rs.n, mo.k, r.H, mo.inter
	slots := make([][]uint32, n)
	for i := range n {
		slots[i] = rs.slotIdx(i).U32s()[:k]
	}
	ent := g4BatchEntries(slots, k)
	nEnt := len(ent) / (2 + moeBatchB)
	r.moeBatchGroups++
	r.moeBatchEnsure(b, n, nEnt, made)
	aq, asc, wgt, xs := b.aq.Int8s(), b.asc.Floats(), b.wgt.Floats(), b.xs.Floats()
	for i := range n {
		copy(aq[i*H:(i+1)*H], rs.mq(i).Int8s()[:H])
		asc[i] = rs.mSc(i).Floats()[0]
		copy(wgt[i*k:(i+1)*k], rs.rWgt(i).Floats()[:k])
		copy(xs[i*H:(i+1)*H], rs.x(i).Floats()[:H])
	}
	copy(b.ent.U32s(), ent)
	b.uN.SetU32(uint32(n))
	guR, downR := moeBatchR(2*I), moeBatchR(H)
	tilesGU, tilesDown := 2*I/(8*guR), H/(8*downR)
	b.uTilesGU.SetU32(uint32(tilesGU))
	b.uTilesDown.SetU32(uint32(tilesDown))
	e.DispatchTG(r.pMoeBatch[guR], nEnt*tilesGU*256, 256, moeBatchB*H*2,
		pool.guW, pool.guS, b.aq, b.asc, b.gu, r.uH, b.ent, mo.uInter2, b.uTilesGU, b.uK, b.uOne)
	e.Dispatch(r.pSwRows, n*k*256, 256, b.gu, b.gu, b.dq, b.dsc, mo.uInter, r.uAct)
	e.DispatchTG(r.pMoeBatch[downR], nEnt*tilesDown*256, 256, moeBatchB*I*2,
		pool.dW, pool.dS, b.dq, b.dsc, b.dacc, mo.uInter, b.ent, r.uH, b.uTilesDown, b.uOne, b.uZero)
	e.Dispatch(r.pMoeCombineAcc, n*H, 256, b.dacc, b.wgt, b.dsc, b.xs, r.uH, b.uK, b.uN)
	for i := range n {
		rs.use(i)
		xi := b.xs.At(4 * i * H)
		r.encodeMoESharedExpert(e, L, xi)
		e.Dispatch(r.pCopyVec, H, 256, xi, rs.x(i), r.uH)
	}
}
