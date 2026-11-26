//go:build darwin

package metal

import (
	"fmt"
	"math"
	"runtime"
	"slices"
)

// forwardMultiOut is forwardMultiInto with an explicit output mode (multiOut).
func (r *resident) forwardMultiOut(seqs []batchSeq, out multiOut) (logits [][]float32, ids []int, err error) {
	argmaxOnly := out == multiArgmax
	if err := r.checkStepRows(seqs); err != nil {
		return nil, nil, err
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	r.stopExec()
	if err := r.kvMapStep(seqs); err != nil { // every row's K/V block, before anything is encoded or reset
		return nil, nil, err
	}
	r.bindStepRows(seqs)
	s := r.beginStep(seqs)
	for l := 0; l < r.nL; l++ {
		if L := &r.layers[l]; r.stepWholeLayer(l) {
			s.wholeLayer(l, L)
		} else {
			s.layerSteps(l, L)
		}
	}
	if out == multiNoHead { // the caller reads nothing from this step but the K/V it wrote
		if err := s.end(); err != nil {
			return nil, nil, err
		}
		r.batch.noHeadSteps++
		return nil, nil, nil
	}
	s.head(out)
	greedyDraw := s.draws()
	devArg := s.deviceArgmax(argmaxOnly, greedyDraw)
	if err := s.end(); err != nil {
		return nil, nil, err
	}
	logits, ids = s.read(out, argmaxOnly, greedyDraw, devArg)
	return logits, ids, nil
}

// checkStepRows is forwardMultiOut's refusal of a step it cannot run, before anything is stopped, mapped or encoded.
func (r *resident) checkStepRows(seqs []batchSeq) error {
	if r.batch == nil {
		return fmt.Errorf("metal: batched decode unavailable")
	}
	B := len(seqs)
	if B < 1 || B > batchMaxSeqs {
		return fmt.Errorf("metal: batched step of %d sequences (1..%d)", B, batchMaxSeqs)
	}
	if r.loraLayers != nil {
		return fmt.Errorf("metal: batched step with an adapter bound")
	}
	// A slot may carry several rows only as ONE sequence's consecutive positions, in row order (a speculative verify:
	// every row's K/V is stored before its attention runs, layer by layer, so each row sees the rows below it as
	// sequential decode would). Two sequences on one slot would overwrite each other.
	last := map[int]int{}
	for _, q := range seqs {
		if q.slot < 0 || q.slot >= len(r.kvSlotBufs) {
			return fmt.Errorf("metal: batched step: slot %d out of range", q.slot)
		}
		if p, ok := last[q.slot]; ok && q.pos != p+1 {
			return fmt.Errorf("metal: batched step: slot %d repeated at position %d after %d (rows on one slot must be consecutive positions of one sequence)", q.slot, q.pos, p)
		}
		last[q.slot] = q.pos
		if q.pos < 0 || q.pos >= r.ctxCap {
			return fmt.Errorf("metal: batched step: position %d outside the resident context %d", q.pos, r.ctxCap)
		}
		if len(q.emb) != r.H+r.nL*r.pleP { // hidden, and on a Gemma 4 E-model the L·P per-layer inputs (metalResident.embLen)
			return fmt.Errorf("metal: batched step: embedding length %d, want %d (hidden %d, and %d per-layer inputs on a Gemma 4 E-model)",
				len(q.emb), r.H+r.nL*r.pleP, r.H, r.nL*r.pleP)
		}
		if q.draw != nil && !r.SampleAvailable() {
			return fmt.Errorf("metal: batched step: a device draw on a resident that cannot sample on-device")
		}
	}
	return nil
}

// bindStepRows writes each row's inputs into the step's buffers: a new sequence's recurrent state zeroed, then its
// embedding, per-layer inputs and learned position, and its position, key count, table offset, query temperature and
// attention split uniforms.
func (r *resident) bindStepRows(seqs []batchSeq) {
	b, B := r.batch, len(seqs)
	for _, q := range seqs {
		if q.pos == 0 { // a new sequence: its slot's recurrent state starts zeroed, as Forward at position 0 does
			r.resetRecurrentSlot(q.slot)
		}
	}
	if r.attnFANKV > 0 {
		r.uAttnFAG.SetU32(uint32(r.nH / r.attnFANKV))
	}
	b.uM.SetU32(uint32(B))
	for m, q := range seqs {
		copy(b.xB.Floats()[m*r.H:(m+1)*r.H], q.emb[:r.H])
		if len(b.pleInB) > 0 {
			copy(b.pleInB[m].Floats(), q.emb[r.H:])
		}
		r.addLearnedPosTo(b.xB.Floats()[m*r.H:(m+1)*r.H], q.pos)
		b.uPos[m].SetU32(uint32(q.pos))
		b.uNKeys[m].SetU32(uint32(q.pos + 1))
		scale := float32(1)
		if r.attnTempBeta != 0 {
			scale = float32(1 + r.attnTempBeta*math.Log1p(math.Floor(float64(q.pos)/r.attnTempOrigMaxPos)))
		}
		b.uQTemp[m].Floats()[0] = scale
		b.posB.U32s()[m] = uint32(q.pos)
		b.nKeysB.U32s()[m] = uint32(q.pos + 1)
		b.tabOffB.U32s()[m] = uint32(q.slot*r.kvTabBlocks + stepSlotOffsetPlantForTest)
		b.qtB.Floats()[m] = scale
		if r.attnFANKV > 0 {
			ns := r.attnFASplitFor(q.pos+1, r.attnFANKV)
			b.uFANSplit[m].SetU32(uint32(ns))
			b.nSplitB.U32s()[m] = uint32(ns)
		}
	}
}

// bufRow is row m of buf's n-float rows. Buffer.At SETS the byte offset from the allocation's start (it does not add to
// one a view already carries), so a sub-row offset is always computed whole (row*n + off), never as row.At(off).
func bufRow(buf Buffer, m, n int) Buffer { return buf.At(4 * m * n) }

// packRows is the activation pack the batched GEMMs read: K int8 activations of each of the step's rows, from aq into aT.
func (b *batchState) packRows(e *Encoder, aq Buffer, K int, uK Buffer) {
	e.Dispatch(b.pack, K*8, 256, aq, b.aT, b.uM, uK)
}

// stepEnc is one batched step while it is encoded: its encoder, its rows, and the kernel and attention choices beginStep
// makes once for every layer's blocks.
type stepEnc struct {
	r    *resident
	b    *batchState
	e    *Encoder
	seqs []batchSeq
	B    int

	rowsQKV, rowsGU, w8Rows, w8AttnRows, w8Attn bool
	pq, pg                                      Pipeline
	nq, ng                                      int

	faRow             []bool // the row runs attention_fa: its depth is past the floor
	nPlain, nFA, maxS int
	faRows            bool // the attention_fa rows run as one dispatch pair
}

// beginStep makes the step's per-row kernel and attention choices and opens its encoder.
func (r *resident) beginStep(seqs []batchSeq) *stepEnc {
	b, B, I := r.batch, len(seqs), r.I
	// Below the calibrated sizes, qkv and gate|up run production's GEMV once per row (the row's int8 activations
	// and scale in, its output row out), which is production's arithmetic by definition.
	rowsQKV, rowsGU := B <= b.rowsQKV, B <= b.rowsGU
	w8Rows := r.w8 && mc3W8RowsOn && B >= 2 // int8 slice 3b: one weight read for all B rows
	// int4mix (slice 4): the attention projections are int8 on r.w8Attn too, its FFN int4 on the step's int4 kernels
	w8AttnRows := (r.w8 || r.w8Attn) && mc3W8RowsOn && B >= 2
	w8Attn := r.w8 || r.w8Attn
	pq, nq := saRowsPick(r.pSABias, r.pSABiasRows, b.qkvRows, r.gemvRows.qkv)
	pg, ng := saRowsPick(r.pSA, r.pSARows, 2*I, r.gemvRows.gu)
	// Each row's attention plan, once per step: batchIneligible admits only uniform layers without a window, so a row's
	// plan depends on its depth alone. Rows on the per-head kernel run in one dispatch (rowmapB); a row at attention_fa
	// depth keeps its own two dispatches.
	faRow := make([]bool, B)
	nPlain, nFA, maxS := 0, 0, 0
	for m, q := range seqs {
		faRow[m] = r.canUseAttnFAAt(0, q.pos+1)
		if !faRow[m] {
			b.rowmapB.U32s()[nPlain] = uint32(m)
			nPlain++
		} else {
			b.rowmapFA.U32s()[nFA] = uint32(m)
			nFA++
			maxS = max(maxS, int(b.nSplitB.U32s()[m]))
		}
	}
	faRows := mc3FARowsOn && nFA > 0 && b.faPartB != (Buffer{})
	if faRows {
		b.uMaxS.SetU32(uint32(maxS))
	}
	return &stepEnc{r: r, b: b, e: r.q.Begin(), seqs: seqs, B: B,
		rowsQKV: rowsQKV, rowsGU: rowsGU, w8Rows: w8Rows, w8AttnRows: w8AttnRows, w8Attn: w8Attn, pq: pq, pg: pg, nq: nq, ng: ng,
		faRow: faRow, nPlain: nPlain, nFA: nFA, maxS: maxS, faRows: faRows}
}

// attnRows is the step's attention block on layer l: every layer the step admits is an attention step and a dense FFN
// step (batchIneligible), and each block emits the dispatches the step always emitted.
func (s *stepEnc) attnRows(l int, L *residLayer) {
	r, b, e, B, seqs := s.r, s.b, s.e, s.B, s.seqs
	H, nHhd, pack := r.H, b.nHhd, b.packRows
	const tgb = batchTGBytes
	g := L.geom
	// the per-row kernels run as one dispatch over all B rows (batch_rows.go): norm+quant, RoPE, ctx quant,
	// SwiGLU+quant. The KV store and attention address each sequence's own slot, so they stay per sequence.
	if r.q4kLane { // decode's lane (encodeAttentionResidualWith): f32 rows, no activation quantization
		e.Dispatch(r.pRmsF32OutRows, B*tgReduceNorm, tgReduceNorm, b.xB, L.preNorm, b.axB, r.uH, r.uEps, r.uAddOne)
		r.q4kStepGemv(e, B, L.qkvSegs, L.qkvBias, b.axB, H, b.qkvB, b.qkvRows, r.uH)
	} else {
		e.Dispatch(r.pRmsRows, B*tgReduceNorm, tgReduceNorm, b.xB, L.preNorm, b.aqB, b.aScB, r.uH, r.uEps, r.uAddOne)
		if s.w8AttnRows {
			e.Dispatch(r.pW8Rows[B], b.qkvRows*32, 32, L.qkvW, L.qkvS, b.aqB, b.aScB, b.qkvB, r.uH, b.uQKV, b.uCount[B], L.qkvBias, b.uMode[1])
		} else if s.rowsQKV && mc3AdjRowsOn && B >= 2 && r.gemvRows.qkv > 0 && !s.w8Attn { // the adjacency kernels read int4
			e.DispatchTG(r.pSABiasRowsAdj, B*s.nq, 256, H*2, L.qkvW, L.qkvS, b.aqB, b.aScB, b.qkvB, L.qkvBias, r.uH, b.uQKV, b.uCount[B])
		} else if s.rowsQKV {
			for m := range B {
				e.DispatchTG(s.pq, s.nq, 256, H*2, L.qkvW, L.qkvS, b.aqB.At(m*H), b.aScB.At(4*m), bufRow(b.qkvB, m, b.qkvRows), L.qkvBias, r.uH)
			}
		} else {
			pack(e, b.aqB, H, r.uH)
			e.DispatchTG(b.bt, b.qkvRows/16*128, 128, tgb, L.qkvW, L.qkvS, b.aT, b.aScB, b.qkvB, r.uH, b.uQKV, b.uM, L.qkvBias, b.uMode[1])
		}
	}
	if r.qkNorm { // Qwen3's per-head Q/K RMSNorm before RoPE, decode's qk_norm over every row, at its width
		e.Dispatch(r.pQKNormRows, B*(r.nH+g.nKV)*tgReduceAttn, tgReduceAttn, b.qkvB, L.qNorm, L.kNorm, r.uNH, g.uNKV, g.uHd, g.uNHhd, r.uEps, r.uAddOne, b.uQKV)
	}
	e.Dispatch(r.pRope2Rows, B*(r.nH*g.half+g.nKV*g.half), 64, b.qkvB, L.invf, g.uHd, b.posB, g.uQtotal, g.uKtotal, g.uHalf, L.mscale, g.uNHhd, b.qtB, b.uQKV, b.uM)
	// KV store and attention, each row over its OWN sequence's blocks: one pool, read from its start, and the row's table in kvTab
	// (tabOffB). One dispatch stores every row's K/V; one runs every per-head-kernel row's attention.
	kcPool, vcPool := r.kc[l].At(0), r.vc[l].At(0)
	if r.kvI8 { // one thread per (row, KV head), as kv_store_i8 is one per head
		e.Dispatch(r.pKvI8Rows, B*g.nKV, 32, b.qkvB.At(b.kOff), b.qkvB.At(b.vOff), kcPool, vcPool, r.ks[l].At(0), r.vs[l].At(0), g.uNKV, g.uHd, b.posB, b.uQKV, b.uM, r.kvTabOf(l), b.tabOffB)
	} else {
		e.Dispatch(r.pKvRows, B*g.kvDim, 64, b.qkvB.At(b.kOff), b.qkvB.At(b.vOff), kcPool, vcPool, g.uKvDim, b.posB, b.uQKV, b.uM, r.kvTabOf(l), b.tabOffB)
	}
	if s.faRows {
		shmBytes := 128 * 6 * (r.nH / g.nKV) * 4
		e.DispatchTG(r.pAttnFARows, s.nFA*g.nKV*s.maxS*128, 128, shmBytes, b.qkvB, kcPool, vcPool, b.faPartB, g.uNKV, r.uAttnFAG,
			b.nKeysB, r.uScale, L.uWindow, b.nSplitB, b.uCount[s.nFA], b.rowmapFA, b.tabOffB, b.uQKV, b.uPartStride, b.uMaxS, r.kvTabOf(l))
		e.Dispatch(r.pAttnFACombineRows, s.nFA*r.nH*g.hd, g.hd, b.faPartB, b.ctxB, r.uAttnFAG, g.uHd, b.nSplitB, r.uNH,
			b.uCount[s.nFA], b.rowmapFA, b.uPartStride, g.uNHhd)
	}
	for m, q := range seqs {
		if !s.faRow[m] || s.faRows {
			continue
		}
		nSplit := r.attnFASplitFor(q.pos+1, g.nKV)
		shmBytes := 128 * 6 * (r.nH / g.nKV) * 4
		e.DispatchTG(r.pAttnFA, g.nKV*nSplit*128, 128, shmBytes, b.qkvB.At(4*m*b.qkvRows), kcPool, vcPool, r.attnFAPartial,
			g.uNKV, r.uAttnFAG, b.uNKeys[m], r.uScale, L.uWindow, b.uFANSplit[m], r.kvSlotKT(l, q.slot))
		e.Dispatch(r.pAttnFACombine, r.nH*g.hd, g.hd, r.attnFAPartial, bufRow(b.ctxB, m, nHhd), r.uAttnFAG, g.uHd, b.uFANSplit[m])
	}
	if s.nPlain > 0 && r.kvI8 {
		e.Dispatch(r.pAttnI8Rows, s.nPlain*r.nH*tgReduceAttn, tgReduceAttn, b.qkvB, kcPool, vcPool, r.ks[l].At(0), r.vs[l].At(0),
			b.ctxB, r.uNH, g.uNKV, g.uHd, b.nKeysB, r.uScale, L.uWindow, b.uCount[s.nPlain], b.rowmapB, r.kvTabOf(l), b.tabOffB)
	} else if s.nPlain > 0 {
		e.Dispatch(r.pAttnRows, s.nPlain*r.nH*tgReduceAttn, tgReduceAttn, b.qkvB, kcPool, vcPool, b.ctxB, r.uNH, g.uNKV, g.uHd,
			b.nKeysB, r.uScale, L.uWindow, L.attnSinks, L.uHasSink, r.kvTabOf(l), b.uCount[s.nPlain], b.rowmapB, b.tabOffB)
	}
	if r.q4kLane { // o-proj + residual reads the f32 ctx rows directly
		r.q4kStepGemv(e, B, L.oSegs, Buffer{}, b.ctxB, nHhd, b.xB, H, g.uNHhd, true)
	} else {
		e.Dispatch(r.pQvRows, B*256, 256, b.ctxB, b.cqB, b.cScB, g.uNHhd)
		if s.w8AttnRows {
			e.Dispatch(r.pW8Rows[B], H*32, 32, L.oW, L.oS, b.cqB, b.cScB, b.xB, g.uNHhd, r.uH, b.uCount[B], b.noBias, b.uMode[2])
		} else if s.w8Attn { // int8 slice 3: decode's o-proj + residual (gemv_w8a8_sa_resid) once per row
			for m := range B {
				e.DispatchTG(r.pSAResid, H*32, 256, nHhd*2, L.oW, L.oS, b.cqB.At(m*nHhd), b.cScB.At(4*m), bufRow(b.xB, m, H), g.uNHhd)
			}
		} else {
			pack(e, b.cqB, nHhd, g.uNHhd)
			e.DispatchTG(b.bt, H/16*128, 128, tgb, L.oW, L.oS, b.aT, b.cScB, b.xB, g.uNHhd, r.uH, b.uM, b.noBias, b.uMode[2])
		}
	}
}

// ffnRows is the step's dense FFN block on one layer.
func (s *stepEnc) ffnRows(L *residLayer) {
	r, b, e, B := s.r, s.b, s.e, s.B
	H, I, pack := r.H, r.I, b.packRows
	const tgb = batchTGBytes
	if r.q4kLane { // norm, fused gate|up, SwiGLU, down + residual, all f32
		e.Dispatch(r.pRmsF32OutRows, B*tgReduceNorm, tgReduceNorm, b.xB, L.postNorm, b.axB, r.uH, r.uEps, r.uAddOne)
		r.q4kStepGemv(e, B, L.guSegs, Buffer{}, b.axB, H, b.guB, 2*I, r.uH)
		e.Dispatch(r.pSwF32Rows, B*I, 256, b.guB, b.guB, b.swB, r.uI, r.uAct, b.uM)
		r.q4kStepGemv(e, B, L.dSegs, Buffer{}, b.swB, I, b.xB, H, r.uI, true)
		return
	}
	e.Dispatch(r.pRmsRows, B*tgReduceNorm, tgReduceNorm, b.xB, L.postNorm, b.mqB, b.mScB, r.uH, r.uEps, r.uAddOne)
	if s.w8Rows {
		e.Dispatch(r.pW8Rows[B], 2*I*32, 32, L.guW, L.guS, b.mqB, b.mScB, b.guB, r.uH, b.uGU, b.uCount[B], b.noBias, b.uMode[0])
	} else if s.rowsGU && mc3AdjRowsOn && B >= 2 && r.gemvRows.gu > 0 && !r.w8 {
		e.DispatchTG(r.pSARowsAdj, B*s.ng, 256, H*2, L.guW, L.guS, b.mqB, b.mScB, b.guB, r.uH, b.uGU, b.uCount[B])
	} else if s.rowsGU {
		for m := range B {
			e.DispatchTG(s.pg, s.ng, 256, H*2, L.guW, L.guS, b.mqB.At(m*H), b.mScB.At(4*m), bufRow(b.guB, m, 2*I), r.uH)
		}
	} else {
		pack(e, b.mqB, H, r.uH)
		e.DispatchTG(b.bt, 2*I/16*128, 128, tgb, L.guW, L.guS, b.aT, b.mScB, b.guB, r.uH, b.uGU, b.uM, b.noBias, b.uMode[0])
	}
	e.Dispatch(r.pSwRows, B*256, 256, b.guB, b.guB, b.dqB, b.dScB, r.uI, r.uAct)
	if s.w8Rows {
		e.Dispatch(r.pW8Rows[B], H*32, 32, L.dW, L.dS, b.dqB, b.dScB, b.xB, r.uI, r.uH, b.uCount[B], b.noBias, b.uMode[2])
	} else if r.w8 { // int8 slice 3: decode's down-proj + residual (gemv_w8a8_body_resid) once per row
		for m := range B {
			e.Dispatch(r.pGemvResid, H*32, 32, L.dW, L.dS, b.dqB.At(m*I), b.dScB.At(4*m), bufRow(b.xB, m, H), r.uI)
		}
	} else {
		pack(e, b.dqB, I, r.uI)
		e.DispatchTG(b.btd, H/16*128, 128, tgb, L.dW, L.dS, b.aT, b.dScB, b.xB, r.uI, r.uH, b.uM)
	}
}

// wholeLayer runs layer l, which the step has no block for (stepWholeLayer), through decode's layer encoder.
func (s *stepEnc) wholeLayer(l int, L *residLayer) {
	r, b, e, B, seqs := s.r, s.b, s.e, s.B, s.seqs
	H := r.H
	// Decode's layer encoder on each row in turn (OP4m-d5a): the row's position uniforms, its slot's K/V, its attention
	// plan and its per-layer embedding input bound at encode time, all restored after.
	kt, pleIn := r.kvBoundTables(), r.pleIn
	// steps runs layer l's steps from..to-1 on each row in turn, with that row's binding.
	steps := func(from, to int) {
		for m, q := range seqs {
			src := m
			if stepAttnRow0PlantForTest {
				src = 0
			}
			r.kvBindTables(q.slot)
			r.encNKeys = seqs[src].pos + 1
			if r.attnFANKV > 0 {
				r.encFANSplit = b.uFANSplit[src]
			}
			if len(b.pleInB) > 0 {
				r.pleIn = b.pleInB[m]
				if stepPLERow0PlantForTest {
					r.pleIn = b.pleInB[0]
				}
			}
			r.encLayer = l
			for i := from; i < to; i++ {
				r.encodeLayerStep(e, L.steps[i], i, l, L, bufRow(b.xB, m, H), b.uPos[src], b.uNKeys[src], b.uQTemp[src], b.uPos[src], rowState(L, q))
			}
		}
	}
	if k := slices.Index(L.steps, stepDenseFFN); k >= 0 && r.gemma4FFNSplits(l) {
		steps(0, k)
		r.encodeGemma4FFNRows(e, l, L, B)
		steps(k+1, len(L.steps))
	} else {
		steps(0, len(L.steps))
	}
	r.kvRestoreTables(kt)
	r.pleIn = pleIn
	r.encNKeys, r.encFANSplit = 0, Buffer{}
}

// layerSteps runs layer l's steps, each by its block.
func (s *stepEnc) layerSteps(l int, L *residLayer) {
	r, b, e, B, seqs := s.r, s.b, s.e, s.B, s.seqs
	H, attnRows, ffnRows := r.H, s.attnRows, s.ffnRows
	for i, st := range L.steps {
		switch st {
		case stepAttention:
			if !L.qGate {
				attnRows(l, L)
				break
			}
			// decode's gated attention on each row in turn: the row's own position uniforms, its slot's K/V bound and
			// its key count as the attention plan's, all at encode time, restored after.
			kt := r.kvBoundTables()
			for m, q := range seqs {
				src := m
				if stepAttnRow0PlantForTest {
					src = 0
				}
				r.kvBindTables(q.slot)
				r.encNKeys = seqs[src].pos + 1
				if r.attnFANKV > 0 {
					r.encFANSplit = b.uFANSplit[src]
				}
				r.encodeLayerStep(e, st, i, l, L, bufRow(b.xB, m, H), b.uPos[src], b.uNKeys[src], b.uQTemp[src], b.uPos[src], stepState{})
			}
			r.kvRestoreTables(kt)
			r.encNKeys, r.encFANSplit = 0, Buffer{}
		case stepDeltaNet, stepMamba:
			for m, q := range seqs { // decode's recurrent mixer on each row in turn, on the row's own slot state (OP4m-d4a, OP4m-d6a)
				r.encodeLayerStep(e, st, i, l, L, bufRow(b.xB, m, H), b.uPos[m], b.uNKeys[m], b.uQTemp[m], b.uPos[m], rowState(L, q))
			}
		case stepNonGatedFFN:
			r.encodeNonGatedFFNRows(e, L, B)
		case stepDenseFFN:
			ffnRows(L)
		case stepMoE:
			r.encLayer = l
			for m := range B { // decode's MoE on each row in turn: norm, route, the k experts and the shared one
				r.encodeLayerStep(e, st, i, l, L, bufRow(b.xB, m, H), b.uPos[m], b.uNKeys[m], b.uQTemp[m], b.uPos[m], stepState{})
			}
		default:
			panic(fmt.Sprintf("metal: the batched step has no block for the %v step, and batchIneligible admits none", st))
		}
	}
}

// end commits the step's command buffer and records its timings and counts.
func (s *stepEnc) end() error {
	r, b, e, B := s.r, s.b, s.e, s.B
	e.End()
	r.recordExecErr(e.Err())
	if err := r.takeExecErr(); err != nil {
		return err
	}
	r.gpuStart, r.gpuEnd, r.kernStart, r.kernEnd = e.GPUStart(), e.GPUEnd(), e.KernStart(), e.KernEnd()
	b.steps, b.seqs, b.maxSeqs = b.steps+1, b.seqs+B, max(b.maxSeqs, B)
	return nil
}

// head is the step's final norm and LM head; with multiLastRow on the q4k lane only the rows read run the head.
func (s *stepEnc) head(out multiOut) {
	r, b, e, B, seqs := s.r, s.b, s.e, s.B, s.seqs
	H := r.H
	if r.q4kLane { // decode's head (gemv_w8f32_head) on each row's f32 final norm
		e.Dispatch(r.pRmsF32OutRows, B*tgReduceNorm, tgReduceNorm, b.xB, r.finalNorm, b.axB, r.uH, r.uEps, r.uAddOne)
		for m, q := range seqs {
			if out == multiLastRow && m != B-1 && q.draw == nil {
				continue // only the last row is read
			}
			e.Dispatch(r.pGemvW8, r.V*32, 32, bufRow(b.axB, m, H), r.aSc, r.lmW, r.lmS, bufRow(b.logitsB, m, r.V), r.uH)
		}
	} else {
		e.Dispatch(r.pRmsRows, B*tgReduceNorm, tgReduceNorm, b.xB, r.finalNorm, b.aqB, b.aScB, r.uH, r.uEps, r.uAddOne)
		e.Dispatch(b.packLM, H*8, 256, b.aqB, b.aTp, b.uM, r.uH)
		e.DispatchTG(b.lm, r.V/128*128, 128, 0, r.lmW, r.lmS, b.aTp, b.aScB, b.logitsB, r.uH, b.uV, b.uM)
	}
}

// Device draws: ForwardSample's two gumbel dispatches, exactly, on each drawing row of logitsB (the same bits as
// r.logits would hold) with that row's uniforms. The partial buffers are reused row after row — dispatches in one
// encoder run in order.
func (s *stepEnc) draws() (greedyDraw []bool) {
	r, b, e, B, seqs := s.r, s.b, s.e, s.B, s.seqs
	greedyDraw = make([]bool, B)
	for m, q := range seqs {
		if q.draw == nil {
			continue
		}
		invT := float32(1 / q.draw.Temperature)
		if math.IsInf(float64(invT), 0) { // ForwardSample's own shortcut: the argmax, as greedy
			greedyDraw[m] = true
			continue
		}
		b.uGInvT[m].Floats()[0] = invT
		b.uGK0[m].SetU32(uint32(q.draw.Seed))
		b.uGK1[m].SetU32(uint32(q.draw.Seed >> 32))
		b.uGD0[m].SetU32(uint32(q.draw.Draw))
		b.uGD1[m].SetU32(uint32(q.draw.Draw >> 32))
		const gbThreads = 256
		const gbShmBytes = gbThreads * 2 * 4
		e.DispatchTG(r.pGumbel1, r.gumbelNB*gbThreads, gbThreads, gbShmBytes,
			b.logitsB.At(4*m*r.V), r.uGumbelV, b.uGInvT[m], b.uGK0[m], b.uGK1[m], b.uGD0[m], b.uGD1[m],
			r.gumbelBKey, r.gumbelBIdx)
		e.DispatchTG(r.pGumbel2, gbThreads, gbThreads, gbShmBytes,
			r.gumbelBKey, r.gumbelBIdx, r.uGumbelNB, b.gOut.At(4*m))
	}
	return greedyDraw
}

// deviceArgmax takes on the GPU the argmax of each row whose id is its raw row's argmax.
func (s *stepEnc) deviceArgmax(argmaxOnly bool, greedyDraw []bool) (devArg []bool) {
	r, b, e, B, seqs := s.r, s.b, s.e, s.B, s.seqs
	devArg = make([]bool, B) // rows whose id is the argmax of the raw row, taken on the GPU
	if mc3DeviceArgmaxOn {
		for m, q := range seqs {
			if !argmaxOnly && !greedyDraw[m] {
				continue // a full-logits row (the caller reads the row) or a sampled one (the gumbel draw gives its id)
			}
			if !argmaxOnly && q.draw.Greedy && r.logitTransform() {
				continue // a greedy generation's id is the argmax of the transformed row: taken on the host below
			}
			devArg[m] = true
			e.DispatchTG(r.pArgRowsPart, batchArgmaxParts*256, 256, 0, b.logitsB.At(4*m*r.V), b.amaxPart.At(8*m*batchArgmaxParts), b.uV, b.uAmaxP)
			e.DispatchTG(r.pArgFinish, 256, 256, 0, b.amaxPart.At(8*m*batchArgmaxParts), b.amaxTok.At(4*m), b.uAmaxP)
		}
	}
	return devArg
}

// read is the step's rows from its committed buffers: each row's logits, or its id (argmaxOnly, or a row that draws).
func (s *stepEnc) read(out multiOut, argmaxOnly bool, greedyDraw, devArg []bool) (logits [][]float32, ids []int) {
	r, b, B, seqs := s.r, s.b, s.B, s.seqs
	logits, ids = make([][]float32, B), make([]int, B)
	lg := b.logitsB.Floats()
	gid := b.gOut.U32s()
	devID := func(m int) int { // the GPU's argmax, or argmaxF32's answer for a row that starts with NaN (0)
		if math.IsNaN(float64(lg[m*r.V])) {
			return 0
		}
		return int(b.amaxTok.U32s()[m])
	}
	if argmaxOnly {
		for m := range seqs {
			if devArg[m] {
				ids[m] = devID(m)
			} else {
				ids[m] = argmaxF32(lg[m*r.V : (m+1)*r.V])
			}
		}
		return nil, ids
	}
	for m, q := range seqs {
		ids[m] = -1
		if q.draw != nil {
			id := int32(gid[m])
			if greedyDraw[m] && devArg[m] {
				id = int32(devID(m))
				if q.draw.Greedy {
					b.greedyDevRows++
				}
			} else if greedyDraw[m] && q.draw.Greedy { // with a softcap or logit scale: the transformed row's argmax
				id = int32(argmaxF32(r.transformedRow(lg[m*r.V : (m+1)*r.V])))
			} else if greedyDraw[m] || id < 0 { // ForwardSample's fallbacks: the argmax of the raw row
				id = int32(argmaxF32(lg[m*r.V : (m+1)*r.V]))
			}
			ids[m] = int(id)
			continue
		}
		if out == multiLastRow && m != B-1 {
			continue // only the last row is read
		}
		logits[m] = r.transformedRow(lg[m*r.V : (m+1)*r.V])
	}
	return logits, ids
}
