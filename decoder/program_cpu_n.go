package decoder

import (
	"context"
	"fmt"
	"math"

	"github.com/townsendmerino/aikit/linalg"
)

// walkBatch runs prog's layers over K pre-embedded rows h ([K*hidden], updated in place) appended as the next K positions of
// cache, then the head's norm, and returns h. Its operations compute what walkToken's do, row by row, with the projections at
// M=K; rows are independent within an operation, so a row's result is the per-token walker's (TestForwardN_matchesSequential).
//
// fastAttn and floorK are runLayersFromEmbedNFloor's: the f32 attention kernel is a choice of the request, not of the model, so
// it is the walker's input; the program's attention order names the exact kernel the request may replace.
func (m *Model) walkBatch(reqCtx context.Context, prog *program, h []float32, cache *KVCache, fastAttn bool, floorK int, ps *prefillPassScratch) ([]float32, error) {
	arch := m.w.arch
	be := m.be
	hidden, nKV, hd := arch.HiddenDim, arch.NumKVHeads, arch.HeadDim
	maxQDim, kvDim, inter := arch.maxHeads()*hd, nKV*hd, arch.IntermediateDim
	K := len(h) / hidden
	startPos := cache.Pos()

	if ps == nil {
		ps = &prefillPassScratch{}
	}
	norm := ps.take(&ps.norm, K*hidden)
	q := ps.take(&ps.q, K*maxQDim)
	k := ps.take(&ps.k, K*kvDim)
	v := ps.take(&ps.v, K*kvDim)
	ctx := ps.take(&ps.ctx, K*maxQDim)
	att := ps.take(&ps.att, K*hidden)
	gate := ps.take(&ps.gate, K*inter)
	up := ps.take(&ps.up, K*inter)
	mlpOut := ps.take(&ps.mlpOut, K*hidden)
	maxKeys := startPos + K
	// The f32 kernel is admitted only above the floor, keyed on the whole suffix's length (runLayersFromEmbedNFloor has the reasons).
	if fastAttn && floorK < fastAttnMinPrompt {
		fastAttn = false
	}
	useAcc64 := !fastAttn
	wantFusedPool := !useAcc64 && cache.treeMask == nil && !prog.anySoftcap() // the fused tile never sees a score to cap
	poolK, poolKeys := K, maxKeys
	if ps.poolKeys > maxKeys {
		poolKeys = ps.poolKeys
	}
	if ps.pool == nil || ps.poolWant != wantFusedPool {
		ps.pool = newHeadWorkerPoolK(m.knobs, prefillAttnWorkersK(m.knobs, poolK, poolKeys, hd, arch.maxHeads()), poolK, poolKeys, hd, wantFusedPool)
		ps.poolWant = wantFusedPool
	}
	attnPool := ps.pool
	var alk, alv []float32
	if cache.localAny || cache.quant.widened() {
		alk = ps.take(&ps.alk, maxKeys*kvDim)
		alv = ps.take(&ps.alv, maxKeys*kvDim)
	}
	ws := &ps.ws
	hq := &ps.hq
	ws.SetThreshold(DefaultDecodeParallelThreshold)
	var qkvOps [3]linalg.W8A8Op
	var guOps [2]linalg.W8A8Op
	gbuf := ps.gbuf
	defer func() { ps.gbuf = gbuf }()

	row := func(b []float32, i, w int) []float32 { return b[i*w : i*w+w] }
	reg := func(r opReg) []float32 {
		switch r {
		case regX:
			return norm
		case regA:
			return att
		case regB:
			return mlpOut
		}
		return h
	}
	rowPos := func(i int) int {
		if cache.treeRowPos != nil {
			return cache.treeRowPos[i]
		}
		return startPos + i
	}

	if m.layerPager != nil {
		defer m.layerPager.finishLayers()
	}
	for l, ops := range prog.layers {
		// Checked per layer: an abandoned client must not keep a prefill running (TestPrefillCancelMidFlight).
		if err := reqCtx.Err(); err != nil {
			return nil, err
		}
		if m.layerPager != nil {
			m.layerPager.enterLayer(l)
		}
		lw := &m.w.Layers[l]
		nH := arch.headsAt(l)
		qDim := nH * hd
		q, ctx := q[:K*qDim], ctx[:K*qDim]
		for oi := range ops {
			o := &ops[oi]
			switch o.kind {
			case opNorm:
				src, dst := reg(o.in), reg(o.out)
				if o.norm.Form == normCopy {
					copy(dst, src)
					continue
				}
				w, b := lw.normWeights(o.norm)
				for i := range K {
					applyNorm(o.norm, row(dst, i, hidden), row(src, i, hidden), w, b, hidden)
				}
			case opProjQKV:
				x := reg(o.in)
				if isW8A8(&lw.QProj) && isW8A8(&lw.KProj) && isW8A8(&lw.VProj) {
					qkvOps[0] = linalg.W8A8Op{BQ: wmInt8(&lw.QProj), Scales: wmScales(&lw.QProj), Dst: q, N: lw.QProj.Rows()}
					qkvOps[1] = linalg.W8A8Op{BQ: wmInt8(&lw.KProj), Scales: wmScales(&lw.KProj), Dst: k, N: lw.KProj.Rows()}
					qkvOps[2] = linalg.W8A8Op{BQ: wmInt8(&lw.VProj), Scales: wmScales(&lw.VProj), Dst: v, N: lw.VProj.Rows()}
					matmulW8A8Batch(be, ws, x, K, lw.QProj.Cols(), qkvOps[:], lw.QProj.ActQuantGroup())
				} else {
					hq.prepareWS(be, ws, &lw.QProj, x, K)
					matmulPreWS(be, ws, &lw.QProj, hq, x, q, K)
					matmulPreWS(be, ws, &lw.KProj, hq, x, k, K)
					matmulPreWS(be, ws, &lw.VProj, hq, x, v, K)
				}
				if o.proj.Bias {
					for i := range K {
						addBias(row(q, i, qDim), lw.QBias)
						addBias(row(k, i, kvDim), lw.KBias)
						addBias(row(v, i, kvDim), lw.VBias)
					}
				}
			case opQKNorm:
				for i := range K {
					applyQKNorm(o.qkNorm, row(q, i, qDim), row(k, i, kvDim), lw, nH, nKV, hd)
				}
			case opRope:
				r := &o.rope
				for i := range K {
					pos := rowPos(i)
					ropeAt(row(q, i, qDim), nH, hd, pos, r.InvFreq, r.Mscale, r.MRopeSection, cache.mropePos, cache.mropeDelta, r.Interleave, r.MRopeInterleaved)
					ropeAt(row(k, i, kvDim), nKV, hd, pos, r.InvFreq, r.Mscale, r.MRopeSection, cache.mropePos, cache.mropeDelta, r.Interleave, r.MRopeInterleaved)
				}
			case opAttnTemp:
				for i := range K {
					scale := float32(1 + o.temp.Beta*math.Log1p(math.Floor(float64(rowPos(i))/o.temp.OrigMaxPos)))
					qi := row(q, i, qDim)
					for j := range qi {
						qi[j] *= scale
					}
				}
			case opAttention:
				if o.attn.Order != attnBatchedAcc64 {
					return nil, fmt.Errorf("decoder: layer %d: attention order %d is not implemented on the CPU's batched walker", l, o.attn.Order)
				}
				isLocal := cache.isLocal(l)
				for i := range K {
					ki, vi := row(k, i, kvDim), row(v, i, kvDim)
					if kvRoundF16ForTest { // F1's reference
						roundF16InPlace(ki)
						roundF16InPlace(vi)
					}
					if !isLocal {
						cache.Append(l, ki, vi) // a local layer's write waits until its window is read (commitBatch)
					}
				}
				switch {
				case isLocal:
					base, nRows := cache.batchReadLocal(l, startPos, K, k, v, alk, alv)
					attendBatchedHeadsP(q, ctx, alk[:nRows*kvDim], alv[:nRows*kvDim], base, cache, l, startPos, K, o.attn, arch, useAcc64, attnPool)
					cache.commitBatch(l, startPos, K, k, v)
				case cache.quant.widened():
					nKeys := cache.dequantGlobalLayer(l, kvDim, alk, alv)
					attendBatchedHeadsP(q, ctx, alk[:nKeys*kvDim], alv[:nKeys*kvDim], 0, cache, l, startPos, K, o.attn, arch, useAcc64, attnPool)
				default:
					attendBatchedHeadsP(q, ctx, cache.Keys(l), cache.Vals(l), 0, cache, l, startPos, K, o.attn, arch, useAcc64, attnPool)
				}
			case opOutGate:
				gRows := lw.GProj.Rows()
				if cap(gbuf) < K*gRows {
					gbuf = make([]float32, K*gRows)
				}
				gb := gbuf[:K*gRows]
				matmulWS(be, ws, &lw.GProj, reg(o.in), gb, K)
				perHead := gRows == nH
				for i := range K {
					if o.gate.Sigmoid {
						applySigmoidGateRow(row(gb, i, gRows), row(ctx, i, qDim), perHead, nH, hd)
					} else {
						applyGateRow(row(gb, i, gRows), row(ctx, i, qDim), perHead, nH, hd)
					}
				}
			case opProjO:
				out := reg(o.out)
				matmulWS(be, ws, &lw.OProj, ctx, out, K)
				if o.proj.Bias {
					for i := range K {
						addBias(row(out, i, hidden), lw.OBias)
					}
				}
			case opFFN:
				if err := m.ffnBatch(&o.ffn, reg(o.in), reg(o.out), o.out, lw, K, ws, hq, gate, up, &guOps); err != nil {
					return nil, err
				}
			case opResidual:
				if o.res.Two {
					addResidual2(h, att, mlpOut)
				} else {
					addResidual(h, att)
				}
			default:
				return nil, fmt.Errorf("decoder: layer %d: operation %v is not implemented on the CPU's batched walker", l, o.kind)
			}
		}
		if ds := cache.deepstack; ds != nil && l < len(ds.rows) {
			addDeepstack(h, ds, l, startPos, K, hidden)
		}
		if cache.captureLayers != nil {
			for ci, cl := range cache.captureLayers {
				if cl == l {
					cache.captured[ci] = append(cache.captured[ci][:0], h...)
				}
			}
		}
	}

	// Advance by K explicitly: a local last layer defers its write (commitBatch does not step pos).
	cache.advanceTo(startPos + K)
	hp := &prog.head
	fb := m.w.FinalNormBias
	if !hp.Norm.Bias {
		fb = nil
	}
	for i := range K {
		applyNorm(hp.Norm, row(h, i, hidden), row(h, i, hidden), m.w.FinalNorm, fb, hidden)
	}
	return h, nil
}

// ffnBatch writes the feed-forward output of K rows of x into out. A MoE layer runs its experts per row (the router picks per
// token) or expert-major in chunks; a dense layer runs gate/up at M=K.
func (m *Model) ffnBatch(p *ffnParams, x, out []float32, outReg opReg, lw *LayerWeights, K int, ws *linalg.Workspace, hq *w4a8Act,
	gate, up []float32, guOps *[2]linalg.W8A8Op) error {
	arch, be := m.w.arch, m.be
	hidden := arch.HiddenDim
	switch p.Form {
	case ffnMoE:
		if outReg == regB {
			// No parallel-block MoE family exists; its joint residual add is not written.
			return errNotImplemented
		}
		emDone := make([]bool, K)
		var emOut []float32
		if m.knobs.moeExpertMajor() {
			emOut = make([]float32, K*hidden)
			for c0 := 0; c0 < K; c0 += moeExpertMajorChunk {
				c1 := min(c0+moeExpertMajorChunk, K)
				ok, err := moeMLPBatchWith(x[c0*hidden:c1*hidden], c1-c0, lw, &p.MoE, p.Act, hidden, be, m.pager, emOut[c0*hidden:c1*hidden])
				if err != nil {
					return err
				}
				if !ok {
					break // refused: every row left goes through the per-row path
				}
				for i := c0; i < c1; i++ {
					emDone[i] = true
				}
			}
		}
		for i := range K {
			dst := out[i*hidden : (i+1)*hidden]
			if emDone[i] {
				copy(dst, emOut[i*hidden:(i+1)*hidden])
				continue
			}
			ff, err := moeMLPWith(x[i*hidden:(i+1)*hidden], lw, &p.MoE, p.Act, hidden, be, nil, m.pager)
			if err != nil {
				return err
			}
			copy(dst, ff)
		}
		return nil
	case ffnNonGated:
		return errNotImplemented // canBatchN keeps a non-gated family on the per-token walker
	}
	if isW8A8(&lw.GateProj) && isW8A8(&lw.UpProj) {
		guOps[0] = linalg.W8A8Op{BQ: wmInt8(&lw.GateProj), Scales: wmScales(&lw.GateProj), Dst: gate, N: lw.GateProj.Rows()}
		guOps[1] = linalg.W8A8Op{BQ: wmInt8(&lw.UpProj), Scales: wmScales(&lw.UpProj), Dst: up, N: lw.UpProj.Rows()}
		matmulW8A8Batch(be, ws, x, K, lw.GateProj.Cols(), guOps[:], lw.GateProj.ActQuantGroup())
	} else {
		hq.prepareWS(be, ws, &lw.GateProj, x, K)
		matmulPreWS(be, ws, &lw.GateProj, hq, x, gate, K)
		matmulPreWS(be, ws, &lw.UpProj, hq, x, up, K)
	}
	if err := activateRows(p.Act, gate, up); err != nil {
		return err
	}
	matmulWS(be, ws, &lw.DownProj, gate, out, K)
	return nil
}

// activateRows is the batched paths' gated activation over gate and up, in place into gate.
func activateRows(act ActKind, gate, up []float32) error {
	switch act {
	case ActGeluTanh:
		if len(gate) < activationFanoutThreshold {
			geglu(gate, up)
		} else {
			parallelElementwise(len(gate), func(lo, hi int) {
				for j := lo; j < hi; j++ {
					gate[j] = geluTanh(gate[j]) * up[j]
				}
			})
		}
	case ActSiLU:
		if len(gate) < activationFanoutThreshold {
			swiglu(gate, up)
		} else {
			parallelElementwise(len(gate), func(lo, hi int) {
				for j := lo; j < hi; j++ {
					gate[j] = silu(gate[j]) * up[j]
				}
			})
		}
	case ActGelu:
		if len(gate) < activationFanoutThreshold {
			gegluExact(gate, up)
		} else {
			parallelElementwise(len(gate), func(lo, hi int) {
				for j := lo; j < hi; j++ {
					gate[j] = geluErf(gate[j]) * up[j]
				}
			})
		}
	default:
		return errNotImplemented
	}
	return nil
}

// anySoftcap reports whether any attention operation caps its scores.
func (p *program) anySoftcap() bool {
	for _, ops := range p.layers {
		for i := range ops {
			if ops[i].kind == opAttention && ops[i].attn.Softcap != 0 {
				return true
			}
		}
	}
	return false
}

// walkMulti runs prog's layers for B independent sequences, one token each, each on its own cache (MC3c), then the head's
// norm; it returns the B rows' hidden states. Its operations compute what walkToken's do for each row, with the projections at
// M=B; the program must be one multiStepEligible admits.
func (m *Model) walkMulti(prog *program, h []float32, caches []*KVCache) ([]float32, error) {
	arch := m.w.arch
	be := m.be
	B := len(caches)
	hidden, nKV, hd := arch.HiddenDim, arch.NumKVHeads, arch.HeadDim
	maxQDim, kvDim, inter := arch.maxHeads()*hd, nKV*hd, arch.IntermediateDim
	row := func(b []float32, i, w int) []float32 { return b[i*w : i*w+w] }

	norm := make([]float32, B*hidden)
	q, k, v := make([]float32, B*maxQDim), make([]float32, B*kvDim), make([]float32, B*kvDim)
	ctx, att := make([]float32, B*maxQDim), make([]float32, B*hidden)
	gate, up := make([]float32, B*inter), make([]float32, B*inter)
	pos := make([]int, B)
	for b, c := range caches {
		pos[b] = c.Pos()
	}
	var ws linalg.Workspace
	ws.SetThreshold(DefaultDecodeParallelThreshold)
	var qkvOps [3]linalg.W8A8Op
	var guOps [2]linalg.W8A8Op
	var ws4 linalg.Workspace // the fused W4A8 calls, at the threshold matmul() gives int4 weights
	var hq w4a8Act           // the B-row normed block for the matmul fallbacks, quantized once per projection group
	ws4.SetThreshold(int4ParThreshold)
	var qkvOps4 [3]linalg.W4A8Op
	var guOps4 [2]linalg.W4A8Op
	reg := func(r opReg) []float32 {
		switch r {
		case regX:
			return norm
		case regA:
			return att
		}
		return h
	}

	for l, ops := range prog.layers {
		lw := &m.w.Layers[l]
		nH := arch.headsAt(l)
		qDim := nH * hd
		q, ctx := q[:B*qDim], ctx[:B*qDim]
		for oi := range ops {
			o := &ops[oi]
			switch o.kind {
			case opNorm:
				src, dst := reg(o.in), reg(o.out)
				w, bias := lw.normWeights(o.norm)
				for b := range B {
					applyNorm(o.norm, row(dst, b, hidden), row(src, b, hidden), w, bias, hidden)
				}
			case opProjQKV:
				x := reg(o.in)
				if isW8A8(&lw.QProj) && isW8A8(&lw.KProj) && isW8A8(&lw.VProj) {
					qkvOps[0] = linalg.W8A8Op{BQ: wmInt8(&lw.QProj), Scales: wmScales(&lw.QProj), Dst: q, N: lw.QProj.Rows()}
					qkvOps[1] = linalg.W8A8Op{BQ: wmInt8(&lw.KProj), Scales: wmScales(&lw.KProj), Dst: k, N: lw.KProj.Rows()}
					qkvOps[2] = linalg.W8A8Op{BQ: wmInt8(&lw.VProj), Scales: wmScales(&lw.VProj), Dst: v, N: lw.VProj.Rows()}
					matmulW8A8Batch(be, &ws, x, B, lw.QProj.Cols(), qkvOps[:], lw.QProj.ActQuantGroup())
				} else if group, ag, ok := w4a8FusedOps(qkvOps4[:], []*linalg.WeightMat{&lw.QProj, &lw.KProj, &lw.VProj}, [][]float32{q, k, v}); ok {
					matmulW4A8Batch(be, &ws4, x, B, lw.QProj.Cols(), group, qkvOps4[:], ag) // q‖k‖v in one fork/join
				} else {
					hq.prepare(be, &lw.QProj, x, B)
					matmulPre(be, &lw.QProj, &hq, x, q, B)
					matmulPre(be, &lw.KProj, &hq, x, k, B)
					matmulPre(be, &lw.VProj, &hq, x, v, B)
				}
				if o.proj.Bias {
					for b := range B {
						addBias(row(q, b, qDim), lw.QBias)
						addBias(row(k, b, kvDim), lw.KBias)
						addBias(row(v, b, kvDim), lw.VBias)
					}
				}
			case opQKNorm:
				for b := range B {
					applyQKNorm(o.qkNorm, row(q, b, qDim), row(k, b, kvDim), lw, nH, nKV, hd)
				}
			case opRope:
				r := &o.rope
				for b, c := range caches {
					ropeAt(row(q, b, qDim), nH, hd, pos[b], r.InvFreq, r.Mscale, r.MRopeSection, c.mropePos, c.mropeDelta, r.Interleave, r.MRopeInterleaved)
					ropeAt(row(k, b, kvDim), nKV, hd, pos[b], r.InvFreq, r.Mscale, r.MRopeSection, c.mropePos, c.mropeDelta, r.Interleave, r.MRopeInterleaved)
				}
			case opAttnTemp:
				for b := range B {
					scale := float32(1 + o.temp.Beta*math.Log1p(math.Floor(float64(pos[b])/o.temp.OrigMaxPos)))
					qb := row(q, b, qDim)
					for j := range qb {
						qb[j] *= scale
					}
				}
			case opAttention:
				if o.attn.Order != attnBatchedAcc64 {
					return nil, fmt.Errorf("decoder: layer %d: attention order %d is not implemented on the CPU's multi-sequence walker", l, o.attn.Order)
				}
				for b, c := range caches { // each sequence's f32 append-forever cache, at K=1 with float64 accumulation
					c.Append(l, row(k, b, kvDim), row(v, b, kvDim))
					nKeys := c.storedRows(l, kvDim)
					pool := c.scr.headWorkerPool(nH, 1, nKeys, hd, false, true)
					attendBatchedHeadsP(row(q, b, qDim), row(ctx, b, qDim), c.Keys(l), c.Vals(l), 0, c, l, pos[b], 1, o.attn, arch, true, pool)
				}
			case opProjO:
				out := reg(o.out)
				matmul(be, &lw.OProj, ctx, out, B)
				if o.proj.Bias {
					for b := range B {
						addBias(row(out, b, hidden), lw.OBias)
					}
				}
			case opFFN:
				if o.ffn.Form != ffnGated {
					return nil, fmt.Errorf("decoder: layer %d: the multi-sequence walker runs a gated dense FFN only", l)
				}
				x, out := reg(o.in), reg(o.out)
				if isW8A8(&lw.GateProj) && isW8A8(&lw.UpProj) {
					guOps[0] = linalg.W8A8Op{BQ: wmInt8(&lw.GateProj), Scales: wmScales(&lw.GateProj), Dst: gate, N: lw.GateProj.Rows()}
					guOps[1] = linalg.W8A8Op{BQ: wmInt8(&lw.UpProj), Scales: wmScales(&lw.UpProj), Dst: up, N: lw.UpProj.Rows()}
					matmulW8A8Batch(be, &ws, x, B, lw.GateProj.Cols(), guOps[:], lw.GateProj.ActQuantGroup())
				} else if group, ag, ok := w4a8FusedOps(guOps4[:], []*linalg.WeightMat{&lw.GateProj, &lw.UpProj}, [][]float32{gate, up}); ok {
					matmulW4A8Batch(be, &ws4, x, B, lw.GateProj.Cols(), group, guOps4[:], ag) // gate‖up in one fork/join
				} else {
					hq.prepare(be, &lw.GateProj, x, B)
					matmulPre(be, &lw.GateProj, &hq, x, gate, B)
					matmulPre(be, &lw.UpProj, &hq, x, up, B)
				}
				if err := activateRows(o.ffn.Act, gate, up); err != nil {
					return nil, err
				}
				matmul(be, &lw.DownProj, gate, out, B)
			case opResidual:
				if o.res.Two {
					return nil, fmt.Errorf("decoder: layer %d: the multi-sequence walker has no parallel block", l)
				}
				addResidual(h, att)
			default:
				return nil, fmt.Errorf("decoder: layer %d: operation %v is not implemented on the CPU's multi-sequence walker", l, o.kind)
			}
		}
	}
	hp := &prog.head
	fb := m.w.FinalNormBias
	if !hp.Norm.Bias {
		fb = nil
	}
	for b := range B {
		applyNorm(hp.Norm, row(h, b, hidden), row(h, b, hidden), m.w.FinalNorm, fb, hidden)
	}
	return h, nil
}

// multiStepEligible is the program-level scope of the multi-sequence walker: every layer pre-norm placed (no post-sublayer norm,
// no copy for post-only placement, no parallel block), a gated dense FFN, no output gate, and no learned positions.
func (p *program) multiStepEligible() error {
	if p.layers == nil {
		return fmt.Errorf("no layer program (an own loop)")
	}
	if p.embed.LearnedPos {
		return fmt.Errorf("learned positions")
	}
	for l, ops := range p.layers {
		for i := range ops {
			o := &ops[i]
			switch {
			case o.kind == opNorm && (o.norm.Slot == slotPostAttn || o.norm.Slot == slotPostMLP || o.norm.Form == normCopy):
				return fmt.Errorf("layer %d: a post-sublayer or post-only norm", l)
			case o.kind == opResidual && o.res.Two:
				return fmt.Errorf("layer %d: a parallel block", l)
			case o.kind == opFFN && o.ffn.Form != ffnGated:
				return fmt.Errorf("layer %d: not a gated dense FFN", l)
			case o.kind == opOutGate:
				return fmt.Errorf("layer %d: an attention output gate", l)
			}
		}
	}
	return nil
}

// batchEligible reports whether the batched walker runs prog: a layer program with no learned positions and no non-gated FFN.
func (p *program) batchEligible() bool {
	if p.layers == nil || p.embed.LearnedPos {
		return false
	}
	for _, ops := range p.layers {
		for i := range ops {
			if ops[i].kind == opFFN && ops[i].ffn.Form == ffnNonGated {
				return false
			}
		}
	}
	return true
}
