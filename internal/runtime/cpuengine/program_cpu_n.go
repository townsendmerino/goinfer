package cpuengine

import (
	"context"
	"fmt"

	"github.com/townsendmerino/aikit/linalg"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/normact"
)

// batchRowPos is the absolute position of row i of a batched pass that starts at startPos: its tree position in a speculative
// tree verify, else startPos+i.
func (c *Cache) batchRowPos(startPos, i int) int {
	if c.TreeRowPos != nil {
		return c.TreeRowPos[i]
	}
	return startPos + i
}

// WalkBatch runs prog's layers over K pre-embedded rows h ([K*hidden], updated in place) appended as the next K positions of
// cache, then the head's norm, and returns h. Its operations compute what walkToken's do, row by row, with the projections at
// M=K; rows are independent within an operation, so a row's result is the per-token walker's (TestForwardN_matchesSequential).
//
// fastAttn and floorK are runLayersFromEmbedNFloor's: the f32 attention kernel is a choice of the request, not of the model, so
// it is the walker's input; the program's attention order names the exact kernel the request may replace. rows carries each row's
// token id and the prompt's bidirectional spans; Gemma 4's operations need it, and every other family passes nil.
func (m *Engine) WalkBatch(reqCtx context.Context, prog *layerProgram, h []float32, cache *Cache, fastAttn bool, floorK int, ps *PrefillPassScratch, rows *BatchRows) ([]float32, error) {
	arch := m.W.Arch
	be := m.Be
	hidden, nKV, hd := arch.HiddenDim, arch.NumKVHeads, arch.HeadDim
	maxQDim, kvDim, inter := arch.MaxHeads()*hd, nKV*hd, arch.IntermediateDim
	K := len(h) / hidden
	startPos := cache.Pos()
	cache.Cost.Tokens(cache.CostPhase, K)

	if ps == nil {
		ps = &PrefillPassScratch{}
	}
	norm := ps.Take(&ps.Norm, K*hidden)
	var q, k, v, ctx []float32 // the generic attention's; Gemma 4's operation allocates its own at each layer's widths
	if prog.AnyOp(func(o *op) bool { return o.Kind == opProjQKV || o.Kind == opGatedAttn }) {
		q, k = ps.Take(&ps.Q, K*maxQDim), ps.Take(&ps.K, K*kvDim)
		v, ctx = ps.Take(&ps.V, K*kvDim), ps.Take(&ps.Ctx, K*maxQDim)
	}
	att := ps.Take(&ps.Att, K*hidden)
	var gate, up []float32
	if prog.AnyOp(func(o *op) bool { return o.Kind == opFFN && o.FFN.Form != ffnGemma4Dense }) {
		gate = ps.Take(&ps.Gate, K*inter)
		up = ps.Take(&ps.Up, K*inter)
	}
	mlpOut := ps.Take(&ps.MLPOut, K*hidden)
	maxKeys := startPos + K
	// The f32 kernel is admitted only above the floor, keyed on the whole suffix's length (runLayersFromEmbedNFloor has the reasons).
	if fastAttn && floorK < FastAttnMinPrompt {
		fastAttn = false
	}
	// qwen3.5's gated attention computes in AttendQuery's order (f64 accumulation) whatever the request asks, so its pool must not
	// be the fused one, whose slots carry no score buffers (TestWalkBatch_gatedAttentionIgnoresFastAttention).
	if fastAttn && prog.AnyOp(func(o *op) bool { return o.Kind == opGatedAttn }) {
		fastAttn = false
	}
	useAcc64 := !fastAttn
	wantFusedPool := !useAcc64 && cache.TreeMask == nil && !prog.AnySoftcap() && !arch.HasAttnSinks() // the fused tile never sees a score to cap or a sink logit
	poolK, poolKeys := K, maxKeys
	if ps.poolKeys > maxKeys {
		poolKeys = ps.poolKeys
	}
	if (ps.Pool == nil || ps.PoolWant != wantFusedPool) && prog.AnyOp(func(o *op) bool { return o.Kind == opAttention || o.Kind == opGatedAttn }) {
		ps.Pool = newHeadWorkerPoolK(m.Knobs, PrefillAttnWorkersK(m.Knobs, poolK, poolKeys, hd, arch.MaxHeads()), poolK, poolKeys, hd, wantFusedPool)
		ps.PoolWant = wantFusedPool
	}
	attnPool := ps.Pool
	var alk, alv []float32
	if cache.LocalAny || cache.Quant.widened() {
		alk = ps.Take(&ps.ALK, maxKeys*kvDim)
		alv = ps.Take(&ps.ALV, maxKeys*kvDim)
	}
	ws := &ps.ws
	hq := &ps.hq
	ws.SetThreshold(DefaultDecodeParallelThreshold)
	var qkvOps [3]linalg.BatchOp
	var guOps [2]linalg.BatchOp
	gbuf := ps.GBuf
	defer func() { ps.GBuf = gbuf }()

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
	var gp *gemma4Pass
	if arch.Gemma4 != nil {
		if rows == nil || len(rows.IDs) != K {
			return nil, fmt.Errorf("decoder: Gemma 4's batched pass needs each row's token id (%d rows)", K)
		}
		gp = m.newGemma4Pass(prog, h, rows, startPos, K)
	}

	if m.LayerPager != nil {
		defer m.LayerPager.FinishLayers()
	}
	for l, ops := range prog.Layers {
		// Checked per layer: an abandoned client must not keep a prefill running (TestPrefillCancelMidFlight).
		if err := reqCtx.Err(); err != nil {
			return nil, err
		}
		if m.LayerPager != nil {
			m.LayerPager.EnterLayer(l)
		}
		lw := &m.W.Layers[l]
		nH := arch.HeadsAt(l)
		qDim := nH * hd
		q, ctx := q[:min(len(q), K*qDim)], ctx[:min(len(ctx), K*qDim)]
		for oi := range ops {
			o := &ops[oi]
			if gp == nil && needsGemma4State(o) {
				return nil, errNoGemma4State(l, o)
			}
			t0 := CostStart(cache)
			switch o.Kind {
			case opNorm:
				src, dst := reg(o.In), reg(o.Out)
				if o.Norm.Form == normCopy {
					copy(dst, src)
					break
				}
				w, b := lw.NormWeights(o.Norm)
				for i := range K {
					ApplyNorm(o.Norm, row(dst, i, hidden), row(src, i, hidden), w, b, hidden)
				}
			case opProjQKV:
				x := reg(o.In)
				if IsW8A8(&lw.QProj) && IsW8A8(&lw.KProj) && IsW8A8(&lw.VProj) {
					qkvOps[0] = linalg.BatchOp{W: &lw.QProj, Dst: q}
					qkvOps[1] = linalg.BatchOp{W: &lw.KProj, Dst: k}
					qkvOps[2] = linalg.BatchOp{W: &lw.VProj, Dst: v}
					MatmulW8A8Batch(be, ws, x, K, lw.QProj.Cols(), qkvOps[:], lw.QProj.ActQuantGroup())
				} else {
					hq.prepareWS(be, ws, &lw.QProj, x, K)
					MatmulPreWS(be, ws, &lw.QProj, hq, x, q, K)
					MatmulPreWS(be, ws, &lw.KProj, hq, x, k, K)
					MatmulPreWS(be, ws, &lw.VProj, hq, x, v, K)
				}
				if o.Proj.Bias {
					for i := range K {
						AddBias(row(q, i, qDim), lw.QBias)
						AddBias(row(k, i, kvDim), lw.KBias)
						AddBias(row(v, i, kvDim), lw.VBias)
					}
				}
			case opQKNorm:
				for i := range K {
					applyQKNorm(o.QKNorm, row(q, i, qDim), row(k, i, kvDim), lw.QNorm, lw.KNorm, nH, nKV, hd)
				}
			case opRope:
				for i := range K {
					applyRope(&o.Rope, row(q, i, qDim), row(k, i, kvDim), nH, nKV, hd, cache.batchRowPos(startPos, i), cache.MRopePos, cache.MRopeDelta)
				}
			case opAttnTemp:
				for i := range K {
					scale := o.Temp.Scale(cache.batchRowPos(startPos, i))
					qi := row(q, i, qDim)
					for j := range qi {
						qi[j] *= scale
					}
				}
			case opAttention:
				if o.Attn.Order != attnBatchedAcc64 {
					return nil, fmt.Errorf("decoder: layer %d: attention order %d is not implemented on the CPU's batched walker", l, o.Attn.Order)
				}
				isLocal := cache.isLocal(l)
				for i := range K {
					ki, vi := row(k, i, kvDim), row(v, i, kvDim)
					if activeHooks().KVRoundF16 { // the f16 exactness gate's reference
						roundF16InPlace(ki)
						roundF16InPlace(vi)
					}
					if !isLocal {
						cache.Append(l, ki, vi) // a local layer's write waits until its window is read (commitBatch)
					}
				}
				ap := o.Attn
				ap.Sinks = lw.AttnSinks
				switch {
				case isLocal:
					base, nRows := cache.BatchReadLocal(l, startPos, K, k, v, alk, alv)
					AttendBatchedHeadsP(q, ctx, alk[:nRows*kvDim], alv[:nRows*kvDim], base, cache, l, startPos, K, ap, arch, useAcc64, attnPool)
					cache.CommitBatch(l, startPos, K, k, v)
				case cache.Quant.widened():
					nKeys := cache.DequantGlobalLayer(l, kvDim, alk, alv)
					AttendBatchedHeadsP(q, ctx, alk[:nKeys*kvDim], alv[:nKeys*kvDim], 0, cache, l, startPos, K, ap, arch, useAcc64, attnPool)
				default:
					keys, vals, rs := cache.LayerRows(l, kvDim)
					AttendBatchedHeadsKV(q, ctx, keys, vals, nil, nil, rs, 0, cache, l, startPos, K, ap, arch, useAcc64, attnPool)
				}
			case opOutGate:
				gRows := lw.GProj.Rows()
				if cap(gbuf) < K*gRows {
					gbuf = make([]float32, K*gRows)
				}
				gb := gbuf[:K*gRows]
				MatmulWS(be, ws, &lw.GProj, reg(o.In), gb, K)
				perHead := gRows == nH
				for i := range K {
					if o.Gate.Sigmoid {
						applySigmoidGateRow(row(gb, i, gRows), row(ctx, i, qDim), perHead, nH, hd)
					} else {
						applyGateRow(row(gb, i, gRows), row(ctx, i, qDim), perHead, nH, hd)
					}
				}
			case opProjO:
				out := reg(o.Out)
				MatmulWS(be, ws, &lw.OProj, ctx, out, K)
				if o.Proj.Bias {
					for i := range K {
						AddBias(row(out, i, hidden), lw.OBias)
					}
				}
			case opFFN:
				if o.FFN.Form == ffnGemma4Dense {
					if err := m.gemma4DenseFFNBatch(o.FFN.Act, reg(o.In), reg(o.Out), lw, arch.FFNAt(l), K, hq); err != nil {
						return nil, err
					}
				} else if err := m.ffnBatch(&o.FFN, reg(o.In), reg(o.Out), o.Out, lw, K, ws, hq, gate, up, &guOps, ps); err != nil {
					return nil, err
				}
			case opGemma4Attn:
				m.gemma4AttnBatch(&o.G4, reg(o.In), reg(o.Out), lw, cache, l, startPos, K, gp, hq)
			case opGemma4MoE:
				for i := range K {
					hi := row(h, i, hidden)
					copy(hi, gemma4MoEFFNWith(be, hidden, o.Norms.Eps, o.Norms.AddOne, hi, lw.Gemma4MoE, m.Pager, gp.moeScratch()))
				}
			case opPLE:
				m.gemma4PLEBatch(h, l, K, lw, &o.Norms, gp)
			case opLayerScale:
				if lw.LayerScalar != 0 {
					for i := range h {
						h[i] *= lw.LayerScalar
					}
				}
			case opDeltaNet:
				m.deltaNetBatch(o.Delta, reg(o.In), reg(o.Out), lw, cache.Delta[l], K, hq, ps.q35())
			case opGatedAttn:
				if o.Gated.Attn.Order != attnQuery {
					return nil, fmt.Errorf("decoder: layer %d: gated attention runs AttendQuery's order only, not order %d", l, o.Gated.Attn.Order)
				}
				m.gatedAttnBatch(&o.Gated, reg(o.In), reg(o.Out), lw, cache, l, startPos, K, q, k, v, ctx, alk, alv, attnPool, hq, ps.q35())
			case opResidual:
				switch {
				case o.Res.Two:
					AddResidual2(h, att, mlpOut)
				case o.Res.Scale != 0:
					AddScaled(h, att, float32(o.Res.Scale))
				default:
					AddResidual(h, att)
				}
			default:
				return nil, fmt.Errorf("decoder: layer %d: operation %v is not implemented on the CPU's batched walker", l, o.Kind)
			}
			if cache.Cost != nil {
				m.countOpN(cache, o, lw, l, K, causalRows(cache, l, startPos, K), t0)
			}
		}
		if ds := cache.Deepstack; ds != nil && l < len(ds.Rows) {
			AddDeepstack(h, ds, l, startPos, K, hidden)
		}
		if cache.CaptureLayers != nil {
			for ci, cl := range cache.CaptureLayers {
				if cl == l {
					cache.Captured[ci] = append(cache.Captured[ci][:0], h...)
				}
			}
		}
	}

	// Advance by K explicitly: a local last layer defers its write (commitBatch does not step pos).
	cache.AdvanceTo(startPos + K)
	hp := &prog.Head
	fb := m.W.FinalNormBias
	if !hp.Norm.Bias {
		fb = nil
	}
	t0 := CostStart(cache)
	for i := range K {
		ApplyNorm(hp.Norm, row(h, i, hidden), row(h, i, hidden), m.W.FinalNorm, fb, hidden)
	}
	if cache.Cost != nil {
		m.CountFinalNorm(cache, K, fb, t0)
	}
	return h, nil
}

// ffnBatch writes the feed-forward output of K rows of x into out. A MoE layer (gpt-oss's included) runs its experts per row (the
// router picks per token) or expert-major in chunks, its buffers in the pass scratch; a dense layer runs gate/up at M=K.
func (m *Engine) ffnBatch(p *ffnParams, x, out []float32, outReg opReg, lw *darch.LayerWeights, K int, ws *linalg.Workspace, hq *W4A8Act,
	gate, up []float32, guOps *[2]linalg.BatchOp, ps *PrefillPassScratch) error {
	arch, be := m.W.Arch, m.Be
	hidden := arch.HiddenDim
	switch p.Form {
	case ffnMoE, ffnGptOss:
		if outReg == regB {
			// No parallel-block MoE family exists; its joint residual add is not written.
			return ErrNotImplemented
		}
		ms := ps.MoEScratch()
		if cap(ms.done) < K {
			ms.done = make([]bool, K)
			MoEScratchGrows.Done.Add(1)
		}
		emDone := ms.done[:K]
		clear(emDone)
		var emOut []float32
		if knobMoEExpertMajor(m.Knobs) {
			emOut = moeGrow(&ms.out, K*hidden, &MoEScratchGrows.Out)
			for c0 := 0; c0 < K; c0 += moeExpertMajorChunk {
				c1 := min(c0+moeExpertMajorChunk, K)
				var ok bool
				var err error
				if p.Form == ffnGptOss {
					ok, err = m.GptOssMoEBatch(x[c0*hidden:c1*hidden], c1-c0, lw, p, hidden, emOut[c0*hidden:c1*hidden], ms)
				} else {
					ok, err = moeMLPBatchWith(x[c0*hidden:c1*hidden], c1-c0, lw, &p.MoE, p.Act, hidden, be, m.Pager, emOut[c0*hidden:c1*hidden], ms)
				}
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
			var ff []float32
			var err error
			if p.Form == ffnGptOss {
				ff, err = m.GptOssMoEWith(x[i*hidden:(i+1)*hidden], lw, p, hidden, nil)
			} else {
				ff, err = moeMLP(x[i*hidden:(i+1)*hidden], lw, &p.MoE, p.Act, hidden, be, nil, m.Pager)
			}
			if err != nil {
				return err
			}
			copy(dst, ff)
		}
		return nil
	case ffnNonGated:
		return ErrNotImplemented // canBatchN keeps a non-gated family on the per-token walker
	}
	if IsW8A8(&lw.GateProj) && IsW8A8(&lw.UpProj) {
		guOps[0] = linalg.BatchOp{W: &lw.GateProj, Dst: gate}
		guOps[1] = linalg.BatchOp{W: &lw.UpProj, Dst: up}
		MatmulW8A8Batch(be, ws, x, K, lw.GateProj.Cols(), guOps[:], lw.GateProj.ActQuantGroup())
	} else {
		hq.prepareWS(be, ws, &lw.GateProj, x, K)
		MatmulPreWS(be, ws, &lw.GateProj, hq, x, gate, K)
		MatmulPreWS(be, ws, &lw.UpProj, hq, x, up, K)
	}
	if err := activateRows(p.Act, gate, up); err != nil {
		return err
	}
	MatmulWS(be, ws, &lw.DownProj, gate, out, K)
	return nil
}

// activateRows is the batched paths' gated activation over gate and up, in place into gate.
func activateRows(act darch.ActKind, gate, up []float32) error {
	switch act {
	case darch.ActGeluTanh:
		if len(gate) < activationFanoutThreshold {
			geglu(gate, up)
		} else {
			parallelElementwise(len(gate), func(lo, hi int) {
				for j := lo; j < hi; j++ {
					gate[j] = normact.GELUTanh(gate[j]) * up[j]
				}
			})
		}
	case darch.ActSiLU:
		if len(gate) < activationFanoutThreshold {
			Swiglu(gate, up)
		} else {
			parallelElementwise(len(gate), func(lo, hi int) {
				for j := lo; j < hi; j++ {
					gate[j] = normact.SiLU(gate[j]) * up[j]
				}
			})
		}
	case darch.ActGelu:
		if len(gate) < activationFanoutThreshold {
			gegluExact(gate, up)
		} else {
			parallelElementwise(len(gate), func(lo, hi int) {
				for j := lo; j < hi; j++ {
					gate[j] = normact.GELUErf(gate[j]) * up[j]
				}
			})
		}
	default:
		return ErrNotImplemented
	}
	return nil
}

// WalkMulti runs prog's layers for B independent sequences, one token each, each on its own cache (the batched CPU decode), then the head's
// norm; it returns the B rows' hidden states. Its operations compute what walkToken's do for each row, with the projections at
// M=B; the program must be one multiStepEligible admits.
func (m *Engine) WalkMulti(prog *layerProgram, h []float32, caches []*Cache) ([]float32, error) {
	arch := m.W.Arch
	be := m.Be
	B := len(caches)
	hidden, nKV, hd := arch.HiddenDim, arch.NumKVHeads, arch.HeadDim
	maxQDim, kvDim, inter := arch.MaxHeads()*hd, nKV*hd, arch.IntermediateDim
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
	var qkvOps [3]linalg.BatchOp
	var guOps [2]linalg.BatchOp
	var ws4 linalg.Workspace // the fused W4A8 calls, at the threshold Matmul() gives int4 weights
	var hq W4A8Act           // the B-row normed block for the Matmul fallbacks, quantized once per projection group
	ws4.SetThreshold(Int4ParThreshold)
	var qkvOps4 [3]linalg.BatchOp
	var guOps4 [2]linalg.BatchOp
	reg := func(r opReg) []float32 {
		switch r {
		case regX:
			return norm
		case regA:
			return att
		}
		return h
	}

	for l, ops := range prog.Layers {
		lw := &m.W.Layers[l]
		nH := arch.HeadsAt(l)
		qDim := nH * hd
		q, ctx := q[:B*qDim], ctx[:B*qDim]
		for oi := range ops {
			o := &ops[oi]
			switch o.Kind {
			case opNorm:
				src, dst := reg(o.In), reg(o.Out)
				w, bias := lw.NormWeights(o.Norm)
				for b := range B {
					ApplyNorm(o.Norm, row(dst, b, hidden), row(src, b, hidden), w, bias, hidden)
				}
			case opProjQKV:
				x := reg(o.In)
				if IsW8A8(&lw.QProj) && IsW8A8(&lw.KProj) && IsW8A8(&lw.VProj) {
					qkvOps[0] = linalg.BatchOp{W: &lw.QProj, Dst: q}
					qkvOps[1] = linalg.BatchOp{W: &lw.KProj, Dst: k}
					qkvOps[2] = linalg.BatchOp{W: &lw.VProj, Dst: v}
					MatmulW8A8Batch(be, &ws, x, B, lw.QProj.Cols(), qkvOps[:], lw.QProj.ActQuantGroup())
				} else if group, ag, ok := W4A8FusedOps(qkvOps4[:], []*linalg.WeightMat{&lw.QProj, &lw.KProj, &lw.VProj}, [][]float32{q, k, v}); ok {
					MatmulW4A8Batch(be, &ws4, x, B, lw.QProj.Cols(), group, qkvOps4[:], ag) // q‖k‖v in one fork/join
				} else {
					hq.Prepare(be, &lw.QProj, x, B)
					MatmulPre(be, &lw.QProj, &hq, x, q, B)
					MatmulPre(be, &lw.KProj, &hq, x, k, B)
					MatmulPre(be, &lw.VProj, &hq, x, v, B)
				}
				if o.Proj.Bias {
					for b := range B {
						AddBias(row(q, b, qDim), lw.QBias)
						AddBias(row(k, b, kvDim), lw.KBias)
						AddBias(row(v, b, kvDim), lw.VBias)
					}
				}
			case opQKNorm:
				for b := range B {
					applyQKNorm(o.QKNorm, row(q, b, qDim), row(k, b, kvDim), lw.QNorm, lw.KNorm, nH, nKV, hd)
				}
			case opRope:
				for b, c := range caches {
					applyRope(&o.Rope, row(q, b, qDim), row(k, b, kvDim), nH, nKV, hd, pos[b], c.MRopePos, c.MRopeDelta)
				}
			case opAttnTemp:
				for b := range B {
					scale := o.Temp.Scale(pos[b])
					qb := row(q, b, qDim)
					for j := range qb {
						qb[j] *= scale
					}
				}
			case opAttention:
				if o.Attn.Order != attnBatchedAcc64 {
					return nil, fmt.Errorf("decoder: layer %d: attention order %d is not implemented on the CPU's multi-sequence walker", l, o.Attn.Order)
				}
				for b, c := range caches { // each sequence's append-forever cache, at K=1 with float64 accumulation, read as attendToken reads it
					kb, vb := row(k, b, kvDim), row(v, b, kvDim)
					if activeHooks().KVRoundF16 {
						roundF16InPlace(kb)
						roundF16InPlace(vb)
					}
					c.Append(l, kb, vb)
					keys, vals, rs := c.LayerRows(l, kvDim)
					nKeys := c.StoredRows(l, kvDim)
					if c.Quant.widened() { // f16 or int8: widened into the cache's own scratch, compact
						lk, lv := c.Scr.LocalBufs(nKeys * kvDim)
						nKeys = c.DequantGlobalLayer(l, kvDim, lk, lv)
						keys, vals, rs = lk[:nKeys*kvDim], lv[:nKeys*kvDim], kvDim
					}
					pool := c.Scr.HeadWorkerPool(nH, 1, nKeys, hd, false, true)
					ap := o.Attn
					ap.Sinks = lw.AttnSinks
					AttendBatchedHeadsKV(row(q, b, qDim), row(ctx, b, qDim), keys, vals, nil, nil, rs, 0, c, l, pos[b], 1, ap, arch, true, pool)
				}
			case opProjO:
				out := reg(o.Out)
				Matmul(be, &lw.OProj, ctx, out, B)
				if o.Proj.Bias {
					for b := range B {
						AddBias(row(out, b, hidden), lw.OBias)
					}
				}
			case opFFN:
				if o.FFN.Form != ffnGated {
					return nil, fmt.Errorf("decoder: layer %d: the multi-sequence walker runs a gated dense FFN only", l)
				}
				x, out := reg(o.In), reg(o.Out)
				if IsW8A8(&lw.GateProj) && IsW8A8(&lw.UpProj) {
					guOps[0] = linalg.BatchOp{W: &lw.GateProj, Dst: gate}
					guOps[1] = linalg.BatchOp{W: &lw.UpProj, Dst: up}
					MatmulW8A8Batch(be, &ws, x, B, lw.GateProj.Cols(), guOps[:], lw.GateProj.ActQuantGroup())
				} else if group, ag, ok := W4A8FusedOps(guOps4[:], []*linalg.WeightMat{&lw.GateProj, &lw.UpProj}, [][]float32{gate, up}); ok {
					MatmulW4A8Batch(be, &ws4, x, B, lw.GateProj.Cols(), group, guOps4[:], ag) // gate‖up in one fork/join
				} else {
					hq.Prepare(be, &lw.GateProj, x, B)
					MatmulPre(be, &lw.GateProj, &hq, x, gate, B)
					MatmulPre(be, &lw.UpProj, &hq, x, up, B)
				}
				if err := activateRows(o.FFN.Act, gate, up); err != nil {
					return nil, err
				}
				Matmul(be, &lw.DownProj, gate, out, B)
			case opResidual:
				if o.Res.Two || o.Res.Scale != 0 {
					return nil, fmt.Errorf("decoder: layer %d: the multi-sequence walker has no parallel block or scaled residual", l)
				}
				AddResidual(h, att)
			default:
				return nil, fmt.Errorf("decoder: layer %d: operation %v is not implemented on the CPU's multi-sequence walker", l, o.Kind)
			}
		}
	}
	hp := &prog.Head
	fb := m.W.FinalNormBias
	if !hp.Norm.Bias {
		fb = nil
	}
	for b := range B {
		ApplyNorm(hp.Norm, row(h, b, hidden), row(h, b, hidden), m.W.FinalNorm, fb, hidden)
	}
	return h, nil
}
