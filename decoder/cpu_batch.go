package decoder

import (
	"fmt"
	"math"

	"github.com/townsendmerino/aikit/linalg"
)

// MC3c step 2 (docs/tasks/task-concurrency-2026-09.md): several CPU generations of one model decoding at once join
// their decode tokens into one batched forward (decodeMultiStep) instead of running N independent forwards. The
// trigger measured it on nobara's 7B: batched B = 4 is 2.25-2.41x J8's 4 independent workers, which barely scale
// there (docs/measurements/concurrency-mc2-2026-09-26.md, "Linux 7B cell").
//
// The coalescing is MC3's (tokenCoalescer, mc3_batch.go): a run starts once every decoding generation has submitted
// its token, or when the straggler window expires. A run of at least two tokens is one decodeMultiStep; a lone token is
// production's own m.forward, so a request served alone takes exactly today's path. Unlike the resident batcher there
// is nothing to hold exclusively — prefill and every non-batched path run outside it, as under step 1.

// CPUBatchDecode values (Options.CPUBatchDecode).
const (
	CPUBatchAuto = 0  // batch an eligible model with at least cpuBatchAutoMinBytes of dense weights
	CPUBatchOn   = 1  // batch every eligible model
	CPUBatchOff  = -1 // never batch: step 1's independent workers
)

// cpuBatchAutoMinBytes is CPUBatchAuto's threshold, pre-registered 2026-09-27. Batching's lead over the workers at
// B = N = 4 measured 0.64-0.75x on the 0.5B (0.47 GB of dense weights), 1.02-1.11x on the 1.5B (1.23 GB) and 2.25-2.41x
// on the 7B (4.93 GB) on nobara, and the workers led on the Mac's 0.5B and 1.5B. On the Mac's 7B batching leads by
// 1.54x (depth 128) and 1.38x (512) (2026-09-27, docs/measurements/concurrency-mc2-2026-09-26.md, "Mac 7B cell"), so
// the threshold holds on darwin too. Below it the workers are no worse off; above it batching wins by a wide margin.
// The range between 2 GiB and the 7B is unmeasured.
const cpuBatchAutoMinBytes = 2 << 30

// cpuBatcher is a Model's MC3c step-2 coordinator (Model.EnableCPUBatch).
type cpuBatcher struct {
	tokenCoalescer
	m *Model
}

// EnableCPUBatch turns on MC3c step 2 for up to n concurrent CPU generations of this model and reports whether it did.
// Call it before any generation starts; each call decides afresh. It stays off, and generations run as step 1's independent workers, when:
//   - n < 2, or the model decodes on a GPU resident (MC3 is that path's batching);
//   - Options.CPUBatchDecode is CPUBatchOff;
//   - the family is outside decodeMultiStep's scope (cpuBatchModelEligible);
//   - under CPUBatchAuto: the model's dense weights are under cpuBatchAutoMinBytes.
func (m *Model) EnableCPUBatch(n int) bool {
	m.cpuBatch = nil // a repeated call (a changed -max-concurrent) decides afresh
	if n < 2 || m.resident != nil || m.cpuBatchMode == CPUBatchOff || m.cpuBatchModelEligible() != nil {
		return false
	}
	if m.cpuBatchMode == CPUBatchAuto && m.ResidentDenseWeightBytes() < cpuBatchAutoMinBytes {
		return false
	}
	b := &cpuBatcher{m: m}
	b.lo, b.hi, b.wait, b.step = 2, n, batchStragglerWait, b.stepCPU
	b.init()
	m.cpuBatch = b
	return true
}

// CPUBatchActive reports whether EnableCPUBatch turned batching on.
func (m *Model) CPUBatchActive() bool { return m.cpuBatch != nil }

// CPUBatchDecodeMode is Options.CPUBatchDecode for this model.
func (m *Model) CPUBatchDecodeMode() int { return m.cpuBatchMode }

// CPUBatchStats is a snapshot of the CPU batcher's counters (runs, steps, step sizes, solo tokens); zero when off.
func (m *Model) CPUBatchStats() ResidentBatchStats {
	if m.cpuBatch == nil {
		return ResidentBatchStats{}
	}
	m.cpuBatch.mu.Lock()
	defer m.cpuBatch.mu.Unlock()
	return m.cpuBatch.stats
}

// stepCPU is the CPU batcher's step: one decodeMultiStep over the run's tokens, each on its own session cache. It runs
// on whichever generation's goroutine started the run; the others are blocked in forward, so their caches are not
// touched concurrently.
func (b *cpuBatcher) stepCPU(run []*batchReq) {
	ids, caches := make([]int, len(run)), make([]*KVCache, len(run))
	for i, r := range run {
		ids[i], caches[i] = r.id, r.cache
	}
	logits, err := runStep(func() ([][]float32, error) { return b.m.decodeMultiStep(ids, caches) })
	for i, r := range run {
		if err != nil {
			r.err = err
		} else {
			r.out = &ResidentBatchOut{Logits: logits[i], ID: -1}
		}
	}
}

// runStep is runSolo for a batched step: a panic becomes every token's error rather than unwinding past the run.
func runStep(f func() ([][]float32, error)) (out [][]float32, err error) {
	defer func() {
		if p := recover(); p != nil {
			out, err = nil, fmt.Errorf("decoder: batched CPU step panicked: %v", p)
		}
	}()
	return f()
}

// cpuBatchModelEligible is decodeMultiStep's model-level scope: the generic forward (no family-specific runLayers), a
// gated dense MLP (no MoE, no non-gated GPT-2-style MLP), the plain pre-norm placement on every layer, no attention
// output gate, no learned positions, a token-embedding table, and no dense weight streaming. It is at least as strict as
// forwardN's canBatchN, whose batched kernels the step reuses. TestCPUBatch_everyEligibleFixtureBitIdentical holds every
// committed family it admits to bit-identity with the production forward.
func (m *Model) cpuBatchModelEligible() error {
	a := m.w.arch
	if _, own := a.ownForward(); own {
		return fmt.Errorf("cpu batch: %s has its own forward", a.Name)
	}
	if a.MoE != nil || a.NonGatedMLP || a.LearnedPosEmbed || a.hasAttnOutputGate() || m.layerPager != nil || m.w.Embed.Rows() == 0 {
		return fmt.Errorf("cpu batch: %s is not a plain gated dense family", a.Name)
	}
	for l := 0; l < a.NumLayers; l++ {
		if p := a.normPlacementAt(l); p == NormSandwich4 || p == NormPostOnly || p == NormParallel {
			return fmt.Errorf("cpu batch: layer %d norm placement %v", l, p)
		}
	}
	return nil
}

// cpuBatchCacheEligible is decodeMultiStep's per-cache scope: f32 append-forever KV (no sliding-window ring, no int8
// KV) with no adapter, tree mask or manual position. A token on any other cache runs production's own forward.
func cpuBatchCacheEligible(c *KVCache) error {
	if c == nil || c.scr == nil || c.localAny || c.quant == kvI8 || c.lora != nil || c.treeMask != nil || c.manualPos {
		return fmt.Errorf("cpu batch: not a plain f32 append-forever cache")
	}
	return nil
}

// cpuBatchFusedW4A8 lets decodeMultiStep run an int4 layer's q‖k‖v, and its gate‖up, each as ONE batched W4A8 call
// over the B rows (MC3c step 2 S1, docs/tasks/task-concurrency-2026-09.md) instead of one matmul per projection. The
// kernel (linalg.MatmulBTW4A8Batch) is numerically identical to calling MatmulBTW4A8Into once per op, so this changes
// only how many fork/joins a step pays: the probe put the small projections (q/o, k/v) at 1.6-3.0x one row's cost at
// M = 4, which is fork/join and compute, not bandwidth. A package variable, not an environment read: the S1
// measurement flips it in-process to interleave its arms.
var cpuBatchFusedW4A8 = true

// w4a8FusedOps fills ops with one linalg.W4A8Op per weight for a fused batched W4A8 call and reports whether the group
// qualifies: fusion on, every weight int4 with its canonical bytes present (wmW4A8Op passes no split-half layout), one
// weight-group size, one K, and one activation-quant group. It returns that group size and activation-quant group.
// Anything else keeps the per-projection matmul calls.
func w4a8FusedOps(ops []linalg.W4A8Op, ws []*linalg.WeightMat, dsts [][]float32) (group, actGroup int, ok bool) {
	if !cpuBatchFusedW4A8 {
		return 0, 0, false
	}
	for i, w := range ws {
		_, _, g, canon := w.Int4F16()
		if !isW4A8(w) || !canon || w.Cols() != ws[0].Cols() || w.ActQuantGroup() != ws[0].ActQuantGroup() {
			return 0, 0, false
		}
		var gi int
		ops[i], gi = wmW4A8Op(w, dsts[i])
		if gi != g || (i > 0 && g != group) {
			return 0, 0, false
		}
		group = g
	}
	return group, ws[0].ActQuantGroup(), true
}

// decodeMultiStep is MC3c step 2's batched step (promoted from MC2's test-only prototype): ONE forward carrying B
// independent sequences, each with its own KV cache at its own position, one token each. The projections, o-proj, MLP
// and LM head run as M = B matmuls through the kernels forwardN uses (bit-identical to M = 1 row for row, by forwardN's
// own contract). Attention runs per sequence over its own cache, exactly as causalAttention's default (f32 KV,
// append-forever) case does at decode. Every row's logits are bit-identical to that sequence's own m.forward
// (TestMC2_decodeMultiStepBitIdentical, TestCPUBatch_everyEligibleFixtureBitIdentical).
func (m *Model) decodeMultiStep(ids []int, caches []*KVCache) ([][]float32, error) {
	if err := m.cpuBatchModelEligible(); err != nil {
		return nil, err
	}
	for i, c := range caches {
		if err := cpuBatchCacheEligible(c); err != nil {
			return nil, fmt.Errorf("cache %d: %w", i, err)
		}
	}
	arch := m.w.arch
	be := m.be
	B := len(ids)
	hidden, nKV, hd := arch.HiddenDim, arch.NumKVHeads, arch.HeadDim
	maxQDim, kvDim, inter := arch.maxHeads()*hd, nKV*hd, arch.IntermediateDim
	row := func(b []float32, i, w int) []float32 { return b[i*w : i*w+w] }

	h := make([]float32, B*hidden)
	for b, id := range ids {
		m.w.Embed.Row(id, row(h, b, hidden))
		if arch.EmbedScale != 0 && arch.EmbedScale != 1 {
			s := float32(arch.EmbedScale)
			for j := range row(h, b, hidden) {
				h[b*hidden+j] *= s
			}
		}
	}
	norm := make([]float32, B*hidden)
	q, k, v := make([]float32, B*maxQDim), make([]float32, B*kvDim), make([]float32, B*kvDim)
	ctx, att := make([]float32, B*maxQDim), make([]float32, B*hidden)
	gate, up, mlpOut := make([]float32, B*inter), make([]float32, B*inter), make([]float32, B*hidden)
	pos := make([]int, B)
	for b, c := range caches {
		pos[b] = c.Pos()
	}
	var ws linalg.Workspace
	ws.SetThreshold(DefaultDecodeParallelThreshold)
	var qkvOps [3]linalg.W8A8Op
	var guOps [2]linalg.W8A8Op
	// ws4: the fused W4A8 calls (S1), at the threshold matmul() gives int4 weights.
	var ws4 linalg.Workspace
	ws4.SetThreshold(int4ParThreshold)
	var qkvOps4 [3]linalg.W4A8Op
	var guOps4 [2]linalg.W4A8Op

	for l := 0; l < arch.NumLayers; l++ {
		lw := &m.w.Layers[l]
		global := arch.isGlobalLayer(l)
		nH := arch.headsAt(l)
		qDim := nH * hd
		q, ctx := q[:B*qDim], ctx[:B*qDim]
		for b := range B {
			normalizeInto(arch, row(norm, b, hidden), row(h, b, hidden), lw.PreAttnNorm, lw.PreAttnNormBias, hidden)
		}
		if isW8A8(&lw.QProj) && isW8A8(&lw.KProj) && isW8A8(&lw.VProj) {
			qkvOps[0] = linalg.W8A8Op{BQ: wmInt8(&lw.QProj), Scales: wmScales(&lw.QProj), Dst: q, N: lw.QProj.Rows()}
			qkvOps[1] = linalg.W8A8Op{BQ: wmInt8(&lw.KProj), Scales: wmScales(&lw.KProj), Dst: k, N: lw.KProj.Rows()}
			qkvOps[2] = linalg.W8A8Op{BQ: wmInt8(&lw.VProj), Scales: wmScales(&lw.VProj), Dst: v, N: lw.VProj.Rows()}
			matmulW8A8Batch(be, &ws, norm, B, lw.QProj.Cols(), qkvOps[:], lw.QProj.ActQuantGroup())
		} else if group, ag, ok := w4a8FusedOps(qkvOps4[:], []*linalg.WeightMat{&lw.QProj, &lw.KProj, &lw.VProj}, [][]float32{q, k, v}); ok {
			// S1: q‖k‖v in one fork/join over the B rows — numerically identical to three matmul calls per op.
			matmulW4A8Batch(be, &ws4, norm, B, lw.QProj.Cols(), group, qkvOps4[:], ag)
		} else {
			matmul(be, &lw.QProj, norm, q, B)
			matmul(be, &lw.KProj, norm, k, B)
			matmul(be, &lw.VProj, norm, v, B)
		}
		invFreq, ms := arch.ropeInvFreq(l), arch.ropeMscale(l)
		noPE := arch.isNoPELayer(l)
		for b, c := range caches {
			qb, kb, vb, cb := row(q, b, qDim), row(k, b, kvDim), row(v, b, kvDim), row(ctx, b, qDim)
			if arch.QKVBias {
				addBias(qb, lw.QBias)
				addBias(kb, lw.KBias)
				addBias(vb, lw.VBias)
			}
			if arch.QKNorm {
				if arch.QKNormWhole {
					rmsNorm(qb, lw.QNorm, 1, nH*hd, arch.NormEps, arch.RMSAddOne)
					rmsNorm(kb, lw.KNorm, 1, nKV*hd, arch.NormEps, arch.RMSAddOne)
				} else {
					rmsNorm(qb, lw.QNorm, nH, hd, arch.NormEps, arch.RMSAddOne)
					rmsNorm(kb, lw.KNorm, nKV, hd, arch.NormEps, arch.RMSAddOne)
				}
			}
			if !noPE {
				ropeAt(qb, nH, hd, pos[b], invFreq, ms, arch.MRopeSection, c.mropePos, c.mropeDelta, arch.ropeInterleave, arch.MRopeInterleaved)
				ropeAt(kb, nKV, hd, pos[b], invFreq, ms, arch.MRopeSection, c.mropePos, c.mropeDelta, arch.ropeInterleave, arch.MRopeInterleaved)
			}
			if arch.AttnTempBeta != 0 {
				scale := float32(1 + arch.AttnTempBeta*math.Log1p(math.Floor(float64(pos[b])/arch.AttnTempOrigMaxPos)))
				for j := range qb {
					qb[j] *= scale
				}
			}
			// causalAttention's default case, verbatim: f32 global, append-forever, acc64.
			c.Append(l, kb, vb)
			nKeys := c.storedRows(l, kvDim)
			pool := c.scr.headWorkerPool(nH, 1, nKeys, hd, false, true)
			attendBatchedHeads(qb, cb, c.Keys(l), c.Vals(l), 0, c, l, pos[b], 1, global, arch, true, pool)
		}
		matmul(be, &lw.OProj, ctx, att, B)
		if arch.OutBias {
			for b := range B {
				addBias(row(att, b, hidden), lw.OBias)
			}
		}
		addResidual(h, att)
		for b := range B {
			normalizeInto(arch, row(norm, b, hidden), row(h, b, hidden), lw.PreMLPNorm, lw.PreMLPNormBias, hidden)
		}
		if isW8A8(&lw.GateProj) && isW8A8(&lw.UpProj) {
			guOps[0] = linalg.W8A8Op{BQ: wmInt8(&lw.GateProj), Scales: wmScales(&lw.GateProj), Dst: gate, N: lw.GateProj.Rows()}
			guOps[1] = linalg.W8A8Op{BQ: wmInt8(&lw.UpProj), Scales: wmScales(&lw.UpProj), Dst: up, N: lw.UpProj.Rows()}
			matmulW8A8Batch(be, &ws, norm, B, lw.GateProj.Cols(), guOps[:], lw.GateProj.ActQuantGroup())
		} else if group, ag, ok := w4a8FusedOps(guOps4[:], []*linalg.WeightMat{&lw.GateProj, &lw.UpProj}, [][]float32{gate, up}); ok {
			// S1: gate‖up in one fork/join over the B rows.
			matmulW4A8Batch(be, &ws4, norm, B, lw.GateProj.Cols(), group, guOps4[:], ag)
		} else {
			matmul(be, &lw.GateProj, norm, gate, B)
			matmul(be, &lw.UpProj, norm, up, B)
		}
		switch arch.Act { // forwardN's activation step, verbatim
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
			return nil, errNotImplemented
		}
		matmul(be, &lw.DownProj, gate, mlpOut, B)
		addResidual(h, mlpOut)
	}
	for b := range B {
		normalize(arch, row(h, b, hidden), m.w.FinalNorm, m.w.FinalNormBias, hidden)
	}
	flat := m.lmHeadN(h, B)
	out := make([][]float32, B)
	for b := range B {
		out[b] = flat[b*arch.VocabSize : (b+1)*arch.VocabSize]
	}
	return out, nil
}
