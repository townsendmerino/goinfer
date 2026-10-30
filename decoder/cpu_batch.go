package decoder

import (
	"fmt"

	"github.com/townsendmerino/aikit/linalg"
)

// MC3c step 2 (docs/tasks/task-concurrency-2026-09.md): several CPU generations of one model decoding at once join their
// decode tokens into one batched forward (decodeMultiStep) instead of running N independent forwards. Measurements:
// docs/measurements/concurrency-mc2-2026-09-26.md.
//
// The coalescing is MC3's (tokenCoalescer, mc3_batch.go): a run starts once every decoding generation has submitted
// its token, or when the straggler window expires. A run of at least two tokens is one decodeMultiStep; a lone token is
// production's own m.forward, so a request served alone takes exactly today's path. Unlike the resident batcher there
// is nothing to hold exclusively: prefill and every non-batched path run outside it, as under step 1.

// CPUBatchDecode values (Options.CPUBatchDecode).
const (
	CPUBatchAuto = 0  // batch an eligible model with at least cpuBatchAutoMinBytes of dense weights
	CPUBatchOn   = 1  // batch every eligible model
	CPUBatchOff  = -1 // never batch: step 1's independent workers
)

// cpuBatchAutoMinBytes is CPUBatchAuto's threshold, pre-registered. Batching beats the independent workers on 7B-class
// dense weights (4.93 GB), loses on 0.5B-class (0.47 GB) and is a wash on 1.5B-class (1.23 GB), on both amd64 and
// darwin; below the threshold the workers are no worse off. The range between 2 GiB and the 7B is unmeasured. Figures:
// docs/measurements/concurrency-mc2-2026-09-26.md.
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

// CPUBatchKVBypass is why this model's decode tokens bypass the CPU batcher although EnableCPUBatch engaged it: its CPU cache is f16 or
// int8, which decodeMultiStep does not read (cpuBatchCacheEligible), so each generation decodes on its own at that precision. "" when the
// cache is f32. Batched decode explicitly on with such a cache refuses the load instead (ResolveKV).
func (m *Model) CPUBatchKVBypass() string { return cpuBatchKVBypass(m.CPUKVPrecision()) }

func cpuBatchKVBypass(prec string) string {
	if prec != kvPrecF16 && prec != kvPrecI8 {
		return ""
	}
	return "the " + prec + " KV cache decodes per sequence; the CPU batcher reads f32 KV only until phase 4"
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
	if a.ownLoopFamily() {
		return fmt.Errorf("cpu batch: %s has its own forward", a.Name)
	}
	if m.layerPager != nil || m.w.Embed.Rows() == 0 {
		return fmt.Errorf("cpu batch: %s is not a plain gated dense family", a.Name)
	}
	if err := m.program().multiStepEligible(); err != nil {
		return fmt.Errorf("cpu batch: %s: %w", a.Name, err)
	}
	return nil
}

// cpuBatchCacheEligible is decodeMultiStep's per-cache scope: f32 append-forever KV (no sliding-window ring, no int8
// KV) with no adapter, tree mask or manual position. A token on any other cache runs production's own forward.
func cpuBatchCacheEligible(c *KVCache) error {
	if c == nil || c.scr == nil || c.localAny || c.quant.widened() || c.lora != nil || c.treeMask != nil || c.manualPos {
		return fmt.Errorf("cpu batch: not a plain f32 append-forever cache")
	}
	return nil
}

// cpuBatchFusedW4A8 lets decodeMultiStep run an int4 layer's q‖k‖v, and its gate‖up, each as one batched W4A8 call over
// the B rows instead of one matmul per projection. The kernel (linalg.MatmulBTW4A8Batch) is numerically identical to
// calling MatmulBTW4A8Into once per op, so this changes only how many fork/joins a step pays. A package variable, not
// an environment read: the measurement (docs/tasks/task-concurrency-2026-09.md, MC3c step 2 S1) flips it in-process to
// interleave its arms.
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

// decodeMultiStep is MC3c step 2's batched step: one forward carrying B independent sequences, each with its own KV
// cache at its own position, one token each. The projections, o-proj, MLP and LM head run as M = B matmuls through the
// kernels forwardN uses (bit-identical to M = 1 row for row, by forwardN's own contract). Attention runs per sequence
// over its own cache, exactly as attendToken's default (f32 KV, append-forever) case does at decode. Every row's
// logits are bit-identical to that sequence's own m.forward (TestMC2_decodeMultiStepBitIdentical,
// TestCPUBatch_everyEligibleFixtureBitIdentical).
func (m *Model) decodeMultiStep(ids []int, caches []*KVCache) ([][]float32, error) {
	if err := m.cpuBatchModelEligible(); err != nil {
		return nil, err
	}
	for i, c := range caches {
		if err := cpuBatchCacheEligible(c); err != nil {
			return nil, fmt.Errorf("cache %d: %w", i, err)
		}
	}
	prog := m.program()
	hidden, vocab := m.w.arch.HiddenDim, m.w.arch.VocabSize
	h := make([]float32, len(ids)*hidden)
	for b, id := range ids {
		hb := h[b*hidden : (b+1)*hidden]
		m.w.Embed.Row(id, hb)
		if s := prog.embed.Scale; s != 0 && s != 1 {
			for j := range hb {
				hb[j] *= float32(s)
			}
		}
	}
	h, err := m.walkMulti(prog, h, caches)
	if err != nil {
		return nil, err
	}
	flat := m.lmHeadN(h, len(ids))
	out := make([][]float32, len(ids))
	for b := range out {
		out[b] = flat[b*vocab : (b+1)*vocab]
	}
	return out, nil
}
