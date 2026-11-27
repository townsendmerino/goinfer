package decoder

import (
	"fmt"

	"github.com/townsendmerino/goinfer/internal/runtime/cpuengine"
)

// The batched CPU decode: several CPU generations of one model decoding at once join their
// decode tokens into one batched forward (decodeMultiStep) instead of running N independent forwards. Measurements:
// docs/measurements/concurrency-mc2-2026-09-26.md.
//
// The coalescing is the resident batcher's (tokenCoalescer, mc3_batch.go): a run starts once every decoding generation has submitted
// its token, or when the straggler window expires. A run of at least two tokens is one decodeMultiStep; a lone token is
// production's own m.forward, so a request served alone takes exactly today's path. Unlike the resident batcher there
// is nothing to hold exclusively: prefill and every non-batched path run outside it, as under the independent CPU workers.

// CPUBatchDecode values (Options.CPUBatchDecode).
const (
	CPUBatchAuto = 0  // batch an eligible model with at least cpuBatchAutoMinBytes of dense weights
	CPUBatchOn   = 1  // batch every eligible model
	CPUBatchOff  = -1 // never batch: the independent CPU workers
)

// cpuBatchAutoMinBytes is CPUBatchAuto's threshold, pre-registered. Batching beats the independent workers on 7B-class
// dense weights (4.93 GB), loses on 0.5B-class (0.47 GB) and is a wash on 1.5B-class (1.23 GB), on both amd64 and
// darwin; below the threshold the workers are no worse off. The range between 2 GiB and the 7B is unmeasured. Figures:
// docs/measurements/concurrency-mc2-2026-09-26.md.
const cpuBatchAutoMinBytes = 2 << 30

// cpuBatcher is a Model's batched-CPU-decode coordinator (Model.EnableCPUBatch).
type cpuBatcher struct {
	tokenCoalescer
	m *Model
}

// EnableCPUBatch turns on the batched CPU decode for up to n concurrent CPU generations of this model and reports whether it did.
// Call it before any generation starts; each call decides afresh. It stays off, and generations run as independent CPU workers, when:
//   - n < 2, or the model decodes on a GPU resident (the resident batcher is that path's batching);
//   - Options.CPUBatchDecode is CPUBatchOff;
//   - the family is outside decodeMultiStep's scope (cpuBatchModelEligible);
//   - under CPUBatchAuto: the model's dense weights are under cpuBatchAutoMinBytes.
func (m *Model) EnableCPUBatch(n int) bool {
	m.cpuBatch = nil // a repeated call (a changed -max-concurrent) decides afresh
	if n < 2 || m.resident != nil || m.opts.CPUBatchDecode == CPUBatchOff || m.cpuBatchModelEligible() != nil {
		return false
	}
	if m.opts.CPUBatchDecode == CPUBatchAuto && m.ResidentDenseWeightBytes() < cpuBatchAutoMinBytes {
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
func (m *Model) CPUBatchDecodeMode() int { return m.opts.CPUBatchDecode }

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
	a := m.w.Arch
	if a.OwnLoopFamily() {
		return fmt.Errorf("cpu batch: %s has its own forward", a.Name)
	}
	if m.layerPager != nil || m.w.Embed.Rows() == 0 {
		return fmt.Errorf("cpu batch: %s is not a plain gated dense family", a.Name)
	}
	if err := m.program().MultiStepEligible(); err != nil {
		return fmt.Errorf("cpu batch: %s: %w", a.Name, err)
	}
	return nil
}

func (m *Model) decodeMultiStep(ids []int, caches []*KVCache) ([][]float32, error) {
	if err := m.cpuBatchModelEligible(); err != nil {
		return nil, err
	}
	cs := make([]*cpuengine.Cache, len(caches))
	for i, c := range caches {
		cs[i] = unwrapCache(c)
	}
	e := m.cpuEngine()
	return e.DecodeMultiStep(ids, cs)
}
