package decoder

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// MC3 (docs/tasks/task-concurrency-2026-09.md): several generations decoding on ONE resident at once, each on its own
// resident KV slot, their decode tokens joined into shared steps.
//
// A resident is one command queue with one set of scratch buffers, so everything that touches it — a prefill, a
// single-sequence forward, a batched step, the slot bookkeeping — runs one at a time, inside the batcher's exclusive
// section. What MC3 adds is WHAT runs there at a decode token: the batcher coalesces the tokens every decoding
// generation submits into one run. A run of at least lo batch-eligible tokens goes to ResidentBatchStepper.StepBatch
// (every row bit-identical to that sequence's own Forward); anything else — a lone generation above all — runs
// production's own per-sequence call, unchanged, so a request served alone takes exactly the path it takes today.
//
// Generations join at token boundaries: a run starts once every decoding generation has submitted, or once the oldest
// submission has waited batchStragglerWait (a generation whose consumer is slow to read its stream must not stall the
// rest). A newcomer's prefill runs between runs, whole, and takes precedence over the next run — chunked prefill
// interleaved with decode is MC5, not this.

// batchStragglerWait bounds how long a submitted token waits for the other decoding generations' tokens before its run
// starts without them. Host work between two tokens is well under a millisecond per generation; a batched step costs
// 20+ ms (MC3 S1), so this is at most a few percent of a step when it fires, and it fires only for a straggler.
const batchStragglerWait = 4 * time.Millisecond

// residentBatcher is a Model's MC3 coordinator (Model.EnableResidentConcurrency).
type residentBatcher struct {
	stepper    ResidentBatchStepper
	lo, hi     int // StepBatch serves lo..hi sequences
	maxHolders int // generations that may hold a slot at once
	wait       time.Duration

	mu          sync.Mutex
	cond        *sync.Cond
	busy        bool // a run or an exclusive section holds the resident
	exclWaiting int  // exclusive sections waiting — a run does not start ahead of them
	holders     int  // generations holding a slot (claimed, not yet released)
	decoders    int  // holders inside their decode loop
	pending     []*batchReq
	freeAt      time.Time // when the resident last came free (a run or exclusive section ended)
	timerAt     time.Time
	slotBusy    []bool // resident KV slots a holder is using
	stats       ResidentBatchStats
}

// ResidentBatchStats counts what MC3's batcher did (Model.ResidentBatchStats): runs, the tokens batched steps served
// and how many steps there were, the tokens that ran as production's per-sequence call, and how many runs started
// because the straggler window expired rather than because every decoding generation had submitted.
type ResidentBatchStats struct {
	Runs, Steps, StepTokens, SoloTokens, StragglerRuns int
	StepSizes                                          [batchStatsSizes]int // StepSizes[b] = steps of b sequences
}

const batchStatsSizes = 17

// ResidentBatchStats is a snapshot of the batcher's counters; zero when MC3 is not enabled.
func (m *Model) ResidentBatchStats() ResidentBatchStats {
	if m.batcher == nil {
		return ResidentBatchStats{}
	}
	m.batcher.mu.Lock()
	defer m.batcher.mu.Unlock()
	return m.batcher.stats
}

// batchReq is one generation's decode token.
type batchReq struct {
	seq  ResidentBatchSeq
	solo func() error      // production's own call for this token; binds the generation's slot first
	out  *ResidentBatchOut // set when a batched step served it
	err  error
	done bool
	at   time.Time
}

// EnableResidentConcurrency lets up to n generations use this model's resident at once (MC3), and returns the n it
// allowed: 1 — nothing enabled — unless the resident can batch (ResidentBatchStepper) over at least two KV slots, and
// never more than the slot count, since each running generation holds a slot. Call it before any generation starts.
func (m *Model) EnableResidentConcurrency(n int) int {
	st, ok := m.resident.(ResidentBatchStepper)
	if !ok || n < 2 || m.hasRecurrentState() {
		return 1
	}
	lo, hi := st.BatchStepRange()
	slots := m.residentSlotCount()
	if hi < 2 || slots < 2 {
		return 1
	}
	n = min(n, slots)
	b := &residentBatcher{stepper: st, lo: max(lo, 2), hi: hi, maxHolders: n, wait: batchStragglerWait,
		slotBusy: make([]bool, slots)}
	b.cond = sync.NewCond(&b.mu)
	m.batcher = b
	return n
}

// ResidentConcurrency is how many generations may use the resident at once: EnableResidentConcurrency's result, 1 when
// it was never enabled.
func (m *Model) ResidentConcurrency() int {
	if m.batcher == nil {
		return 1
	}
	return m.batcher.maxHolders
}

// ResidentBatchCapable reports whether EnableResidentConcurrency could enable anything on this model.
func (m *Model) ResidentBatchCapable() bool {
	st, ok := m.resident.(ResidentBatchStepper)
	if !ok || m.hasRecurrentState() || m.residentSlotCount() < 2 {
		return false
	}
	_, hi := st.BatchStepRange()
	return hi >= 2
}

// claim takes one of the batcher's holder places. It fails while an exclusive resident claim (tryClaimResident: the
// paths MC3 does not batch — speculation, vision, adapters) is held, and when every place is taken.
func (b *residentBatcher) claim(resBusy *int32) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	if atomic.LoadInt32(resBusy) != 0 || b.holders >= b.maxHolders {
		return false
	}
	b.holders++
	return true
}

// release gives a holder place back.
func (b *residentBatcher) release() {
	b.mu.Lock()
	b.holders--
	b.cond.Broadcast()
	b.mu.Unlock()
}

// claimExclusive is tryClaimResident under a batcher: the whole resident, only while no holder has a slot.
func (b *residentBatcher) claimExclusive(resBusy *int32) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.holders == 0 && atomic.CompareAndSwapInt32(resBusy, 0, 1)
}

// exclusive runs fn with the resident to itself, between runs: prefill, slot bookkeeping. It goes ahead of the next run.
func (b *residentBatcher) exclusive(fn func()) {
	b.mu.Lock()
	b.exclWaiting++
	for b.busy {
		b.cond.Wait()
	}
	b.exclWaiting--
	b.busy = true
	b.mu.Unlock()
	defer func() {
		b.mu.Lock()
		b.busy = false
		b.freeAt = time.Now()
		b.cond.Broadcast()
		b.mu.Unlock()
	}()
	fn()
}

// prefillTailMin is chunked prefill's shortest final pass (Options.ResidentPrefillChunk; docs/tasks/task-concurrency-
// 2026-09.md, chunked prefill, unparked by the owner 2026-09-27): residentPrefillSeed takes a suffix of at least 8
// tokens through the batched prefill, and the chunks must all run that same path. Chunking is sound only where the
// resident's prefill is chunk-invariant — the same bits whole or in chunks (TestMC5_prefillChunkInvariance on Metal) —
// so a reply never depends on whether others were decoding.
const prefillTailMin = 8

// decoding reports how many holders are inside their decode loop.
func (b *residentBatcher) decoding() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.decoders
}

// yieldToDecode lets the decoding generations run one step before a prefilling generation takes the resident for its
// next chunk: it returns once a run has started (or when nobody is decoding, or after a bounded wait — a decoder whose
// consumer has stalled must not hold the prefill up indefinitely).
func (b *residentBatcher) yieldToDecode() {
	b.mu.Lock()
	defer b.mu.Unlock()
	start := b.stats.Runs
	deadline := time.Now().Add(50 * time.Millisecond)
	for b.stats.Runs == start && b.decoders > 0 && time.Now().Before(deadline) {
		b.armLocked(deadline)
		b.cond.Wait()
	}
}

// enterDecode / exitDecode bracket a holder's decode loop: a run waits for tokens from exactly the holders inside one.
func (b *residentBatcher) enterDecode() {
	b.mu.Lock()
	b.decoders++
	b.mu.Unlock()
}

func (b *residentBatcher) exitDecode() {
	b.mu.Lock()
	b.decoders--
	b.cond.Broadcast() // a run may have been waiting for this holder's token
	b.mu.Unlock()
}

// markSlot records a holder's slot as in use (true) or free (false).
func (b *residentBatcher) markSlot(slot int, busy bool) {
	b.mu.Lock()
	if slot >= 0 && slot < len(b.slotBusy) {
		b.slotBusy[slot] = busy
	}
	b.mu.Unlock()
}

// busySlots is a snapshot of the slots holders are using, for residentAcquire (called inside exclusive).
func (b *residentBatcher) busySlots() []bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return append([]bool(nil), b.slotBusy...)
}

// forward submits one decode token and returns once a run has served it: q.out is set when a batched step did, and
// otherwise q.solo ran (production's own call, with its own results).
func (b *residentBatcher) forward(q *batchReq) error {
	b.mu.Lock()
	q.at = time.Now()
	b.pending = append(b.pending, q)
	b.cond.Broadcast()
	for !q.done {
		if !b.busy && b.exclWaiting == 0 && len(b.pending) > 0 {
			// The straggler window opens when a run COULD start: at the oldest submission or when the resident came
			// free, whichever is later. Timed from submission alone, a token submitted during a run has "waited" the
			// whole run when it ends, starts one at once without the others, and the generations phase-lock into
			// split runs every token (measured 2026-09-26: 4 generations ran as 3 batched + 1 solo on all 128 tokens,
			// 255 of 256 runs straggler-started).
			start := b.pending[0].at
			if b.freeAt.After(start) {
				start = b.freeAt
			}
			if all := len(b.pending) >= b.decoders; all || time.Since(start) >= b.wait {
				if !all {
					b.stats.StragglerRuns++
				}
				b.stats.Runs++
				n := min(len(b.pending), b.hi)
				run := b.pending[:n:n]
				b.pending = append([]*batchReq(nil), b.pending[n:]...)
				b.busy = true
				b.mu.Unlock()
				b.runTokens(run)
				b.mu.Lock()
				b.busy = false
				b.freeAt = time.Now()
				for _, r := range run {
					r.done = true
				}
				b.cond.Broadcast()
				continue
			}
			b.armLocked(start.Add(b.wait))
		}
		b.cond.Wait()
	}
	b.mu.Unlock()
	return q.err
}

// armLocked wakes the waiters at deadline (so a straggler's absence is noticed); b.mu held.
func (b *residentBatcher) armLocked(deadline time.Time) {
	if !b.timerAt.IsZero() && !deadline.Before(b.timerAt) {
		return
	}
	b.timerAt = deadline
	time.AfterFunc(time.Until(deadline), func() {
		b.mu.Lock()
		if b.timerAt.Equal(deadline) {
			b.timerAt = time.Time{}
		}
		b.cond.Broadcast()
		b.mu.Unlock()
	})
}

// runTokens serves one run, with the resident held: in one StepBatch when it has at least lo tokens, else every token
// through its own production call.
func (b *residentBatcher) runTokens(run []*batchReq) {
	if len(run) >= b.lo {
		seqs := make([]ResidentBatchSeq, len(run))
		for i, r := range run {
			seqs[i] = r.seq
		}
		out, err := b.stepper.StepBatch(seqs)
		if err == nil && len(out) != len(run) {
			err = fmt.Errorf("decoder: StepBatch returned %d rows for %d sequences", len(out), len(run))
		}
		for i, r := range run {
			if err != nil {
				r.err = err
			} else {
				r.out = &out[i]
			}
		}
		b.mu.Lock()
		b.stats.Steps++
		b.stats.StepTokens += len(run)
		b.stats.StepSizes[min(len(run), batchStatsSizes-1)]++
		b.mu.Unlock()
		return
	}
	for _, r := range run {
		r.err = runSolo(r.solo)
		b.mu.Lock()
		b.stats.SoloTokens++
		b.mu.Unlock()
	}
}

// runSolo runs one token's production call, turning a panic into its error: the run holds the resident for every
// generation in it, and must release it.
func runSolo(solo func() error) (err error) {
	defer func() {
		if p := recover(); p != nil {
			err = fmt.Errorf("decoder: resident forward panicked: %v", p)
		}
	}()
	return solo()
}
