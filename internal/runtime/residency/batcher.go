package residency

import (
	"errors"
	"fmt"
	"github.com/townsendmerino/goinfer/decoder/resident"
	"runtime"
	"slices"
	"sync"
	"sync/atomic"
	"time"
)

// The batched resident decode: several generations decoding on one resident at once, each on its own
// resident KV slot, their decode tokens joined into shared steps.
//
// A resident is one command queue with one set of scratch buffers, so everything that touches it (a prefill, a
// single-sequence forward, a batched step, the slot bookkeeping) runs one at a time, inside the batcher's exclusive
// section. What the batcher adds is what runs there at a decode token: the batcher coalesces the tokens every decoding
// generation submits into one run. A run of at least lo batch-eligible tokens goes to BatchStepperAvailable.StepBatch
// (every row bit-identical to that sequence's own Forward); anything else, a lone generation above all, runs
// production's own per-sequence call unchanged, so a request served alone takes exactly the path it takes without the batcher.
//
// Generations join at token boundaries: a run starts once every decoding generation has submitted, or once the oldest
// submission has waited BatchStragglerWait (a generation whose consumer is slow to read its stream must not stall the
// rest). A newcomer's prefill runs between runs, whole or in chunks (Options.ResidentPrefillChunk), and takes precedence
// over the next run. On a resident that takes ragged steps a chunk instead rides the next run, one step with the decode
// rows; elsewhere yieldToDecode
// lets one run go between chunks.

// BatchStragglerWait bounds how long a submitted token waits for the other decoding generations' tokens before its run
// starts without them. Host work between two tokens is far below a batched step's cost, so this is a small fraction of
// a step when it fires, and it fires only for a straggler.
const BatchStragglerWait = 4 * time.Millisecond

// tokenCoalescer is the token-joining core shared by the resident's residentBatcher and the CPU's cpuBatcher
// (cpu_batch.go): generations submit one decode token each, and a run serves the tokens pending once every decoding
// generation has submitted or the straggler window has expired — in one batched step (step) when it has at least lo
// tokens, else each through its own production call. What it does not know is what a run holds: the resident batcher
// adds slot holders and an exclusive section; the CPU batcher adds nothing, since a CPU model has no single device.
type TokenCoalescer struct {
	Lo, Hi int // a batched step serves lo..hi tokens
	Wait   time.Duration
	// step serves a run of at least lo tokens in one batched step, setting every token's out or err.
	Step func(run []*BatchReq)
	// ride serves a run that carries a prefill chunk: one step of the chunk and the run's tokens, setting the chunk's err
	// and every token's out or err. nil where chunks do not ride (rideChunk is then never called).
	Ride func(c *ChunkReq, run []*BatchReq)

	mu          sync.Mutex
	cond        *sync.Cond
	busy        bool // a run or an exclusive section is in progress
	exclWaiting int  // exclusive sections waiting — a run does not start ahead of them
	decoders    int  // generations inside their decode loop
	pending     []*BatchReq
	chunks      []*ChunkReq // prefill chunks waiting for a run; a run carries at most one
	freeAt      time.Time   // when the last run or exclusive section ended
	timerAt     time.Time
	stats       BatchStats
}

func (c *TokenCoalescer) Init() { c.cond = sync.NewCond(&c.mu) }

// residentBatcher is a Model's batched-decode coordinator (Model.EnableResidentConcurrency): the shared coalescing core plus the
// resident's slot holders. A resident is one command queue, so its runs and exclusive sections are one at a time.
type Batcher struct {
	TokenCoalescer
	stepper    resident.Forward
	stepSeqs   func([]resident.BatchSeq) ([]resident.BatchOut, error)
	stepRagged func([]resident.RaggedSeq) ([]resident.RaggedOut, error)
	maxHolders int // generations that may hold a slot at once

	holders  int    // generations holding a slot (claimed, not yet released); guarded by mu
	adaptive int    // adaptive speculative generations active on the batcher; guarded by mu
	slotBusy []bool // resident KV slots a holder is using; guarded by mu
	// holding: the one holder, decoding alone, keeps the resident (busy) across its tokens (holdSolo), so a chained forward
	// stays queued on the device between them; guarded by mu.
	holding bool
}

// BatchStats counts what the resident batcher did (Model.BatchStats): runs, the tokens batched steps served
// and how many steps there were, the tokens that ran as production's per-sequence call, and how many runs started
// because the straggler window expired rather than because every decoding generation had submitted.
//
// The durations are wall time with the resident held: RunNs inside decode runs, ExclusiveNs inside every exclusive
// section, and PrefillNs inside the ones that are prefill passes (PrefillPasses of them). ExclusiveNs - PrefillNs is
// slot bookkeeping. They attribute a served cell's time without timing anything else.
type BatchStats struct {
	Runs, Steps, StepTokens, SoloTokens, StragglerRuns int
	// HeldTokens are the tokens a lone holder ran with the resident held across them (holdSolo: the chained decode), and
	// Holds how many times a hold was taken. They are not in Runs or SoloTokens.
	HeldTokens, Holds             int
	StepSizes                     [batchStatsSizes]int // StepSizes[b] = steps of b sequences
	PrefillPasses                 int
	RunNs, ExclusiveNs, PrefillNs int64
	// RidePasses are the prefill chunks that rode a run (rideChunk), RideRows the decode rows those runs carried beside them,
	// and RideNs their wall time, which RunNs does not include.
	RidePasses, RideRows int
	RideNs               int64
}

const batchStatsSizes = 17

// batchReq is one generation's decode token: a resident one (seq) or a CPU one (id on cache).
type BatchReq struct {
	Seq   resident.BatchSeq
	ID    int                // CPU (cpuBatcher): the token to decode
	Cache any                // CPU: the generation's own session cache
	Solo  func() error       // production's own call for this token (resident: binds the generation's slot first)
	Out   *resident.BatchOut // set when a batched step served it
	Err   error
	done  bool
	at    time.Time
}

// chunkReq is one prefill chunk waiting to ride a run: its ragged sequence (a Prefill one) and the step's verdict on it.
type ChunkReq struct {
	Seq  resident.RaggedSeq
	Err  error
	done bool
	at   time.Time
}

// claim takes one of the batcher's holder places. It fails while an exclusive resident claim (tryClaimResident: the
// paths the batcher does not batch — speculation, vision, adapters) is held, and when every place is taken.
func (b *Batcher) Claim(resBusy *int32) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	if atomic.LoadInt32(resBusy) != 0 || b.holders >= b.maxHolders {
		return false
	}
	b.holders++
	return true
}

// release gives a holder place back.
func (b *Batcher) Release() {
	b.mu.Lock()
	b.holders--
	b.cond.Broadcast()
	b.mu.Unlock()
}

// holdSolo runs this holder's token fn with the resident kept between calls, while it is the only generation that wants
// the resident: one holder, decoding, nothing pending and no exclusive section waiting. That is how a chained decode
// runs under the batcher: the chain keeps the next token's forward queued on the device between tokens, and nothing else may
// touch the resident until it stops. ok is false, with fn not run, when the holder is not alone; if a hold was kept
// from an earlier call, release (the chain's Stop) runs first with the resident still held, then the hold ends. The
// caller then submits the token through forward as usual. A newcomer (another holder's claim, its prefill's exclusive
// section) therefore waits at most until this holder's next token.
func (b *Batcher) HoldSolo(fn func() error, release func()) (ok bool, err error) {
	b.mu.Lock()
	alone := b.holders == 1 && b.decoders == 1 && b.exclWaiting == 0 && len(b.pending) == 0
	switch {
	case b.holding && !alone:
		b.mu.Unlock()
		b.ReleaseHold(release)
		return false, nil
	case !b.holding && (!alone || b.busy):
		b.mu.Unlock()
		return false, nil
	case !b.holding:
		b.busy, b.holding = true, true
		b.stats.Holds++
	}
	b.mu.Unlock()
	t0 := time.Now()
	err = runSolo(fn)
	b.mu.Lock()
	b.stats.HeldTokens++
	b.stats.RunNs += time.Since(t0).Nanoseconds()
	b.mu.Unlock()
	return true, err
}

// releaseHold ends a hold holdSolo kept: release (the chain's Stop) runs first, with the resident still held, then the
// resident is free again. A no-op when no hold is kept. Every exit from a decode loop that may hold calls it before
// anything else on the batcher (the commit's exclusive section waits for the resident, so a kept hold would deadlock it).
func (b *Batcher) ReleaseHold(release func()) {
	b.mu.Lock()
	held := b.holding
	b.mu.Unlock()
	if !held {
		return
	}
	defer func() {
		b.mu.Lock()
		b.holding, b.busy = false, false
		b.freeAt = time.Now()
		b.cond.Broadcast()
		b.mu.Unlock()
	}()
	release()
}

// MC3HoldSendGrace is how long a holder keeping the resident (holdSolo) waits for its consumer to take a token before it
// lets go of the hold: a consumer that is not reading must not keep a newcomer waiting on the resident. Streaming
// clients read at once, so the grace is short and the hold is kept across them.
const MC3HoldSendGrace = 2 * time.Millisecond

// isHolding reports whether a hold is kept (the decode loop asks before a token send that might block).
func (b *Batcher) IsHolding() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.holding
}

// claimExclusive is tryClaimResident under a batcher: the whole resident, only while no holder or adaptive
// generation is active and no batch/exclusive run is in progress.
func (b *Batcher) ClaimExclusive(resBusy *int32) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.holders == 0 && b.adaptive == 0 && !b.busy && atomic.CompareAndSwapInt32(resBusy, 0, 1)
}

// claimAdaptive registers an adaptive speculative generation on the batcher. It fails when a whole-generation
// exclusive caller (tryClaimResident: adaptive == 0 with resBusy set) currently owns the resident.
func (b *Batcher) ClaimAdaptive(resBusy *int32) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.adaptive == 0 && atomic.LoadInt32(resBusy) != 0 {
		return false
	}
	b.adaptive++
	return true
}

// releaseAdaptive unregisters an adaptive speculative generation from the batcher.
func (b *Batcher) ReleaseAdaptive() {
	b.mu.Lock()
	b.adaptive--
	b.cond.Broadcast()
	b.mu.Unlock()
}

// claimAdaptiveRound is an adaptive speculative generation's per-round probe for whether it is the sole
// resident user: it succeeds only with no batched holders and no other adaptive generation active, setting
// resBusy so no batched holder joins mid-round while preserving exclusive()'s own serialization.
func (b *Batcher) ClaimAdaptiveRound(resBusy *int32) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.holders == 0 && b.adaptive == 1 && atomic.CompareAndSwapInt32(resBusy, 0, 1)
}

// exclusive runs fn with the resident to itself, between runs: prefill, slot bookkeeping. It goes ahead of the next run.
func (b *TokenCoalescer) Exclusive(fn func()) { b.exclusiveKind(false, fn) }

// prefillExclusive is exclusive for a prefill pass, counted in PrefillPasses and PrefillNs as well.
func (b *TokenCoalescer) PrefillExclusive(fn func()) { b.exclusiveKind(true, fn) }

func (b *TokenCoalescer) exclusiveKind(prefill bool, fn func()) {
	b.mu.Lock()
	b.exclWaiting++
	for b.busy {
		b.cond.Wait()
	}
	b.exclWaiting--
	b.busy = true
	b.mu.Unlock()
	t0 := time.Now()
	defer func() {
		d := time.Since(t0).Nanoseconds()
		b.mu.Lock()
		b.stats.ExclusiveNs += d
		if prefill {
			b.stats.PrefillPasses++
			b.stats.PrefillNs += d
		}
		b.busy = false
		b.freeAt = time.Now()
		b.cond.Broadcast()
		b.mu.Unlock()
	}()
	fn()
}

// PrefillTailMin is chunked prefill's shortest final pass (Options.ResidentPrefillChunk): residentPrefillSeed takes a suffix of at least 8 tokens through the batched
// prefill, and the chunks must all run that same path. Chunking is sound only where the resident's prefill is
// chunk-invariant, the same bits whole or in chunks (TestMC5_prefillChunkInvariance on Metal), so a reply never depends
// on whether others were decoding.
const PrefillTailMin = 8

// decoding reports how many holders are inside their decode loop.
func (b *TokenCoalescer) Decoding() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.decoders
}

// yieldToDecode lets the decoding generations run one step before a prefilling generation takes the resident for its
// next chunk: it returns once a run has started (or when nobody is decoding, or after a bounded wait — a decoder whose
// consumer has stalled must not hold the prefill up indefinitely).
func (b *TokenCoalescer) YieldToDecode() {
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
func (b *TokenCoalescer) EnterDecode() {
	b.mu.Lock()
	b.decoders++
	b.mu.Unlock()
}

func (b *TokenCoalescer) ExitDecode() {
	b.mu.Lock()
	b.decoders--
	b.cond.Broadcast() // a run may have been waiting for this holder's token
	b.mu.Unlock()
}

// markSlot records a holder's slot as in use (true) or free (false).
func (b *Batcher) MarkSlot(slot int, busy bool) {
	b.mu.Lock()
	if slot >= 0 && slot < len(b.slotBusy) {
		b.slotBusy[slot] = busy
	}
	b.mu.Unlock()
	if slot >= 0 {
		ResolveSubsystems(b.stepper).KV.SetKVSlotBusy(slot, busy)
	}
}

// busySlots is a snapshot of the slots holders are using, for residentAcquire (called inside exclusive).
func (b *Batcher) BusySlots() []bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return slices.Clone(b.slotBusy)
}

// forward submits one decode token and returns once a run has served it: q.Out is set when a batched step did, and
// otherwise q.Solo ran (production's own call, with its own results).
func (b *TokenCoalescer) Forward(q *BatchReq) error {
	b.mu.Lock()
	q.at = time.Now()
	b.pending = append(b.pending, q)
	b.cond.Broadcast()
	b.serveLocked(func() bool { return q.done })
	b.mu.Unlock()
	return q.Err
}

// rideChunk submits one prefill chunk and returns once a run has carried it in one step with the decode tokens pending
// then. It joins a run as a token does: once every decoding generation has submitted, or the straggler window has expired,
// or nobody is decoding. A non-nil error means the step did not take the chunk; the run's tokens were served without it
// and the caller prefills from the chunk's start another way.
func (b *TokenCoalescer) RideChunk(c *ChunkReq) error {
	b.mu.Lock()
	c.at = time.Now()
	b.chunks = append(b.chunks, c)
	b.cond.Broadcast()
	b.serveLocked(func() bool { return c.done })
	b.mu.Unlock()
	return c.Err
}

// serveLocked waits until done reports true, starting a run whenever one can start: whichever waiter finds the resident
// free runs the pending tokens, and the oldest pending chunk with them, for everyone. b.mu is held.
func (b *TokenCoalescer) serveLocked(done func() bool) {
	for !done() {
		if !b.busy && b.exclWaiting == 0 && (len(b.pending) > 0 || len(b.chunks) > 0) {
			// The straggler window opens when a run could start: at the oldest submission or when the resident came free,
			// whichever is later. Timed from submission alone, a token submitted during a run has already "waited" the whole run
			// when it ends, starts one at once without the others, and the generations phase-lock into split runs every token.
			var start time.Time
			if len(b.pending) > 0 {
				start = b.pending[0].at
			}
			if len(b.chunks) > 0 && (start.IsZero() || b.chunks[0].at.Before(start)) {
				start = b.chunks[0].at
			}
			if b.freeAt.After(start) {
				start = b.freeAt
			}
			if all := len(b.pending) >= b.decoders; all || time.Since(start) >= b.Wait {
				if !all {
					b.stats.StragglerRuns++
				}
				b.stats.Runs++
				n := min(len(b.pending), b.Hi)
				run := b.pending[:n:n]
				b.pending = slices.Clone(b.pending[n:])
				var c *ChunkReq
				if len(b.chunks) > 0 {
					c = b.chunks[0]
					b.chunks = slices.Clone(b.chunks[1:])
				}
				b.busy = true
				b.mu.Unlock()
				t0 := time.Now()
				bug := b.ExecRun(c, run)
				b.mu.Lock()
				if d := time.Since(t0).Nanoseconds(); c != nil {
					b.stats.RideNs += d
					if c.Err == nil {
						b.stats.RidePasses++
						b.stats.RideRows += len(run)
					}
				} else {
					b.stats.RunNs += d
				}
				b.busy = false
				b.freeAt = time.Now()
				for _, r := range run {
					r.done = true
				}
				if c != nil {
					c.done = true
				}
				b.cond.Broadcast()
				if bug != nil {
					// A programming bug crashes loudly, as runGen's recover lets it (mem.RepanicBug), but only after busy is clear,
					// the run's waiters are failed and b.mu is released: a caller up the stack that recovers it (net/http does)
					// must not inherit a held lock or a resident marked busy forever.
					b.mu.Unlock()
					panic(bug)
				}
				continue
			}
			b.armLocked(start.Add(b.Wait))
		}
		b.cond.Wait()
	}
}

// armLocked wakes the waiters at deadline (so a straggler's absence is noticed); b.mu held.
func (b *TokenCoalescer) armLocked(deadline time.Time) {
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

// execRun runs a batched step or prefill ride, converting a panic into an error for every waiting caller in the run so
// serveLocked always releases b.busy and wakes waiters instead of deadlocking. A runtime.Error (a programming bug) is also
// returned, for serveLocked to re-panic once the batcher is consistent again.
func (b *TokenCoalescer) ExecRun(c *ChunkReq, run []*BatchReq) (bug any) {
	defer func() {
		if p := recover(); p != nil {
			if _, ok := p.(runtime.Error); ok {
				bug = p
			}
			err := fmt.Errorf("decoder: batched step panicked: %v", p)
			if c != nil && c.Err == nil {
				c.Err = err
			}
			for _, r := range run {
				if r.Out == nil && r.Err == nil {
					r.Err = err
				}
			}
		}
	}()
	if c != nil {
		b.Ride(c, run)
	} else {
		b.runTokens(run)
	}
	return nil
}

// runTokens serves one run: in one batched step (step) when it has at least lo tokens, else every token through its
// own production call. The resident batcher holds the resident across it.
func (b *TokenCoalescer) runTokens(run []*BatchReq) {
	if len(run) >= b.Lo {
		b.Step(run)
		b.mu.Lock()
		b.stats.Steps++
		b.stats.StepTokens += len(run)
		b.stats.StepSizes[min(len(run), batchStatsSizes-1)]++
		b.mu.Unlock()
		return
	}
	for _, r := range run {
		r.Err = runSolo(r.Solo)
		b.mu.Lock()
		b.stats.SoloTokens++
		b.mu.Unlock()
	}
}

// stepResident is the resident batcher's step: one StepBatch over the run's sequences.
func (b *Batcher) stepResident(run []*BatchReq) {
	seqs := make([]resident.BatchSeq, len(run))
	for i, r := range run {
		seqs[i] = r.Seq
	}
	out, err := b.stepSeqs(seqs)
	if errors.Is(err, ErrKVPoolFull) {
		// A row past its share found no block. The step maps every row's blocks before it writes anything, so it wrote nothing:
		// run each token on its own, and only the row that cannot get a block ends.
		for _, r := range run {
			r.Err = runSolo(r.Solo)
		}
		return
	}
	if err == nil && len(out) != len(run) {
		err = fmt.Errorf("decoder: StepBatch returned %d rows for %d sequences", len(out), len(run))
	}
	for i, r := range run {
		if err != nil {
			r.Err = err
		} else {
			r.Out = &out[i]
		}
	}
}

// rideResident is the resident batcher's ride: one StepRagged of the chunk (first) and one decode row for each of the
// run's tokens. A decode row equals its StepBatch row bit for bit, and the chunk equals the same pass of its prompt's
// prefill (RaggedStepperAvailable's contract). When the step fails, the chunk takes the error back to mc3Prefill and the
// run is served as a run without it: a step that fails has written nothing, or the bits a re-run writes at the same
// positions.
func (b *Batcher) rideResident(c *ChunkReq, run []*BatchReq) {
	seqs := make([]resident.RaggedSeq, 0, len(run)+1)
	seqs = append(seqs, c.Seq)
	for _, r := range run {
		seqs = append(seqs, resident.RaggedSeq{Slot: r.Seq.Slot, Pos: r.Seq.Pos, Embs: [][]float32{r.Seq.Emb}, Draw: r.Seq.Draw})
	}
	outs, err := b.stepRagged(seqs)
	if err == nil && len(outs) != len(seqs) {
		err = fmt.Errorf("decoder: StepRagged returned %d sequences for %d", len(outs), len(seqs))
	}
	for i := 0; err == nil && i < len(run); i++ {
		if o := outs[i+1]; run[i].Seq.Draw == nil && len(o.Logits) != 1 {
			err = fmt.Errorf("decoder: StepRagged returned %d logits rows for a decode row", len(o.Logits))
		}
	}
	if err != nil {
		c.Err = err
		b.runTokens(run)
		return
	}
	for i, r := range run {
		o := outs[i+1]
		if r.Seq.Draw != nil {
			r.Out = &resident.BatchOut{ID: o.ID}
		} else {
			r.Out = &resident.BatchOut{Logits: o.Logits[0], ID: -1}
		}
	}
}

// runSolo runs one token's production call, turning a panic into its error: a run serves every generation in it
// (and, on the resident, holds the resident for them), so it must not unwind past them.
func runSolo(solo func() error) (err error) {
	defer func() {
		if p := recover(); p != nil {
			err = fmt.Errorf("decoder: forward panicked: %v", p)
		}
	}()
	return solo()
}

// NewBatcher constructs a resident Batcher over stepper with n concurrent holders.
func NewBatcher(stepper resident.Forward, n, slots, lo, hi int,
	stepSeqs func([]resident.BatchSeq) ([]resident.BatchOut, error),
	stepRagged func([]resident.RaggedSeq) ([]resident.RaggedOut, error)) *Batcher {
	b := &Batcher{
		stepper:    stepper,
		stepSeqs:   stepSeqs,
		stepRagged: stepRagged,
		maxHolders: n,
		slotBusy:   make([]bool, slots),
	}
	b.Lo, b.Hi, b.Wait, b.Step = max(lo, 2), hi, BatchStragglerWait, b.stepResident
	if stepRagged != nil {
		b.Ride = b.rideResident
	}
	b.Init()
	return b
}

// Stats returns a snapshot of the batcher's counters.
func (c *TokenCoalescer) Stats() BatchStats {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.stats
}

// MaxHolders returns the configured concurrency cap.
func (b *Batcher) MaxHolders() int { return b.maxHolders }

// ErrKVPoolFull is returned when a resident KV pool has no free block for a row past its share.
var ErrKVPoolFull = errors.New("decoder: resident KV pool is full")

// NewTestBatcher constructs a minimal Batcher for concurrency/unit tests.
func NewTestBatcher(lo, hi int, wait time.Duration) *Batcher {
	b := &Batcher{maxHolders: hi, slotBusy: make([]bool, hi)}
	b.Lo, b.Hi, b.Wait = lo, hi, wait
	b.Init()
	return b
}
