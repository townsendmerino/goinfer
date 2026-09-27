package decoder

import (
	"context"
	"fmt"
	"math"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// mc3Fake is a fake MC3 resident: several KV slots, each holding a positional history, and a batched step. A forward
// at pos on a slot writes that position (dropping everything after it, as attention reads only pos+1 keys) and returns
// logits whose argmax hashes the slot's whole history — so a token routed to the wrong slot, a stale position, or a
// batched row that differs from Forward's changes every later token. It panics on concurrent use: every call into a
// real resident must come from inside the batcher's exclusive section.
type mc3Fake struct {
	*fakeResident
	inUse    int32
	n, bound int
	hist     [][]uint64
	steps    int // StepBatch calls
	stepSeqs int // sequences served by them
	maxStep  int
	soloFwds int
	lo, hi   int
	stepErr  error
	delay    time.Duration
}

// delay, when set, is how long a StepBatch or Forward takes (a GPU's step time) — long against the batcher's straggler
// window, which is the regime the window has to get right.
func (f *mc3Fake) wait() {
	if f.delay > 0 {
		time.Sleep(f.delay)
	}
}

func newMC3Fake(vocab, slots int) *mc3Fake {
	return &mc3Fake{fakeResident: &fakeResident{vocab: vocab}, n: slots, hist: make([][]uint64, slots), lo: 2, hi: 8}
}

func (f *mc3Fake) enter() {
	if !atomic.CompareAndSwapInt32(&f.inUse, 0, 1) {
		panic("mc3Fake: the resident was entered concurrently — a call escaped the batcher's exclusive section")
	}
}
func (f *mc3Fake) leave() { atomic.StoreInt32(&f.inUse, 0) }

func (f *mc3Fake) fwd(slot int, emb []float32, pos int) []float32 {
	h := uint64(14695981039346656037)
	for _, x := range emb {
		h = (h ^ uint64(math.Float32bits(x))) * 1099511628211
	}
	for len(f.hist[slot]) <= pos {
		f.hist[slot] = append(f.hist[slot], 0)
	}
	f.hist[slot] = f.hist[slot][:pos+1]
	f.hist[slot][pos] = h
	acc := uint64(1469598103934665603)
	for _, x := range f.hist[slot] {
		acc = (acc ^ x) * 1099511628211
	}
	out := make([]float32, f.vocab)
	out[acc%uint64(f.vocab)] = 1
	return out
}

func (f *mc3Fake) Forward(emb []float32, pos int) ([]float32, error) {
	f.enter()
	defer f.leave()
	f.soloFwds++
	f.wait()
	return f.fwd(f.bound, emb, pos), nil
}
func (f *mc3Fake) KVSlots() int { return f.n }
func (f *mc3Fake) UseKVSlot(i int) error {
	f.enter()
	defer f.leave()
	f.bound = i
	return nil
}
func (f *mc3Fake) BatchStepRange() (int, int) { return f.lo, f.hi }
func (f *mc3Fake) StepBatch(seqs []ResidentBatchSeq) ([][]float32, error) {
	f.enter()
	defer f.leave()
	if f.stepErr != nil {
		return nil, f.stepErr
	}
	f.steps++
	f.wait()
	f.stepSeqs += len(seqs)
	f.maxStep = max(f.maxStep, len(seqs))
	out := make([][]float32, len(seqs))
	for i, q := range seqs {
		out[i] = f.fwd(q.Slot, q.Emb, q.Pos)
	}
	return out, nil
}

// mc3FakeBackend builds an mc3Fake as the model's resident.
type mc3FakeBackend struct {
	Backend
	rf    *mc3Fake
	slots int
}

func (b *mc3FakeBackend) BuildResident(m *Model) (ResidentForward, bool, error) {
	_, _, _, _, _, _, vocab := m.Dims()
	b.rf = newMC3Fake(vocab, b.slots)
	return b.rf, true, nil
}
func (b *mc3FakeBackend) Close() error { return nil }

var mc3FakeBackends atomic.Int32

func loadWithMC3Fake(t *testing.T, slots int) (*Model, *mc3Fake) {
	t.Helper()
	be := &mc3FakeBackend{slots: slots}
	name := fmt.Sprintf("fake-mc3-%d", mc3FakeBackends.Add(1))
	RegisterBackend(name, func() (Backend, error) {
		cpu, err := NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be.Backend = cpu
		return be, nil
	})
	m, err := Load(tinyFixture(t), Options{Backend: name})
	if err != nil {
		t.Fatalf("Load with the MC3 fake: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	if !m.ResidentActive() {
		t.Skip("fixture is not resident-eligible")
	}
	return m, be.rf
}

type mc3Turn struct {
	ids    []int
	reused int
}

// mc3Conversation plays conversation c for `turns` turns and returns every turn's ids and PrefillReused.
func mc3Conversation(t *testing.T, m *Model, c, turns, maxTok int) []mc3Turn {
	vocab := m.w.arch.VocabSize
	prompt := []int{1, 2, 3, (c*17 + 5) % vocab, (c*29 + 9) % vocab, (c*31 + 3) % vocab}
	var out []mc3Turn
	for tn := range turns {
		ch, gen := m.Generate(context.Background(), prompt, maxTok, SamplingParams{})
		var ids []int
		for id := range ch {
			ids = append(ids, id)
		}
		if err := gen.Err(); err != nil {
			t.Errorf("conversation %d turn %d: %v", c, tn, err)
			return out
		}
		out = append(out, mc3Turn{ids, gen.PrefillReused})
		prompt = append(append(slices.Clone(prompt), ids...), (c*7+tn)%vocab, (c*11+tn)%vocab)
	}
	return out
}

// TestMC3_concurrentGenerationsMatchAlone is MC3's decoder-side identity gate on a fake resident: 4 conversations of 3
// turns, generating at once through the batcher, each emit exactly the ids and reuse exactly the prefix they do when
// run one after another on the same slots — and their tokens really were batched. Run it under -race.
func TestMC3_concurrentGenerationsMatchAlone(t *testing.T) {
	const nConv, turns, maxTok = 4, 3, 24
	mAlone, _ := loadWithMC3Fake(t, nConv)
	alone := make([][]mc3Turn, nConv)
	for c := range nConv {
		alone[c] = mc3Conversation(t, mAlone, c, turns, maxTok)
	}
	m, rf := loadWithMC3Fake(t, nConv)
	if got := m.EnableResidentConcurrency(nConv); got != nConv {
		t.Fatalf("EnableResidentConcurrency(%d) = %d", nConv, got)
	}
	concurrent := make([][]mc3Turn, nConv)
	var wg sync.WaitGroup
	for c := range nConv {
		wg.Add(1)
		go func(c int) {
			defer wg.Done()
			concurrent[c] = mc3Conversation(t, m, c, turns, maxTok)
		}(c)
	}
	wg.Wait()
	for c := range nConv {
		for tn := range alone[c] {
			if tn >= len(concurrent[c]) {
				t.Fatalf("conversation %d: %d turns concurrently, %d alone", c, len(concurrent[c]), len(alone[c]))
			}
			a, b := alone[c][tn], concurrent[c][tn]
			if !slices.Equal(a.ids, b.ids) {
				t.Errorf("conversation %d turn %d: concurrent %v, alone %v", c, tn, b.ids, a.ids)
			}
			if a.reused != b.reused {
				t.Errorf("conversation %d turn %d: concurrent reused %d, alone %d", c, tn, b.reused, a.reused)
			}
			if tn > 0 && b.reused == 0 {
				t.Errorf("conversation %d turn %d reused nothing — its slot's history was not kept", c, tn)
			}
		}
	}
	if rf.steps == 0 || rf.maxStep < 2 {
		t.Errorf("StepBatch ran %d times (largest %d sequences): the concurrent tokens were never batched", rf.steps, rf.maxStep)
	}
	t.Logf("batched steps %d serving %d tokens (largest %d); solo forwards %d", rf.steps, rf.stepSeqs, rf.maxStep, rf.soloFwds)
}

// TestMC3_aloneRunsProductionPath pins the solo guard's mechanism: with concurrency enabled, a generation running by
// itself never takes the batched step — every token is production's own per-sequence call.
func TestMC3_aloneRunsProductionPath(t *testing.T) {
	m, rf := loadWithMC3Fake(t, 4)
	m.EnableResidentConcurrency(4)
	got := mc3Conversation(t, m, 0, 2, 16)
	if len(got) != 2 || len(got[0].ids) == 0 {
		t.Fatalf("conversation produced %v", got)
	}
	if rf.steps != 0 {
		t.Errorf("a lone generation took the batched step %d times", rf.steps)
	}
}

// TestMC3_exclusiveClaimWaitsForNoHolder: while an MC3 generation holds a place, the exclusive resident claim (the
// paths MC3 does not batch) fails instead of sharing the resident, and succeeds once no place is held.
func TestMC3_exclusiveClaimWaitsForNoHolder(t *testing.T) {
	m, _ := loadWithMC3Fake(t, 2)
	m.EnableResidentConcurrency(2)
	if !m.batcher.claim(&m.resBusy) {
		t.Fatal("first MC3 claim failed")
	}
	if m.tryClaimResident() {
		t.Fatal("the exclusive claim succeeded while an MC3 generation held a place")
	}
	m.batcher.release()
	if !m.tryClaimResident() {
		t.Fatal("the exclusive claim failed with no MC3 place held")
	}
	if m.batcher.claim(&m.resBusy) {
		t.Fatal("an MC3 claim succeeded while the exclusive claim was held")
	}
	atomic.StoreInt32(&m.resBusy, 0)
}

// TestMC3_stepErrorFailsTheGeneration: an error from the batched step reaches the generations it served — it is not
// swallowed into wrong tokens.
func TestMC3_stepErrorFailsTheGeneration(t *testing.T) {
	m, rf := loadWithMC3Fake(t, 4)
	m.EnableResidentConcurrency(4)
	rf.stepErr = fmt.Errorf("forced step failure")
	var wg sync.WaitGroup
	errs := make([]error, 4)
	for c := range 4 {
		wg.Add(1)
		go func(c int) {
			defer wg.Done()
			ch, gen := m.Generate(context.Background(), []int{1, 2, 3, 4 + c}, 64, SamplingParams{})
			for range ch {
			}
			errs[c] = gen.Err()
		}(c)
	}
	wg.Wait()
	failed := 0
	for _, e := range errs {
		if e != nil {
			failed++
		}
	}
	if failed == 0 {
		t.Error("four concurrent 64-token generations all succeeded with StepBatch failing — no step ran, or its error was dropped")
	}
}

// TestMC3_concurrentGenerationsFillSteps pins the straggler window's timing. With a step that takes far longer than the
// window (a GPU's), 4 generations decoding together must run as full 4-wide steps, token after token. The window is
// timed from when the resident came free: timed from each token's submission, a token submitted during a run has
// "waited" the whole run when it ends, runs at once without the others, and the generations phase-lock into split
// runs — measured on Metal as 3 batched + 1 solo on every token, 255 of 256 runs started by the timeout.
func TestMC3_concurrentGenerationsFillSteps(t *testing.T) {
	m, rf := loadWithMC3Fake(t, 4)
	rf.delay = 12 * time.Millisecond // 3x the straggler window
	m.EnableResidentConcurrency(4)
	var wg sync.WaitGroup
	for c := range 4 {
		wg.Add(1)
		go func(c int) {
			defer wg.Done()
			if c == 3 {
				// A late joiner: its prefill runs while the other three are decoding, and its first token arrives
				// just as the resident comes free — the case that phase-locked on Metal.
				time.Sleep(300 * time.Millisecond) // past the others' three 48 ms prefills, well into their decode
			}
			ch, gen := m.Generate(context.Background(), []int{1, 2, 3, 4 + c}, 60, SamplingParams{})
			for range ch {
			}
			if err := gen.Err(); err != nil {
				t.Errorf("conversation %d: %v", c, err)
			}
		}(c)
	}
	wg.Wait()
	st := m.ResidentBatchStats()
	t.Logf("runs %d (straggler-started %d), steps %d by size %v, solo tokens %d", st.Runs, st.StragglerRuns, st.Steps, st.StepSizes[:6], st.SoloTokens)
	// The 4th joins ~13 tokens late and so also finishes ~13 tokens late: those ends run 3-wide and solo. Everything in
	// between must be 4-wide.
	if st.Steps == 0 || st.StepSizes[4]*10 < st.Steps*6 {
		t.Errorf("only %d of %d steps ran all 4 generations (sizes %v): the generations did not join shared steps", st.StepSizes[4], st.Steps, st.StepSizes[:6])
	}
	if st.StragglerRuns*10 > st.Runs {
		t.Errorf("%d of %d runs started on the straggler timeout: the window is firing on generations that were not late", st.StragglerRuns, st.Runs)
	}
}
