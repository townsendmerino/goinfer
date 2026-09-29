package decoder

import (
	"context"
	"fmt"
	"math"
	"slices"
	"strings"
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
	inUse     int32
	n, bound  int
	hist      [][]uint64
	steps     int // StepBatch calls
	stepSeqs  int // sequences served by them
	maxStep   int
	soloFwds  int
	lo, hi    int
	stepErr   error
	delay     time.Duration
	stepDraws int    // rows a StepBatch drew on-device
	events    []byte // P: a PrefillLast call, S: a StepBatch, F: a single-sequence Forward
}

// PrefillLast makes mc3Fake a batched prefiller (Prefiller): the positions from startPos on, in order, as Forward would
// write them — chunk-invariant by construction, as a real prefill must be for chunking to be sound.
func (f *mc3Fake) PrefillLast(ctx context.Context, embs [][]float32, startPos int) ([]float32, error) {
	f.enter()
	defer f.leave()
	f.events = append(f.events, 'P')
	f.wait()
	var lg []float32
	for i, e := range embs {
		lg = f.fwd(f.bound, e, startPos+i)
	}
	return lg, nil
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
	f.events = append(f.events, 'F')
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
func (f *mc3Fake) StepBatch(seqs []ResidentBatchSeq) ([]ResidentBatchOut, error) {
	f.enter()
	defer f.leave()
	if f.stepErr != nil {
		return nil, f.stepErr
	}
	f.steps++
	f.events = append(f.events, 'S')
	f.wait()
	f.stepSeqs += len(seqs)
	f.maxStep = max(f.maxStep, len(seqs))
	out := make([]ResidentBatchOut, len(seqs))
	for i, q := range seqs {
		lg := f.fwd(q.Slot, q.Emb, q.Pos)
		if q.Draw != nil {
			f.stepDraws++
			out[i] = ResidentBatchOut{ID: fakeDraw(lg, q.Draw.Seed, q.Draw.Draw)}
		} else {
			out[i] = ResidentBatchOut{Logits: lg, ID: -1}
		}
	}
	return out, nil
}

// SampleAvailable / ForwardSample make mc3Fake a device sampler (ResidentSample), so temperature-only generations take
// the device-draw path; fakeDraw is its draw, a function of the row and (seed, draw) only, as the real one is.
func (f *mc3Fake) SampleAvailable() bool { return true }
func (f *mc3Fake) ForwardSample(emb []float32, pos int, temperature float64, seed, draw uint64) (int, error) {
	f.enter()
	defer f.leave()
	f.soloFwds++
	f.wait()
	return fakeDraw(f.fwd(f.bound, emb, pos), seed, draw), nil
}

func fakeDraw(logits []float32, seed, draw uint64) int {
	h := (seed*0x9E3779B97F4A7C15 ^ draw*0xBF58476D1CE4E5B9) + 0x94D049BB133111EB
	for i, x := range logits {
		if x != 0 {
			h ^= uint64(i) * 0xD6E8FEB86659FD93
		}
	}
	return int(h % uint64(len(logits)))
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
	// A step time, so a decoding generation yields while its token runs. With an instant fake, the one generation
	// decoding runs every token at once and never blocks: on a starved scheduler (a busy CI runner, or -cpu 1, 10 of 10
	// runs) it finishes its conversation before the others are scheduled, and nothing ever batches.
	rf.delay = 3 * time.Millisecond
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
	rf.delay = 3 * time.Millisecond // so the generations overlap on a starved scheduler (TestMC3_concurrentGenerationsMatchAlone)
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

// TestMC3_sampledConcurrentMatchesAlone: temperature-only generations, each with its own seed, draw their tokens
// on-device. Concurrently they join shared steps, where each row's draw is the one its own ForwardSample would have
// made (same seed and draw counter), so each conversation emits exactly the ids it emits alone — and the draws really
// were made in steps.
func TestMC3_sampledConcurrentMatchesAlone(t *testing.T) {
	const nConv, maxTok = 4, 32
	run := func(m *Model, c int) []int {
		ch, gen := m.Generate(context.Background(), []int{1, 2, 3, 10 + c}, maxTok, SamplingParams{Temperature: 0.8, Seed: int64(100 + c)})
		var ids []int
		for id := range ch {
			ids = append(ids, id)
		}
		if err := gen.Err(); err != nil {
			t.Errorf("conversation %d: %v", c, err)
		}
		if gen.DeviceSampled == 0 {
			t.Errorf("conversation %d: no token was drawn on-device — the test is not exercising the device-draw path", c)
		}
		return ids
	}
	mAlone, _ := loadWithMC3Fake(t, nConv)
	alone := make([][]int, nConv)
	for c := range nConv {
		alone[c] = run(mAlone, c)
	}
	m, rf := loadWithMC3Fake(t, nConv)
	rf.delay = 3 * time.Millisecond // so the generations overlap on a starved scheduler (TestMC3_concurrentGenerationsMatchAlone)
	m.EnableResidentConcurrency(nConv)
	together := make([][]int, nConv)
	var wg sync.WaitGroup
	for c := range nConv {
		wg.Add(1)
		go func(c int) { defer wg.Done(); together[c] = run(m, c) }(c)
	}
	wg.Wait()
	for c := range nConv {
		if !slices.Equal(alone[c], together[c]) {
			t.Errorf("conversation %d: concurrent %v, alone %v", c, together[c], alone[c])
		}
	}
	if rf.stepDraws == 0 {
		t.Error("no row was drawn inside a batched step: sampled tokens still run solo")
	}
	t.Logf("steps %d, rows drawn in steps %d, solo calls %d", rf.steps, rf.stepDraws, rf.soloFwds)
}

// TestMC3_longPrefillChunksWhileOthersDecode: a newcomer with a long prompt arrives while two generations decode. Its
// prefill must come in chunks (prefillChunkTokens) with decode steps between them, so the decoders keep producing,
// and every reply — the newcomer's included — must equal the same conversation served alone, where the newcomer's
// prompt is prefilled in one pass.
func TestMC3_longPrefillChunksWhileOthersDecode(t *testing.T) {
	vocabPrompt := func(m *Model, n, salt int) []int {
		v := m.w.arch.VocabSize
		p := make([]int, n)
		for i := range p {
			p[i] = (i*7 + salt) % v
		}
		return p
	}
	gen := func(m *Model, prompt []int, maxTok int) []int {
		ch, g := m.Generate(context.Background(), prompt, maxTok, SamplingParams{})
		var ids []int
		for id := range ch {
			ids = append(ids, id)
		}
		if err := g.Err(); err != nil {
			t.Errorf("generate: %v", err)
		}
		return ids
	}
	mAlone, _ := loadWithMC3Fake(t, 4)
	d0, d1 := []int{1, 2, 3, 40}, []int{1, 2, 3, 50}
	aloneD0, aloneD1 := gen(mAlone, d0, 60), gen(mAlone, d1, 60)
	aloneNew := gen(mAlone, vocabPrompt(mAlone, 1200, 3), 4)

	m, rf := loadWithMC3Fake(t, 4)
	rf.delay = 3 * time.Millisecond
	m.prefillChunk = 256 // Options.ResidentPrefillChunk
	m.EnableResidentConcurrency(4)
	var got0, got1, gotNew []int
	var wg sync.WaitGroup
	wg.Add(3)
	go func() { defer wg.Done(); got0 = gen(m, d0, 60) }()
	go func() { defer wg.Done(); got1 = gen(m, d1, 60) }()
	go func() {
		defer wg.Done()
		time.Sleep(40 * time.Millisecond) // the two are decoding by now
		gotNew = gen(m, vocabPrompt(m, 1200, 3), 4)
	}()
	wg.Wait()
	for _, c := range []struct {
		name      string
		got, want []int
	}{{"decoder 0", got0, aloneD0}, {"decoder 1", got1, aloneD1}, {"newcomer", gotNew, aloneNew}} {
		if !slices.Equal(c.got, c.want) {
			t.Errorf("%s: concurrent %v, alone %v", c.name, c.got, c.want)
		}
	}
	ev := string(rf.events)
	chunks := strings.Count(ev, "P")
	i, j := strings.Index(ev, "P"), strings.LastIndex(ev, "P")
	between := 0
	if i >= 0 && j > i {
		between = strings.Count(ev[i:j], "S")
	}
	t.Logf("resident events: %d prefill calls, %d steps between the first and last of them", chunks, between)
	if chunks < 4 { // 1200 = 256 x 4 + 176 (a chunk is cut while a chunk plus prefillTailMin remain)
		t.Errorf("the newcomer's 1200-token prompt went in %d prefill calls: it was not chunked", chunks)
	}
	if between < chunks-1 {
		t.Errorf("only %d decode steps ran between the %d prefill chunks: the decoders were stalled for the prefill", between, chunks)
	}
	// The time attribution the served per-pass prefill measurement reads (ResidentBatchStats): every prefill call the
	// resident saw is one counted pass, and the durations nest — prefill inside exclusive, runs timed apart.
	st := m.ResidentBatchStats()
	t.Logf("stats: %d prefill passes %.1f ms, exclusive %.1f ms, runs %.1f ms over %d runs",
		st.PrefillPasses, float64(st.PrefillNs)/1e6, float64(st.ExclusiveNs)/1e6, float64(st.RunNs)/1e6, st.Runs)
	// One seed pass per generation (the two 4-token decoders seed through per-token Forward, under the 8-token batched
	// floor, so the fake logs them as F, not P), plus the newcomer's chunks before its seed.
	if want := 3 + (chunks - 1); st.PrefillPasses != want {
		t.Errorf("PrefillPasses = %d, want %d (3 seed passes + %d earlier chunks)", st.PrefillPasses, want, chunks-1)
	}
	if st.PrefillNs <= 0 || st.ExclusiveNs < st.PrefillNs {
		t.Errorf("prefill %d ns, exclusive %d ns: prefill passes must be timed, inside the exclusive total", st.PrefillNs, st.ExclusiveNs)
	}
	if st.Steps > 0 && st.RunNs < int64(st.Steps)*rf.delay.Nanoseconds() {
		t.Errorf("runs timed %d ns over %d steps of %v each: RunNs misses the steps", st.RunNs, st.Steps, rf.delay)
	}
}

// TestMC3_prefillChunkOffByDefault: with Options.ResidentPrefillChunk unset, a long newcomer is prefilled in one pass
// even while others decode — chunked prefill ships off (its first candidate missed a gate).
func TestMC3_prefillChunkOffByDefault(t *testing.T) {
	m, rf := loadWithMC3Fake(t, 4)
	rf.delay = 3 * time.Millisecond
	m.EnableResidentConcurrency(4)
	v := m.w.arch.VocabSize
	long := make([]int, 1200)
	for i := range long {
		long[i] = (i*7 + 3) % v
	}
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		ch, _ := m.Generate(context.Background(), []int{1, 2, 3, 40}, 60, SamplingParams{})
		for range ch {
		}
	}()
	go func() {
		defer wg.Done()
		time.Sleep(40 * time.Millisecond)
		ch, _ := m.Generate(context.Background(), long, 4, SamplingParams{})
		for range ch {
		}
	}()
	wg.Wait()
	if n := strings.Count(string(rf.events), "P"); n != 1 {
		t.Errorf("the 1200-token prompt went in %d prefill calls with chunking unset, want 1", n)
	}
}
