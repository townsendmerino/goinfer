package decoder

import (
	"context"
	"fmt"
	"math"
	"slices"
	"sync/atomic"
	"testing"
)

// KV8a through Generate (docs/measurements/kv8a-2026-10/README.md, gates 2 and 4, decoder half): a recurrent model's turn 2 that
// re-renders turn 1's reply restores the state saved at the end of turn 1's prompt and continues from there, and its reply equals a
// cold run's; the save is taken after the prefill and before the first decode step, chained or not.

// recurrentFake is a one-slot resident whose state is a running hash of every row it forwards, reset only at position 0, as a
// recurrent model's is. The next token is the state's argmax, so a state that is not the one a cold prefill reaches changes the reply.
type recurrentFake struct {
	*fakeResident
	st, saved  uint64
	keeps      bool // keeps a checkpoint (RecurrentCheckpointBytes > 0)
	noRestore  bool // PLANT: RestoreRecurrentCheckpoint reports success and copies nothing
	chain      bool // offers the device greedy chain
	chainOpen  bool
	chainPos   int
	chainLast  int
	events     []string
	restores   int
	afterPre   uint64 // the state when the last PrefillLast returned
	savedAfter bool   // the last save copied the state PrefillLast left
}

func (f *recurrentFake) step(emb []float32, pos int) []float32 {
	if pos == 0 {
		f.st = 1469598103934665603
	}
	h := f.st
	for _, x := range emb {
		h = (h ^ uint64(math.Float32bits(x))) * 1099511628211
	}
	f.st = (h ^ uint64(pos)) * 1099511628211
	out := make([]float32, f.vocab)
	out[f.st%uint64(f.vocab)] = 1
	return out
}

func (f *recurrentFake) Forward(emb []float32, pos int) ([]float32, error) {
	f.events = append(f.events, fmt.Sprintf("forward %d", pos))
	return f.step(emb, pos), nil
}

func (f *recurrentFake) ForwardN(embs [][]float32, startPos int) ([][]float32, error) {
	rows := make([][]float32, len(embs))
	for i, e := range embs {
		rows[i] = f.step(e, startPos+i)
	}
	return rows, nil
}

func (f *recurrentFake) PrefillLast(ctx context.Context, embs [][]float32, startPos int) ([]float32, error) {
	var last []float32
	for i, e := range embs {
		last = f.step(e, startPos+i)
	}
	f.events = append(f.events, fmt.Sprintf("prefill %d+%d", startPos, len(embs)))
	f.afterPre = f.st
	return last, nil
}

func (f *recurrentFake) RecurrentCheckpointBytes() int64 {
	if f.keeps {
		return 64
	}
	return 0
}
func (f *recurrentFake) SaveRecurrentCheckpoint() error {
	f.saved, f.savedAfter = f.st, f.st == f.afterPre
	f.events = append(f.events, "save")
	return nil
}
func (f *recurrentFake) RestoreRecurrentCheckpoint() error {
	f.restores++
	f.events = append(f.events, "restore")
	if !f.noRestore {
		f.st = f.saved
	}
	return nil
}

// The device chain: the next token from the state, fed as a one-element embedding of the previous id.
func (f *recurrentFake) GreedyChainAvailable() bool { return f.chain }
func (f *recurrentFake) GreedyChainStart(id, pos int) error {
	f.events = append(f.events, fmt.Sprintf("chain start %d", pos))
	f.chainOpen, f.chainPos = true, pos
	f.chainLast = id
	return nil
}
func (f *recurrentFake) GreedyChainNext() (int, error) {
	row := f.step([]float32{float32(f.chainLast)}, f.chainPos)
	f.chainPos++
	f.chainLast = slices.Index(row, 1)
	return f.chainLast, nil
}
func (f *recurrentFake) GreedyChainStop() { f.chainOpen = false }

type recurrentFakeBackend struct {
	Backend
	rf             *recurrentFake
	keeps, chained bool
}

func (b *recurrentFakeBackend) BuildResident(m *Model) (ResidentForward, bool, error) {
	_, _, _, _, _, _, vocab := m.Dims()
	b.rf = &recurrentFake{fakeResident: &fakeResident{vocab: vocab}, keeps: b.keeps, chain: b.chained}
	return b.rf, true, nil
}
func (b *recurrentFakeBackend) Close() error { return nil }

var recurrentFakeBackends atomic.Int32

func loadRecurrentFake(t *testing.T, keeps, chain bool) (*Model, *recurrentFake) {
	t.Helper()
	be := &recurrentFakeBackend{keeps: keeps, chained: chain}
	name := fmt.Sprintf("fake-kv8a-%d", recurrentFakeBackends.Add(1))
	RegisterBackend(name, func() (Backend, error) {
		cpu, err := NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be.Backend = cpu
		return be, nil
	})
	m, err := Load("../testdata/qwen35-tiny", Options{Backend: name})
	if err != nil {
		t.Fatalf("load qwen35-tiny on the fake: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	if !m.ResidentActive() || !m.hasRecurrentState() {
		t.Fatalf("resident %v, recurrent %v: the fixture must load as a resident recurrent model", m.ResidentActive(), m.hasRecurrentState())
	}
	return m, be.rf
}

func kv8aRun(t *testing.T, m *Model, prompt []int) ([]int, *Generation) {
	t.Helper()
	ch, gen := m.Generate(context.Background(), prompt, 8, SamplingParams{})
	var ids []int
	for id := range ch {
		ids = append(ids, id)
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("generate: %v", err)
	}
	return ids, gen
}

// kv8aTurns is turn 1's prompt and, from its reply, turn 2's: the prompt, the reply as a client re-renders it (changed after 3
// tokens), and a new message long enough to clear the final-pass minimum.
func kv8aTurns(m *Model, reply []int) ([]int, []int) {
	_, _, _, _, _, _, vocab := m.Dims()
	p1 := make([]int, 40)
	for i := range p1 {
		p1[i] = (i*37 + 11) % vocab
	}
	re := slices.Clone(reply)
	if len(re) > 3 {
		re[3] = (re[3] + 1) % vocab
	}
	p2 := append(append(slices.Clone(p1), re...), 5, 9, 2, 7, 3, 8, 1, 6, 4, 10, 12, 14)
	return p1, p2
}

func TestKV8a_restoreEqualsCold(t *testing.T) {
	for _, plant := range []bool{false, true} {
		m, rf := loadRecurrentFake(t, true, false)
		rf.noRestore = plant
		p1, _ := kv8aTurns(m, nil)
		r1, g1 := kv8aRun(t, m, p1)
		if !g1.CheckpointSaved || g1.PrefillReused != 0 {
			t.Fatalf("turn 1: saved %v, reused %d; want a cold prefill and a save", g1.CheckpointSaved, g1.PrefillReused)
		}
		_, p2 := kv8aTurns(m, r1)
		got, g2 := kv8aRun(t, m, p2)
		if g2.PrefillReused != len(p1) || g2.CheckpointRestored != len(p1) || rf.restores != 1 {
			t.Fatalf("turn 2: reused %d, restored at %d, %d restores; want the checkpoint at %d", g2.PrefillReused, g2.CheckpointRestored, rf.restores, len(p1))
		}
		cm, _ := loadRecurrentFake(t, true, false)
		want, _ := kv8aRun(t, cm, p2)
		same := slices.Equal(got, want)
		t.Logf("plant=%v: turn 2 from the checkpoint %v, cold %v", plant, got, want)
		switch {
		case !plant && !same:
			t.Errorf("turn 2 restored from the checkpoint %v, a cold run %v", got, want)
		case plant && same:
			t.Errorf("a restore that copies nothing gave the cold reply: the comparison cannot see the state")
		}
	}
}

// Without a checkpoint kept (the resident declined it at load), turn 2 prefills cold, as before KV8a.
func TestKV8a_noCheckpointRunsCold(t *testing.T) {
	m, rf := loadRecurrentFake(t, false, false)
	p1, _ := kv8aTurns(m, nil)
	r1, g1 := kv8aRun(t, m, p1)
	_, p2 := kv8aTurns(m, r1)
	_, g2 := kv8aRun(t, m, p2)
	if g1.CheckpointSaved || g2.PrefillReused != 0 || rf.restores != 0 {
		t.Fatalf("saved %v, turn 2 reused %d with %d restores: want no save and a cold turn 2", g1.CheckpointSaved, g2.PrefillReused, rf.restores)
	}
}

// Gate 4(a): the save copies the state the prefill left, before the first decode step, on the device-chained path and the plain one.
func TestKV8a_saveBeforeTheFirstDecodeStep(t *testing.T) {
	for _, chain := range []bool{false, true} {
		m, rf := loadRecurrentFake(t, true, chain)
		p1, _ := kv8aTurns(m, nil)
		kv8aRun(t, m, p1)
		save := slices.Index(rf.events, "save")
		first := slices.IndexFunc(rf.events, func(e string) bool {
			return e == fmt.Sprintf("forward %d", len(p1)) || e == fmt.Sprintf("chain start %d", len(p1))
		})
		t.Logf("chain=%v: %v", chain, rf.events)
		if save < 0 || first < 0 || save > first || !rf.savedAfter {
			t.Errorf("chain=%v: save at event %d, first decode step at %d, saved the prefill's state %v; want the save first, of that state", chain, save, first, rf.savedAfter)
		}
		if chain && !slices.ContainsFunc(rf.events, func(e string) bool { return e == fmt.Sprintf("chain start %d", len(p1)) }) {
			t.Errorf("the chained run never opened the chain: the case did not test it")
		}
	}
}

// slottedRecurrentFake is recurrentFake with a state and a checkpoint per KV slot, as a resident that keeps both.
type slottedRecurrentFake struct {
	*recurrentFake
	bound        int
	states, ckpt []uint64
	held         []bool
}

func (f *slottedRecurrentFake) KVSlots() int { return len(f.states) }
func (f *slottedRecurrentFake) UseKVSlot(i int) error {
	f.states[f.bound], f.bound = f.st, i
	f.st = f.states[i]
	return nil
}
func (f *slottedRecurrentFake) RecurrentCheckpointHeld(slot int) bool { return f.held[slot] }
func (f *slottedRecurrentFake) Reset() {
	f.fakeResident.Reset()
	f.held[f.bound] = false
}
func (f *slottedRecurrentFake) SaveRecurrentCheckpoint() error {
	f.ckpt[f.bound], f.held[f.bound] = f.st, true
	return nil
}
func (f *slottedRecurrentFake) RestoreRecurrentCheckpoint() error {
	f.restores++
	f.st = f.ckpt[f.bound]
	return nil
}

type slottedRecurrentBackend struct {
	Backend
	rf *slottedRecurrentFake
}

func (b *slottedRecurrentBackend) Name() string { return "cuda" } // a table that keeps recurrent state per slot
func (b *slottedRecurrentBackend) BuildResident(m *Model) (ResidentForward, bool, error) {
	_, _, _, _, _, _, vocab := m.Dims()
	b.rf = &slottedRecurrentFake{recurrentFake: &recurrentFake{fakeResident: &fakeResident{vocab: vocab}, keeps: true}, states: make([]uint64, 2), ckpt: make([]uint64, 2), held: make([]bool, 2)}
	return b.rf, true, nil
}
func (b *slottedRecurrentBackend) Close() error { return nil }

// TestKV8_severalSlotsRestoreTheirCheckpoints: on a load of several slots, a follow-up that diverges at its prompt's end scores its slot's checkpoint, so it keeps that
// slot and restores it, for each of two conversations in turn. Each reply equals that conversation's alone on one slot. It reads the decoder's half with no device; on the
// card the same is cuda's TestRecurrentSlots_restoreThroughGenerate.
func TestKV8_severalSlotsRestoreTheirCheckpoints(t *testing.T) {
	be := &slottedRecurrentBackend{}
	name := fmt.Sprintf("fake-kv8-%d", recurrentFakeBackends.Add(1))
	RegisterBackend(name, func() (Backend, error) {
		cpu, err := NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be.Backend = cpu
		return be, nil
	})
	m, err := Load("../testdata/qwen35-tiny", Options{Backend: name, ResidentKVSlots: 2})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = m.Close() })
	if got := m.ResidentKVSlots(); got != 2 {
		t.Fatalf("%d slots, want 2", got)
	}
	vocab := m.w.arch.VocabSize
	PA, _ := kv8aTurns(m, nil)
	PB := make([]int, len(PA))
	for i := range PB {
		PB[i] = (i*17 + 101) % vocab
	}
	follow := func(p, reply []int) []int {
		re := slices.Clone(reply)
		re[0] = (re[0] + 1) % vocab // diverges right at the prompt's end
		return append(append(slices.Clone(p), re...), 5, 9, 2, 7, 3, 8, 1, 6, 4, 10, 12, 14)
	}
	ra, _ := kv8aRun(t, m, PA)
	rb, _ := kv8aRun(t, m, PB)
	fa, fb := follow(PA, ra), follow(PB, rb)
	got := map[string][]int{}
	for _, c := range []struct {
		name   string
		prompt []int
	}{{"A", fa}, {"B", fb}} {
		reply, g := kv8aRun(t, m, c.prompt)
		t.Logf("%s: restored at %d on slot %d, declined %q", c.name, g.CheckpointRestored, m.kv.Bound(), g.CheckpointDeclined)
		if g.CheckpointRestored != len(PA) {
			t.Errorf("%s: restored at %d (declined %q), want its prompt's end, %d", c.name, g.CheckpointRestored, g.CheckpointDeclined, len(PA))
		}
		got[c.name] = reply
	}
	for name, c := range map[string][2][]int{"A": {PA, fa}, "B": {PB, fb}} {
		alone, _ := loadRecurrentFake(t, true, false)
		kv8aRun(t, alone, c[0])
		want, g := kv8aRun(t, alone, c[1])
		if g.CheckpointRestored != len(PA) {
			t.Errorf("%s alone restored at %d: the comparison reads something else", name, g.CheckpointRestored)
		}
		if !slices.Equal(got[name], want) {
			t.Errorf("%s: the reply %v differs from that conversation's alone on one slot, %v", name, got[name], want)
		}
	}
}

// oneCopyRecurrentFake is slottedRecurrentFake on a device with room for one checkpoint: a save takes it from whichever slot held it, and a restore on any other slot fails
// and leaves the live state zeroed, as a slot whose restore fails then prefills from 0.
type oneCopyRecurrentFake struct {
	*slottedRecurrentFake
	owner int
}

func (f *oneCopyRecurrentFake) SaveRecurrentCheckpoint() error {
	f.owner, f.saved = f.bound, f.st
	return nil
}
func (f *oneCopyRecurrentFake) RestoreRecurrentCheckpoint() error {
	f.restores++
	if f.owner != f.bound {
		return fmt.Errorf("slot %d holds no recurrent checkpoint", f.bound)
	}
	f.st = f.saved
	return nil
}
func (f *oneCopyRecurrentFake) RecurrentCheckpointHeld(slot int) bool { return f.owner == slot }
func (f *oneCopyRecurrentFake) Reset() {
	f.fakeResident.Reset()
	if f.owner == f.bound {
		f.owner = -1
	}
}

type oneCopyRecurrentBackend struct {
	Backend
	rf *oneCopyRecurrentFake
}

func (b *oneCopyRecurrentBackend) Name() string { return "cuda" }
func (b *oneCopyRecurrentBackend) BuildResident(m *Model) (ResidentForward, bool, error) {
	_, _, _, _, _, _, vocab := m.Dims()
	b.rf = &oneCopyRecurrentFake{slottedRecurrentFake: &slottedRecurrentFake{recurrentFake: &recurrentFake{fakeResident: &fakeResident{vocab: vocab}, keeps: true}, states: make([]uint64, 3), ckpt: make([]uint64, 3), held: make([]bool, 3)}, owner: -1}
	return b.rf, true, nil
}
func (b *oneCopyRecurrentBackend) Close() error { return nil }

// TestKV8_aTakenCheckpointIsNotScored: three slots and one copy. A saves, B saves and takes the copy. A branch of A's prompt then matches slot 0 through its checkpoint's position,
// but slot 0 holds no checkpoint any more, so it takes the empty slot, attempts no restore, and leaves slot 0's sequence alive for A's own next turn.
func TestKV8_aTakenCheckpointIsNotScored(t *testing.T) {
	be := &oneCopyRecurrentBackend{}
	name := fmt.Sprintf("fake-kv8-%d", recurrentFakeBackends.Add(1))
	RegisterBackend(name, func() (Backend, error) {
		cpu, err := NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be.Backend = cpu
		return be, nil
	})
	m, err := Load("../testdata/qwen35-tiny", Options{Backend: name, ResidentKVSlots: 3})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = m.Close() })
	vocab := m.w.arch.VocabSize
	PA, _ := kv8aTurns(m, nil)
	PB := make([]int, len(PA))
	for i := range PB {
		PB[i] = (i*17 + 101) % vocab
	}
	ra, _ := kv8aRun(t, m, PA)
	slotA := m.kv.Bound()
	kv8aRun(t, m, PB)
	branch := append(slices.Clone(PA), (ra[0]+1)%vocab, 5, 9, 2, 7, 3, 8, 1, 6, 4, 10, 12, 14)
	_, g := kv8aRun(t, m, branch)
	t.Logf("the branch: slot %d, restored at %d, reused %d, declined %q, %d restores tried", m.kv.Bound(), g.CheckpointRestored, g.PrefillReused, g.CheckpointDeclined, be.rf.restores)
	if be.rf.restores != 0 || g.CheckpointDeclined != "" {
		t.Errorf("a restore was tried on a slot with no checkpoint: %d restores, declined %q", be.rf.restores, g.CheckpointDeclined)
	}
	if m.kv.Bound() == slotA {
		t.Errorf("the branch took slot %d, A's, with an empty slot free", slotA)
	}
	ext := append(append(slices.Clone(PA), ra...), 21, 22)
	if _, g := kv8aRun(t, m, ext); g.PrefillReused != len(PA)+len(ra) {
		t.Errorf("A's own next turn reused %d of %d rows: the branch took its slot's sequence", g.PrefillReused, len(PA)+len(ra))
	}
}

// TestKV8_aResetFreesTheCheckpoint: a reset on the slot that holds the checkpoint starts a new sequence there, so the slot no longer scores it, though its record still names the
// position until the commit that follows. Nothing tries a restore of it.
func TestKV8_aResetFreesTheCheckpoint(t *testing.T) {
	be := &oneCopyRecurrentBackend{}
	name := fmt.Sprintf("fake-kv8-%d", recurrentFakeBackends.Add(1))
	RegisterBackend(name, func() (Backend, error) {
		cpu, err := NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be.Backend = cpu
		return be, nil
	})
	m, err := Load("../testdata/qwen35-tiny", Options{Backend: name, ResidentKVSlots: 3})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = m.Close() })
	PA, _ := kv8aTurns(m, nil)
	ra, _ := kv8aRun(t, m, PA)
	slot := m.kv.Bound()
	if !m.checkpointHeld(slot) {
		t.Fatal("the save left the slot holding nothing: the test reads nothing")
	}
	m.resident.Reset()
	if m.checkpointHeld(slot) {
		t.Error("a reset left the slot holding its checkpoint")
	}
	branch := append(slices.Clone(PA), (ra[0]+1)%m.w.arch.VocabSize, 5, 9, 2, 7, 3, 8, 1, 6, 4, 10, 12, 14)
	if _, g := kv8aRun(t, m, branch); be.rf.restores != 0 || g.CheckpointRestored != 0 || g.CheckpointDeclined != "" {
		t.Errorf("after the reset: %d restores, restored at %d, declined %q; want none tried", be.rf.restores, g.CheckpointRestored, g.CheckpointDeclined)
	}
}
