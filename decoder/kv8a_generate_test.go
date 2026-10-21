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
