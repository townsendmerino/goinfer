package decoder

import (
	"context"
	"fmt"
	"sync"
	"testing"
)

// slotHiddenLastResident is a two-slot resident that answers HiddenLast, recording which slot every write lands in: the bound
// slot when Forward, ForwardN or HiddenLast runs. Its numerics are fakeResident's.
type slotHiddenLastResident struct {
	*fakeResident
	hiddenDim int

	mu        sync.Mutex
	n, bound  int
	hlSlots   []int // the slot bound at each HiddenLast call
	firstPos  []int // the first position each Forward/ForwardN batch wrote (one entry per call)
	lastWrite []string
}

func (f *slotHiddenLastResident) KVSlots() int { return f.n }
func (f *slotHiddenLastResident) UseKVSlot(i int) error {
	f.mu.Lock()
	f.bound = i
	f.mu.Unlock()
	return nil
}

func (f *slotHiddenLastResident) note(who string, pos int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.lastWrite[f.bound] = who
	if who == "chat" {
		f.firstPos = append(f.firstPos, pos)
	}
}

func (f *slotHiddenLastResident) Forward(e []float32, pos int) ([]float32, error) {
	f.note("chat", pos)
	return f.fakeResident.Forward(e, pos)
}

func (f *slotHiddenLastResident) ForwardN(es [][]float32, startPos int) ([][]float32, error) {
	f.note("chat", startPos)
	return f.fakeResident.ForwardN(es, startPos)
}

func (f *slotHiddenLastResident) HiddenLast(ctx context.Context, es [][]float32, startPos int) ([]float32, error) {
	f.mu.Lock()
	f.hlSlots = append(f.hlSlots, f.bound)
	f.mu.Unlock()
	f.note("embed", startPos)
	out := make([]float32, f.hiddenDim)
	out[0] = 1
	return out, nil
}

type slotHiddenLastBackend struct {
	Backend
	rf *slotHiddenLastResident
}

func (b *slotHiddenLastBackend) BuildResident(m *Model) (ResidentForward, bool, error) {
	hidden, _, _, _, _, _, vocab := m.Dims()
	b.rf = &slotHiddenLastResident{fakeResident: &fakeResident{vocab: vocab}, hiddenDim: hidden, n: 2, lastWrite: make([]string, 2)}
	return b.rf, true, nil
}

func (b *slotHiddenLastBackend) Close() error { return nil }

func loadWithSlotHiddenLastResident(t *testing.T) (*Model, *slotHiddenLastBackend) {
	t.Helper()
	be := &slotHiddenLastBackend{}
	name := fmt.Sprintf("fake-slot-hiddenlast-%s", t.Name())
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
		t.Fatalf("Load with fake slotted resident: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	if !m.ResidentActive() || m.residentSlotCount() < 2 {
		t.Skip("fixture is not resident-eligible with two slots")
	}
	return m, be
}

func drainGenerate(t *testing.T, m *Model, prompt []int, n int) []int {
	t.Helper()
	stream, g := m.Generate(context.Background(), prompt, n, SamplingParams{})
	var out []int
	for id := range stream {
		out = append(out, id)
	}
	if err := g.Err(); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	return out
}

// TestHiddenLast_residentKeepsTheChatsSlot is F-N01's gate: on a resident with a free KV slot, an embeddings call
// (HiddenLast, PromptHidden) runs in that free slot, so the chat parked in the bound slot keeps its rows and its next turn
// reuses them instead of prefilling from position 0.
func TestHiddenLast_residentKeepsTheChatsSlot(t *testing.T) {
	m, be := loadWithSlotHiddenLastResident(t)
	rf := be.rf
	turn1 := []int{1, 2, 3, 4, 5}
	reply := drainGenerate(t, m, turn1, 3)
	chatSlot := rf.bound

	if _, err := m.HiddenLast([]int{7, 8, 9}); err != nil {
		t.Fatalf("HiddenLast: %v", err)
	}
	if _, err := m.PromptHidden(context.Background(), []int{9, 8, 7}); err != nil {
		t.Fatalf("PromptHidden: %v", err)
	}
	for i, s := range rf.hlSlots {
		if s == chatSlot {
			t.Errorf("embeddings call %d ran in slot %d, the chat's, with slot %d free: the chat's rows are overwritten", i, s, 1-chatSlot)
		}
	}
	if rf.lastWrite[chatSlot] != "chat" {
		t.Errorf("slot %d (the chat's) was last written by %q, want the chat", chatSlot, rf.lastWrite[chatSlot])
	}

	turn2 := append(append(append([]int(nil), turn1...), reply...), 6)
	rf.mu.Lock()
	rf.firstPos = rf.firstPos[:0]
	rf.mu.Unlock()
	drainGenerate(t, m, turn2, 2)
	if rf.bound != chatSlot {
		t.Errorf("turn 2 ran in slot %d, want the chat's slot %d", rf.bound, chatSlot)
	}
	if len(rf.firstPos) == 0 || rf.firstPos[0] == 0 {
		t.Errorf("turn 2's first resident write was at position %v, want past 0 (the turn-1 prefix reused)", rf.firstPos)
	}
}

// TestHiddenLast_residentRacesChat runs embeddings calls against chat turns on one resident model, for -race: each call either
// takes the resident (a free slot) or falls to the CPU, and none fails.
func TestHiddenLast_residentRacesChat(t *testing.T) {
	m, _ := loadWithSlotHiddenLastResident(t)
	var wg sync.WaitGroup
	errs := make(chan error, 64)
	wg.Add(2)
	go func() {
		defer wg.Done()
		prompt := []int{1, 2, 3}
		for range 8 {
			stream, g := m.Generate(context.Background(), prompt, 3, SamplingParams{})
			for id := range stream {
				prompt = append(prompt, id)
			}
			if err := g.Err(); err != nil {
				errs <- err
			}
			prompt = append(prompt, 4)
		}
	}()
	go func() {
		defer wg.Done()
		for i := range 16 {
			if _, err := m.HiddenLast([]int{5, 6, 7 + i%3}); err != nil {
				errs <- err
			}
		}
	}()
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

// TestHiddenLast_scratchSlotPlantIsRed plants F-N01 back (residentForgetScratch writes the bound slot) and requires the gate
// above to see it: the embeddings call lands in the chat's slot.
func TestHiddenLast_scratchSlotPlantIsRed(t *testing.T) {
	scratchSlotPlantForTest = true
	t.Cleanup(func() { scratchSlotPlantForTest = false })
	m, be := loadWithSlotHiddenLastResident(t)
	drainGenerate(t, m, []int{1, 2, 3, 4, 5}, 3)
	chatSlot := be.rf.bound
	if _, err := m.HiddenLast([]int{7, 8, 9}); err != nil {
		t.Fatalf("HiddenLast: %v", err)
	}
	if len(be.rf.hlSlots) != 1 || be.rf.hlSlots[0] != chatSlot {
		t.Fatalf("with the plant, HiddenLast ran in slots %v, want the chat's slot %d: the gate cannot see the bug", be.rf.hlSlots, chatSlot)
	}
}
