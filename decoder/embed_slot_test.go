package decoder

import (
	"context"
	"fmt"
	"sync"
	"testing"
)

// slotHiddenLastResident is a two-slot resident that answers HiddenLast and ResidualAll, recording which slot every write
// lands in: the bound slot when Forward, ForwardN, HiddenLast or ResidualAll runs. Its numerics are fakeResident's. Read
// its recorded fields through snap, which takes the lock.
type slotHiddenLastResident struct {
	*fakeResident
	hiddenDim int

	mu        sync.Mutex
	n, bound  int
	embeds    []embedCall // each HiddenLast or ResidualAll call, in order
	firstPos  []int       // the first position each Forward/ForwardN batch wrote (one entry per call)
	lastWrite []string
}

// embedCall is one embeddings call the resident answered: which seam, and the slot bound when it ran.
type embedCall struct {
	seam string // "HiddenLast" or "ResidualAll"
	slot int
}

// slotSnap is a copy of slotHiddenLastResident's recorded fields, taken under its lock.
type slotSnap struct {
	bound     int
	embeds    []embedCall
	firstPos  []int
	lastWrite []string
}

func (f *slotHiddenLastResident) snap() slotSnap {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slotSnap{
		bound:     f.bound,
		embeds:    append([]embedCall(nil), f.embeds...),
		firstPos:  append([]int(nil), f.firstPos...),
		lastWrite: append([]string(nil), f.lastWrite...),
	}
}

func (f *slotHiddenLastResident) clearFirstPos() {
	f.mu.Lock()
	f.firstPos = f.firstPos[:0]
	f.mu.Unlock()
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
	f.embedded("HiddenLast", startPos)
	out := make([]float32, f.hiddenDim)
	out[0] = 1
	return out, nil
}

func (f *slotHiddenLastResident) ResidualAll(ctx context.Context, es [][]float32, startPos int) ([][]float32, error) {
	f.embedded("ResidualAll", startPos)
	out := make([][]float32, len(es))
	for i := range out {
		out[i] = make([]float32, f.hiddenDim)
		out[i][0] = 1
	}
	return out, nil
}

func (f *slotHiddenLastResident) embedded(seam string, pos int) {
	f.mu.Lock()
	f.embeds = append(f.embeds, embedCall{seam, f.bound})
	f.mu.Unlock()
	f.note("embed", pos)
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
		t.Fatalf("the fake builds a two-slot resident, but the model has resident %v with %d slots", m.ResidentActive(), m.residentSlotCount())
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

// TestHiddenLast_residentKeepsTheChatsSlot is F-N01's gate: on a resident with a free KV slot, each embeddings call
// (HiddenLast, PromptHidden, PromptHiddenAll) runs on the resident in that free slot, so the chat parked in the bound slot
// keeps its rows and its next turn reuses them instead of prefilling from position 0. Each call gets its own model, so
// it starts with the chat's slot bound.
func TestHiddenLast_residentKeepsTheChatsSlot(t *testing.T) {
	calls := []struct {
		name, seam string
		run        func(m *Model) error
	}{
		{"HiddenLast", "HiddenLast", func(m *Model) error { _, err := m.HiddenLast([]int{7, 8, 9}); return err }},
		{"PromptHidden", "HiddenLast", func(m *Model) error {
			_, err := m.PromptHidden(context.Background(), []int{9, 8, 7})
			return err
		}},
		{"PromptHiddenAll", "ResidualAll", func(m *Model) error {
			_, err := m.PromptHiddenAll(context.Background(), []int{8, 9, 7})
			return err
		}},
	}
	for _, c := range calls {
		t.Run(c.name, func(t *testing.T) {
			m, be := loadWithSlotHiddenLastResident(t)
			rf := be.rf
			turn1 := []int{1, 2, 3, 4, 5}
			reply := drainGenerate(t, m, turn1, 3)
			chatSlot := rf.snap().bound

			if err := c.run(m); err != nil {
				t.Fatalf("%s: %v", c.name, err)
			}
			got := rf.snap()
			if len(got.embeds) != 1 || got.embeds[0].seam != c.seam {
				t.Fatalf("the resident answered %+v, want one %s call: %s fell back to the CPU", got.embeds, c.seam, c.name)
			}
			if got.embeds[0].slot == chatSlot {
				t.Errorf("%s ran in slot %d, the chat's, with slot %d free: the chat's rows are overwritten", c.name, chatSlot, 1-chatSlot)
			}
			if got.lastWrite[chatSlot] != "chat" {
				t.Errorf("slot %d (the chat's) was last written by %q, want the chat", chatSlot, got.lastWrite[chatSlot])
			}

			rf.clearFirstPos()
			drainGenerate(t, m, append(append(append([]int(nil), turn1...), reply...), 6), 2)
			got = rf.snap()
			if got.bound != chatSlot {
				t.Errorf("turn 2 ran in slot %d, want the chat's slot %d", got.bound, chatSlot)
			}
			if len(got.firstPos) == 0 || got.firstPos[0] == 0 {
				t.Errorf("turn 2's first resident write was at position %v, want past 0 (the turn-1 prefix reused)", got.firstPos)
			}
		})
	}
}

// TestHiddenLast_residentLRUWhenNoSlotIsFree: with both slots holding a chat, an embeddings call takes the least recently
// used slot, so the chat that ran last keeps its rows and its next turn reuses them.
func TestHiddenLast_residentLRUWhenNoSlotIsFree(t *testing.T) {
	m, be := loadWithSlotHiddenLastResident(t)
	rf := be.rf
	chatA := []int{1, 2, 3, 4, 5}
	drainGenerate(t, m, chatA, 3)
	slotA := rf.snap().bound
	chatB := []int{11, 12, 13, 14, 15}
	replyB := drainGenerate(t, m, chatB, 3)
	slotB := rf.snap().bound
	if slotA == slotB {
		t.Fatalf("chats A and B shared slot %d; the setup needs one in each", slotA)
	}

	if _, err := m.HiddenLast([]int{7, 8, 9}); err != nil {
		t.Fatalf("HiddenLast: %v", err)
	}
	got := rf.snap()
	if len(got.embeds) != 1 {
		t.Fatalf("the resident answered %d embeddings calls (%+v), want 1: HiddenLast fell back to the CPU", len(got.embeds), got.embeds)
	}
	if got.embeds[0].slot != slotA {
		t.Errorf("with no slot free, HiddenLast ran in slot %d, want %d (chat A's, the least recently used)", got.embeds[0].slot, slotA)
	}

	rf.clearFirstPos()
	drainGenerate(t, m, append(append(append([]int(nil), chatB...), replyB...), 6), 2)
	got = rf.snap()
	if got.bound != slotB || len(got.firstPos) == 0 || got.firstPos[0] == 0 {
		t.Errorf("chat B's turn 2 ran in slot %d from position %v, want slot %d past 0 (its prefix kept)", got.bound, got.firstPos, slotB)
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
	chatSlot := be.rf.snap().bound
	if _, err := m.HiddenLast([]int{7, 8, 9}); err != nil {
		t.Fatalf("HiddenLast: %v", err)
	}
	if got := be.rf.snap().embeds; len(got) != 1 || got[0].slot != chatSlot {
		t.Fatalf("with the plant, HiddenLast ran as %+v, want once in the chat's slot %d: the gate cannot see the bug", got, chatSlot)
	}
}
