package decoder

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"
)

// TestGeneration_residentDeclinedSaysWhyItRanOnTheCPU: on a resident model, a generation that runs on the CPU says why in
// Generation.ResidentDeclined (its prompt passed the resident's context, or another generation held the resident), and one
// that runs on the resident leaves it empty.
func TestGeneration_residentDeclinedSaysWhyItRanOnTheCPU(t *testing.T) {
	run := func(m *Model, prompt []int) *Generation {
		ch, g := m.Generate(context.Background(), prompt, 2, SamplingParams{})
		for range ch {
		}
		if g.Err() != nil {
			t.Fatalf("Generate: %v", g.Err())
		}
		return g
	}
	m, be := loadWithFakeResident(t)
	if !m.ResidentActive() {
		t.Skip("fixture is not resident-eligible")
	}
	if g := run(m, []int{1, 2, 3}); g.ResidentDeclined != "" || be.rf.forwards == 0 {
		t.Errorf("a resident run: ResidentDeclined %q, %d resident forwards; want empty and some", g.ResidentDeclined, be.rf.forwards)
	}
	be.rf.capPos = 3
	if g := run(m, []int{1, 2, 3, 4, 5}); !strings.Contains(g.ResidentDeclined, "context") {
		t.Errorf("a prompt past the resident's context: ResidentDeclined %q, want the cap named", g.ResidentDeclined)
	}
	be.rf.capPos = 0
	if !atomic.CompareAndSwapInt32(&m.resBusy, 0, 1) {
		t.Fatal("test setup: resBusy already claimed")
	}
	g := run(m, []int{1, 2, 3})
	atomic.StoreInt32(&m.resBusy, 0)
	if !strings.Contains(g.ResidentDeclined, "another generation") {
		t.Errorf("the resident held by another generation: ResidentDeclined %q", g.ResidentDeclined)
	}
}
