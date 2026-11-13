package decoder

import (
	"context"
	"slices"
	"sync"
	"testing"
	"time"
)

// TestModel_concurrentGenerationsBeyondTheResidentsPlacesRunOnTheCPU pins decoder.Model's concurrency contract past the
// resident's capacity: with resident concurrency enabled for 2 places, 4 overlapping generations all complete without error,
// each emitting exactly what it emits alone on the resident or alone on the CPU, and the ones the resident had no place for
// ran on the CPU.
func TestModel_concurrentGenerationsBeyondTheResidentsPlacesRunOnTheCPU(t *testing.T) {
	const n, places, maxTok = 4, 2, 16
	prompts := make([][]int, n)
	for c := range n {
		prompts[c] = []int{1, 2, 3, 4 + c, 9 + 2*c}
	}
	gen := func(m *Model, p []int) ([]int, error) {
		ch, g := m.Generate(context.Background(), p, maxTok, SamplingParams{})
		var ids []int
		for id := range ch {
			ids = append(ids, id)
		}
		return ids, g.Err()
	}
	cpu, err := Load(tinyFixture(t), Options{Backend: "cpu"})
	if err != nil {
		t.Fatal(err)
	}
	defer cpu.Close()
	mAlone, _ := loadWithMC3Fake(t, n)
	onCPU, onRes := make([][]int, n), make([][]int, n)
	for c, p := range prompts {
		if onCPU[c], err = gen(cpu, p); err != nil {
			t.Fatal(err)
		}
		if onRes[c], err = gen(mAlone, p); err != nil {
			t.Fatal(err)
		}
		if slices.Equal(onCPU[c], onRes[c]) {
			t.Fatalf("VOID: prompt %d emits the same ids on the fake resident and the CPU, so the test cannot tell them apart", c)
		}
	}
	m, rf := loadWithMC3Fake(t, n)
	rf.delay = 3 * time.Millisecond // a resident token takes long enough that the four generations overlap
	if got := m.EnableResidentConcurrency(places); got != places {
		t.Fatalf("EnableResidentConcurrency(%d) = %d", places, got)
	}
	got := make([][]int, n)
	errs := make([]error, n)
	var wg sync.WaitGroup
	for c, p := range prompts {
		wg.Add(1)
		go func() { defer wg.Done(); got[c], errs[c] = gen(m, p) }()
	}
	wg.Wait()
	cpuRuns := 0
	for c := range n {
		switch {
		case errs[c] != nil:
			t.Errorf("generation %d failed: %v", c, errs[c])
		case slices.Equal(got[c], onCPU[c]):
			cpuRuns++
		case !slices.Equal(got[c], onRes[c]):
			t.Errorf("generation %d emitted %v: neither its resident %v nor its CPU %v", c, got[c], onRes[c], onCPU[c])
		}
	}
	if cpuRuns == 0 {
		t.Errorf("all %d overlapping generations ran on the resident's %d places; none took the CPU", n, places)
	}
	t.Logf("%d of %d ran on the CPU", cpuRuns, n)
}
