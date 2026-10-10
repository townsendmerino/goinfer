//go:build cuda && goinfer_testhooks

package cuda

// TestSpecAdaptiveConcurrentCUDA gates -spec-adaptive: N CONCURRENT greedy GenerateNgramSpeculative generations on a REAL
// CUDA resident with SetSpecAdaptive(true) must emit exactly what each emits alone. decoder/spec_adaptive_switch_test.go
// checks the same property on mc3Fake, which has no numerics and no real slots, so it cannot see the race this guards: a
// round that claims only resBusy runs its bind-then-verify while another generation's exclusive section (slot pick,
// prefill, commit) binds a different slot, and each writes the other's (docs/measurements/mc4-candidate-cuda-2026-10-01.md).
// Red without the fix, green with it.

import (
	"context"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

func specIDs(t *testing.T, m *decoder.Model, prompt []int, maxTok int) []int {
	ch, gen, err := m.GenerateNgramSpeculative(context.Background(), prompt, maxTok, &decoder.NgramDrafter{}, 4, decoder.SamplingParams{})
	if err != nil {
		t.Errorf("GenerateNgramSpeculative: %v", err)
		return nil
	}
	var ids []int
	for id := range ch {
		ids = append(ids, id)
	}
	if err := gen.Err(); err != nil {
		t.Errorf("generation: %v", err)
	}
	return ids
}

func TestSpecAdaptiveConcurrentCUDA(t *testing.T) {
	path := filepath.Join("..", "testdata", "llama-tiny")
	requireDeviceAndFixture(t, path)
	const nCl = 4
	load := func(slots int) *decoder.Model {
		m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentKVSlots: slots, ResidentContext: 512})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		if _, ok := m.ResidentForwardForTest().(*cudaResident); !ok {
			m.Close()
			t.Fatalf("not CUDA-resident: %s", m.ResidentDecline())
		}
		return m
	}
	// prompts with internal repetition so the n-gram drafter proposes
	mkPrompt := func(c int) []int {
		base := []int{3 + c, 9 + c, 14 + c, 20 + c, 31 + c, 7 + c}
		var p []int
		for range 3 {
			p = append(p, base...)
		}
		return append(p, base[:3]...)
	}
	const maxTok = 48

	mRef := load(1)
	ref := make([][]int, nCl)
	for c := range nCl {
		ref[c] = specIDs(t, mRef, mkPrompt(c), maxTok)
	}
	mRef.Close()

	m := load(4)
	defer m.Close()
	if got := m.EnableResidentConcurrency(4); got < 2 {
		t.Fatalf("EnableResidentConcurrency = %d (the resident cannot batch?)", got)
	}
	m.SetSpecAdaptive(true)
	bad := 0
	const reps = 6
	for rep := range reps {
		got := make([][]int, nCl)
		var wg sync.WaitGroup
		for c := range nCl {
			wg.Go(func() { ; got[c] = specIDs(t, m, mkPrompt(c), maxTok) })
		}
		wg.Wait()
		for c := range nCl {
			if !slices.Equal(got[c], ref[c]) {
				bad++
				t.Logf("rep %d client %d DIFFERS: len got %d ref %d; got %v\n   ref %v", rep, c, len(got[c]), len(ref[c]), got[c][:min(16, len(got[c]))], ref[c][:min(16, len(ref[c]))])
			}
		}
	}
	t.Logf("%d of %d generations differ from their lone reference", bad, reps*nCl)
	if bad > 0 {
		t.Errorf("-spec-adaptive under concurrency is not lossless: %d of %d generations differ from their lone reference", bad, reps*nCl)
	}
}
