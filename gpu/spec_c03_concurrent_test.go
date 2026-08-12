//go:build gpu

package gpu_test

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/gpu"
)

// TestSpeculative_C03_concurrentResidentClaim is the gate for audit C-03: GenerateSpeculative must
// claim the shared resident KV (resBusy CAS) before any device write, exactly as generateInto and
// the n-gram path do. Before the fix it skipped the claim, so a second concurrent generation on the
// same *Model — which the Model doc explicitly permits for distinct sequences — prefilled into the
// SAME positional device KV, interleaving writes and silently corrupting both streams.
//
// The bug is a missing mutual-exclusion claim, so it only manifests under real concurrency: this
// runs a plain resident Generate and a GenerateSpeculative CONCURRENTLY on one *Model. Whichever
// loses the CAS falls back to the staged CPU cache — separate state — so each stream is a VALID
// greedy decode on EITHER the resident or the CPU backend (the two can differ by a token at a
// near-tie: the documented resident-vs-CPU parity gap, NOT a bug). The test asserts each output
// equals one of those two references; a C-03 corruption (interleaved resident KV) matches neither.
// Deterministically passes on the fixed code regardless of which side wins the race.
//
// Target = 1.5B (resident), draft = 0.5B (CPU), both dense Qwen2 and vocab-matched — the same models
// as TestSpeculativeResident_parity. Heavy + webgpu gated.
func TestSpeculative_C03_concurrentResidentClaim(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1 to opt in (loads a multi-GB model from ~/models)")
	}
	if _, err := gpu.New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	home, _ := os.UserHomeDir()
	tpath := os.Getenv("GOINFER_SPEC_TARGET")
	if tpath == "" {
		tpath = filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	dpath := os.Getenv("GOINFER_SPEC_DRAFT")
	if dpath == "" {
		dpath = filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	}
	for _, p := range []string{tpath, dpath} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("missing model %s: %v", p, err)
		}
	}

	target, err := decoder.Load(tpath, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load target: %v", err)
	}
	defer target.Close()
	if !target.ResidentActive() {
		t.Skip("target not GPU-resident (ineligible / no residency)")
	}
	draft, err := decoder.Load(dpath, decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load draft: %v", err)
	}
	defer draft.Close()

	collect := func(ch <-chan int) []int {
		var v []int
		for id := range ch {
			v = append(v, id)
		}
		return v
	}
	ctx := context.Background()
	greedy := decoder.SamplingParams{Temperature: 0}
	const n = 24
	promptA := []int{785, 6722, 315, 9625, 374}      // drives the plain resident Generate
	promptB := []int{1654, 264, 729, 311, 912, 1378} // drives GenerateSpeculative

	resGreedy := func(prompt []int) []int {
		ch, _ := target.Generate(ctx, prompt, n, greedy)
		return collect(ch)
	}
	// Resident-path greedy references (no contention).
	resRefA := resGreedy(promptA)
	resRefB := resGreedy(promptB)

	// CPU-path greedy references. The CAS loser falls back to the STAGED CPU path (model.go:782),
	// whose greedy output can differ from the resident's at a near-tied token (documented
	// resident-vs-CPU parity gap) — so a contended stream may legitimately land on EITHER sequence.
	// A C-03 corruption (interleaved resident KV) matches NEITHER. We assert membership in the two
	// valid backends; that distinguishes a backend switch (fine) from corruption (the bug).
	cpuTarget, err := decoder.Load(tpath, decoder.Options{Backend: "cpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load cpu target (for staged reference): %v", err)
	}
	cpuGreedy := func(prompt []int) []int {
		ch, _ := cpuTarget.Generate(ctx, prompt, n, greedy)
		return collect(ch)
	}
	cpuRefA := cpuGreedy(promptA)
	cpuRefB := cpuGreedy(promptB)
	cpuTarget.Close() // free before the concurrent rounds

	validPaths := func(what string, got, resRef, cpuRef []int, round int) {
		if slices.Equal(got, resRef) || slices.Equal(got, cpuRef) {
			return
		}
		t.Fatalf("round %d: concurrent %s output matches NEITHER the resident nor the CPU greedy — "+
			"interleaved resident KV, the C-03 corruption\n got %v\n res %v\n cpu %v", round, what, got, resRef, cpuRef)
	}

	// Run several overlapping rounds — the resident claim is contended, so each round exercises the
	// CAS: one of the two takes the resident, the other falls back to staged CPU rather than sharing
	// the device KV. Each output must be a VALID greedy decode on one backend or the other.
	for round := range 4 {
		var gotA, gotB []int
		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			ch, _ := target.Generate(ctx, promptA, n, greedy)
			gotA = collect(ch)
		}()
		go func() {
			defer wg.Done()
			ch, g, err := target.GenerateSpeculative(ctx, promptB, n, draft, 4, greedy)
			if err != nil {
				t.Errorf("round %d: GenerateSpeculative: %v", round, err)
				return
			}
			gotB = collect(ch)
			if g.Err() != nil {
				t.Errorf("round %d: spec stream err: %v", round, g.Err())
			}
		}()
		wg.Wait()

		validPaths("Generate", gotA, resRefA, cpuRefA, round)
		validPaths("GenerateSpeculative", gotB, resRefB, cpuRefB, round)
	}
	t.Logf("C-03 OK: %d concurrent Generate‖GenerateSpeculative rounds — every stream a valid resident-or-CPU greedy decode, no interleaved-KV corruption", 4)
}
