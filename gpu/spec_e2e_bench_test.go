//go:build gpu

package gpu_test

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/gpu"
)

// TestSpeculativeResident_e2eThroughput is the Lever 2 KILL-GATE measurement: real
// end-to-end tokens/sec for resident GPU speculative decoding (1.5B target on webgpu,
// 0.5B CPU draft) vs plain resident greedy on the SAME target, on code prompts where
// acceptance is high. It also re-asserts the parity invariant (spec output ==
// greedy output) so a throughput regression can't hide behind a correctness break.
//
// This measures Stage A (the batched ForwardN amortizes only the (K-1) Submit/Poll
// syncs, not the K×~420 dispatch records — see gpu-next-levers-assessment.md §7), so a
// speedup BELOW ~1.3× here is the expected, documented signal that Stage B (the true
// M=K GEMM verify) is what unlocks the predicted 1.4–2.0×. The number is logged, not
// asserted as a floor (Stage A is not where the win lives); only parity is hard-gated.
//
//	GOINFER_SPEC_TARGET=~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
//	GOINFER_SPEC_DRAFT=~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf \
//	  go test -tags gpu ./gpu/ -run TestSpeculativeResident_e2eThroughput -v -timeout 30m
func TestSpeculativeResident_e2eThroughput(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1 to opt in (loads a multi-GB model from ~/models)")
	}
	if testing.Short() {
		t.Skip("throughput measurement: skipped in -short")
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
	// Draft is ALSO GPU-resident: a CPU draft is slower per token than the GPU target
	// (measured), which makes speculation a net loss regardless of verify cost. Two
	// resident models = two Contexts/devices sharing the 8 GB pool.
	draft, err := decoder.Load(dpath, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load draft: %v", err)
	}
	defer draft.Close()
	t.Logf("draft resident on GPU: %v", draft.ResidentActive())

	collect := func(ch <-chan int) []int {
		var v []int
		for id := range ch {
			v = append(v, id)
		}
		return v
	}
	ctx := context.Background()
	greedy := decoder.SamplingParams{Temperature: 0}

	// A code-shaped prompt (Qwen2.5-coder BPE ids): "def quicksort(arr):\n    if len(arr)"
	// — the regime where draft/target agree often, so acceptance (and the spec ceiling)
	// is high. Generate enough tokens that per-token wall dominates prefill.
	prompt := []int{750, 4271, 4427, 9669, 982, 262, 421, 2422, 9669}
	const n = 160

	// time1 runs gen once (after a warm-up call to JIT pipelines + fill page caches)
	// and returns (tokens, wall, tokens/sec).
	timeRun := func(run func() ([]int, *decoder.Generation)) ([]int, time.Duration, float64, *decoder.Generation) {
		run() // warm-up (discarded): first call compiles pipelines / warms caches
		t0 := time.Now()
		toks, g := run()
		dt := time.Since(t0)
		return toks, dt, float64(len(toks)) / dt.Seconds(), g
	}

	// Baseline: plain resident greedy on the target.
	ref, refDT, refTPS, _ := timeRun(func() ([]int, *decoder.Generation) {
		ch, g := target.Generate(ctx, prompt, n, greedy)
		return collect(ch), g
	})
	t.Logf("plain resident greedy: %d tok in %v = %.2f tok/s", len(ref), refDT.Round(time.Millisecond), refTPS)

	for _, K := range []int{4, 8} {
		var spec *decoder.Generation
		got, dt, tps, g := timeRun(func() ([]int, *decoder.Generation) {
			ch, gg, err := target.GenerateSpeculative(ctx, prompt, n, draft, K, greedy)
			if err != nil {
				t.Fatalf("GenerateSpeculative K=%d: %v", K, err)
			}
			return collect(ch), gg
		})
		spec = g
		if spec.Err() != nil {
			t.Fatalf("K=%d: stream err %v", K, spec.Err())
		}
		// PARITY (hard gate): resident speculative must be token-identical to greedy.
		if !slices.Equal(got, ref) {
			t.Fatalf("K=%d: resident speculative != resident greedy\n got %v\n ref %v", K, got, ref)
		}
		speedup := tps / refTPS
		acc, tpr := 0.0, 0.0
		if spec.Spec != nil {
			acc, tpr = spec.Spec.AcceptanceRate(), spec.Spec.TokensPerRound()
		}
		verdict := "✅ ≥1.3× (Stage A already clears the kill-gate)"
		if speedup < 1.3 {
			verdict = "⚠️  <1.3× — expected for Stage A; the win is Stage B (M=K GEMM verify)"
		}
		t.Logf("resident spec K=%d: %d tok in %v = %.2f tok/s | speedup %.2f× vs plain | acceptance %.3f, %.2f tok/round | %s",
			K, len(got), dt.Round(time.Millisecond), tps, speedup, acc, tpr, verdict)
	}
}
