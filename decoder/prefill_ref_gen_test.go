//go:build goinfer_testhooks

package decoder

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestPrefillGateReference is Phase A of docs/task-prefill-gap.md §3.1's L1 re-run: build the CPU
// f32-activation reference logits that BOTH Metal arms (metal/prefill_gate_ref_test.go, Phase B)
// are scored against. §3's first form scored Metal's fast (f16-activation) path against Metal's
// own exact (int8-per-row-activation) path and called the exact path truth — but the exact path is
// itself a quantisation, guaranteed to disagree with the fast path for a reason that has nothing
// to do with a defect (see the doc's §3.1 correction). This reference is what actually has no
// activation-precision loss to attribute anything to.
//
// Runs in its OWN process, deliberately separate from the Metal gate: a 7B CPU reference and a 7B
// Metal resident sharing 16GB at once on this machine is exactly the failure mode that produced a
// real kernel panic here before a model was ever intentionally run oversized. Two models:
//   - S (1.5B): Options{Backend:"cpu", Quant:""} — f32 weights, f32 activations, ~6GB.
//   - D7 (7B):  Options{Backend:"cpu", Quant:"int8"} — weight-only per-row int8, f32 activations,
//     ~7.6GB. The f32 weights for D7 would be ~28GB and do not fit; per the brief, if the q4_k_m
//     GGUF cannot load under "int8" this subtest logs why and skips — D7 is the confirmation
//     model, S is the one the decision rests on.
//
// The reference is the CPU backend's own prefill (PrefillLogitsForTest, the batched prompt path —
// weights streamed once and reused across all K positions, ~1.7-2x faster than a naive per-token
// loop and bit-identical to it) with GOINFER_CPU_FAST_ATTENTION forced to "0", i.e. the exact
// f64-accumulating attention kernel, never the f32-fast one that is default ON elsewhere in this
// tree. The 64-token greedy continuation past the prompt still runs one token at a time
// (ForwardForTest) — inherent to greedy decoding (each step needs the previous step's own choice),
// and cheap next to K up to 3900.
//
// PARALLEL ACROSS PROMPTS: the 10 prompts per (model, K) run concurrently, one goroutine per CPU
// (capped), each with its own *KVCache. This is safe because a CPU forward's mutable scratch state
// lives on the KVCache it's given (decoder/model.go's cache.scr), not on the shared *Model — the
// only mutex on *Model guards LoRA adapter swaps, a different concern. Verified empirically, not
// just argued: before the real run, a short preflight sends the SAME prompt through N concurrent
// workers on independent caches and requires bit-identical seed logits; a real corruption would
// show up there in seconds rather than being discovered hours into results that already cost the
// wall-clock this parallelism exists to avoid.
//
// Output is NOT written into the repo — these are large, per-machine, per-session binaries
// (10 prompts x 65 x vocab float32s per cell) meant only for the Phase B run on this box, not a
// committed artifact. Written to ~/goinfer-logs/prefill-ref/<model>-K<k>-p<i>.bin via
// decoder.WritePrefillReferenceForTest.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./decoder/ -run TestPrefillGateReference -v -timeout 4h
func TestPrefillGateReference(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads two real checkpoints on CPU, S and D7)")
	}
	if testing.Short() {
		t.Skip("long-running reference build: skipped in -short")
	}
	t.Setenv("GOINFER_CPU_FAST_ATTENTION", "0") // exact f64-accumulating attention, not the fast f32 default

	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("home dir: %v", err)
	}
	outDir := filepath.Join(home, "goinfer-logs", "prefill-ref")
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", outDir, err)
	}

	const continuationN = 64
	workers := min(8, runtime.NumCPU())
	models := []struct {
		name        string
		pathEnv     string
		defaultPath string
		quant       string // "" = f32 weights+activations (S); "int8" = weight-only, f32 activations (D7)
		ks          []int  // decision-set + confirmation K's for THIS model, in run order
	}{
		{"S", "GOINFER_CPU_MODEL", "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", "", []int{256, 1024, 3900}},
		{"D7", "GOINFER_CPU_MODEL_D7", "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf", d7RefQuant(), []int{256, 1024}},
	}

	for _, mc := range models {
		t.Run(mc.name, func(t *testing.T) {
			path := os.Getenv(mc.pathEnv)
			if path == "" {
				path = os.ExpandEnv(mc.defaultPath)
			}
			if _, err := os.Stat(path); err != nil {
				t.Skipf("no fixture at %s (set %s)", path, mc.pathEnv)
			}
			m, err := Load(path, Options{Backend: "cpu", Quant: mc.quant})
			if err != nil {
				if mc.name == "D7" {
					t.Skipf("D7 CPU quant=%q load failed (%v) — D7 is the confirmation model, "+
						"S decides; run S alone", mc.quant, err)
				}
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			tk, err := tokenizer.LoadGGUF(path)
			if err != nil {
				t.Fatalf("load tokenizer: %v", err)
			}

			maxK := mc.ks[0]
			for _, k := range mc.ks {
				if k > maxK {
					maxK = k
				}
			}
			prompts := make([][]int, 0, len(PrefillGateProseFiles))
			for _, f := range PrefillGateProseFiles {
				prompts = append(prompts, PrefillGateProseIDsForTest(t, tk, f, maxK))
			}

			preflightConcurrencySafety(t, m, prompts[0], workers)

			for _, K := range mc.ks {
				runPrefillReferenceKConcurrent(t, m, mc.name, K, prompts, outDir, continuationN, workers)
			}
		})
	}
}

// d7RefQuant picks D7's reference weight precision. The default is "int8" — weight-only per-row
// int8 with f32 ACTIVATIONS — because the machine this generator was written on has 16 GB and a
// f32 7B is ~28 GB. That default is correct there and is left alone.
//
// GOINFER_CPU_REF_QUANT_D7="" selects true f32 weights, which is what the 62 GB Linux box uses for
// the CUDA gate (docs/task-prefill-gap.md Phase 3 names `Options{Backend:"cpu", Quant:""}` for both
// models there). Either is a valid reference for the property that matters: §3.1's point is that
// the reference must have f32 ACTIVATIONS, since that is the axis both arms differ from it on, and
// the weight-requantisation error is common-mode across the two arms — they share identical int4
// weights, so it cancels out of the PAIRED comparison the gate actually decides on.
//
// A reference built at one precision is not comparable to one built at another, so the choice is
// recorded in the measurement doc for the run that used it.
func d7RefQuant() string {
	if v, ok := os.LookupEnv("GOINFER_CPU_REF_QUANT_D7"); ok {
		return v
	}
	return "int8"
}

// preflightConcurrencySafety sends the SAME (short) prompt through `workers` concurrent goroutines,
// each with its own *KVCache, and requires bit-identical seed logits from every one. It exists to
// verify — not merely argue — that concurrent CPU forward on independent caches doesn't corrupt
// shared state, before the real run spends hours of wall-clock trusting that. A short prompt (<=64
// tokens) keeps this to a few seconds regardless of model size.
func preflightConcurrencySafety(t *testing.T, m *Model, prompt []int, workers int) {
	t.Helper()
	probe := prompt
	if len(probe) > 64 {
		probe = probe[:64]
	}
	results := make([][]float32, workers)
	errs := make([]error, workers)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		w := w
		wg.Add(1)
		go func() {
			defer wg.Done()
			cache := m.NewCache(len(probe) + 4)
			lg, err := m.PrefillLogitsForTest(context.Background(), probe, cache)
			if err != nil {
				errs[w] = err
				return
			}
			results[w] = append([]float32(nil), lg...)
		}()
	}
	wg.Wait()
	for w, err := range errs {
		if err != nil {
			t.Fatalf("concurrency preflight: worker %d: %v", w, err)
		}
	}
	for w := 1; w < workers; w++ {
		if len(results[w]) != len(results[0]) {
			t.Fatalf("CONCURRENCY UNSAFE: worker %d returned %d logits, worker 0 returned %d — "+
				"parallel CPU forward on this model is corrupting shared state; do not trust the "+
				"parallel run below", w, len(results[w]), len(results[0]))
		}
		for i := range results[0] {
			if results[w][i] != results[0][i] {
				t.Fatalf("CONCURRENCY UNSAFE: worker %d's seed logits differ from worker 0's on an "+
					"IDENTICAL prompt at index %d (%v vs %v) — parallel CPU forward on this model is "+
					"corrupting shared state; do not trust the parallel run below", w, i, results[w][i], results[0][i])
			}
		}
	}
	fmt.Printf("[ref] concurrency preflight OK: %d parallel workers, identical prompt, bit-identical seed logits\n", workers)
}

// runPrefillReferenceKConcurrent processes every prompt for one (model, K) cell across a bounded
// worker pool, each worker owning its own *KVCache (see the concurrency-safety note on
// TestPrefillGateReference and the preflight above).
func runPrefillReferenceKConcurrent(t *testing.T, m *Model, modelName string, K int, prompts [][]int, outDir string, continuationN, workers int) {
	t.Helper()
	type job struct {
		pi  int
		ids []int
	}
	jobs := make(chan job, len(prompts))
	for pi, ids := range prompts {
		if len(ids) < K {
			t.Fatalf("prompt %d: only %d tokens, need >= %d", pi, len(ids), K)
		}
		jobs <- job{pi, ids[:K]}
	}
	close(jobs)

	var (
		mu       sync.Mutex
		firstErr error
		done     int
		wg       sync.WaitGroup
	)
	t0 := time.Now()
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				seedLogits, refTokens, refLogits, err := prefillReferenceCell(m, j.ids, continuationN, K)
				if err != nil {
					mu.Lock()
					if firstErr == nil {
						firstErr = fmt.Errorf("prompt %d: %w", j.pi, err)
					}
					mu.Unlock()
					continue
				}
				outPath := filepath.Join(outDir, fmt.Sprintf("%s-K%d-p%d.bin", modelName, K, j.pi))
				if err := WritePrefillReferenceForTest(outPath, seedLogits, refTokens, refLogits); err != nil {
					mu.Lock()
					if firstErr == nil {
						firstErr = fmt.Errorf("write %s: %w", outPath, err)
					}
					mu.Unlock()
					continue
				}
				mu.Lock()
				done++
				fmt.Printf("[ref] %s K=%d prompt %2d done (%d/%d) -> %s elapsed=%s\n",
					modelName, K, j.pi+1, done, len(prompts), outPath, time.Since(t0).Round(time.Second))
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	if firstErr != nil {
		t.Fatalf("%s K=%d: %v", modelName, K, firstErr)
	}
}

// prefillReferenceCell runs the CPU batched prefill over ids, then continuationN more greedy
// steps, returning the seed logits, the greedy continuation tokens, and that continuation's own
// per-position logits — all cloned at capture time (no assumption about buffer ownership across
// calls, the exact class of bug metal/prefill_gate_test.go's runPrefillGateCell already had to fix
// once). Plain error return, not *testing.T: called from worker goroutines, and t.Fatalf from a
// non-test goroutine is unsafe.
func prefillReferenceCell(m *Model, ids []int, continuationN, K int) (seedLogits []float32, refTokens []int, refLogits [][]float32, err error) {
	cache := m.NewCache(K + continuationN + 4)
	seed, err := m.PrefillLogitsForTest(context.Background(), ids, cache)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("prefill K=%d: %w", K, err)
	}
	seed = append([]float32(nil), seed...)

	refTokens = make([]int, continuationN)
	refLogits = make([][]float32, continuationN)
	cur := seed
	for i := 0; i < continuationN; i++ {
		refTokens[i] = refArgmax(cur)
		refLogits[i] = cur
		if i == continuationN-1 {
			break
		}
		lg, err := m.ForwardForTest(refTokens[i], cache)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("continuation step=%d: %w", i, err)
		}
		cur = append([]float32(nil), lg...)
	}
	return seed, refTokens, refLogits, nil
}

func refArgmax(v []float32) int {
	bi, bv := 0, v[0]
	for i, x := range v {
		if x > bv {
			bv, bi = x, i
		}
	}
	return bi
}
