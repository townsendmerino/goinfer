//go:build goinfer_testhooks

package decoder

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestPrefillGateReference is Phase A of docs/completed/task-prefill-gap.md §3.1's L1 re-run: it builds the CPU
// f32-activation reference logits that BOTH Metal arms (metal/prefill_gate_ref_test.go, Phase B) are scored against.
// Metal's exact path (int8-per-row activation) is itself a quantisation, so scoring the fast path against it blames a
// defect for a disagreement that has nothing to do with one; this reference has no activation-precision loss to attribute
// anything to.
//
// Runs in its OWN process, deliberately separate from the Metal gate: a 7B CPU reference and a 7B Metal resident sharing
// 16GB is a kernel-panic risk. Two models:
//   - S (1.5B): Options{Backend:"cpu", Quant:""}: f32 weights, f32 activations. The one the decision rests on.
//   - D7 (7B): Options{Backend:"cpu", Quant:"int8"}: weight-only per-row int8, f32 activations (f32 weights would not
//     fit). The confirmation model: if the q4_k_m GGUF cannot load under "int8" the subtest logs why and skips.
//
// The reference is the CPU backend's own batched prefill (PrefillLogitsForTest, bit-identical to a per-token loop) with
// GOINFER_CPU_FAST_ATTENTION forced to "0": the exact f64-accumulating attention kernel, never the f32-fast one. The
// 64-token greedy continuation past the prompt runs one token at a time (ForwardForTest), as greedy decoding requires.
//
// PARALLEL ACROSS PROMPTS: the prompts of a (model, K) cell run concurrently, one goroutine per CPU (capped), each with its
// own *KVCache. That is safe because a CPU forward's mutable scratch lives on the KVCache it is given (cache.scr in
// model.go), not on the shared *Model; the only mutex on *Model guards LoRA adapter swaps. preflightConcurrencySafety
// checks it before the real run. At depth concurrency can lose: see refWorkers.
//
// Output is NOT written into the repo: these are large, per-machine binaries (prompts x 65 x vocab float32s per cell) for the
// Phase B run on this box. They go to ~/goinfer-logs/prefill-ref/<model>-K<k>-p<i>.bin via decoder.WritePrefillReferenceForTest.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./decoder/ -run TestPrefillGateReference -v -timeout 4h
func TestPrefillGateReference(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads real checkpoints on CPU: S and D7, or the cells GOINFER_CPU_REF_MODELS names)")
	}
	if testing.Short() {
		t.Skip("long-running reference build: skipped in -short")
	}
	t.Setenv("GOINFER_CPU_FAST_ATTENTION", "0") // exact f64-accumulating attention, not the fast f32 default

	setLabel, promptFiles := PrefillGatePromptSet()

	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("home dir: %v", err)
	}
	// Set A keeps its historical directory name (prefill-ref) so existing reference files (and
	// anything that already reads from there) are untouched; set B gets its own directory so the
	// two never collide or get scored against the wrong prompts. See PrefillGatePromptSet.
	refDirName := "prefill-ref"
	if setLabel != "a" {
		refDirName = "prefill-ref-" + setLabel
	}
	outDir := filepath.Join(home, "goinfer-logs", refDirName)
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", outDir, err)
	}
	t.Logf("prompt set %q (%d files) -> %s", setLabel, len(promptFiles), outDir)

	const continuationN = 64
	workers := refWorkers(min(8, runtime.NumCPU()))
	// K=512 is in the decision set because a floor must be measured at the depth it is set to, not interpolated between cells
	// (docs/completed/task-prefill-gap.md §3 and §4 L1).
	models := []struct {
		name        string
		pathEnv     string
		defaultPath string
		quant       string // "" = f32 weights+activations (S); "int8" = weight-only, f32 activations (D7)
		ks          []int  // decision-set + confirmation K's for THIS model, in run order
	}{
		{"S", "GOINFER_CPU_MODEL", "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", "", refKs([]int{256, 512, 1024, 3900})},
		{"D7", "GOINFER_CPU_MODEL_D7", "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf", d7RefQuant(), refKs([]int{256, 512, 1024})},
		// Q05, the 0.5B (head dim 64): references for B-P01's fidelity gate (docs/tasks/task-metal-audit-2026-10.md), f32
		// like S. Built only when GOINFER_CPU_REF_MODELS names it, so the standing S/D7 run is unchanged.
		{"Q05", "GOINFER_CPU_MODEL_Q05", "$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", "", refKs([]int{2048, 3900})},
		// Q35, Qwen3.5-0.8B (a Gated-DeltaNet hybrid, safetensors): references for D-B01's fidelity gate, f32 like S. Built
		// only when GOINFER_CPU_REF_MODELS names it.
		{"Q35", "GOINFER_CPU_MODEL_Q35", "$HOME/models/qwen3.5-0.8b", "", refKs([]int{256, 512, 1024})},
	}
	// GOINFER_CPU_REF_MODELS (comma-separated cell names) narrows the run; unset builds S and D7, as before Q05.
	want := map[string]bool{"S": true, "D7": true}
	if v := strings.TrimSpace(os.Getenv("GOINFER_CPU_REF_MODELS")); v != "" {
		want = map[string]bool{}
		for f := range strings.SplitSeq(v, ",") {
			want[strings.TrimSpace(f)] = true
		}
	}

	for _, mc := range models {
		if !want[mc.name] {
			continue
		}
		t.Run(mc.name, func(t *testing.T) {
			path := os.Getenv(mc.pathEnv)
			if path == "" {
				path = os.ExpandEnv(mc.defaultPath)
			}
			if _, err := os.Stat(path); err != nil {
				t.Skipf("no fixture at %s (set %s)", path, mc.pathEnv)
			}

			maxK := mc.ks[0]
			for _, k := range mc.ks {
				if k > maxK {
					maxK = k
				}
			}
			// ResidentContext is a GPU-resident-KV concept (Options doc: "Ignored off the residency path"). On the CPU backend it only
			// changes what fitCheckFor prices the load's KV term at (effCtx in fitguard.go: the pinned ResidentContext when set, else
			// the model's own MaxPositions). Pin it to what this run touches (maxK+continuationN) so the guard prices a context the run
			// actually reaches.
			m, err := Load(path, Options{Backend: "cpu", Quant: mc.quant, ResidentContext: maxK + continuationN})
			if err != nil {
				if mc.name == "D7" {
					t.Skipf("D7 CPU quant=%q load failed (%v) — D7 is the confirmation model, "+
						"S decides; run S alone", mc.quant, err)
				}
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			tk, err := loadRefTokenizer(path)
			if err != nil {
				t.Fatalf("load tokenizer: %v", err)
			}
			prompts := make([][]int, 0, len(promptFiles))
			for _, f := range promptFiles {
				prompts = append(prompts, PrefillGateProseIDsForTest(t, tk, f, maxK))
			}

			preflightConcurrencySafety(t, m, prompts[0], workers)

			for _, K := range mc.ks {
				runPrefillReferenceKConcurrent(t, m, mc.name, path, mc.quant, K, prompts, outDir, continuationN, workers)
			}
		})
	}
}

// refKs lets a run build references at depths beyond the standing set, via GOINFER_CPU_REF_KS (comma-separated). The
// defaults are the §3 decision + confirmation cells and do not move. It exists because a FLOOR must be measured at the
// depth it is set to: a floor placed between two measured cells interpolates a fidelity result nobody took.
func refKs(def []int) []int {
	v := os.Getenv("GOINFER_CPU_REF_KS")
	if strings.TrimSpace(v) == "" {
		return def
	}
	out := make([]int, 0, 4)
	for f := range strings.SplitSeq(v, ",") {
		if k, err := strconv.Atoi(strings.TrimSpace(f)); err == nil && k > 0 {
			out = append(out, k)
		}
	}
	if len(out) == 0 {
		return def
	}
	return out
}

// refWorkers lets a run cap how many prompts are prefilled CONCURRENTLY, via GOINFER_CPU_REF_WORKERS. The default (8) is
// right for the shallow standing cells.
//
// AT DEPTH, CONCURRENCY IS A LOSS, NOT A WIN. The exact f64-accumulating attention materialises K x K score rows per head,
// which at K=8000 sit far outside a desktop CPU's L3; concurrent prompts doing that contend for memory bandwidth until each
// is slower than running them in series. One worker is the safe choice at the deep cells; the break-even lies between K=2048
// and K=8000. The measurements and the run that found it: docs/measurements/vsum-split-fidelity-2026-09-13.md, deviation D2.
func refWorkers(def int) int {
	if v, err := strconv.Atoi(strings.TrimSpace(os.Getenv("GOINFER_CPU_REF_WORKERS"))); err == nil && v > 0 {
		return v
	}
	return def
}

// d7RefQuant picks D7's reference weight precision. The default is "int8" (weight-only per-row int8, f32 ACTIVATIONS)
// because a f32 7B is ~28 GB and does not fit the 16 GB machine this generator targets.
//
// GOINFER_CPU_REF_QUANT_D7="" selects true f32 weights, as on the 62 GB Linux box for the CUDA gate
// (docs/completed/task-prefill-gap.md Phase 3). Either is a valid reference for the property that matters: the reference
// must have f32 ACTIVATIONS (§3.1), since that is the axis both arms differ from it on, while the weight-requantisation error
// is common-mode (both arms share identical int4 weights) and cancels out of the PAIRED comparison the gate decides on. A
// reference built at one precision is not comparable to one built at another, so record the choice in the measurement doc for
// the run that used it.
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
	for w := range workers {
		wg.Go(func() {
			cache := m.NewCache(len(probe) + 4)
			lg, err := m.PrefillLogitsForTest(context.Background(), probe, cache)
			if err != nil {
				errs[w] = err
				return
			}
			results[w] = append([]float32(nil), lg...)
		})
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
//
// CONTENT-KEYED AND RESUMABLE (TE6(a)/TE8, docs/tasks/task-test-efficiency-2026-09.md): each prompt's reference is
// looked up in prefill_ref_cache_testhook.go's cache by checkpoint sha256 + the prompt's ids + the reference path's
// source + arch/quant/continuation. A hit is linked into outDir and skipped; a miss is computed, stored atomically, then
// linked. So a rerun after an interruption computes only the prompts not yet stored, and a reference for different
// text, weights or code can never be picked up by accident.
func runPrefillReferenceKConcurrent(t *testing.T, m *Model, modelName, checkpointPath, quant string, K int, prompts [][]int, outDir string, continuationN, workers int) {
	t.Helper()
	type job struct {
		pi    int
		ids   []int
		parts PrefillRefKeyParts
	}
	jobs := make(chan job, len(prompts))
	cached := 0
	for pi, ids := range prompts {
		if len(ids) < K {
			t.Fatalf("prompt %d: only %d tokens, need >= %d", pi, len(ids), K)
		}
		parts, err := PrefillRefKeyForTest(checkpointPath, quant, ids[:K], continuationN)
		if err != nil {
			t.Fatalf("prompt %d: reference key: %v", pi, err)
		}
		outPath := filepath.Join(outDir, fmt.Sprintf("%s-K%d-p%d.bin", modelName, K, pi))
		if cp, ok := LookupPrefillRefForTest(parts); ok {
			if err := LinkPrefillRefForTest(cp, outPath, parts); err != nil {
				t.Fatalf("link cached reference %s -> %s: %v", cp, outPath, err)
			}
			cached++
			continue
		}
		jobs <- job{pi, ids[:K], parts}
	}
	close(jobs)
	fmt.Printf("[ref] %s K=%d: %d of %d prompts already in the reference cache, %d to compute\n",
		modelName, K, cached, len(prompts), len(prompts)-cached)

	var (
		mu       sync.Mutex
		firstErr error
		done     int
		wg       sync.WaitGroup
	)
	t0 := time.Now()
	for range workers {
		wg.Go(func() {
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
				cp, err := StorePrefillRefForTest(j.parts, seedLogits, refTokens, refLogits)
				if err == nil {
					err = LinkPrefillRefForTest(cp, outPath, j.parts)
				}
				if err != nil {
					mu.Lock()
					if firstErr == nil {
						firstErr = fmt.Errorf("store %s: %w", outPath, err)
					}
					mu.Unlock()
					continue
				}
				mu.Lock()
				done++
				fmt.Printf("[ref] %s K=%d prompt %2d done (%d/%d computed) -> %s elapsed=%s\n",
					modelName, K, j.pi+1, done, len(prompts)-cached, outPath, time.Since(t0).Round(time.Second))
				mu.Unlock()
			}
		})
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
	for i := range continuationN {
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

// loadRefTokenizer reads a checkpoint's tokenizer: a .gguf carries its own; a safetensors directory has tokenizer.json.
func loadRefTokenizer(path string) (*tokenizer.Tokenizer, error) {
	if st, err := os.Stat(path); err == nil && st.IsDir() {
		return tokenizer.Load(filepath.Join(path, "tokenizer.json"))
	}
	return tokenizer.LoadGGUF(path)
}
