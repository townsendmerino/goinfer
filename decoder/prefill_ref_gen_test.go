//go:build goinfer_testhooks

package decoder

import (
	"fmt"
	"os"
	"path/filepath"
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
// The reference is the plain sequential per-token forward (ForwardForTest — the same seam the GPU
// resident-vs-CPU parity gates already use), which carries the f64-accumulating attention
// unconditionally: GOINFER_CPU_FAST_ATTENTION only ever gates the BATCHED/chunked prefill attention
// path, which this per-token loop never engages, so there is nothing to disable here.
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

	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("home dir: %v", err)
	}
	outDir := filepath.Join(home, "goinfer-logs", "prefill-ref")
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", outDir, err)
	}

	const continuationN = 64
	models := []struct {
		name        string
		pathEnv     string
		defaultPath string
		quant       string // "" = f32 weights+activations (S); "int8" = weight-only, f32 activations (D7)
		ks          []int  // decision-set + confirmation K's for THIS model, in run order
	}{
		{"S", "GOINFER_CPU_MODEL", "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", "", []int{256, 1024, 3900}},
		{"D7", "GOINFER_CPU_MODEL_D7", "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf", "int8", []int{256, 1024}},
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

			for _, K := range mc.ks {
				t0 := time.Now()
				for pi, ids := range prompts {
					if len(ids) < K {
						t.Fatalf("prompt %d: only %d tokens, need >= %d", pi, len(ids), K)
					}
					seedLogits, refTokens, refLogits := prefillReferenceCell(t, m, ids[:K], continuationN, K)
					outPath := filepath.Join(outDir, fmt.Sprintf("%s-K%d-p%d.bin", mc.name, K, pi))
					if err := WritePrefillReferenceForTest(outPath, seedLogits, refTokens, refLogits); err != nil {
						t.Fatalf("write %s: %v", outPath, err)
					}
					fmt.Printf("[ref] %s K=%d prompt %2d/%2d -> %s elapsed=%s\n",
						mc.name, K, pi+1, len(prompts), outPath, time.Since(t0).Round(time.Second))
				}
			}
		})
	}
}

// prefillReferenceCell runs the sequential CPU forward over ids, then continuationN more greedy
// steps, returning the seed logits, the greedy continuation tokens, and that continuation's own
// per-position logits — all cloned at capture time (ForwardForTest's underlying buffer ownership
// is not guaranteed across calls; cloning defensively costs nothing at this scale and is the exact
// class of bug metal/prefill_gate_test.go's runPrefillGateCell already had to fix once).
func prefillReferenceCell(t *testing.T, m *Model, ids []int, continuationN, K int) (seedLogits []float32, refTokens []int, refLogits [][]float32) {
	t.Helper()
	cache := m.NewCache(K + continuationN + 4)

	lastLog := time.Now()
	var seed []float32
	for i, id := range ids {
		lg, err := m.ForwardForTest(id, cache)
		if err != nil {
			t.Fatalf("forward pos=%d: %v", i, err)
		}
		seed = append([]float32(nil), lg...)
		if time.Since(lastLog) > 20*time.Second {
			fmt.Printf("[ref]   ... K=%d at pos %d/%d\n", K, i+1, K)
			lastLog = time.Now()
		}
	}

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
			t.Fatalf("continuation forward step=%d: %v", i, err)
		}
		cur = append([]float32(nil), lg...)
	}
	return seed, refTokens, refLogits
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
