//go:build gpu && goinfer_testhooks

package gpu_test

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/gpu"
)

// TestResidentPrefillLast_TTFT is the real end-to-end number the whole
// task-gpu-batched-prefill.md build was for: sequential per-token Forward (today's
// shipped residentPrefillSeed loop) vs one PrefillLast call, at realistic prompt
// lengths, on the actual resident decode pipeline (not an isolated matmul
// microbenchmark like TestTiledDP4A_microbench — that one only measured the
// projection GEMM in isolation and found DP4A's own contribution modest; this is
// the trustworthy comparison for whether batching prefill is worth shipping).
//
//	GOINFER_RESIDENT_GGUF=~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf \
//	  GOINFER_HEAVY_TESTS=1 go test -tags 'gpu goinfer_testhooks' ./gpu/ -run TestResidentPrefillLast_TTFT -v -timeout 20m
func TestResidentPrefillLast_TTFT(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1 to opt in (loads a multi-GB model from ~/models)")
	}
	if _, err := gpu.New(); err != nil {
		if gpu.GPUEverAvailable() {
			t.Fatalf("GPU exhausted by this test binary (not a missing GPU) — this gate must not skip: %v", err)
		}
		t.Skipf("no WebGPU adapter: %v", err)
	}
	home, _ := os.UserHomeDir()
	path := os.Getenv("GOINFER_RESIDENT_GGUF")
	if path == "" {
		path = filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no resident model at %s: %v", path, err)
	}

	m, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load %s: %v", path, err)
	}
	defer m.Close()
	if !m.ResidentActive() {
		t.Skip("model not GPU-resident (ineligible arch / no residency)")
	}
	rf := m.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("ResidentActive but no resident forward")
	}
	pf, ok := rf.(decoder.Prefiller)
	if !ok {
		t.Skip("resident forward does not implement Prefiller (arch outside PrefillLastW8A8's dense-W8A8 scope)")
	}
	cap := 0
	if capper, ok := rf.(interface{ ContextCap() int }); ok {
		cap = capper.ContextCap()
	}
	hidden, _, _, _, _, _, _ := m.Dims()
	rng := rand.New(rand.NewSource(1))
	emb := func() []float32 {
		e := make([]float32, hidden)
		for j := range e {
			e[j] = float32(rng.NormFloat64()) * 0.5
		}
		return e
	}

	fmt.Printf("\nPREFILL TTFT (webgpu, int8int8) — %s\n", path)
	fmt.Printf("%-6s %-16s %-16s %-10s\n", "P", "sequential_ms", "batched_ms", "speedup")
	for _, p := range []int{64, 256, 1024} {
		if cap > 0 && p >= cap {
			t.Logf("P=%d skipped: exceeds this model's resident context cap %d", p, cap)
			continue
		}
		embs := make([][]float32, p)
		for i := range embs {
			embs[i] = emb()
		}

		// SEQUENTIAL — today's shipped default (decoder/model.go's residentPrefillSeed
		// per-token loop): one Forward call per prompt token, from an empty cache.
		rf.Reset()
		t0 := time.Now()
		for i := range p {
			if _, e := rf.Forward(embs[i], i); e != nil {
				t.Fatalf("sequential P=%d pos=%d: %v", p, i, e)
			}
		}
		seq := time.Since(t0)

		// BATCHED — one PrefillLast call over the same P embeddings, same positions.
		rf.Reset()
		t1 := time.Now()
		if _, e := pf.PrefillLast(context.Background(), embs, 0); e != nil {
			// See prefilllast_resident_parity_test.go's identical skip: a decline
			// (out-of-scope model, e.g. Qwen2 bias — runModelToModelW's doc
			// comment) is not a bug here, just means this checkpoint has no TTFT
			// number to report from this path.
			t.Skipf("PrefillLast declined or failed at P=%d (falls back to sequential in production): %v", p, e)
		}
		batched := time.Since(t1)

		seqMs := float64(seq.Microseconds()) / 1000
		batchedMs := float64(batched.Microseconds()) / 1000
		speedup := seqMs / batchedMs
		fmt.Printf("%-6d %-16.2f %-16.2f %-10.2fx\n", p, seqMs, batchedMs, speedup)
		t.Logf("P=%d sequential=%.2fms batched=%.2fms speedup=%.2fx", p, seqMs, batchedMs, speedup)
	}
}
