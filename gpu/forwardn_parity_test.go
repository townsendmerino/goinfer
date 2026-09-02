//go:build gpu && goinfer_testhooks

package gpu_test

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/gpu"
)

// TestResidentForwardN_parity is THE gate for the batched resident verify (Lever 2,
// Stage A): ForwardN(K embeddings, startPos) must be bit-for-bit equal to K sequential
// Forward calls at the same positions. They share the resident KV cache, so this
// specifically validates that the single command buffer's intra-pass barriers make each
// row's kv-store visible to the next row's attention (causal: row i sees [0, startPos+i]).
// Inputs are random [hidden] vectors — RMSNorm normalizes the input scale, so token
// lookup is unnecessary; the cross-row KV dependency is exercised regardless.
//
//	GOINFER_RESIDENT_GGUF=~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf \
//	  go test -tags gpu ./gpu/ -run TestResidentForwardN_parity -v
func TestResidentForwardN_parity(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1 to opt in (loads a multi-GB model from ~/models)")
	}
	if _, err := gpu.New(); err != nil {
		// A device that worked earlier in this process and is now gone means the test
		// binary exhausted it — skipping there is how this gate silently stopped gating.
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
	hidden, _, _, _, _, _, vocab := m.Dims()

	const K = 5
	rng := rand.New(rand.NewSource(1))
	embs := make([][]float32, K)
	for i := range embs {
		e := make([]float32, hidden)
		for j := range e {
			e[j] = float32(rng.NormFloat64()) * 0.5
		}
		embs[i] = e
	}

	// Batched: K rows in one command buffer, positions 0..K-1.
	rf.Reset()
	batch, err := rf.ForwardN(embs, 0)
	if err != nil {
		t.Fatalf("ForwardN: %v", err)
	}
	if len(batch) != K {
		t.Fatalf("ForwardN returned %d rows, want %d", len(batch), K)
	}

	// Sequential: same positions overwrite the same KV with identical values, so each
	// Forward(embs[i], i) must reproduce ForwardN's row i.
	rf.Reset()
	for i := range K {
		seq, err := rf.Forward(embs[i], i)
		if err != nil {
			t.Fatalf("Forward[%d]: %v", i, err)
		}
		if len(seq) != vocab || len(batch[i]) != vocab {
			t.Fatalf("row %d: len seq=%d batch=%d, want vocab=%d", i, len(seq), len(batch[i]), vocab)
		}
		var maxAbs, dot, nb, ns float64
		for j := range seq {
			d := math.Abs(float64(batch[i][j] - seq[j]))
			if d > maxAbs {
				maxAbs = d
			}
			dot += float64(batch[i][j]) * float64(seq[j])
			nb += float64(batch[i][j]) * float64(batch[i][j])
			ns += float64(seq[j]) * float64(seq[j])
		}
		cos := dot / (math.Sqrt(nb)*math.Sqrt(ns) + 1e-30)
		t.Logf("row %d: cosine=%.6f maxAbsDiff=%.4g", i, cos, maxAbs)
		if cos < 0.99999 || maxAbs > 1e-2 {
			t.Errorf("row %d: ForwardN != sequential Forward (cosine %.6f, maxAbsDiff %.4g)", i, cos, maxAbs)
		}
	}
}
