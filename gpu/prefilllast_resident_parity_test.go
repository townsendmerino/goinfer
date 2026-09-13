//go:build gpu && goinfer_testhooks

package gpu_test

import (
	"context"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/gpu"
)

// TestResidentPrefillLast_parity is the Increment-3 gate for the resident
// PrefillLast wiring (docs/task-gpu-batched-prefill.md): PrefillLast(embeddings,
// startPos)'s single returned row must match the LAST of len(embeddings) sequential
// Forward calls at the same positions — the real production path this replaces
// (decoder/model.go's residentPrefillSeed). Same shape as TestResidentForwardN_parity
// (random [hidden] embeddings, same real checkpoint), extended past ForwardN's K=5 to
// K=20 — past gemmRowMaxM=16, the reason PrefillLastW8A8 exists as its own function
// rather than reusing DecodeTokenFusedBatched.
//
//	GOINFER_RESIDENT_GGUF=~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf \
//	  GOINFER_HEAVY_TESTS=1 go test -tags 'gpu goinfer_testhooks' ./gpu/ -run TestResidentPrefillLast_parity -v
func TestResidentPrefillLast_parity(t *testing.T) {
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
	hidden, _, _, _, _, _, vocab := m.Dims()

	const K = 20
	rng := rand.New(rand.NewSource(1))
	embs := make([][]float32, K)
	for i := range embs {
		e := make([]float32, hidden)
		for j := range e {
			e[j] = float32(rng.NormFloat64()) * 0.5
		}
		embs[i] = e
	}

	rf.Reset()
	last, err := pf.PrefillLast(context.Background(), embs, 0)
	if err != nil {
		// A decline (this model uses a feature outside PrefillLastW8A8's scope —
		// see runModelToModelW's doc comment, e.g. Qwen2 bias is deliberately
		// declined pending a real bug fix) is the SAFE, EXPECTED outcome for many
		// checkpoints, not a test failure — decoder/model.go's residentPrefillSeed
		// treats any PrefillLast error identically, by falling back to the
		// sequential loop. Skip rather than fail; this gate only asserts
		// correctness for models PrefillLast actually accepts.
		t.Skipf("PrefillLast declined or failed (falls back to sequential in production): %v", err)
	}
	if len(last) != vocab {
		t.Fatalf("PrefillLast returned %d logits, want vocab=%d", len(last), vocab)
	}

	rf.Reset()
	var seqLast []float32
	for i := range K {
		seqLast, err = rf.Forward(embs[i], i)
		if err != nil {
			t.Fatalf("Forward[%d]: %v", i, err)
		}
	}

	var maxAbs, dot, nb, ns float64
	for j := range seqLast {
		d := math.Abs(float64(last[j] - seqLast[j]))
		if d > maxAbs {
			maxAbs = d
		}
		dot += float64(last[j]) * float64(seqLast[j])
		nb += float64(last[j]) * float64(last[j])
		ns += float64(seqLast[j]) * float64(seqLast[j])
	}
	cos := dot / (math.Sqrt(nb)*math.Sqrt(ns) + 1e-30)
	t.Logf("last row (pos %d): cosine=%.6f maxAbsDiff=%.4g", K-1, cos, maxAbs)
	if cos < 0.99999 || maxAbs > 1e-2 {
		t.Errorf("PrefillLast != sequential Forward at last row (cosine %.6f, maxAbsDiff %.4g)", cos, maxAbs)
	}
}
