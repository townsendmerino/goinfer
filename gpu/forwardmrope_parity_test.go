//go:build gpu && goinfer_testhooks

package gpu

import (
	"math/rand"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestForwardMRoPE_constantShiftInvariance is ForwardMRoPE's FIRST real correctness test on the
// WebGPU backend — the CUDA twin lives in cuda/forwardmrope_parity_test.go; see that file's doc
// comment for the full rationale (RoPE's relative-angle invariance: a decode run where every
// step's rope angle is shifted by the same constant delta must match delta=0 exactly, a property
// independent of any specific family's m-RoPE usage).
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'gpu goinfer_testhooks' ./gpu/ -run TestForwardMRoPE_constantShiftInvariance -v -timeout 20m
func TestForwardMRoPE_constantShiftInvariance(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1")
	}
	if _, err := New(); err != nil {
		t.Skipf("no webgpu: %v", err)
	}
	home, _ := os.UserHomeDir()
	path := os.Getenv("GOINFER_QWEN15_CKPT")
	if path == "" {
		path = home + "/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s: %v", path, err)
	}

	for _, delta := range []int{17, -13} {
		t.Run(map[bool]string{true: "positive", false: "negative"}[delta > 0], func(t *testing.T) {
			mc, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer mc.Close()
			rf := mc.ResidentForwardForTest()
			if rf == nil {
				t.Skip("model not resident-eligible on this build")
			}
			mrope, ok := rf.(decoder.ResidentMRoPE)
			if !ok {
				t.Fatal("residentDecoder does not satisfy decoder.ResidentMRoPE — interface wiring broken")
			}
			_, _, _, _, _, _, vocab := mc.Dims()

			const N = 24
			rng := rand.New(rand.NewSource(11))
			embs := make([][]float32, N)
			for i := range embs {
				embs[i] = mc.EmbedResidentForTest(rng.Intn(vocab - 1))
			}

			var want []float32
			for i, e := range embs {
				l, err := rf.Forward(e, i)
				if err != nil {
					t.Fatalf("ordinary Forward step %d: %v", i, err)
				}
				want = l
			}

			rf.Reset()

			var got []float32
			for i, e := range embs {
				l, err := mrope.ForwardMRoPE(e, i, i+delta)
				if err != nil {
					t.Fatalf("ForwardMRoPE step %d (ropePos=%d): %v", i, i+delta, err)
				}
				got = l
			}

			cos, maxAbs := cosSim(want, got)
			t.Logf("delta=%d: cosine(ordinary, constant-shifted) = %.8f maxAbs=%.4g", delta, cos, maxAbs)
			if cos < 0.9999 {
				t.Errorf("delta=%d: cosine = %.8f, want ~1.0 — RoPE's relative-angle invariance is violated, ForwardMRoPE's ropePos wiring is likely wrong", delta, cos)
			}
		})
	}
}

// TestForwardMRoPE_equalArgsMatchesForward checks the interface's own stated contract
// (decoder/residency.go: "ForwardMRoPE(emb, pos, pos) must equal Forward(emb, pos) exactly").
func TestForwardMRoPE_equalArgsMatchesForward(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1")
	}
	if _, err := New(); err != nil {
		t.Skipf("no webgpu: %v", err)
	}
	home, _ := os.UserHomeDir()
	path := os.Getenv("GOINFER_QWEN15_CKPT")
	if path == "" {
		path = home + "/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s: %v", path, err)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest()
	if rf == nil {
		t.Skip("model not resident-eligible on this build")
	}
	mrope, ok := rf.(decoder.ResidentMRoPE)
	if !ok {
		t.Fatal("residentDecoder does not satisfy decoder.ResidentMRoPE")
	}
	_, _, _, _, _, _, vocab := mc.Dims()
	rng := rand.New(rand.NewSource(3))
	const N = 8
	embs := make([][]float32, N)
	for i := range embs {
		embs[i] = mc.EmbedResidentForTest(rng.Intn(vocab - 1))
	}

	var want []float32
	for i, e := range embs {
		l, err := rf.Forward(e, i)
		if err != nil {
			t.Fatalf("Forward step %d: %v", i, err)
		}
		want = l
	}
	rf.Reset()
	var got []float32
	for i, e := range embs {
		l, err := mrope.ForwardMRoPE(e, i, i) // ropePos == pos
		if err != nil {
			t.Fatalf("ForwardMRoPE step %d: %v", i, err)
		}
		got = l
	}
	if len(want) != len(got) {
		t.Fatalf("length %d vs %d", len(want), len(got))
	}
	for i := range want {
		if want[i] != got[i] {
			t.Fatalf("logit %d: Forward=%v ForwardMRoPE(pos,pos)=%v — must be bit-identical, same dispatch", i, want[i], got[i])
		}
	}
}
