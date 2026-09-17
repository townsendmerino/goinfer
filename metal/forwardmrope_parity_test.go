//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

func cosSim(a, b []float32) (cos float64, maxAbs float32) {
	var dot, sa, sb float64
	for i := range a {
		ai, bi := float64(a[i]), float64(b[i])
		dot += ai * bi
		sa += ai * ai
		sb += bi * bi
		diff := float32(math.Abs(ai - bi))
		if diff > maxAbs {
			maxAbs = diff
		}
	}
	denom := math.Sqrt(sa) * math.Sqrt(sb)
	if denom == 0 {
		return 0, maxAbs
	}
	return dot / denom, maxAbs
}

func resolveMRoPETestModel(t *testing.T) string {
	t.Helper()
	candidates := []string{
		os.Getenv("GOINFER_QWEN15_CKPT"),
	}
	if os.Getenv("GOINFER_HEAVY_TESTS") != "" {
		candidates = append(candidates, modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"))
	}
	candidates = append(candidates, "../testdata/llama-tiny")
	for _, p := range candidates {
		if p == "" {
			continue
		}
		if f, err := os.Open(p); err == nil {
			f.Close()
			return p
		}
	}
	t.Skip("no accessible model fixture")
	return ""
}

// TestForwardMRoPE_constantShiftInvariance is ForwardMRoPE's (decoder.ResidentMRoPE) FIRST real
// correctness test on the Metal backend — the CUDA twin lives in cuda/forwardmrope_parity_test.go;
// the WebGPU twin lives in gpu/forwardmrope_parity_test.go.
//
// Relative-angle invariance: attention(q, k) depends only on the relative rotation angle between
// query and key (the mathematical foundation of RoPE). A decode run where EVERY step's rope angle
// is shifted by the same constant delta must produce identical output to delta=0 — the KV position
// (pos) is unchanged in both arms, only the rotary angle (ropePos = pos + delta) differs.
//
// ARM A (ordinary): Forward(emb_i, i) for i in [0, N) — delta=0.
// ARM B (shifted):  ForwardMRoPE(emb_i, i, i+delta) for the SAME embeddings — delta != 0.
func TestForwardMRoPE_constantShiftInvariance(t *testing.T) {
	path := resolveMRoPETestModel(t)

	for _, delta := range []int{17, -13} {
		t.Run(map[bool]string{true: "positive", false: "negative"}[delta > 0], func(t *testing.T) {
			mc, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4"})
			if err != nil {
				t.Fatalf("load metal: %v", err)
			}
			defer mc.Close()
			rf := mc.ResidentForwardForTest()
			if rf == nil {
				t.Skip("model not resident-eligible on this build")
			}
			mrope, ok := rf.(decoder.ResidentMRoPE)
			if !ok {
				t.Fatal("metalResident does not satisfy decoder.ResidentMRoPE — interface wiring broken")
			}
			_, _, _, _, _, _, vocab := mc.Dims()

			const N = 24
			rng := rand.New(rand.NewSource(11))
			embs := make([][]float32, N)
			for i := range embs {
				embs[i] = mc.EmbedResidentForTest(rng.Intn(vocab - 1))
			}

			// ARM A: ordinary, delta=0.
			var want []float32
			for i, e := range embs {
				l, err := rf.Forward(e, i)
				if err != nil {
					t.Fatalf("ordinary Forward step %d: %v", i, err)
				}
				want = l
			}

			rf.Reset()

			// ARM B: every step's rope angle shifted by the same constant delta.
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
				t.Errorf("delta=%d: cosine = %.8f, want ~1.0 — RoPE relative-angle invariance violated", delta, cos)
			}
		})
	}
}

// TestForwardMRoPE_equalArgsMatchesForward checks the interface contract:
// ForwardMRoPE(emb, pos, pos) must equal Forward(emb, pos) exactly.
func TestForwardMRoPE_equalArgsMatchesForward(t *testing.T) {
	path := resolveMRoPETestModel(t)

	mc, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest()
	if rf == nil {
		t.Skip("model not resident-eligible on this build")
	}
	mrope, ok := rf.(decoder.ResidentMRoPE)
	if !ok {
		t.Fatal("metalResident does not satisfy decoder.ResidentMRoPE")
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
		l, err := mrope.ForwardMRoPE(e, i, i)
		if err != nil {
			t.Fatalf("ForwardMRoPE step %d: %v", i, err)
		}
		got = l
	}
	if len(want) != len(got) {
		t.Fatalf("length mismatch: want %d, got %d", len(want), len(got))
	}
	for i := range want {
		if want[i] != got[i] {
			t.Fatalf("logit %d: Forward=%v ForwardMRoPE(pos,pos)=%v — must be bit-identical", i, want[i], got[i])
		}
	}
}
