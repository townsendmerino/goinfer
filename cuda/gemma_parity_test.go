//go:build cuda

package cuda

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemma3ResidentParity is the real-checkpoint gate for Gemma 3 on CUDA: the CUDA resident
// forward must agree with the CPU forward on a real gemma-3-4b-it, under the repo's 3%
// near-tie rule (gpu/kv_i8_parity_test.go). Gemma exercises five features no other CUDA-
// admitted family does: (1+w) RMS, sandwich norms, GeGLU, embed scale, per-layer RoPE base.
func TestGemma3ResidentParity(t *testing.T) {
	residentCosineParity(t, os.ExpandEnv("$HOME/models/gemma-3-4b-it"),
		[]int{2, 651, 6037, 529, 6081, 603, 12545, 235265, 714, 6398})
}

// TestDenseResidentParity is the CONTROL for the Gemma gate: the same harness on the
// known-good dense Qwen path, so "is 0.999 good?" has an answer measured on this box rather
// than assumed. Without it, a Gemma cosine cannot be judged.
func TestDenseResidentParity(t *testing.T) {
	residentCosineParity(t, os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
		[]int{785, 3840, 315, 6137, 448, 264, 1467, 315, 264, 3070})
}

func residentCosineParity(t *testing.T, path string, prompt []int) {
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("cuda resident DECLINED — admission says it should be admitted")
	}
	t.Logf("admitted; required features = %v", mc.RequiredResidentFeatures())

	mcpu, err := decoder.Load(path, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cpu): %v", err)
	}
	defer mcpu.Close()

	_, _, _, _, _, _, vocab := mc.Dims()
	minCos := 1.0
	cache := mcpu.NewCache(len(prompt) + 2)
	exact, hard := 0, 0
	worst := 0.0
	for i, tok := range prompt {
		cpuL, err := mcpu.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu pos %d: %v", i, err)
		}
		gpuL, err := rf.Forward(mc.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("cuda pos %d: %v", i, err)
		}
		// Cosine is the real signal here: argmax over a 262k vocab is a coarse, high-variance
		// statistic, so a low exact-match rate can be pure near-tie churn rather than a bug.
		var dot, na, nb float64
		for j := range cpuL {
			dot += float64(cpuL[j]) * float64(gpuL[j])
			na += float64(cpuL[j]) * float64(cpuL[j])
			nb += float64(gpuL[j]) * float64(gpuL[j])
		}
		c := dot / (math.Sqrt(na) * math.Sqrt(nb))
		if c < minCos {
			minCos = c
		}
		t.Logf("  pos %2d cosine %.6f", i, c)
		ca, ga := argmaxF(cpuL), argmaxF(gpuL)
		if ca == ga {
			exact++
			continue
		}
		lo, hi := cpuL[0], cpuL[0]
		for _, v := range cpuL {
			if v < lo {
				lo = v
			}
			if v > hi {
				hi = v
			}
		}
		gap := float64(cpuL[ca]-cpuL[ga]) / (float64(hi-lo) + 1e-9)
		if gap > worst {
			worst = gap
		}
		if gap > 0.03 {
			hard++
			t.Errorf("pos %d: CPU=%d CUDA=%d gap=%.3f%% > 3%% (vocab %d)", i, ca, ga, gap*100, vocab)
		}
	}
	t.Logf("CUDA vs CPU: %d/%d exact | worst near-tie %.3f%% | hard fails %d | min cosine %.6f",
		exact, len(prompt), worst*100, hard, minCos)
	// The GATE is the repo's own rule (gpu/kv_i8_parity_test.go): argmax must match, or differ
	// only inside a 3% near-tie — asserted per position above. Cosine is logged as a DIAGNOSTIC,
	// with only a gross-breakage floor: an early draft of this test asserted cosine ≥ 0.999 and
	// that bar failed the SHIPPED dense Qwen path (min 0.9936), which is why the control below
	// exists. W4A8 int4 does not reproduce CPU int4 to 0.999; a tighter floor here would encode a
	// number no backend meets.
	if minCos < 0.95 {
		t.Errorf("logit cosine %.6f < 0.95 — far below the dense control (~0.99); that is gross breakage, not int4 noise", minCos)
	}
}
