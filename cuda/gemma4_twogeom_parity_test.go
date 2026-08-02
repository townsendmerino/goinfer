//go:build cuda

package cuda

import (
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// promptIDs are pin_gemma4_dense_twogeom's fixed PROMPT — same for every variant.
var twoGeomPrompt = []int{1, 7, 42, 100, 5, 200, 13, 88}

// twoGeomParity drives the fixed prompt through the cuda resident runner (env on) and the CPU
// forward for the checkpoint in dir, returning per-position cosine/maxAbs (int4 both sides).
func twoGeomParity(t *testing.T, dir string) (minCos, maxMaxAbs float64, exact, n int, admitted bool) {
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture (%s) — run scripts/pin_gemma4_dense_twogeom.py or scripts/dbg_twogeom_variants.py", dir)
	}
	mc, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest()
	if rf == nil {
		return 0, 0, 0, len(twoGeomPrompt), false
	}
	mcpu, err := decoder.Load(dir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cpu): %v", err)
	}
	defer mcpu.Close()
	cache := mcpu.NewCache(len(twoGeomPrompt))
	minCos = 1.0
	for i, tok := range twoGeomPrompt {
		cpuL, err := mcpu.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu pos %d: %v", i, err)
		}
		gpuL, err := rf.Forward(mc.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("cuda pos %d: %v", i, err)
		}
		var dot, na, nb, maxAbs float64
		for j := range cpuL {
			if d := math.Abs(float64(cpuL[j]) - float64(gpuL[j])); d > maxAbs {
				maxAbs = d
			}
			dot += float64(cpuL[j]) * float64(gpuL[j])
			na += float64(cpuL[j]) * float64(cpuL[j])
			nb += float64(gpuL[j]) * float64(gpuL[j])
		}
		c := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30)
		if c < minCos {
			minCos = c
		}
		if maxAbs > maxMaxAbs {
			maxMaxAbs = maxAbs
		}
		if argmaxF(cpuL) == argmaxF(gpuL) {
			exact++
		}
		t.Logf("  pos %2d cosine %.6f maxAbs %.4e  argmax cpu=%d cuda=%d", i, c, maxAbs, argmaxF(cpuL), argmaxF(gpuL))
	}
	return minCos, maxMaxAbs, exact, len(twoGeomPrompt), true
}

// TestGemma4DenseTwoGeom_residentParity is the Split-A end-to-end gate (9a-P2 piece 3): the first
// two-live-variant launchToken run — local head_dim=16 / global head_dim=512 with K=V on the
// global layer — resident vs CPU at the 3% near-tie rule. strengthen()'s non-trivial norms
// (Phase 1c) are what make the V=v_norm(raw k) ordering observable here.
func TestGemma4DenseTwoGeom_residentParity(t *testing.T) {
	minCos, maxAbs, exact, n, ok := twoGeomParity(t, "../testdata/gemma4-dense-twogeom-tiny")
	if !ok {
		t.Fatal("cuda resident DECLINED dense Gemma 4 with env on — admission regressed")
	}
	t.Logf("two-geometry K=V resident parity: minCosine=%.6f maxAbs=%.4e exact-argmax %d/%d", minCos, maxAbs, exact, n)
	if minCos < 0.97 {
		t.Errorf("minCosine %.6f < 0.97 — the resident two-geometry/K=V forward diverges from CPU", minCos)
	}
}
