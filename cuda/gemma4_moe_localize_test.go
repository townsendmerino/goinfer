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

// TestGemma4MoE_localize is the localization harness the task-2c steer asked for BEFORE the gate:
// at pos 0 it diffs the four gemma4-MoE-layer buffers (rn / wgt / x1 / x2) resident-vs-CPU, so a
// whole-forward miss points at router vs dense branch vs expert branch (and the join, by elimination)
// in one run instead of a cosine that only says "wiring". Debug at pos 0 (smallest error), per the
// steer. Diagnostic: logs, never fails (the gate is TestGemma4MoE_residentParity).
func TestGemma4MoE_localize(t *testing.T) {
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	t.Setenv("GOINFER_G4_CAPTURE", "1")
	const dir = "../testdata/gemma4-moe-tiny"
	if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture (%s)", dir)
	}
	mc, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("resident declined")
	}
	r, ok := rf.(*cudaResident)
	if !ok {
		t.Fatalf("resident is %T, not *cudaResident", rf)
	}
	mcpu, err := decoder.Load(dir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mcpu.Close()

	decoder.SetRouterCaptureForTest(true)
	defer decoder.SetRouterCaptureForTest(false)
	tok := 1 // prompt[0]
	if _, err := mcpu.ForwardForTest(tok, mcpu.NewCache(8)); err != nil {
		t.Fatalf("cpu forward: %v", err)
	}
	if _, err := rf.Forward(mc.EmbedResidentForTest(tok), 0); err != nil {
		t.Fatalf("resident forward: %v", err)
	}

	_, cpuRn := decoder.RouterCaptureForTest()
	cpuWts, cpuX1, cpuX2 := decoder.Gemma4MoECaptureForTest()
	// resident captures are indexed by layer; collect the MoE layers in order.
	var resRn, resWgt, resX1, resX2 [][]float32
	for l := range r.g4capRn {
		if r.g4capRn[l] != nil {
			resRn = append(resRn, r.g4capRn[l])
			resWgt = append(resWgt, r.g4capWgt[l])
			resX1 = append(resX1, r.g4capX1[l])
			resX2 = append(resX2, r.g4capX2[l])
		}
	}
	n := len(cpuRn)
	if len(resRn) != n {
		t.Fatalf("MoE-decision count mismatch: cpu %d vs resident %d", n, len(resRn))
	}
	// rn is NOT asserted: the resident buffer is the UNSCALED rmsnorm_nw(h) (routerScale·hidden^-0.5 is
	// folded into RouterProjScaled), while CPU captures the SCALED rn — different vectors by design.
	// The router OUTPUT (wgt) matching subsumes it: if the fold were wrong, wgt would diverge.
	for k := 0; k < n; k++ {
		c, m := cosMaxAbs(cpuRn[k], resRn[k])
		t.Logf("  rn   decision %d: cosine %.6f maxAbs %.4e (UNSCALED vs scaled — not gated; wgt subsumes it)", k, c, m)
	}
	// wgt / x1 / x2 ARE the localized kernel gate: router, dense branch, expert branch, each vs CPU.
	assertBranch := func(name string, cpu, res [][]float32, floor float64) {
		for k := 0; k < n; k++ {
			c, m := cosMaxAbs(cpu[k], res[k])
			t.Logf("  %-4s decision %d: cosine %.6f maxAbs %.4e", name, k, c, m)
			if c < floor {
				t.Errorf("%s decision %d cosine %.6f < %.2f — the resident %s branch diverges from CPU", name, k, c, floor, name)
			}
		}
	}
	assertBranch("wgt", cpuWts, resWgt, 0.99)
	assertBranch("x1", cpuX1, resX1, 0.99)
	assertBranch("x2", cpuX2, resX2, 0.99)
}

func cosMaxAbs(a, b []float32) (cos, maxAbs float64) {
	var dot, na, nb float64
	for i := range a {
		if i >= len(b) {
			break
		}
		if d := math.Abs(float64(a[i]) - float64(b[i])); d > maxAbs {
			maxAbs = d
		}
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30), maxAbs
}
