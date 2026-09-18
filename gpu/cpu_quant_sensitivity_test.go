//go:build gpu && goinfer_testhooks

package gpu

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestCPUQuantSensitivity is the control for TestPhi3SynthResidentParityWebGPU: NO GPU at
// all. It runs the CPU int4 forward on two copies of the same checkpoint that differ in ONE
// f32 norm weight by one part in 2^20 (~1e-6 relative; GOINFER_PARITY_CKPT_B is that copy) and
// reports the logit cosine between them, per position. Whatever this prints is the floor a
// resident-vs-CPU comparison of this quantized forward can EVER reach: the W4A8/W8A8 path
// re-quantizes activations to int8 at every projection, and a perturbation far below one int8
// step flips a fraction of rounding decisions proportional to its size while each flip is a
// whole step — a square-root amplifier of tiny numerical differences, compounding per layer.
// A GPU that reproduces the CPU's f32 arithmetic only to ~1e-7 (reduction order) is such a
// perturbation. See docs/tasks/task-webgpu-nogqa-decode-bug.md.
func TestCPUQuantSensitivity(t *testing.T) {
	dirA, dirB := os.Getenv("GOINFER_PARITY_CKPT"), os.Getenv("GOINFER_PARITY_CKPT_B")
	if dirA == "" {
		t.Skip("set GOINFER_PARITY_CKPT (and optionally GOINFER_PARITY_CKPT_B, a hand-nudged copy; otherwise the second load uses the GOINFER_NORM_ULP_NOISE diagnostic)")
	}
	useDiag := dirB == ""
	if useDiag {
		dirB = dirA
	}
	quant := os.Getenv("GOINFER_PARITY_QUANT")
	if quant == "" {
		quant = "int4"
	}
	prompt := []int{1, 7, 42, 100, 5, 200, 13, 88}
	ma, err := decoder.Load(dirA, decoder.Options{Quant: quant})
	if err != nil {
		t.Fatal(err)
	}
	defer ma.Close()
	if useDiag { // dense ±1-ULP noise on every norm vector at load — decoder/normnoise.go
		os.Setenv("GOINFER_NORM_ULP_NOISE", "1")
	}
	mb, err := decoder.Load(dirB, decoder.Options{Quant: quant})
	if useDiag {
		os.Unsetenv("GOINFER_NORM_ULP_NOISE")
	}
	if err != nil {
		t.Fatal(err)
	}
	defer mb.Close()
	{ // how many norm elements actually differ between the two loads (sanity for the diagnostic)
		wa, wb := ma.Weights(), mb.Weights()
		diff := 0
		for i := range wa.Layers {
			for j := range wa.Layers[i].PreAttnNorm {
				if wa.Layers[i].PreAttnNorm[j] != wb.Layers[i].PreAttnNorm[j] {
					diff++
				}
			}
		}
		t.Logf("PreAttnNorm elements differing between the two loads: %d", diff)
	}
	ca, cb := ma.NewCache(len(prompt)+8), mb.NewCache(len(prompt)+8)
	minCos := 1.0
	rel := func(ref, got []float32) float64 { // ||got-ref|| / ||ref||
		var d, n float64
		for j := range ref {
			e := float64(got[j]) - float64(ref[j])
			d += e * e
			n += float64(ref[j]) * float64(ref[j])
		}
		return math.Sqrt(d) / (math.Sqrt(n) + 1e-30)
	}
	for i, tok := range prompt {
		la, aAttn, aMLP, aCtx, _, err := ma.ForwardSubCaptureLogitsForTest(tok, ca)
		if err != nil {
			t.Fatal(err)
		}
		lb, bAttn, bMLP, bCtx, _, err := mb.ForwardSubCaptureLogitsForTest(tok, cb)
		if err != nil {
			t.Fatal(err)
		}
		for l := range aAttn {
			t.Logf("      L%02d ctx rel %.2e | attn-contrib rel %.2e | mlp-contrib rel %.2e", l, rel(aCtx[l], bCtx[l]), rel(aAttn[l], bAttn[l]), rel(aMLP[l], bMLP[l]))
		}
		cos, maxAbs := cosine(la, lb)
		if cos < minCos {
			minCos = cos
		}
		t.Logf("  pos %2d: CPU-vs-CPU(±1-ULP norm noise) cosine %.6f maxAbs %.3e argmax %d/%d", i, cos, maxAbs, argmaxF(la), argmaxF(lb))
	}
	t.Logf("CPU self-sensitivity floor (%s): min cosine %.6f", quant, minCos)
}
