//go:build gpu && goinfer_testhooks

package webgpu

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestCPUQuantSensitivity is the control for TestResidentCaptureParityWebGPU: no GPU at all. It runs the CPU int4 forward
// on two loads of the same checkpoint whose f32 norms differ by about one part in 2^20 (GOINFER_PARITY_CKPT_B, a
// hand-nudged copy differing in ONE norm weight; or, when unset, the GOINFER_NORM_ULP_NOISE load diagnostic) and logs the
// logit cosine between them per position. It asserts nothing: what it logs is the floor a resident-vs-CPU comparison of a
// quantized forward can reach, because the W4A8/W8A8 path re-quantizes activations to int8 at every projection and a
// perturbation far below one int8 step flips a fraction of rounding decisions, each flip a whole step, compounding per
// layer. A GPU that reproduces the CPU's f32 arithmetic only to ~1e-7 (reduction order) is such a perturbation. See
// docs/code-notes/gpu.md#webgpuBackend.BuildResident.nogqa.
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

// TestCPUQuantSensitivityREADME is TestCPUQuantSensitivity on TestPairwiseRoPERealWebGPU's own sequence: README.md's first 64
// tokens and a 16-token greedy continuation the clean CPU load chooses, both loads teacher-forced on it. It logs the
// per-position logit cosine between the clean load and the ±1-ULP-norm-noise load, their mean and worst, and every position
// under 0.95: the checkpoint's own floor on the prompt where the resident's worst position is read. It asserts nothing.
//
//	GOINFER_PARITY_CKPT=~/models/command-r7b go test -tags 'gpu goinfer_testhooks' ./backend/webgpu -run TestCPUQuantSensitivityREADME -v
func TestCPUQuantSensitivityREADME(t *testing.T) {
	dir := os.Getenv("GOINFER_PARITY_CKPT")
	if dir == "" {
		t.Skip("set GOINFER_PARITY_CKPT to a checkpoint directory with a tokenizer.json")
	}
	quant := os.Getenv("GOINFER_PARITY_QUANT")
	if quant == "" {
		quant = "int4"
	}
	readme, err := os.ReadFile("../../README.md")
	if err != nil {
		t.Fatal(err)
	}
	tk, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	const nPrompt, nCont = 64, 16
	ids, err := tk.Encode(string(readme), false)
	if err != nil || len(ids) < nPrompt {
		t.Fatalf("encode README: %d ids, %v", len(ids), err)
	}
	toks := append([]int(nil), ids[:nPrompt]...)
	ma, err := decoder.Load(dir, decoder.Options{Quant: quant})
	if err != nil {
		t.Fatal(err)
	}
	defer ma.Close()
	ca := ma.NewCache(nPrompt + nCont)
	var ref [][]float32
	for i := 0; i < nPrompt+nCont; i++ {
		l, err := ma.ForwardForTest(toks[i], ca)
		if err != nil {
			t.Fatal(err)
		}
		ref = append(ref, append([]float32(nil), l...))
		if i >= nPrompt-1 && len(toks) < nPrompt+nCont {
			toks = append(toks, argmaxF(l))
		}
	}
	t.Setenv("GOINFER_NORM_ULP_NOISE", "1") // dense ±1-ULP noise on every norm vector at load — decoder/normnoise.go
	mb, err := decoder.Load(dir, decoder.Options{Quant: quant})
	os.Unsetenv("GOINFER_NORM_ULP_NOISE") // the clean arm is loaded; t.Setenv restores the original at cleanup
	if err != nil {
		t.Fatal(err)
	}
	defer mb.Close()
	cb := mb.NewCache(nPrompt + nCont)
	var mean float64
	worst, wp := 1.0, -1
	for i, tok := range toks {
		l, err := mb.ForwardForTest(tok, cb)
		if err != nil {
			t.Fatal(err)
		}
		cos, _ := cosine(ref[i], l)
		if math.IsNaN(cos) {
			cos = -2
		}
		mean += cos
		if cos < worst {
			worst, wp = cos, i
		}
		if cos < 0.95 {
			t.Logf("  position %d (%s): CPU-vs-CPU cosine %.6f", i, map[bool]string{true: "prompt", false: "continuation"}[i < nPrompt], cos)
		}
	}
	t.Logf("CPU self-sensitivity on the README sequence (%s, %d positions): mean cosine %.6f, worst %.6f at position %d", quant, len(toks), mean/float64(len(toks)), worst, wp)
}
