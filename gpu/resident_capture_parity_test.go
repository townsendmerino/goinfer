//go:build gpu && goinfer_testhooks

package gpu

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestResidentCaptureParityWebGPU runs docs/tasks/task-webgpu-nogqa-decode-bug.md's exact
// experiment (resident Forward vs CPU ForwardForTest at the same quant, the 8-token arbitrary
// prompt, then a greedy continuation) through the PRODUCTION load path — decoder.Load with
// Backend:"webgpu", so BuildResident's uploadProj fast path, any fused-tensor split the family
// does at load, residentDecoder.Forward and the generic CPU forward with aikit's kernels are all
// the real ones — on ANY checkpoint (GOINFER_PARITY_CKPT: a safetensors dir or a .gguf), and with
// GOINFER_GPU_CAPTURE=1 it differences the two forwards PER SUBLAYER PER LAYER (attention
// context, attention contribution, MLP contribution — the runner's capture buffers vs
// decoder.ForwardSubCaptureLogitsForTest) so a divergence is localised in one run instead of
// argued from the final logits. Prints, does not assert a bar: what a cosine means for a given
// checkpoint is what TestCPUQuantSensitivity (same file's sibling) measures for it.
//
// GOINFER_PARITY_QUANT: int4 (default), int8, int8int8, or native (CPU f32 vs the resident
// W8A8-at-upload path — NOT a same-quant comparison; see the task doc for what that shows).
// GOINFER_INT4_F16_SCALES=1 makes both sides carry the f16-rounded int4 group scales the GPU
// stores, which is the only single-variable form of the int4 comparison. Written against
// scripts/mk_phi3_synth.py's phi3-mini-shaped synthetic checkpoint; runs on the real one too.
func TestResidentCaptureParityWebGPU(t *testing.T) {
	dir := os.Getenv("GOINFER_PARITY_CKPT")
	if dir == "" {
		t.Skip("set GOINFER_PARITY_CKPT to a phi3-shaped safetensors dir")
	}
	if c, err := New(); err != nil {
		t.Skipf("no webgpu device: %v", err)
	} else {
		c.Close()
	}
	prompt := []int{1, 7, 42, 100, 5, 200, 13, 88}
	quant := os.Getenv("GOINFER_PARITY_QUANT") // "" ⇒ int4 (the doc's case); "native" for f32-vs-W8A8
	if quant == "" {
		quant = "int4"
	}
	cpuQuant, gpuQuant := quant, quant
	if quant == "native" { // Options.Quant "" is the f32 load; the resident path then W8A8-quantizes at upload
		cpuQuant, gpuQuant = "", ""
	}

	mg, err := decoder.Load(dir, decoder.Options{Backend: "webgpu", Quant: gpuQuant})
	if err != nil {
		t.Fatalf("load webgpu: %v", err)
	}
	defer mg.Close()
	rf := mg.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("did not go resident — decode path %q; decline: %s", mg.DecodePath(), mg.ResidentDecline())
	}
	t.Logf("resident decode path: %s", mg.DecodePath())

	mcpu, err := decoder.Load(dir, decoder.Options{Quant: cpuQuant})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mcpu.Close()
	t.Logf("cpu decode path: %s", mcpu.DecodePath())

	cache := mcpu.NewCache(len(prompt) + 16)
	minCos := 1.0
	var lastCPU, lastGPU []float32
	// GOINFER_GPU_CAPTURE=1: per-layer, per-sublayer differencing (CLAUDE.md: "prefer
	// differencing per layer over reasoning from final logits") — the resident runner's capture
	// buffers vs decoder.ForwardSubCaptureLogitsForTest, from the SAME token on each side.
	capture := os.Getenv("GOINFER_GPU_CAPTURE") != ""
	runner := rf.(*residentDecoder).runner
	for i, tok := range prompt {
		var cpuL []float32
		var cAttn, cMLP, cCtx [][]float32
		if capture {
			cpuL, cAttn, cMLP, cCtx, _, err = mcpu.ForwardSubCaptureLogitsForTest(tok, cache)
		} else {
			cpuL, err = mcpu.ForwardForTest(tok, cache)
		}
		if err != nil {
			t.Fatalf("cpu pos %d: %v", i, err)
		}
		x0 := mg.EmbedResidentForTest(tok)
		gpuL, err := rf.Forward(x0, i)
		if err != nil {
			t.Fatalf("resident forward pos %d: %v", i, err)
		}
		cos, maxAbs := cosine(cpuL, gpuL)
		if cos < minCos {
			minCos = cos
		}
		t.Logf("  prompt pos %2d tok %4d: cosine %.6f maxAbs %.3e argmax cpu=%d gpu=%d", i, tok, cos, maxAbs, argmaxF(cpuL), argmaxF(gpuL))
		if capture {
			gCtx, gAttn, gMLP, err := runner.ReadCapture()
			if err != nil {
				t.Fatalf("ReadCapture pos %d: %v", i, err)
			}
			// Reconstruct the CPU residual stream from the sublayer contributions: h_in(0) = the
			// embedding (identical on both sides), h_attn(l) = h_in(l) + attn(l), h_out(l) =
			// h_attn(l) + mlp(l). The GPU captures the residual itself; its contributions are
			// the differences. Both views are printed: the residual cosine shows inherited error,
			// the contribution cosine shows the sublayer's OWN error at that layer.
			hIn := append([]float32(nil), x0...)
			sub := func(a, b []float32) []float32 {
				d := make([]float32, len(a))
				for j := range d {
					d[j] = a[j] - b[j]
				}
				return d
			}
			add := func(a, b []float32) []float32 {
				d := make([]float32, len(a))
				for j := range d {
					d[j] = a[j] + b[j]
				}
				return d
			}
			for l := range gMLP {
				hAttnCPU := add(hIn, cAttn[l])
				hOutCPU := add(hAttnCPU, cMLP[l])
				rel := func(ref, got []float32) float64 { // ||got-ref|| / ||ref||
					var d, n float64
					for j := range ref {
						e := float64(got[j]) - float64(ref[j])
						d += e * e
						n += float64(ref[j]) * float64(ref[j])
					}
					return math.Sqrt(d) / (math.Sqrt(n) + 1e-30)
				}
				cCos, _ := cosine(cCtx[l], gCtx[l])
				aCos, _ := cosine(cAttn[l], sub(gAttn[l], hIn))
				hACos, _ := cosine(hAttnCPU, gAttn[l])
				mCos, _ := cosine(cMLP[l], sub(gMLP[l], gAttn[l]))
				hOCos, _ := cosine(hOutCPU, gMLP[l])
				t.Logf("      L%02d ctx cos %.8f rel %.2e | attn-contrib cos %.8f rel %.2e (resid %.8f) | mlp-contrib cos %.8f rel %.2e (resid %.8f)",
					l, cCos, rel(cCtx[l], gCtx[l]), aCos, rel(cAttn[l], sub(gAttn[l], hIn)), hACos, mCos, rel(cMLP[l], sub(gMLP[l], gAttn[l])), hOCos)
				hIn = gMLP[l] // next layer's input on the GPU side is its own residual; use it so each
				//              layer's contribution cosine isolates that layer's error, not inherited drift
				_ = hOutCPU
			}
		}
		lastCPU = append(lastCPU[:0], cpuL...)
		lastGPU = append(lastGPU[:0], gpuL...)
	}
	// Greedy continuation, each side fed its OWN argmax (the doc's protocol), plus the
	// teacher-forced cosine (both fed the CPU's token) so a first flip does not hide what follows.
	cpuTok, gpuTok := argmaxF(lastCPU), argmaxF(lastGPU)
	pos := len(prompt)
	for step := 0; step < 6; step++ {
		cpuL, err := mcpu.ForwardForTest(cpuTok, cache)
		if err != nil {
			t.Fatalf("cpu gen %d: %v", step, err)
		}
		gpuL, err := rf.Forward(mg.EmbedResidentForTest(gpuTok), pos)
		if err != nil {
			t.Fatalf("gpu gen %d: %v", step, err)
		}
		cos, _ := cosine(cpuL, gpuL)
		t.Logf("  greedy step %d: fed cpu=%d gpu=%d → cosine %.6f next cpu=%d gpu=%d", step, cpuTok, gpuTok, cos, argmaxF(cpuL), argmaxF(gpuL))
		cpuTok, gpuTok = argmaxF(cpuL), argmaxF(gpuL)
		pos++
	}
	t.Logf("phi3-synth (%s): min prompt cosine %.6f", quant, minCos)
	if math.IsNaN(minCos) {
		t.Errorf("NaN cosine")
	}
}
