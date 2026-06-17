//go:build gpu

package gpu_test

import (
	"context"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/gpu"
)

// TestGLMResidency_matchesCPU is the end-to-end gate for Lever C5 (partial RoPE): the
// tiny GLM-4.5 checkpoints (glm4_moe — sigmoid routing + selection bias + ungated shared
// expert + dense prefix + partial_rotary_factor 0.5) must go GPU-resident — proving the
// partial-RoPE eligibility relaxation + the rotaryDim/2 rope dispatch — and decode
// greedily in agreement with the CPU f32 forward. Two fixtures: plain, and with q/k/v
// bias + qk-norm. The resident path runs int8, the CPU path f32, so a tiny-random model's
// logits eventually flip an argmax under int8 rounding; the gate is the FIRST generated
// token, with full sequences logged. Before C5, GLM was off the resident path (partial
// rotary tripped the RotaryDim==HeadDim check) — stranding the C3d shared-expert support.
func TestGLMResidency_matchesCPU(t *testing.T) {
	if _, err := gpu.New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	for _, ckpt := range []string{"../testdata/glm-tiny", "../testdata/glm-tiny-bias"} {
		t.Run(ckpt, func(t *testing.T) {
			if _, err := os.Stat(ckpt); err != nil {
				t.Skipf("no checkpoint at %s", ckpt)
			}
			ids := []int{1, 5, 9, 2, 7}
			const N = 8
			greedy := decoder.SamplingParams{Temperature: 0}

			gen := func(backend string, opts decoder.Options) ([]int, bool) {
				opts.Backend = backend
				m, err := decoder.Load(ckpt, opts)
				if err != nil {
					t.Fatalf("Load(%s, %s): %v", ckpt, backend, err)
				}
				defer m.Close()
				resident := m.ResidentActive()
				ch, _ := m.Generate(context.Background(), ids, N, greedy)
				var toks []int
				for id := range ch {
					toks = append(toks, id)
				}
				return toks, resident
			}

			cpuToks, _ := gen("", decoder.Options{})
			gpuToks, resident := gen("webgpu", decoder.Options{Quant: "int8int8"})
			if !resident {
				t.Fatal("glm-tiny did not go GPU-resident — partial-RoPE eligibility regressed")
			}
			t.Logf("GLM resident: cpu=%v webgpu=%v", cpuToks, gpuToks)
			if len(cpuToks) == 0 || len(gpuToks) == 0 {
				t.Fatalf("empty generation: cpu=%v gpu=%v", cpuToks, gpuToks)
			}
			if cpuToks[0] != gpuToks[0] {
				t.Errorf("GLM resident first-token mismatch: cpu=%d webgpu=%d", cpuToks[0], gpuToks[0])
			}
		})
	}
}
