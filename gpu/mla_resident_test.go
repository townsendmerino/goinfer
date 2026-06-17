//go:build gpu

package gpu_test

import (
	"context"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/gpu"
)

// TestMLAResidency_matchesCPU is the end-to-end bridge gate for Lever C4d: the tiny
// DeepSeek-V3 checkpoint (q-LoRA + compressed-KV latent attention + group-limited
// DeepSeekMoE + dense prefix + ungated shared expert) loaded on the webgpu backend must
// go GPU-resident (proving decodeRunnerEligible + BuildResident accept MLA) and decode
// greedily in agreement with the CPU f32 forward. The resident path runs int8, the CPU
// path f32, so a tiny-random model's logits can eventually flip an argmax under int8
// rounding — the gate is the FIRST generated token (single forward, minimal quant
// accumulation), with the full sequences logged. Numerical correctness of the MLA kernels
// themselves is pinned bit-identically by TestDecodeRunnerMLA_parity.
func TestMLAResidency_matchesCPU(t *testing.T) {
	const ckpt = "../testdata/deepseek-tiny"
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no checkpoint at %s — run scripts/pin_deepseek_tiny.py", ckpt)
	}
	if _, err := gpu.New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}

	ids := []int{1, 7, 3, 42, 9, 5} // arbitrary in-vocab (vocab 128) prompt ids
	const N = 8
	greedy := decoder.SamplingParams{Temperature: 0}

	gen := func(backend string, opts decoder.Options) ([]int, bool) {
		opts.Backend = backend
		m, err := decoder.Load(ckpt, opts)
		if err != nil {
			t.Fatalf("Load(%s): %v", backend, err)
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
		t.Fatal("deepseek-tiny did not go GPU-resident — MLA bridge/eligibility regressed")
	}
	t.Logf("MLA resident: cpu=%v webgpu=%v", cpuToks, gpuToks)
	if len(cpuToks) == 0 || len(gpuToks) == 0 {
		t.Fatalf("empty generation: cpu=%v gpu=%v", cpuToks, gpuToks)
	}
	if cpuToks[0] != gpuToks[0] {
		t.Errorf("MLA resident first-token mismatch: cpu=%d webgpu=%d", cpuToks[0], gpuToks[0])
	}
}
