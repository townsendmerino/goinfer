//go:build gpu

package gpu

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestE2E_webgpu loads a real model on both the CPU and WebGPU backends and
// compares (1) greedy output — the GPU W8A8 path must track the CPU's tokens —
// and (2) prefill + decode timing. Gated on GOINFER_E2E_MODEL (a .gguf).
//
// Honest scope: only the NON-fused W8A8 matmuls (o_proj, down, lm_head, router)
// route to the GPU today; the fused qkv / gate-up batch dispatches stay on CPU
// (QuantBackend doesn't cover MatmulBTW8A8Batch yet). And each GPU matmul still
// pays one sync, which the Stage-2 finding showed is the decode floor. So this
// measures partial offload — the number that scopes the remaining work, not a
// finished decode win.
func TestE2E_webgpu(t *testing.T) {
	path := os.Getenv("GOINFER_E2E_MODEL")
	if path == "" {
		t.Skip("set GOINFER_E2E_MODEL=<.gguf> for the E2E GPU test")
	}
	load := func(backend string) *decoder.Model {
		m, err := decoder.Load(path, decoder.Options{Backend: backend, Quant: "int8int8"})
		if err != nil {
			t.Fatalf("load (%s): %v", backend, err)
		}
		return m
	}
	cpu := load("cpu")
	gpu := load("webgpu")

	gen := func(m *decoder.Model, prompt []int, maxTok int) ([]int, time.Duration) {
		t0 := time.Now()
		ch, _ := m.Generate(context.Background(), prompt, maxTok, decoder.SamplingParams{Temperature: 0})
		var toks []int
		for id := range ch {
			toks = append(toks, id)
		}
		return toks, time.Since(t0)
	}

	// Correctness: greedy tokens from the GPU path should track the CPU path
	// (tiny int8-rounding differences can eventually flip an argmax; a healthy
	// prefix must match, and a bug would diverge immediately).
	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8}
	cpuToks, _ := gen(cpu, prompt, 16)
	gpuToks, _ := gen(gpu, prompt, 16)
	match := 0
	for match < len(cpuToks) && match < len(gpuToks) && cpuToks[match] == gpuToks[match] {
		match++
	}
	t.Logf("greedy agreement: %d/%d tokens match\n  cpu=%v\n  gpu=%v", match, len(cpuToks), cpuToks, gpuToks)
	if match == 0 {
		t.Errorf("GPU diverges from CPU on the first token — wiring bug")
	}

	// Decode rate: short prompt, many tokens (decode-dominated).
	_, cpuDec := gen(cpu, []int{1, 2, 3, 4}, 48)
	_, gpuDec := gen(gpu, []int{1, 2, 3, 4}, 48)
	// Prefill / TTFT: long prompt, 1 token.
	long := make([]int, 256)
	for i := range long {
		long[i] = i%4000 + 1
	}
	_, cpuPre := gen(cpu, long, 1)
	_, gpuPre := gen(gpu, long, 1)

	t.Logf("decode 48 tok:  CPU %6.1f tok/s  |  GPU %6.1f tok/s  (%.2f×)",
		48/cpuDec.Seconds(), 48/gpuDec.Seconds(), float64(cpuDec)/float64(gpuDec))
	t.Logf("prefill 256 tok (TTFT):  CPU %6.1f ms  |  GPU %6.1f ms  (%.2f×)",
		ms(cpuPre), ms(gpuPre), float64(cpuPre)/float64(gpuPre))
}
