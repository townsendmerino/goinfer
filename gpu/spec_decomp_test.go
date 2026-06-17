//go:build gpu

package gpu_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/gpu"
)

// TestSpeculativeResident_decomp decomposes the per-round cost so the kill-gate
// verdict points at the right culprit: it times (a) the CPU draft's standalone
// tokens/sec, (b) the GPU target's plain resident tokens/sec, and (c) a single
// target ForwardN(K+1) verify pass. With a GPU target and a CPU draft, if the draft
// is slower per token than the target, speculation is a net loss independent of how
// cheap the verify is — which Stage B (M=K verify) cannot fix.
func TestSpeculativeResident_decomp(t *testing.T) {
	if testing.Short() {
		t.Skip("decomposition: skipped in -short")
	}
	if _, err := gpu.New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	home, _ := os.UserHomeDir()
	tpath := os.Getenv("GOINFER_SPEC_TARGET")
	if tpath == "" {
		tpath = filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	dpath := os.Getenv("GOINFER_SPEC_DRAFT")
	if dpath == "" {
		dpath = filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	}
	for _, p := range []string{tpath, dpath} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("missing model %s: %v", p, err)
		}
	}

	ctx := context.Background()
	greedy := decoder.SamplingParams{Temperature: 0}
	prompt := []int{750, 4271, 4427, 9669, 982, 262, 421, 2422, 9669}

	// msPerTok loads a model with the given options and returns its standalone greedy
	// ms/token (warm-up DRAINED first — an undrained Generate goroutine keeps issuing
	// GPU submits and races the timed run on the non-thread-safe wgpu queue → crash).
	// Only one model is exercised at a time so two resident models never contend.
	drain := func(ch <-chan int) int {
		n := 0
		for range ch {
			n++
		}
		return n
	}
	msPerTok := func(path string, opts decoder.Options, label string) (float64, bool) {
		m, err := decoder.Load(path, opts)
		if err != nil {
			t.Fatalf("load %s: %v", label, err)
		}
		defer m.Close()
		if opts.Backend == "webgpu" && !m.ResidentActive() {
			t.Logf("%s: not GPU-resident (ineligible) — skipped", label)
			return 0, false
		}
		w, _ := m.Generate(ctx, prompt, 16, greedy)
		drain(w) // warm-up, fully drained
		t0 := time.Now()
		ch, _ := m.Generate(ctx, prompt, 64, greedy)
		n := drain(ch)
		ms := time.Since(t0).Seconds() * 1000 / float64(n)
		t.Logf("%s: %.1f ms/tok (%.2f tok/s)", label, ms, 1000/ms)
		return ms, true
	}

	cpuOpt := decoder.Options{Quant: "int8int8"}
	gpuOpt := decoder.Options{Backend: "webgpu", Quant: "int8int8"}

	draftCPU, _ := msPerTok(dpath, cpuOpt, "(a) CPU draft  (0.5B)")
	targetGPU, _ := msPerTok(tpath, gpuOpt, "(b) GPU target (1.5B)")
	draftGPU, gpuDraftOK := msPerTok(dpath, gpuOpt, "(c) GPU draft  (0.5B)")

	// Spec ceiling: per round = K·draft + verify; ideal Stage-B verify ≈ 1 target
	// token. speedup ≈ (tokens/round) / (round_ms / target_ms). Use a representative
	// tokens/round ≈ 3.3 at K=4 (measured acceptance 0.58).
	const K, tpr = 4, 3.3
	ceiling := func(draftMs float64) float64 {
		roundMs := K*draftMs + targetGPU // ideal verify == one target token
		return tpr / (roundMs / targetGPU)
	}
	t.Logf("draft/target cost ratio: CPU draft %.2f× target  (must be < ~1/K for spec to win)", draftCPU/targetGPU)
	t.Logf("ideal-Stage-B spec ceiling, CPU draft: %.2f× plain  (round=%.0fms vs %.0fms to decode %.1f tok directly)",
		ceiling(draftCPU), K*draftCPU+targetGPU, tpr*targetGPU, tpr)
	if gpuDraftOK {
		t.Logf("draft/target cost ratio: GPU draft %.2f× target", draftGPU/targetGPU)
		t.Logf("ideal-Stage-B spec ceiling, GPU draft: %.2f× plain", ceiling(draftGPU))
	}
}
