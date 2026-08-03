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

// TestStagedGPU_nonDense is Lever A / B14 (docs/task-benchmark-refresh.md): the honest
// staged-path GPU decode number for residency-INELIGIBLE families (MoE / Mamba-2 hybrid
// / MLA). These never enter the resident DecodeRunner, but under -tags gpu their int8
// (W8A8) matmuls still dispatch to the GPU backend (decoder/weightmat.go matmul →
// QuantBackend.MatmulW8A8), CPU glue between — so a real "staged GPU" tok/s exists today
// with no code. It is glue/dispatch-bound, NOT the resident megakernel path; report it
// in its own table, NEVER as the GPU headline (which is dense-only residency).
//
// int8 ONLY: the int4 (W4A8) matmul branch dispatches straight to linalg.MatmulBTW4A8
// with no backend routing, so -tags gpu does nothing for an int4 model (that gap is
// Lever B). Loads here are Quant "int8int8" so the matmuls actually reach the GPU.
//
// Pairs staged-GPU vs CPU per family so the staged-vs-CPU delta is visible — on a
// hybrid the Mamba-2 scan stays on the CPU regardless, so the GPU only accelerates the
// attention/MLP matmuls; a glue-bound staged path may not beat tuned CPU SIMD, which is
// itself a finding (the doc calls this out).
//
//	go test -tags gpu ./gpu/ -run TestStagedGPU_nonDense -v -timeout 60m
func TestStagedGPU_nonDense(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1 to opt in (loads a multi-GB model from ~/models)")
	}
	if testing.Short() {
		t.Skip("staged-path benchmark: skipped in -short")
	}
	if _, err := gpu.New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	home, _ := os.UserHomeDir()
	type fam struct {
		name, axis, env, def string
	}
	fams := []fam{
		{"deepseek-v2-lite", "MoE + MLA", "GOINFER_DEEPSEEK_V2LITE", filepath.Join(home, "models", "deepseek-v2-lite")},
		{"granite", "Mamba-2 hybrid + MoE", "GOINFER_GRANITE", filepath.Join(home, "models", "granite")},
		{"nemotron", "Mamba-2 hybrid", "GOINFER_NEMOTRON", filepath.Join(home, "models", "nemotron")},
	}
	prompt := []int{785, 6722, 315, 9625, 374} // "The capital of France is" (Llama-ish ids; coherence irrelevant — we time decode)
	const n = 32

	// decodeTPS loads model at int8 on the given backend, warms once, and times n
	// greedy tokens. Returns tokens/sec and whether it was GPU-resident (must be false
	// here — these are ineligible; if a family ever becomes eligible this flags it).
	decodeTPS := func(path, backend string) (float64, bool, error) {
		m, err := decoder.Load(path, decoder.Options{Backend: backend, Quant: "int8int8"})
		if err != nil {
			return 0, false, err
		}
		defer m.Close()
		resident := m.ResidentActive()
		ctx := context.Background()
		greedy := decoder.SamplingParams{Temperature: 0}
		drain := func(ch <-chan int) int {
			k := 0
			for range ch {
				k++
			}
			return k
		}
		w, _ := m.Generate(ctx, prompt, 8, greedy)
		drain(w) // warm-up
		t0 := time.Now()
		ch, _ := m.Generate(ctx, prompt, n, greedy)
		k := drain(ch)
		return float64(k) / time.Since(t0).Seconds(), resident, nil
	}

	t.Logf("=== staged-path GPU vs CPU, int8 (residency-ineligible families) ===")
	t.Logf("%-18s %-22s %10s %10s %8s", "family", "axis", "cpu tok/s", "gpu tok/s", "gpu/cpu")
	for _, f := range fams {
		path := os.Getenv(f.env)
		if path == "" {
			path = f.def
		}
		if _, err := os.Stat(path); err != nil {
			t.Logf("%-18s %-22s  (skip — no checkpoint at %s)", f.name, f.axis, path)
			continue
		}
		cpuTPS, _, err := decodeTPS(path, "cpu")
		if err != nil {
			t.Logf("%-18s %-22s  (cpu load failed: %v)", f.name, f.axis, err)
			continue
		}
		gpuTPS, resident, err := decodeTPS(path, "webgpu")
		if err != nil {
			t.Logf("%-18s %-22s  (gpu load failed: %v)", f.name, f.axis, err)
			continue
		}
		if resident {
			t.Errorf("%s unexpectedly GPU-resident — not a staged-path family", f.name)
		}
		t.Logf("%-18s %-22s %10.2f %10.2f %7.2f×", f.name, f.axis, cpuTPS, gpuTPS, gpuTPS/cpuTPS)
	}
}
