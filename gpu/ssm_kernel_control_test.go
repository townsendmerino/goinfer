//go:build gpu

package gpu

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// D3 — the last localization control (docs/ssm-int8-quality.md). The CPU experiments
// (decoder/ssm_precision_localize_test.go) showed every precision/quant change lands at
// 95-100% vs the f32 reference: E1 f32-SSM=100%, D1 int8-mamba=97%, R2 int8-attn/MoE=95%.
// None reproduce the resident's 66%. D3 stacks ALL of them on the staged path — int8
// mamba (ssmQ8CPU) + int8 attn/MoE (staged webgpu) + (f32 SSM is equivalent to f64 per
// E1, so left f64) — with the mamba COMPUTE on the CPU (mamba2Step), not the GPU kernels.
// If D3 ≈ 95% while the int8 GPU-resident is 66%, the only remaining variable is the GPU
// mamba kernels (conv/ssm/gatedNorm) — i.e. the gap is a kernel discrepancy, not precision.
func TestSSMKernelControlD3(t *testing.T) {
	if os.Getenv("GOINFER_SSM_QUALITY") == "" {
		t.Skip("ssm D3 kernel control — slow (cpu + staged granite load); set GOINFER_SSM_QUALITY=1")
	}
	if _, err := New(); err != nil {
		t.Skipf("no webgpu: %v", err)
	}
	path := os.ExpandEnv("$HOME/models/granite/granite-4.0-h-tiny-Q8_0.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no granite model: %v", err)
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	ids, _ := tk.Encode(ssmHeldOut, true) // shared with ssm_int8_quality_test.go
	if len(ids) > 220 {
		ids = ids[:220]
	}
	N := len(ids) - 1
	t.Logf("held-out: %d tokens scored", N)

	// ---- R1: f32 CPU reference (per-token argmax + dist) ----
	os.Unsetenv("GOINFER_SSM_RESIDENT")
	decoder.SetSSMQ8CPU(false)
	decoder.SetSSMForceF32(false)
	r1, err := decoder.Load(path, decoder.Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("load R1: %v", err)
	}
	r1Arg := make([]int, N)
	r1Dist := make([][]float64, N)
	c1 := r1.NewCache(len(ids) + 1)
	for i := 0; i < N; i++ {
		lg, e := r1.ForwardForTest(ids[i], c1)
		if e != nil {
			t.Fatal(e)
		}
		r1Dist[i] = softmaxOf(lg)
		r1Arg[i] = argmaxOf(lg)
	}
	r1.Close()

	score := func(name string, dist [][]float64, arg []int) {
		agree, kl := 0, 0.0
		for i := 0; i < N; i++ {
			if arg[i] == r1Arg[i] {
				agree++
			}
			for v, p := range r1Dist[i] {
				if p > 1e-9 {
					kl += p * (math.Log(p) - math.Log(math.Max(dist[i][v], 1e-12)))
				}
			}
		}
		t.Logf("%-40s agreement=%.1f%%  meanKL=%.4f", name, 100*float64(agree)/float64(N), kl/float64(N))
	}

	// One staged webgpu load (cpu mamba + int8 GPU attn/MoE matmuls); run it twice —
	// plain (R2) and with the int8-mamba round-trip (D3) — fresh cache each.
	r2, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load staged: %v", err)
	}
	defer r2.Close()
	stagedRun := func() ([][]float64, []int) {
		c := r2.NewCache(len(ids) + 1)
		dist := make([][]float64, N)
		arg := make([]int, N)
		for i := 0; i < N; i++ {
			lg, e := r2.ForwardForTest(ids[i], c)
			if e != nil {
				t.Fatal(e)
			}
			dist[i] = softmaxOf(lg)
			arg[i] = argmaxOf(lg)
		}
		return dist, arg
	}

	decoder.SetSSMQ8CPU(false)
	d, a := stagedRun()
	score("R2: staged (int8 attn/MoE, f32 mamba)", d, a)

	decoder.SetSSMQ8CPU(true) // add int8 mamba on top → the full resident arithmetic, CPU mamba
	d, a = stagedRun()
	score("D3: staged + int8 mamba (= resident, CPU kernels)", d, a)
	decoder.SetSSMQ8CPU(false)
}
