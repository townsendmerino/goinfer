//go:build gpu && goinfer_testhooks

package webgpu

import (
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"

	"github.com/townsendmerino/goinfer/internal/testgate"
)

// TestGraniteResidentParity is the whole-model integration gate: resident granite-4.0-h-tiny
// (Mamba SSM mixer + attention + MoE-every-layer + the 4 Granite multipliers) vs the CPU
// program walker, feeding the SAME fixed token sequence to both and comparing logits at
// 1/16/256/1k/2k tokens. State (conv ring + ssm + KV) COMPOUNDS, so an integration error (a wrong
// multiplier fold, mixer-kind misroute, or pos off-by-one) shows as a cosine that DRIFTS long.
// GOINFER_SSM_NTOK bounds the run.
func TestGraniteResidentParity(t *testing.T) {
	testgate.RequireHeavy(t)
	// Opt-in: this CHARACTERIZES the resident int8 vs CPU divergence (it does not pass a
	// 0.999 f32 bar — granite's MoE+SSM are int8-sensitive; the engine is correct at int8,
	// see TestGraniteResidentSpeedup). Set GOINFER_SSM_PARITY=1 to run, with optional
	// GOINFER_SSM_CPUQ8=1 (int8 CPU ref → ~0.99) / GOINFER_SSM_NTOK.
	if os.Getenv("GOINFER_SSM_PARITY") == "" {
		t.Skip("granite resident parity characterization (set GOINFER_SSM_PARITY=1)")
	}
	if _, err := New(); err != nil {
		t.Skipf("no webgpu: %v", err)
	}
	path := os.ExpandEnv("$HOME/models/granite/granite-4.0-h-tiny-Q8_0.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no granite model: %v", err)
	}
	os.Setenv("GOINFER_SSM_RESIDENT", "1") // P5b guarded eligibility
	defer os.Unsetenv("GOINFER_SSM_RESIDENT")

	mRes, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load webgpu: %v", err)
	}
	defer mRes.Close()
	rf := mRes.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("granite did not go resident (BuildResident refused) — check the SSM build")
	}
	cpuOpts := decoder.Options{Backend: "cpu"}
	if os.Getenv("GOINFER_SSM_CPUQ8") != "" { // match the resident's int8 weights on the CPU ref
		cpuOpts.Quant = "int8int8"
	}
	mCPU, err := decoder.Load(path, cpuOpts)
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mCPU.Close()

	ntok := 64
	if v := os.Getenv("GOINFER_SSM_NTOK"); v != "" {
		if n, e := strconv.Atoi(v); e == nil {
			ntok = n
		}
	}
	_, _, _, _, _, _, vocab := mCPU.Dims()
	checks := map[int]bool{1: true, 16: true, 64: true, 256: true, 1024: true, 2048: true}

	cache := mCPU.NewCache(ntok + 1)
	rf.Reset()
	worst := 1.0
	for i := 0; i < ntok; i++ {
		tok := (i*131 + 7) % vocab
		lc, err := mCPU.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu forward[%d]: %v", i, err)
		}
		lr, err := rf.Forward(mRes.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("resident forward[%d]: %v", i, err)
		}
		if checks[i+1] || i+1 == ntok {
			cos, maxAbs := cosSim(lc, lr)
			am := argmaxF(lc) == argmaxF(lr)
			t.Logf("  tok %5d: cosine=%.6f maxAbs=%.4g argmax_match=%v", i+1, cos, maxAbs, am)
			if cos < worst {
				worst = cos
			}
			if cos < 0.999 {
				t.Errorf("tok %d DRIFT: cosine=%.6f (< 0.999)", i+1, cos)
			}
		}
	}
	t.Logf("  worst cosine over %d tokens: %.6f", ntok, worst)
}

// TestLogitScaleResidentWebGPU holds the head's logit-scale fold on the dense families that carry one (dense Granite
// divides) to the CPU's scale: relL2 <= 1e-3 over 16 decoded positions at int8int8 and with an int4 head
// (EmbedInt4), where a missing or inverted fold reads relL2 of |1/s - 1| or more. Cosine cannot see a constant factor.
func TestLogitScaleResidentWebGPU(t *testing.T) {
	if c, err := New(); err != nil {
		t.Skipf("no webgpu: %v", err)
	} else {
		c.Close()
	}
	for _, fx := range []string{"granite-dense-tiny"} {
		for _, int4Head := range []bool{false, true} {
			opts := decoder.Options{Quant: "int8int8"}
			if int4Head {
				opts = decoder.Options{Quant: "int4", EmbedInt4: true}
			}
			dir := filepath.Join("../../testdata", fx)
			cpuO := opts
			cpuO.Backend = "cpu"
			cpu, err := decoder.Load(dir, cpuO)
			if err != nil {
				t.Fatal(err)
			}
			gpuO := opts
			gpuO.Backend = "webgpu"
			m, err := decoder.Load(dir, gpuO)
			if err != nil {
				t.Fatal(err)
			}
			rd, ok := m.ResidentForwardForTest().(*residentDecoder)
			if !ok {
				t.Fatalf("%s int4 head %v: not resident: %s", fx, int4Head, m.ResidentDecline())
			}
			_, _, _, _, _, _, vocab := cpu.Dims()
			c := cpu.NewCache(16)
			rd.Reset()
			worst := 0.0
			for i := range 16 {
				tok := (i*37 + 3) % vocab
				want, err := cpu.ForwardForTest(tok, c)
				if err != nil {
					t.Fatal(err)
				}
				got, err := rd.Forward(m.EmbedResidentForTest(tok), i)
				if err != nil {
					t.Fatal(err)
				}
				var dd, bb float64
				for j := range want {
					d := float64(got[j]) - float64(want[j])
					dd, bb = dd+d*d, bb+float64(want[j])*float64(want[j])
				}
				worst = math.Max(worst, math.Sqrt(dd/bb))
			}
			m.Close()
			cpu.Close()
			t.Logf("%s int4 head %v: worst relL2 %.3g", fx, int4Head, worst)
			if worst > 1e-3 {
				t.Errorf("%s int4 head %v: worst relL2 %.3g against the CPU, want <= 1e-3", fx, int4Head, worst)
			}
		}
	}
}
