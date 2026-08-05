//go:build gpu && goinfer_testhooks

package gpu

import (
	"os"
	"strconv"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGraniteResidentParity is the whole-model integration gate (P5b.3 short / P5b.4 long):
// resident granite-4.0-h-tiny (Mamba SSM mixer + attention + MoE-every-layer + the 4 Granite
// multipliers) vs the CPU runLayersGranite, feeding the SAME fixed token sequence to both and
// comparing logits at 1/16/256/1k/2k tokens. State (conv ring + ssm + KV) COMPOUNDS, so an
// integration error — a wrong multiplier fold, mixer-kind misroute, or pos off-by-one — shows
// as a cosine that DRIFTS long. GOINFER_SSM_NTOK bounds the run (P5b.3 a few; P5b.4 the ladder).
func TestGraniteResidentParity(t *testing.T) {
	requireHeavyModel(t)
	// Opt-in: this CHARACTERIZES the resident int8 vs CPU divergence (it does not pass a
	// 0.999 f32 bar — granite's MoE+SSM are int8-sensitive; the engine is correct at int8,
	// see TestGraniteResidentSpeedup). Set GOINFER_SSM_PARITY=1 to run, with optional
	// GOINFER_SSM_CPUQ8=1 (int8 CPU ref → ~0.99) / GOINFER_SSM_NTOK / GOINFER_SSM_STOP_LAYER.
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
