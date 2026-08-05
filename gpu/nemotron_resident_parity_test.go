//go:build gpu && goinfer_testhooks

package gpu

import (
	"os"
	"strconv"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// P3/P4 whole-model parity: resident Nemotron-H (single-op-per-block Mamba-2 / NoPE-GQA / relu²
// MLP) vs the CPU runLayersNemotron, feeding the SAME fixed token sequence to both and comparing
// logits at 1/16/256/1k/2k. State (conv ring + ssm + KV) COMPOUNDS, so a wiring error — a
// block-kind misroute, NoPE slip, single-op pairing bug, or pos off-by-one — shows as a cosine
// that DRIFTS long. Matched precision (resident int8 vs CPU int8 via GOINFER_SSM_CPUQ8) isolates
// WIRING from int8 quality: the engine should track ~flat near the int8-activation floor (the
// resident quantizes the relu²/conv/proj activations the CPU keeps f32). Opt-in (tiny fixture).
func TestNemotronResidentParity(t *testing.T) {
	if os.Getenv("GOINFER_SSM_PARITY") == "" {
		t.Skip("nemotron resident parity (set GOINFER_SSM_PARITY=1; +GOINFER_SSM_CPUQ8=1 for matched int8)")
	}
	if _, err := New(); err != nil {
		t.Skipf("no webgpu: %v", err)
	}
	ckpt := "../testdata/nemotron-tiny"
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no nemotron-tiny fixture: %v", err)
	}
	os.Setenv("GOINFER_SSM_RESIDENT", "1")
	defer os.Unsetenv("GOINFER_SSM_RESIDENT")

	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load webgpu: %v", err)
	}
	defer mRes.Close()
	rf := mRes.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("nemotron did not go resident (BuildResident refused) — check the single-op wiring")
	}
	cpuOpts := decoder.Options{Backend: "cpu"}
	if os.Getenv("GOINFER_SSM_CPUQ8") != "" { // match the resident int8 weights+activations on CPU
		cpuOpts.Quant = "int8int8"
		decoder.SetSSMQ8CPU(true)
		defer decoder.SetSSMQ8CPU(false)
	}
	mCPU, err := decoder.Load(ckpt, cpuOpts)
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mCPU.Close()

	ntok := 256
	if v := os.Getenv("GOINFER_SSM_NTOK"); v != "" {
		if n, e := strconv.Atoi(v); e == nil {
			ntok = n
		}
	}
	_, _, _, _, _, _, vocab := mCPU.Dims()
	checks := map[int]bool{1: true, 16: true, 64: true, 256: true, 1024: true, 2048: true}

	cache := mCPU.NewCache(ntok + 1)
	rf.Reset()
	worst, first := 1.0, 1.0
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
			if i+1 == 1 {
				first = cos
			}
			if cos < worst {
				worst = cos
			}
		}
	}
	t.Logf("  worst cosine over %d tokens: %.6f (tok1 %.6f, drift %.4f)", ntok, worst, first, first-worst)
	// Wiring gate: no DRIFT (the matched-precision cosine must not decay materially over the run).
	if os.Getenv("GOINFER_SSM_CPUQ8") != "" && first-worst > 0.02 {
		t.Errorf("DRIFT over %d tokens: tok1 %.6f → worst %.6f (Δ%.4f > 0.02) — state-threading bug", ntok, first, worst, first-worst)
	}
}
