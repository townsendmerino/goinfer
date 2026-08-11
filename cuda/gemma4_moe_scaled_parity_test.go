//go:build cuda && goinfer_testhooks

package cuda

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemma4MoEScaled_residentParity closes audit Check A: until this landed there was NO asserting
// parity gate for the Gemma-4 resident forward against the CPU path at real width.
//
// What existed, and why none of it covered this:
//   - TestGemma4MoE_residentParity — resident-vs-CPU, but on gemma4-moe-tiny (2 layers, hidden 256,
//     4 experts). Real assertions, toy width.
//   - TestGemma4DenseScaled_residentParity — real head geometry, but DENSE and random-weight.
//   - the real-26B gates (cache/graphs bit-exactness) — compare the resident path against ITSELF
//     with one knob moved. Nothing anchored either arm to CPU.
//   - TestGemma4_12B_logitParity — CPU vs HF bf16, never touches the resident path.
//
// So every real Gemma-4 checkpoint that reached the resident path was compared only to itself, and
// the shipped "26B decodes coherently at ~17 tok/s" rested on a distinct-trigram degeneracy score —
// a forward that was numerically wrong but non-repetitive would have passed everything. That is why
// GOINFER_GEMMA4_RESIDENT could not be defaulted on: the flag had become load-bearing by accident.
//
// The fixture keeps hidden=2816 / moe_inter=704 / head_dim 256 local, 512 global K=V — the real
// 26B's per-expert and per-head geometry — and shrinks only expert count and depth. Its per-group
// weight scales are TRANSPLANTED from the real 26B (log2std 0.32, 24.1x spread on experts vs 0.27 /
// 5.0x for random init), because a fused-multiply-add defect that cost 84% stream divergence in
// v0.9.0 was invisible on uniform random weights. See scripts/pin_gemma4_moe_scaled.py.
//
// Instruments are the calibrated pair the other scaled gates use, not a picked absolute floor:
//
//  1. pos-0 CUDA-vs-CPU-int4 — no KV accumulation, so the MoE block + both attention geometries
//     must compose correctly at the first token.
//  2. run-mean CUDA-vs-CPU-int4 measured against the fixture's OWN CPU-int4-vs-f32 curve. CUDA and
//     CPU-int4 share weights and differ only in W4A8 activation rounding, so CUDA must track or sit
//     above that baseline. Dropping BELOW it is a real divergence no conditioning explains.
func TestGemma4MoEScaled_residentParity(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("GOINFER_HEAVY_TESTS unset (loads three copies of a 1.9 GB fixture)")
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	dir := os.Getenv("GOINFER_MOE_SCALED_FIXTURE")
	if dir == "" {
		dir = "../testdata/gemma4-moe-scaled"
	}
	// Stat the WEIGHTS, not the directory (the config JSONs are not gitignored; see loadG4MoECache).
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture weights (%s/model.safetensors) — run scripts/pin_gemma4_moe_scaled.py", dir)
	}

	mc, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("cuda resident DECLINED scaled gemma4 MoE with env on — admission regressed " +
			"(the runtime prints the reason unconditionally; see stderr)")
	}
	mc4, err := decoder.Load(dir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cpu int4): %v", err)
	}
	defer mc4.Close()
	mcF, err := decoder.Load(dir, decoder.Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("load (cpu f32): %v", err)
	}
	defer mcF.Close()

	prompt := []int{1, 7, 42, 100, 5, 200, 13, 88, 3, 71, 128, 9, 250, 17, 60, 200} // len 16 > window 8

	cpu := func(m *decoder.Model) [][]float32 {
		out := make([][]float32, len(prompt))
		c := m.NewCache(len(prompt))
		for i, tok := range prompt {
			l, err := m.ForwardForTest(tok, c)
			if err != nil {
				t.Fatalf("cpu pos %d: %v", i, err)
			}
			out[i] = append([]float32(nil), l...)
		}
		return out
	}
	cpu4, cpuF := cpu(mc4), cpu(mcF)
	cuda := make([][]float32, len(prompt))
	for i, tok := range prompt {
		l, err := rf.Forward(mc.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("cuda pos %d: %v", i, err)
		}
		cuda[i] = append([]float32(nil), l...)
	}

	cos := func(a, b []float32) float64 { c, _ := cosMaxAbs(a, b); return c }
	pos0, exact, sumCuda, sumCpu := 0.0, 0, 0.0, 0.0
	for i := range prompt {
		cVs4, c4VsF := cos(cpu4[i], cuda[i]), cos(cpuF[i], cpu4[i])
		if i == 0 {
			pos0 = cVs4
		}
		sumCuda += cVs4
		sumCpu += c4VsF
		if argmaxF(cuda[i]) == argmaxF(cpu4[i]) {
			exact++
		}
		t.Logf("  pos %2d  CUDA-vs-CPUint4 %.6f | CPUint4-vs-f32 %.6f  argmax cuda=%d cpu4=%d",
			i, cVs4, c4VsF, argmaxF(cuda[i]), argmaxF(cpu4[i]))
	}
	meanCuda, meanCpu := sumCuda/float64(len(prompt)), sumCpu/float64(len(prompt))
	t.Logf("scaled MoE (hidden 2816, moe_inter 704, 32 experts top-8, 4 layers): pos0=%.6f "+
		"exact-argmax %d/%d | mean CUDA-vs-CPUint4=%.6f  CPUint4-vs-f32=%.6f",
		pos0, exact, len(prompt), meanCuda, meanCpu)

	if pos0 < 0.97 {
		t.Errorf("pos-0 CUDA-vs-CPUint4 %.6f < 0.97 — the resident MoE forward diverges from CPU at "+
			"the first token, at the real per-expert row geometry", pos0)
	}
	if meanCuda < meanCpu {
		t.Errorf("mean CUDA-vs-CPUint4 %.6f < mean CPUint4-vs-f32 %.6f — the resident path diverges "+
			"FASTER than the fixture's own int4 quantization: a real bug, not conditioning",
			meanCuda, meanCpu)
	}
}
