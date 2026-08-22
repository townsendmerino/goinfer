//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestSAGemvLargeK — proves the dynamic-As Stage A GEMV is correct at K=4096 (Mistral-7B's
// o-proj / all gemvs: nH*hd = 32*128 = 4096), the large-K case the old hardcoded As[1536]
// overflowed. Dispatches the production gemv_w4a8_sa with threadgroup memory sized to K, vs a
// CPU dequant reference on the same packed nibbles. (Cheap: synthetic weights, no 7B model —
// a full Mistral-7B run doesn't fit comfortably in 16 GB.)
func TestSAGemvLargeK(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Fatalf("device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "gemv_w4a8_sa")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	const N, K = 256, 4096 // K=4096 > old As[1536] cap
	rng := rand.New(rand.NewSource(23))
	// int8 activation.
	a := make([]float32, K)
	for i := range a {
		a[i] = rng.Float32()*2 - 1
	}
	amx := float32(0)
	for _, v := range a {
		if x := float32(math.Abs(float64(v))); x > amx {
			amx = x
		}
	}
	aSc := amx / 127
	aq := make([]int8, K)
	for i, v := range a {
		aq[i] = int8(math.Max(-127, math.Min(127, math.Round(float64(v/aSc)))))
	}
	// int4 weights (validated packer) + f16 scales; CPU dequant reference.
	words := make([]uint32, N*(K/8))
	scalesH := make([]uint16, N*(K/32))
	ref := make([]float32, N)
	for n := range N {
		row := make([]float32, K)
		for k := range row {
			row[k] = rng.Float32()*2 - 1
		}
		w, s := packW4A8Row(row)
		copy(words[n*(K/8):(n+1)*(K/8)], w)
		var acc float64
		for g := range K / 32 {
			scalesH[n*(K/32)+g] = f32ToF16(s[g])
			sc := float64(f16ToF32(scalesH[n*(K/32)+g]))
			for e := range 32 {
				k := g*32 + e
				nib := int((w[k/8]>>(4*uint(k%8)))&0xF) - 8
				acc += float64(nib) * float64(aq[k]) * sc
			}
		}
		ref[n] = float32(acc) * aSc
	}
	q := d.NewCommandQueue()
	out := d.NewBufferLen(N)
	// dynamic As of K shorts (2*K bytes) — the fix; reps=1.
	q.Run1DBatchTG(pipe, N*32, 256, 1, K*2,
		NewBufferUint32s(d, words), NewBufferU16s(d, scalesH), NewBufferInt8(d, aq),
		NewBufferFloats(d, []float32{aSc}), out, NewBufferU32(d, K))
	got := out.Floats()

	var dot, na, nb, maxRel float64
	for n := range N {
		dot += float64(got[n]) * float64(ref[n])
		na += float64(got[n]) * float64(got[n])
		nb += float64(ref[n]) * float64(ref[n])
		if dd := math.Abs(float64(got[n] - ref[n])); dd > 1e-3 {
			if rel := dd / (math.Abs(float64(ref[n])) + 1e-3); rel > maxRel {
				maxRel = rel
			}
		}
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	mustFinite(t, "Stage A GEMV cosine", cos)
	if cos < 0.9999 || maxRel > 5e-3 {
		t.Fatalf("Stage A GEMV K=%d parity FAIL: cos=%.7f maxRel=%.4f", K, cos, maxRel)
	}
	t.Logf("Stage A GEMV (dynamic As) K=%d vs CPU: cos=%.7f maxRel=%.4f — PARITY ✓ (As-cap fix at 4096)", K, cos, maxRel)
}
