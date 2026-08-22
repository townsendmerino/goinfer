//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestSAGemv_OutlierRegime is the decisive experiment for the Gemma o-proj amplitude bug
// (docs/prompts/gemma-metal-signflip-bisect.md, Fork 2). The CUDA box showed Metal's o-proj
// contribution inflates 2–6× and flips on the secondary channels while CUDA's dp4a stays clean,
// and named it a Metal W4A8 GEMV scale bug. TestSAGemvLargeK already proves the kernel correct at
// K=4096 — but with a BENIGN random activation. This drives the same production kernel in the
// regime that actually triggers the bug: at Gemma's o-proj K=2048, with an activation carrying
// ONE massive outlier channel that sets the int8 scale (~558) and crushes every other channel to
// near-zero int8 — exactly the post-quant state of the attention context feeding the o-proj.
//
// The CPU reference computes the kernel's OWN formula bit-for-bit (dequant nibble × int8 act ×
// f16 group scale × asc). So this isolates the KERNEL from the quant scheme:
//   - Metal == CPU here ⇒ the o-proj GEMV arithmetic is FAITHFUL even in the outlier regime, and
//     Metal's divergence-vs-CUDA lives upstream (a different attention context) or in the quant
//     policy, NOT this kernel — which would redirect the fix.
//   - Metal ≠ CPU here ⇒ the kernel mis-handles the outlier regime (group-scale accumulation or
//     asc application under extreme dynamic range) — the bug, localized.
//
// Reported per-channel on the crushed rows, because a whole-vector cosine is dominated by the one
// massive output and would hide a 6× error on the secondary rows (the sink lesson).
func TestSAGemv_OutlierRegime(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "gemv_w4a8_sa")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	const N, K = 512, 2048 // Gemma-3-4b o-proj: nH*hd = 8*256 = 2048
	rng := rand.New(rand.NewSource(4433))

	// Activation with ONE massive outlier channel; the rest are mid-magnitude (~hundreds relative
	// to the outlier's ~70k), so int8 quant (scale = max/127) crushes them to 0..±2 — the regime.
	a := make([]float32, K)
	for i := range a {
		a[i] = (rng.Float32()*2 - 1) * 300
	}
	a[733] = 70000 // the 443-class massive channel that sets the scale
	amx := float32(0)
	for _, v := range a {
		if x := float32(math.Abs(float64(v))); x > amx {
			amx = x
		}
	}
	aSc := amx / 127
	aq := make([]int8, K)
	var crushed int
	for i, v := range a {
		aq[i] = int8(math.Max(-127, math.Min(127, math.Round(float64(v/aSc)))))
		if aq[i] == 0 {
			crushed++
		}
	}
	t.Logf("outlier regime: aSc=%.2f, %d/%d channels crushed to int8 0 (the near-zero regime)", aSc, crushed, K)

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
	q.Run1DBatchTG(pipe, N*32, 256, 1, K*2,
		NewBufferUint32s(d, words), NewBufferU16s(d, scalesH), NewBufferInt8(d, aq),
		NewBufferFloats(d, []float32{aSc}), out, NewBufferU32(d, K))
	got := out.Floats()

	// Worst per-row divergence AND the worst on the small-output rows (the ones the flip hits).
	var worstAbs, worstRel float64
	var worstRow int
	for n := range N {
		dd := math.Abs(float64(got[n] - ref[n]))
		if dd > worstAbs {
			worstAbs = dd
		}
		if rel := dd / (math.Abs(float64(ref[n])) + 1e-3); rel > worstRel {
			worstRel, worstRow = rel, n
		}
	}
	t.Logf("Metal gemv_w4a8_sa vs bit-faithful CPU W4A8 (outlier regime, K=%d): worst |Δ|=%.4g, worst rel=%.4g at row %d (metal=%.4g cpu=%.4g)",
		K, worstAbs, worstRel, worstRow, got[worstRow], ref[worstRow])
	// If the kernel is faithful, Metal reproduces its own formula to f32 rounding even here.
	if worstRel > 1e-3 {
		t.Errorf("Metal o-proj GEMV DIVERGES from the CPU W4A8 model in the outlier regime (rel %.4g) "+
			"— kernel bug localized to the outlier-regime arithmetic", worstRel)
	} else {
		t.Logf("KERNEL FAITHFUL: Metal == CPU W4A8 even in the outlier regime — the o-proj GEMV arithmetic " +
			"is correct, so the Metal-vs-CUDA divergence is upstream (attention context) or quant policy, not this kernel")
	}
}
