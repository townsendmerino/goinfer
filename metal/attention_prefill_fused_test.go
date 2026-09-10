//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestAttentionPrefillFused checks attention_prefill_fused (the simdgroup_matrix flash-attention
// twin of attention_prefill, L2-Metal — docs/task-prefill-gap.md §4) against the exact scalar
// kernel across GQA, non-multiple-of-8 M (ragged last row-tile), sliding window, a mid-context
// startPos offset, and a small hd (single MMA tile). NOT bit-identical by design (online-softmax
// rescale reorders the sum vs. the exact kernel's single final normalize — same P19 category as
// the CUDA L2 twin), so this asserts closeness (cosine + max-abs-diff), not equality.
func TestAttentionPrefillFused(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(prefillKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pExact, err := d.NewComputePipeline(lib, "attention_prefill")
	if err != nil {
		t.Fatalf("pipeline attention_prefill: %v", err)
	}
	pFused, err := d.NewComputePipeline(lib, "attention_prefill_fused")
	if err != nil {
		t.Fatalf("pipeline attention_prefill_fused: %v", err)
	}

	type cfg struct {
		name                string
		nH, nKV, hd         int
		M, startPos, window int
	}
	cases := []cfg{
		{"gqa-ragged-tail", 12, 2, 128, 37, 0, 0},
		{"no-gqa-hd64-offset", 4, 4, 64, 16, 100, 0},
		{"gqa-window-midctx", 8, 2, 128, 20, 500, 64},
		{"single-tile-hd8", 2, 2, 8, 9, 0, 0},
		{"exact-multiple-of-8", 6, 1, 128, 32, 7, 0},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			qDim := c.nH * c.hd
			kvDim := c.nKV * c.hd
			qStride := qDim
			cacheLen := c.startPos + c.M

			rng := rand.New(rand.NewSource(int64(1000 + c.M)))
			rndHalf := func(n int) []uint16 {
				s := make([]uint16, n)
				for i := range s {
					s[i] = f32ToF16(rng.Float32()*2 - 1)
				}
				return s
			}
			qkvData := rndHalf(c.M * qStride)
			kcData := rndHalf(cacheLen * kvDim)
			vcData := rndHalf(cacheLen * kvDim)

			dQkv := NewBufferU16s(d, qkvData)
			dKc := NewBufferU16s(d, kcData)
			dVc := NewBufferU16s(d, vcData)
			dOutExact := NewBufferU16s(d, make([]uint16, c.M*qDim))
			dOutFused := NewBufferU16s(d, make([]uint16, c.M*qDim))

			uNH := NewBufferU32(d, uint32(c.nH))
			uNKV := NewBufferU32(d, uint32(c.nKV))
			uHd := NewBufferU32(d, uint32(c.hd))
			uStartPos := NewBufferU32(d, uint32(c.startPos))
			uScale := NewBufferFloats(d, []float32{1.0 / float32(math.Sqrt(float64(c.hd)))})
			uQStride := NewBufferU32(d, uint32(qStride))
			uWindow := NewBufferU32(d, uint32(c.window))
			uM := NewBufferU32(d, uint32(c.M))

			q := d.NewCommandQueue()
			e := q.Begin()
			e.Dispatch(pExact, c.M*c.nH*128, 128, dQkv, dKc, dVc, dOutExact, uNH, uNKV, uHd, uStartPos, uScale, uQStride, uWindow)
			numRowTiles := (c.M + 7) / 8
			const sgpt = 4
			total := c.nH * numRowTiles
			total = (total + sgpt - 1) / sgpt * sgpt * 32
			e.Dispatch(pFused, total, sgpt*32, dQkv, dKc, dVc, dOutFused, uNH, uNKV, uHd, uStartPos, uScale, uQStride, uWindow, uM)
			e.End()

			exact := dOutExact.U16s()
			fused := dOutFused.U16s()

			var dot, na, nb, maxAbs float64
			for i := range exact {
				a := float64(f16ToF32(exact[i]))
				b := float64(f16ToF32(fused[i]))
				dot += a * b
				na += a * a
				nb += b * b
				if diff := math.Abs(a - b); diff > maxAbs {
					maxAbs = diff
				}
			}
			cos := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-12)
			t.Logf("%s: cosine=%.6f maxAbsDiff=%.6f", c.name, cos, maxAbs)
			if cos < 0.999 {
				t.Errorf("%s: cosine %.6f below 0.999 (exact vs fused diverge)", c.name, cos)
			}
			if maxAbs > 0.05 {
				t.Errorf("%s: max abs diff %.6f exceeds 0.05", c.name, maxAbs)
			}
		})
	}
}
