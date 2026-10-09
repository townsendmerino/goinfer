//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// G-IP1 (S17's Metal image prefill, docs/tasks/task-multimodal-support-2026-10.md): attention_prefill_img against a float64
// reference of the CPU's rule (KVCache.attendHi, WindowStart): a query inside the image block [imgStart, imgEnd) sees keys up
// to imgEnd-1, every other query is causal, and the sliding window's lower bound stays at the causal position's. Max abs diff
// <= 4e-3 (f16 output). Control: attention_prefill (causal everywhere) on the same input must miss by at least 10x on the
// block's rows, so the case cannot pass without the mask.
func TestAttentionPrefillImg_matchesReference(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(prefillKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile prefillKernels: %v", err)
	}
	pipe := func(n string) Pipeline {
		p, err := d.NewComputePipeline(lib, n)
		if err != nil {
			t.Fatalf("pipeline %s: %v", n, err)
		}
		return p
	}
	q := d.NewCommandQueue()
	for _, tc := range []struct {
		name                       string
		startPos, M, imgS, imgE, w int
	}{
		{"block at 3..19 of 24 rows, full attention", 0, 24, 3, 19, 0},
		{"block after a cached prefix (startPos 5), rows 5..28", 5, 24, 9, 21, 0},
		{"sliding window 8 cutting into the block", 0, 24, 2, 18, 8},
	} {
		const nH, nKV, hd = 4, 2, 16
		rng := rand.New(rand.NewSource(int64(31 + tc.imgS)))
		total := tc.startPos + tc.M
		qStride := (nH + 2*nKV) * hd
		kvDim := nKV * hd
		qkv := make([]uint16, tc.M*qStride)
		kc := make([]uint16, total*kvDim)
		vc := make([]uint16, total*kvDim)
		for i := range qkv {
			qkv[i] = f32ToF16(rng.Float32()*2 - 1)
		}
		for i := range kc {
			kc[i], vc[i] = f32ToF16(rng.Float32()*2-1), f32ToF16(rng.Float32()*2-1)
		}
		scale := float32(1 / math.Sqrt(hd))
		ref := make([]float64, tc.M*nH*hd)
		for m := range tc.M {
			pos := tc.startPos + m
			hi := pos
			if pos >= tc.imgS && pos < tc.imgE {
				hi = tc.imgE - 1
			}
			lo := 0
			if tc.w > 0 && pos+1 > tc.w {
				lo = pos + 1 - tc.w
			}
			for h := range nH {
				kvh := h / (nH / nKV)
				sc := make([]float64, hi+1)
				mx := math.Inf(-1)
				for s := lo; s <= hi; s++ {
					var a float64
					for dd := range hd {
						a += float64(f16ToF32(qkv[m*qStride+h*hd+dd])) * float64(f16ToF32(kc[s*kvDim+kvh*hd+dd]))
					}
					sc[s] = a * float64(scale)
					mx = math.Max(mx, sc[s])
				}
				var sum float64
				for s := lo; s <= hi; s++ {
					sc[s] = math.Exp(sc[s] - mx)
					sum += sc[s]
				}
				for dd := range hd {
					var a float64
					for s := lo; s <= hi; s++ {
						a += sc[s] * float64(f16ToF32(vc[s*kvDim+kvh*hd+dd]))
					}
					ref[m*nH*hd+h*hd+dd] = a / sum
				}
			}
		}
		run := func(kernel string, withImg bool) []float32 {
			out := NewBufferU16s(d, make([]uint16, tc.M*nH*hd))
			args := []Buffer{NewBufferU16s(d, qkv), NewBufferU16s(d, kc), NewBufferU16s(d, vc), out, NewBufferU32(d, nH), NewBufferU32(d, nKV),
				NewBufferU32(d, hd), NewBufferU32(d, uint32(tc.startPos)), NewBufferFloats(d, []float32{scale}), NewBufferU32(d, uint32(qStride)),
				NewBufferU32(d, uint32(tc.w))}
			if withImg {
				args = append(args, NewBufferUint32s(d, []uint32{uint32(tc.imgS), uint32(tc.imgE)}))
			}
			q.Run1D(pipe(kernel), tc.M*nH*32, 32, args...)
			o := make([]float32, tc.M*nH*hd)
			for i, h := range out.U16s()[:len(o)] {
				o[i] = f16ToF32(h)
			}
			return o
		}
		diff := func(got []float32, rowsInBlockOnly bool) float64 {
			mx := 0.0
			for m := range tc.M {
				pos := tc.startPos + m
				if rowsInBlockOnly && (pos < tc.imgS || pos >= tc.imgE-1) {
					continue // the block's last row sees the same keys either way
				}
				for i := m * nH * hd; i < (m+1)*nH*hd; i++ {
					mx = math.Max(mx, math.Abs(float64(got[i])-ref[i]))
				}
			}
			return mx
		}
		img := diff(run("attention_prefill_img", true), false)
		causal := diff(run("attention_prefill", false), true)
		t.Logf("%-52s attention_prefill_img max|diff| %.3g (tol 4e-3) | causal control on the block's rows %.3g", tc.name, img, causal)
		if img > 4e-3 {
			t.Errorf("%s: attention_prefill_img max|diff| %.3g > 4e-3", tc.name, img)
		}
		if causal < 10*4e-3 {
			t.Errorf("%s: the causal control is within 10x of the tolerance (%.3g): the case cannot see the mask", tc.name, causal)
		}
	}
}
