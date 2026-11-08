//go:build gpu

package kernels

import (
	"math"
	"testing"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// TestRope_matchesCPUAngles holds the rotation every RoPE kernel applies to the CPU's float64 cos/sin of the same f32 angle, to about
// f32 accuracy on any adapter: the rope kernel (Q), ropeStore (K) and the fused qkvFinalize (Q and K). A WGSL builtin sin/cos may
// carry an absolute error far larger than f32's (2^-11 is conformant), so the kernels use ropeCosSin. The positions keep the angle
// under 8, where the f32 angle itself is good to 5e-7.
func TestRope_matchesCPUAngles(t *testing.T) {
	c := newOrSkipHW(t)
	defer c.Close()
	if err := c.ensureAttn(); err != nil {
		t.Fatalf("ensureAttn: %v", err)
	}
	const nH, nKV, hd = 4, 2, 128
	const half, maxLen = hd / 2, 8
	kvDim := nKV * hd
	invFreq := make([]float32, half)
	for i := range invFreq {
		invFreq[i] = float32(math.Pow(10000, -2*float64(i)/hd))
	}
	rot := func(src []float32, heads, pos int) []float32 {
		out := make([]float32, len(src))
		for h := range heads {
			off := h * hd
			for d := range half {
				th := float64(float32(pos)) * float64(invFreq[d])
				cs, sn := math.Cos(th), math.Sin(th)
				x1, x2 := float64(src[off+d]), float64(src[off+half+d])
				out[off+d] = float32(x1*cs - x2*sn)
				out[off+half+d] = float32(x2*cs + x1*sn)
			}
		}
		return out
	}
	// bar: a few f32 roundings of the inputs' size, far under a 6e-5 sin/cos error.
	check := func(label string, pos int, got, want, src []float32) {
		t.Helper()
		for i := range want {
			if e, tol := math.Abs(float64(got[i]-want[i])), 2e-6*(1+math.Abs(float64(src[i]))); e > tol {
				t.Errorf("%s pos %d: element %d = %v, CPU %v (|diff| %.3g > %.3g)", label, pos, i, got[i], want[i], e, tol)
				return
			}
		}
	}
	rng := i8rng()
	for _, pos := range []int{1, 2, 3, 5, 7} {
		q, k, v := randF(rng, nH*hd), randF(rng, kvDim), randF(rng, kvDim)

		qb := c.sbuf(q)
		uni := c.ubuf([]uint32{nH, hd, half, uint32(pos), math.Float32bits(1), 0, 0, 0})
		if err := c.dispatchI8(c.ropePipeline, c.ropeLayout, (nH*half+63)/64, []*wgpu.Buffer{qb, c.sbuf(invFreq)}, uni); err != nil {
			t.Fatalf("rope: %v", err)
		}
		check("rope Q", pos, c.readF(qb, nH*hd), rot(q, nH, pos), q)

		kc := c.zbuf(maxLen * kvDim)
		suni := c.ubuf([]uint32{nKV, hd, half, uint32(pos), math.Float32bits(1), uint32(pos * kvDim), 0, 0})
		if err := c.dispatchI8(c.ropeStorePipeline, c.ropeStoreLayout, (nKV*half+63)/64, []*wgpu.Buffer{c.sbuf(k), c.sbuf(invFreq), kc}, suni); err != nil {
			t.Fatalf("ropeStore: %v", err)
		}
		check("ropeStore K", pos, c.readF(kc, maxLen*kvDim)[pos*kvDim:(pos+1)*kvDim], rot(k, nKV, pos), k)

		qb2, kc2, vc2 := c.sbuf(q), c.zbuf(maxLen*kvDim), c.zbuf(maxLen*kvDim)
		funi := c.ubuf([]uint32{nH, nKV, hd, half, uint32(pos), uint32(pos * kvDim), math.Float32bits(1), uint32(kvDim)})
		if err := c.dispatchI8(c.qkvFinPipeline, c.qkvFinLayout, (max(nH*half, kvDim)+63)/64,
			[]*wgpu.Buffer{qb2, c.sbuf(k), c.sbuf(v), c.sbuf(invFreq), kc2, vc2}, funi); err != nil {
			t.Fatalf("qkvFinalize: %v", err)
		}
		check("qkvFinalize Q", pos, c.readF(qb2, nH*hd), rot(q, nH, pos), q)
		check("qkvFinalize K", pos, c.readF(kc2, maxLen*kvDim)[pos*kvDim:(pos+1)*kvDim], rot(k, nKV, pos), k)
	}
}
