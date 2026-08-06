//go:build gpu

package gpu

import (
	"math"
	"testing"

	"github.com/cogentcore/webgpu/wgpu"
)

// TestRopeStorePartialRotary_tailStored gates C4: for partial rotary (rotaryDim < headDim — GLM,
// some Phi), the K-store kernels must store the un-rotated pass-through tail [2*half, headDim), not
// just the rotated span. Dropping it makes attention read zeros for those key dims at every cached
// position — plausible-looking, silently wrong logits. Compares the resident f32 ropeStore + the
// fused qkvFinalize against a CPU reference that rotates [0,2*half) and passes the tail through.
func TestRopeStorePartialRotary_tailStored(t *testing.T) {
	c := newOrSkipHW(t)
	defer c.Close()
	if err := c.ensureAttn(); err != nil { // compiles the rope/store kernels (validates the C4 WGSL)
		t.Fatalf("ensureAttn: %v", err)
	}

	const nH, nKV, hd, rotaryDim = 4, 2, 64, 32
	const half = rotaryDim / 2 // 16 ⇒ tail [32, 64) is un-rotated
	const pos, maxLen = 3, 8
	kvDim := nKV * hd
	invFreq := make([]float32, half)
	for i := range invFreq {
		invFreq[i] = float32(math.Pow(10000, -2*float64(i)/float64(rotaryDim)))
	}
	// CPU reference: rotate [0,2*half), pass the tail [2*half,hd) through unchanged.
	ropeFull := func(head []float32) []float32 {
		out := make([]float32, hd)
		for d := range half {
			th := float64(pos) * float64(invFreq[d])
			cs, sn := float32(math.Cos(th)), float32(math.Sin(th))
			out[d] = head[d]*cs - head[half+d]*sn
			out[half+d] = head[half+d]*cs + head[d]*sn
		}
		for d := 2 * half; d < hd; d++ {
			out[d] = head[d] // pass-through
		}
		return out
	}
	wantK := func(src []float32) []float32 {
		var w []float32
		for h := range nKV {
			w = append(w, ropeFull(src[h*hd:h*hd+hd])...)
		}
		return w
	}
	// assertTail fails if any tail dim was dropped (the C4 bug wrote zeros there).
	assertTail := func(label string, got, src []float32) {
		for h := range nKV {
			for d := 2 * half; d < hd; d++ {
				if g, w := got[h*hd+d], src[h*hd+d]; g != w {
					t.Fatalf("%s: tail [h=%d d=%d] = %v, want src %v (un-rotated pass-through, C4)", label, h, d, g, w)
				}
			}
		}
	}

	rng := i8rng()
	kSrc := randF(rng, kvDim)

	// --- standalone f32 ropeStore ---
	kCache := c.zbuf(maxLen * kvDim)
	uni := c.ubuf([]uint32{nKV, hd, half, pos, math.Float32bits(1.0), uint32(pos * kvDim), 0, 0})
	groups := (nKV*(hd-half) + 63) / 64
	if err := c.dispatchI8(c.ropeStorePipeline, c.ropeStoreLayout, groups, []*wgpu.Buffer{c.sbuf(kSrc), c.sbuf(invFreq), kCache}, uni); err != nil {
		t.Fatalf("ropeStore dispatch: %v", err)
	}
	got := c.readF(kCache, maxLen*kvDim)[pos*kvDim : pos*kvDim+kvDim]
	if cos := cosF(got, wantK(kSrc)); cos < 0.99999 {
		t.Errorf("ropeStore partial-rotary vs CPU: cosine %.6f", cos)
	}
	assertTail("ropeStore", got, kSrc)

	// --- fused qkvFinalize (f32-KV, the default GLM path) ---
	qSrc := randF(rng, nH*hd)
	vSrc := randF(rng, kvDim)
	kCache2 := c.zbuf(maxLen * kvDim)
	vCache2 := c.zbuf(maxLen * kvDim)
	quni := c.ubuf([]uint32{nH, nKV, hd, half, pos, uint32(pos * kvDim), math.Float32bits(1.0), uint32(kvDim)})
	fgroups := (max3(nH*half, kvDim, nKV*(hd-2*half)) + 63) / 64
	if err := c.dispatchI8(c.qkvFinPipeline, c.qkvFinLayout, fgroups,
		[]*wgpu.Buffer{c.sbuf(qSrc), c.sbuf(kSrc), c.sbuf(vSrc), c.sbuf(invFreq), kCache2, vCache2}, quni); err != nil {
		t.Fatalf("qkvFinalize dispatch: %v", err)
	}
	got2 := c.readF(kCache2, maxLen*kvDim)[pos*kvDim : pos*kvDim+kvDim]
	if cos := cosF(got2, wantK(kSrc)); cos < 0.99999 {
		t.Errorf("qkvFinalize partial-rotary K vs CPU: cosine %.6f", cos)
	}
	assertTail("qkvFinalize", got2, kSrc)
}

func max3(a, b, c int) int { return max(a, max(b, c)) }
