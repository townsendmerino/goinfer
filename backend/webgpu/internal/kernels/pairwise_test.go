//go:build gpu

package kernels

import (
	"math"
	"math/rand"
	"testing"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// pairwiseRef is GPT-J pairwise RoPE of one token's heads×hd vector at pos, in float64 over the kernels' f32 angle:
// pair (2d, 2d+1) for d < half rotated by pos·invFreq[d], the tail [2·half, hd) unchanged.
func pairwiseRef(x, invFreq []float32, heads, hd, half, pos int, scale float32) []float64 {
	out := make([]float64, len(x))
	for h := range heads {
		off := h * hd
		for d := range half {
			theta := float64(float32(pos) * invFreq[d])
			c, s := math.Cos(theta)*float64(scale), math.Sin(theta)*float64(scale)
			a, b := float64(x[off+2*d]), float64(x[off+2*d+1])
			out[off+2*d], out[off+2*d+1] = a*c-b*s, b*c+a*s
		}
		for e := 2 * half; e < hd; e++ {
			out[off+e] = float64(x[off+e])
		}
	}
	return out
}

// TestPairwiseTwins_matchGPTJ holds every pairwise twin to pairwiseRef at a partial rotary width (the tail passes through):
// rope and qkvFinalize's q in place, ropeStore and qkvFinalize's k into the cache at base, both to 1e-5 of the vector's
// scale; the f16 store to f16 rounding; the int8 store to half a step of its per-head scale. The NeoX kernel on the same
// input misses the f32 bar, so the reference tells the two rotations apart.
func TestPairwiseTwins_matchGPTJ(t *testing.T) {
	c, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	defer c.Close()
	if err := c.ensureAttn(); err != nil {
		t.Fatal(err)
	}
	if err := c.ensurePairwise(); err != nil {
		t.Fatal(err)
	}
	const heads, hd, half, pos, at = 3, 32, 12, 37, 2 // at: the cache position stored to
	const scale = float32(1.25)
	n, base := heads*hd, at*heads*hd
	rng := rand.New(rand.NewSource(5))
	x, invFreq := randF(rng, n), make([]float32, half)
	for d := range invFreq {
		invFreq[d] = float32(math.Pow(10000, -2*float64(d)/float64(2*half)))
	}
	want := pairwiseRef(x, invFreq, heads, hd, half, pos, scale)
	sb := math.Float32bits(scale)
	check := func(name string, got []float32, tol float64) {
		t.Helper()
		var e, m float64
		for i := range want {
			e, m = max(e, math.Abs(float64(got[i])-want[i])), max(m, math.Abs(want[i]))
		}
		if !(e/m <= tol) {
			t.Errorf("%s: max error %.3g of the vector's scale, want <= %g", name, e/m, tol)
		}
	}
	runRope := func(p *wgpu.ComputePipeline, ly *wgpu.BindGroupLayout) []float32 {
		v := c.sbuf(x)
		if err := c.dispatchI8(p, ly, (heads*half+63)/64, []*wgpu.Buffer{v, c.sbuf(invFreq)}, c.ubuf([]uint32{heads, hd, half, pos, sb, 0, 0, 0})); err != nil {
			t.Fatal(err)
		}
		return c.readF(v, n)
	}
	check("rope", runRope(c.pairwise.rope.pl, c.pairwise.rope.ly), 1e-5)
	var e, m float64
	for i, v := range runRope(c.ropePipeline, c.ropeLayout) {
		e, m = max(e, math.Abs(float64(v)-want[i])), max(m, math.Abs(want[i]))
	}
	if e/m <= 1e-5 {
		t.Errorf("the NeoX rope matches the pairwise reference (%.3g): the reference cannot tell the rotations apart", e/m)
	}

	cache := func() *wgpu.Buffer { return c.zbuf(base + n) }
	kc := cache()
	if err := c.dispatchI8(c.pairwise.ropeStore.pl, c.pairwise.ropeStore.ly, (heads*(hd-half)+63)/64,
		[]*wgpu.Buffer{c.sbuf(x), c.sbuf(invFreq), kc}, c.ubuf([]uint32{heads, hd, half, pos, sb, uint32(base), 0, 0})); err != nil {
		t.Fatal(err)
	}
	check("ropeStore", c.readF(kc, base+n)[base:], 1e-5)

	q, kq, vq := c.sbuf(x), cache(), cache()
	if err := c.dispatchI8(c.pairwise.qkvFin.pl, c.pairwise.qkvFin.ly, (n+63)/64,
		[]*wgpu.Buffer{q, c.sbuf(x), c.sbuf(x), c.sbuf(invFreq), kq, vq},
		c.ubuf([]uint32{heads, heads, hd, half, pos, uint32(base), sb, uint32(n)})); err != nil {
		t.Fatal(err)
	}
	check("qkvFinalize q", c.readF(q, n), 1e-5)
	check("qkvFinalize k", c.readF(kq, base+n)[base:], 1e-5)

	f16 := c.zbuf((base + n) / 2)
	if err := c.dispatchI8(c.pairwise.ropeStoreF16.pl, c.pairwise.ropeStoreF16.ly, (n/2+63)/64,
		[]*wgpu.Buffer{c.sbuf(x), c.sbuf(invFreq), f16}, c.ubuf([]uint32{heads, hd, half, pos, sb, uint32(base), 0, 0})); err != nil {
		t.Fatal(err)
	}
	words := c.readWords(f16, (base+n)/2)[base/2:]
	got := make([]float32, n)
	for i, w := range words {
		got[2*i], got[2*i+1] = f16to32(uint16(w)), f16to32(uint16(w>>16))
	}
	check("ropeStoreF16", got, 1e-3)

	i8, sc := c.zbuf((base+n)/4), c.zbuf(heads*(at+1))
	if err := c.dispatchI8(c.pairwise.ropeStoreI8.pl, c.pairwise.ropeStoreI8.ly, 1,
		[]*wgpu.Buffer{c.sbuf(x), c.sbuf(invFreq), i8, sc}, c.ubuf([]uint32{heads, hd, half, pos, sb, uint32(base), heads, at})); err != nil {
		t.Fatal(err)
	}
	words, scales := c.readWords(i8, (base+n)/4)[base/4:], c.readF(sc, heads*(at+1))[at*heads:]
	for h := range heads { // each head's scale is its rotated row's max |x| / 127
		var amax float64
		for e := h * hd; e < (h+1)*hd; e++ {
			amax = max(amax, math.Abs(want[e]))
		}
		if d := math.Abs(float64(scales[h]) - amax/127); !(d <= 1e-5*amax/127) {
			t.Errorf("ropeStoreI8 head %d: scale %g, want max|x|/127 = %g", h, scales[h], amax/127)
		}
	}
	for i := range n {
		code := float64(int8(words[i/4] >> (8 * (i % 4))))
		s := float64(scales[i/hd])
		if d := math.Abs(code*s - want[i]); !(d <= 0.5*s*(1+1e-4)) {
			t.Errorf("ropeStoreI8 element %d: %g, want %g within half a step (%g)", i, code*s, want[i], s/2)
		}
	}
}
