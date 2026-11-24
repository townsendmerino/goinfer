//go:build gpu

package kernels

import (
	"math"
	"math/rand"
	"testing"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/aikit/linalg"
)

// actG32Input is a K-wide activation with Phi-3's shape of trouble: unit-scale values and two outliers about 100x the
// rest, so a per-vector scale rounds nearly every other element to zero.
func actG32Input(rng *rand.Rand, K int) []float32 {
	a := randF(rng, K)
	a[37], a[K-200] = 95, -120
	return a
}

// actG32Codes unpacks n packed int8 codes.
func actG32Codes(words []uint32, n int) []int8 {
	out := make([]int8, n)
	for i := range out {
		out[i] = int8(words[i/4] >> (8 * (i % 4)))
	}
	return out
}

// packI8 packs int8 codes four to a word, as the quantizers write them.
func packI8(q []int8) []float32 {
	out := make([]float32, len(q)/4)
	for i := range out {
		var w uint32
		for j := range 4 {
			w |= uint32(uint8(q[4*i+j])) << (8 * j)
		}
		out[i] = math.Float32frombits(w)
	}
	return out
}

// groupedRef is linalg's per-32 W8A8 product for one activation row: bS[n] · Σ_g aS[g] · Σ_{k∈g} aq[k]·w[n,k], in float64.
func groupedRef(aq []int8, aS []float32, bq []int8, bS []float32, N, K int) []float64 {
	out := make([]float64, N)
	for n := range N {
		var acc float64
		for g := 0; g < K/32; g++ {
			var isum int32
			for k := g * 32; k < g*32+32; k++ {
				isum += int32(aq[k]) * int32(bq[n*K+k])
			}
			acc += float64(isum) * float64(aS[g])
		}
		out[n] = acc * float64(bS[n])
	}
	return out
}

// maxRelErr is max |got-want| over max |want|.
func maxRelErr(got []float32, want []float64) float64 {
	var e, m float64
	for i := range want {
		e = max(e, math.Abs(float64(got[i])-want[i]))
		m = max(m, math.Abs(want[i]))
	}
	return e / m
}

// TestActG32_quantizeAndGEMVMatchLinalg holds the per-32 twins to aikit's CPU per-32 path on an outlier input: the
// identity quantizer's codes and scales are linalg.QuantizeActivationsGroupedInto's exactly; the GEMV, its bias and
// residual epilogues are the grouped product to f32 accumulation; the planted per-vector scale index
// (ActG32PerVectorScaleForTest) fails the same bar.
func TestActG32_quantizeAndGEMVMatchLinalg(t *testing.T) {
	c, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	defer c.Close()
	if err := c.ensureActG32(); err != nil {
		t.Fatal(err)
	}
	const K, N = 512, 96
	rng := rand.New(rand.NewSource(32))
	a := actG32Input(rng, K)
	src, q, s := c.sbuf(a), c.zbuf(K/4), c.zbuf(K/32)
	if err := c.dispatchI8(c.actG32.quantize.pl, c.actG32.quantize.ly, 1, []*wgpu.Buffer{src, q, s}, c.ubuf([]uint32{1, K, K, 0})); err != nil {
		t.Fatal(err)
	}
	gotQ, gotS := actG32Codes(c.readWords(q, K/4), K), c.readF(s, K/32)
	wantQ, wantS := make([]int8, K), make([]float32, K/32)
	linalg.QuantizeActivationsGroupedInto(wantQ, wantS, a, 1, K, 32)
	for g := range wantS {
		if gotS[g] != wantS[g] {
			t.Errorf("group %d: scale %g, linalg %g", g, gotS[g], wantS[g])
		}
	}
	for k := range wantQ {
		if gotQ[k] != wantQ[k] {
			t.Errorf("element %d: code %d, linalg %d", k, gotQ[k], wantQ[k])
		}
	}

	bq, bS := linalg.QuantizeRowsInt8(randF(rng, N*K), N, K)
	rm, err := c.UploadW8A8(bq, bS, N, K)
	if err != nil {
		t.Fatal(err)
	}
	defer rm.Release()
	want := groupedRef(wantQ, wantS, bq, bS, N, K)
	bias, resid := randF(rng, N), randF(rng, N)
	gx, gy := GemvGrid(N)
	run := func(p g32Pipe, withBias, addResidual bool) []float32 {
		aqB, asB, dst := c.sbuf(packI8(wantQ)), c.sbuf(wantS), c.sbuf(resid)
		dims := []uint32{1, uint32(rm.kp), N, 0}
		if addResidual {
			dims[3] = 1
		}
		bufs := []*wgpu.Buffer{aqB, rm.bq, asB, rm.bScales, dst}
		tail := []*wgpu.Buffer{c.ubuf(dims)} // bindings 5 (dims) and, for the bias twin, 6
		if withBias {
			tail = append(tail, c.sbuf(bias))
		}
		if err := c.dispatchI8Grid(p.pl, p.ly, int(gx), int(gy), bufs, tail...); err != nil {
			t.Fatal(err)
		}
		return c.readF(dst, N)
	}
	const bar = 1e-5
	if e := maxRelErr(run(c.actG32.gemv, false, false), want); e > bar {
		t.Errorf("gemv: max rel err %.3g, want <= %g", e, bar)
	}
	withResid := make([]float64, N)
	withBias := make([]float64, N)
	for n := range N {
		withResid[n], withBias[n] = want[n]+float64(resid[n]), want[n]+float64(bias[n])
	}
	if e := maxRelErr(run(c.actG32.gemv, false, true), withResid); e > bar {
		t.Errorf("gemv addResidual: max rel err %.3g, want <= %g", e, bar)
	}
	if e := maxRelErr(run(c.actG32.gemvBias, true, false), withBias); e > bar {
		t.Errorf("gemv bias: max rel err %.3g, want <= %g", e, bar)
	}

	ActG32PerVectorScaleForTest = true
	c.actG32 = nil
	err = c.ensureActG32()
	ActG32PerVectorScaleForTest = false
	if err != nil {
		t.Fatal(err)
	}
	if e := maxRelErr(run(c.actG32.gemv, false, false), want); e <= bar {
		t.Errorf("plant (group 0's scale for every group): max rel err %.3g still passes the %g bar", e, bar)
	}
}

// TestActG32_fusedQuantizersMatchCPU holds the RMSNorm and gated quantizers' per-32 twins to a float64 evaluation of
// their value functions, quantized by linalg.QuantizeActivationsGroupedInto: every scale to 1e-6 and every code within
// one (the GPU's f32 RMSNorm inverse and exp differ from the float64 ones by an ulp, which can move a rounding).
func TestActG32_fusedQuantizersMatchCPU(t *testing.T) {
	c, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	defer c.Close()
	if err := c.ensureActG32(); err != nil {
		t.Fatal(err)
	}
	const K = 768
	rng := rand.New(rand.NewSource(33))
	check := func(name string, p g32Pipe, inputs [][]float32, uni []uint32, val func(k int) float64) {
		t.Helper()
		ref := make([]float32, K)
		for k := range ref {
			ref[k] = float32(val(k))
		}
		wantQ, wantS := make([]int8, K), make([]float32, K/32)
		linalg.QuantizeActivationsGroupedInto(wantQ, wantS, ref, 1, K, 32)
		bufs := make([]*wgpu.Buffer, 0, len(inputs)+2)
		for _, in := range inputs {
			bufs = append(bufs, c.sbuf(in))
		}
		q, s := c.zbuf(K/4), c.zbuf(K/32)
		bufs = append(bufs, q, s)
		if err := c.dispatchI8(p.pl, p.ly, 1, bufs, c.ubuf(uni)); err != nil {
			t.Fatal(err)
		}
		gotQ, gotS := actG32Codes(c.readWords(q, K/4), K), c.readF(s, K/32)
		for g := range wantS {
			if d := math.Abs(float64(gotS[g]-wantS[g])) / float64(wantS[g]); d > 1e-6 {
				t.Errorf("%s group %d: scale %g, want %g", name, g, gotS[g], wantS[g])
			}
		}
		for k := range wantQ {
			if d := int(gotQ[k]) - int(wantQ[k]); d < -1 || d > 1 {
				t.Errorf("%s element %d: code %d, want %d", name, k, gotQ[k], wantQ[k])
			}
		}
	}
	x, w := actG32Input(rng, K), randF(rng, K)
	const eps = 1e-5
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	inv := 1 / math.Sqrt(ss/K+eps)
	check("rmsnormQuantG32", c.actG32.rmsQuant, [][]float32{x, w}, []uint32{K, math.Float32bits(eps), 0, K},
		func(k int) float64 { return float64(x[k]) * inv * float64(w[k]) })
	check("rmsnormQuantG32 addOne", c.actG32.rmsQuant, [][]float32{x, w}, []uint32{K, math.Float32bits(eps), 1, K},
		func(k int) float64 { return float64(x[k]) * inv * (float64(w[k]) + 1) })
	gate, up := actG32Input(rng, K), randF(rng, K)
	check("swigluQuantG32", c.actG32.swigluQuant, [][]float32{gate, up}, []uint32{K, K, 0, 0},
		func(k int) float64 { g := float64(gate[k]); return g / (1 + math.Exp(-g)) * float64(up[k]) })
	check("gegluQuantG32", c.actG32.gegluQuant, [][]float32{gate, up}, []uint32{K, K, 0, 0},
		func(k int) float64 {
			g := float64(gate[k])
			a := 0.7978845608028654 * (g + 0.044715*g*g*g)
			return 0.5 * g * (1 + math.Tanh(max(-15, min(15, a)))) * float64(up[k])
		})
}
