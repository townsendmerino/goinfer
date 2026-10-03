//go:build darwin

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// TestAttentionKernelsPastTileBound is what keeps metalCtxCapMax (32768) a fact for decode: the three decode attention
// kernels that hold scores in `threadgroup float sc[4096]` (attention on the f16 cache, attention_f32, attention_i8)
// handle more keys than one tile by online-softmax tiling, and this runs the shipped kernels across that boundary
// against a float64 reference. It replaces TestMetalCtxCapWithinKernelBound, which asserted two constants and touched
// no kernel (F-G03, docs/audit-metal-2026-09-30.md). The library is compiled as buildResident compiles it, and the
// launch is encodeAttention's: one tgReduceAttn-wide threadgroup per query head.
//
// Each case also checks that the test can fail: the answer a kernel would give if it stopped after its first tile
// must miss the bar. The exact prefill kernel does not tile; PrefillLast declines it past prefillExactAttnMaxKeys
// (TestPrefill_exactAttentionDeclinesPast4096Keys).
func TestAttentionKernelsPastTileBound(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer d.ReleaseObjects()
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	q := d.NewCommandQueue()
	type tc struct {
		nKeys, window int
		sink          bool
	}
	cases := []tc{
		{4096, 0, false}, {4097, 0, false}, {8192, 0, false}, {8193, 0, false}, {12289, 0, false},
		{9000, 4096, false}, // a window inside one tile, starting past key 4096
		{9000, 6000, false}, // a window spanning two tiles
		{8193, 0, true},     // an attention sink, which joins the max and the denominator
	}
	const nH, nKV, tol = 4, 2, 5e-5
	rng := rand.New(rand.NewSource(20261001))
	var checked int
	for _, kernel := range []string{"attention", "attention_f32", "attention_i8"} {
		pipe, err := d.NewComputePipeline(lib, kernel)
		if err != nil {
			t.Fatalf("pipeline %s: %v", kernel, err)
		}
		for _, hd := range []int{64, 128} {
			for _, c := range cases {
				if c.sink && kernel == "attention_f32" {
					continue // attention_f32 takes no sink arguments
				}
				name := fmt.Sprintf("%s hd=%d nKeys=%d window=%d sink=%v", kernel, hd, c.nKeys, c.window, c.sink)
				kvDim := nKV * hd
				qv := randUnit(rng, nH*hd)
				kf, vf := randUnit(rng, c.nKeys*kvDim), randUnit(rng, c.nKeys*kvDim)
				winStart := 0
				if c.window > 0 && c.nKeys > c.window {
					winStart = c.nKeys - c.window
				}
				if c.nKeys-winStart > attnScoreTileBound {
					// Plant one dominant key in the last tile (its score is several units above the rest for the group's
					// first head), so dropping a later tile moves the answer far past the bar rather than by one key's
					// 1/4097 share of random scores.
					for kvh := range nKV {
						h0 := kvh * (nH / nKV)
						for i := range hd {
							kf[(c.nKeys-1)*kvDim+kvh*hd+i] = 4 * qv[h0*hd+i]
						}
					}
				}
				sinks := []float32{0, 0, 0, 0}
				if c.sink {
					sinks = []float32{1.5, -0.5, 3, 0.25}
				}
				// The cache as the kernel stores it, and the values the reference must use.
				var kBuf, vBuf, ksBuf, vsBuf Buffer
				switch kernel {
				case "attention":
					kh, vh := make([]uint16, len(kf)), make([]uint16, len(vf))
					for i := range kf {
						kh[i], vh[i] = f32ToF16(kf[i]), f32ToF16(vf[i])
						kf[i], vf[i] = f16ToF32(kh[i]), f16ToF32(vh[i])
					}
					kBuf, vBuf = NewBufferU16s(d, kh), NewBufferU16s(d, vh)
				case "attention_f32":
					kBuf, vBuf = NewBufferFloats(d, kf), NewBufferFloats(d, vf)
				case "attention_i8":
					kq, ks := quantKVRowsI8(kf, c.nKeys, nKV, hd)
					vq, vs := quantKVRowsI8(vf, c.nKeys, nKV, hd)
					for i := range kf {
						p, h := i/kvDim, (i%kvDim)/hd
						kf[i], vf[i] = float32(kq[i])*ks[p*nKV+h], float32(vq[i])*vs[p*nKV+h]
					}
					kBuf, vBuf, ksBuf, vsBuf = NewBufferInt8(d, kq), NewBufferInt8(d, vq), NewBufferFloats(d, ks), NewBufferFloats(d, vs)
				}
				scale := float32(1 / math.Sqrt(float64(hd)))
				out := d.NewBufferLen(nH * hd)
				u := func(v int) Buffer { return NewBufferU32(d, uint32(v)) }
				hasSink := 0
				if c.sink {
					hasSink = 1
				}
				var bufs []Buffer
				switch kernel {
				case "attention":
					bufs = []Buffer{NewBufferFloats(d, qv), kBuf, vBuf, out, u(nH), u(nKV), u(hd), u(c.nKeys),
						NewBufferFloats(d, []float32{scale}), u(c.window), NewBufferFloats(d, sinks), u(hasSink)}
				case "attention_f32":
					bufs = []Buffer{NewBufferFloats(d, qv), kBuf, vBuf, out, u(nH), u(nKV), u(hd), u(c.nKeys),
						NewBufferFloats(d, []float32{scale}), u(c.window)}
				case "attention_i8":
					bufs = []Buffer{NewBufferFloats(d, qv), kBuf, vBuf, ksBuf, vsBuf, out, u(nH), u(nKV), u(hd), u(c.nKeys),
						NewBufferFloats(d, []float32{scale}), u(c.window), NewBufferFloats(d, sinks), u(hasSink)}
				}
				e := q.Begin()
				e.Dispatch(pipe, nH*tgReduceAttn, tgReduceAttn, bufs...)
				e.End()
				if err := e.Err(); err != nil {
					t.Fatalf("%s: dispatch: %v", name, err)
				}
				got := append([]float32(nil), out.Floats()[:nH*hd]...)
				d.ReleaseAll() // this case's buffers; the next case allocates its own
				want := attnRef64(qv, kf, vf, sinks, c.sink, nH, nKV, hd, winStart, c.nKeys, float64(scale))
				firstTile := attnRef64(qv, kf, vf, sinks, c.sink, nH, nKV, hd, winStart, min(c.nKeys, winStart+attnScoreTileBound), float64(scale))
				if e := maxAbsDiff(got, want); !(e <= tol) {
					t.Errorf("%s: max |kernel - float64 reference| = %.3g, want <= %g", name, e, tol)
				}
				if c.nKeys-winStart > attnScoreTileBound {
					if e := maxAbsDiff(firstTile, want); e <= 10*tol {
						t.Errorf("%s: a kernel that stopped after its first tile would differ by only %.3g; the case cannot catch it", name, e)
					}
				}
				checked++
			}
		}
	}
	t.Logf("%d kernel × head-dim × key-count cases within %g of the float64 reference", checked, tol)
}

func randUnit(rng *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = rng.Float32()*2 - 1
	}
	return v
}

// quantKVRowsI8 quantizes [nKeys][nKV][hd] values as kv_store_i8 does: one max/127 scale per position and KV head.
func quantKVRowsI8(x []float32, nKeys, nKV, hd int) ([]int8, []float32) {
	q, sc := make([]int8, len(x)), make([]float32, nKeys*nKV)
	for p := range nKeys {
		for h := range nKV {
			row := x[(p*nKV+h)*hd : (p*nKV+h+1)*hd]
			var mx float32
			for _, v := range row {
				mx = max(mx, float32(math.Abs(float64(v))))
			}
			s := mx / 127
			if s == 0 {
				s = 1
			}
			sc[p*nKV+h] = s
			for i, v := range row {
				q[(p*nKV+h)*hd+i] = int8(max(-127, min(127, math.Round(float64(v*(1/s))))))
			}
		}
	}
	return q, sc
}

// attnRef64 is single-query attention in float64 over keys [from, to), with an optional per-head sink logit that
// joins the max and the denominator only.
func attnRef64(q, k, v, sinks []float32, sink bool, nH, nKV, hd, from, to int, scale float64) []float32 {
	out := make([]float32, nH*hd)
	kvDim := nKV * hd
	for h := range nH {
		kvh := h / (nH / nKV)
		scores := make([]float64, to-from)
		mx := math.Inf(-1)
		for s := from; s < to; s++ {
			var a float64
			for i := range hd {
				a += float64(q[h*hd+i]) * float64(k[s*kvDim+kvh*hd+i])
			}
			scores[s-from] = a * scale
			mx = math.Max(mx, scores[s-from])
		}
		if sink {
			mx = math.Max(mx, float64(sinks[h]))
		}
		var den float64
		for i := range scores {
			scores[i] = math.Exp(scores[i] - mx)
			den += scores[i]
		}
		if sink {
			den += math.Exp(float64(sinks[h]) - mx)
		}
		for i := range hd {
			var a float64
			for s := from; s < to; s++ {
				a += scores[s-from] * float64(v[s*kvDim+kvh*hd+i])
			}
			out[h*hd+i] = float32(a / den)
		}
	}
	return out
}

func maxAbsDiff(a, b []float32) float64 {
	var m float64
	for i := range a {
		d := math.Abs(float64(a[i]) - float64(b[i]))
		if math.IsNaN(d) {
			return math.NaN()
		}
		m = math.Max(m, d)
	}
	return m
}
