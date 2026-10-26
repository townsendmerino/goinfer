//go:build darwin

package gpudev

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"testing"
)

// metal_vit_test.go gates each Metal encoder kernel INDIVIDUALLY against a Go reference
// that reimplements vision/encoder.go's arithmetic — the same discipline as
// cuda_vit_test.go. Gating one op at a time is the point: a tower cosine stays high while
// a single op is subtly wrong. The tolerances differ from the CUDA bars where a kernel's
// reduction ran in double on CUDA (MSL has no double); each such case states the achieved
// f32 number and the reason. The exact ones (int8 dot, elementwise adds, byte-exact quant)
// hold the same tight bounds.

func vitSetupM(t testing.TB) (*Device, Queue, ViT) {
	t.Helper()
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no Metal device: %v", err)
	}
	t.Cleanup(func() { d.ReleaseObjects(); d.ReleaseAll() })
	v, err := d.NewViT()
	if err != nil {
		t.Fatalf("NewViT: %v", err)
	}
	return d, d.NewCommandQueue(), v
}

func randF32M(rng *rand.Rand, n int, scale float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(rng.NormFloat64()) * scale
	}
	return out
}

func maxAbsDiffM(a, b []float32) float64 {
	worst := 0.0
	for i := range a {
		if d := math.Abs(float64(a[i]) - float64(b[i])); d > worst {
			worst = d
		}
	}
	return worst
}

// i32b/f32b bind an int/float scalar as a 1-element buffer — Metal has no by-value arg.
func i32b(d *Device, v int) Buffer     { return NewBufferOf(d, []uint32{uint32(int32(v))}) }
func f32b(d *Device, v float32) Buffer { return NewBufferOf(d, []float32{v}) }

func mtg(n int) int {
	if n < ViTBlock {
		return n
	}
	return ViTBlock
}

// run1d / run1dTG pin the OS thread across the dispatch: Run1D allocs+drains an
// NSAutoreleasePool, which objc requires on one thread (a migrating goroutine crashes).
func run1d(q Queue, p Pipeline, n, tg int, bufs ...Buffer) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	q.Run1D(p, n, tg, bufs...)
}
func run1dTG(q Queue, p Pipeline, n, tg, tgBytes int, bufs ...Buffer) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	q.Run1DTG(p, n, tg, tgBytes, bufs...)
}
func run2d(q Queue, p Pipeline, gx, gy, tgx, tgy int, bufs ...Buffer) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	q.Run2D(p, gx, gy, tgx, tgy, bufs...)
}

func TestMetal_vitLayerNorm(t *testing.T) {
	d, q, v := vitSetupM(t)
	rng := rand.New(rand.NewSource(1))
	worstAll := 0.0
	for _, shape := range []struct{ rows, dim int }{{1, 32}, {16, 32}, {7, 257}, {64, 1152}} {
		x := randF32M(rng, shape.rows*shape.dim, 2)
		w := randF32M(rng, shape.dim, 1)
		b := randF32M(rng, shape.dim, 1)
		const eps = 1e-6

		want := make([]float32, len(x))
		for r := range shape.rows {
			xr := x[r*shape.dim : (r+1)*shape.dim]
			var mean float64
			for _, val := range xr {
				mean += float64(val)
			}
			mean /= float64(shape.dim)
			var variance float64
			for _, val := range xr {
				dv := float64(val) - mean
				variance += dv * dv
			}
			variance /= float64(shape.dim)
			inv := 1.0 / math.Sqrt(variance+eps)
			for i := range shape.dim {
				want[r*shape.dim+i] = float32((float64(xr[i])-mean)*inv)*w[i] + b[i]
			}
		}
		dx, dw, db := NewBufferOf(d, x), NewBufferOf(d, w), NewBufferOf(d, b)
		out := d.NewBufferLen(len(x))
		run1d(q, v.LayerNorm, shape.rows*ViTBlock, ViTBlock,
			dx, dw, db, out, i32b(d, shape.rows), i32b(d, shape.dim), f32b(d, eps))
		if dmax := maxAbsDiffM(out.Floats(), want); dmax > worstAll {
			worstAll = dmax
		}
	}
	t.Logf("layernorm f32-reduction worst Δ %.3g vs double CPU (CUDA bar was 1e-5)", worstAll)
	// f32 pairwise reduction, not double — allow a looser, stated bound.
	if worstAll > 5e-5 {
		t.Errorf("layernorm worst Δ %.3g exceeds the stated f32 bound 5e-5", worstAll)
	}
}

func TestMetal_vitGELU(t *testing.T) {
	d, q, v := vitSetupM(t)
	rng := rand.New(rand.NewSource(2))
	x := randF32M(rng, 4096, 3)
	const c = 0.7978845608028654
	want := make([]float32, len(x))
	for i, val := range x {
		vv := float64(val)
		want[i] = float32(0.5 * vv * (1.0 + math.Tanh(c*(vv+0.044715*vv*vv*vv))))
	}
	dx := NewBufferOf(d, x)
	run1d(q, v.GELUTanh, len(x), mtg(len(x)), dx, i32b(d, len(x)))
	got := append([]float32(nil), dx.Floats()...)
	dmax := maxAbsDiffM(got, want)
	t.Logf("gelu_tanh f32 worst Δ %.3g vs double CPU (CUDA bar was 1e-6)", dmax)
	if dmax > 1e-5 {
		t.Errorf("gelu worst Δ %.3g exceeds the stated f32 bound 1e-5", dmax)
	}
	// Break-it-first: erf-GELU must be rejected by the discriminating bound — proving
	// the kernel is tanh-GELU, not just "a GELU". erf vs tanh differ ~5e-4.
	erf := make([]float32, len(x))
	for i, val := range x {
		vv := float64(val)
		erf[i] = float32(0.5 * vv * (1.0 + math.Erf(vv/math.Sqrt2)))
	}
	if dmax := maxAbsDiffM(erf, want); dmax <= 1e-5 {
		t.Error("break-it-first vacuous: erf-GELU indistinguishable at 1e-5")
	} else {
		t.Logf("erf-GELU rejected at Δ %.3g — the gate is tanh-specific", dmax)
	}
}

func TestMetal_vitQuantRows(t *testing.T) {
	d, q, v := vitSetupM(t)
	rng := rand.New(rand.NewSource(3))
	const rows, dim = 12, 97
	x := randF32M(rng, rows*dim, 2)
	for i := range dim {
		x[5*dim+i] = 0 // all-zero row
	}
	wantQ := make([]int8, rows*dim)
	wantS := make([]float32, rows)
	for r := range rows {
		xr := x[r*dim : (r+1)*dim]
		var maxAbs float32
		for _, val := range xr {
			if val < 0 {
				val = -val
			}
			if val > maxAbs {
				maxAbs = val
			}
		}
		if maxAbs == 0 {
			continue
		}
		scale := maxAbs / 127
		wantS[r] = scale
		inv := 1.0 / scale
		for i, val := range xr {
			vv := math.Round(float64(val * inv))
			if vv > 127 {
				vv = 127
			} else if vv < -127 {
				vv = -127
			}
			wantQ[r*dim+i] = int8(vv)
		}
	}
	dx := NewBufferOf(d, x)
	dq := NewBufferOf(d, make([]int8, rows*dim))
	ds := d.NewBufferLen(rows)
	run1d(q, v.QuantRows, rows*ViTBlock, ViTBlock, dx, dq, ds, i32b(d, rows), i32b(d, dim))
	gotQ, gotS := dq.Int8s(), ds.Floats()
	for i := range wantQ {
		if gotQ[i] != wantQ[i] {
			t.Fatalf("q[%d]=%d want %d (row %d)", i, gotQ[i], wantQ[i], i/dim)
		}
	}
	for r := range rows {
		if gotS[r] != wantS[r] {
			t.Fatalf("scale[%d]=%v want %v", r, gotS[r], wantS[r])
		}
	}
	if gotS[5] != 0 {
		t.Errorf("all-zero row scale=%v want 0", gotS[5])
	}
	t.Log("quant_rows byte-exact vs linalg quantizeRowInt8 (incl. all-zero row)")
}

func TestMetal_vitGEMMs(t *testing.T) {
	d, q, v := vitSetupM(t)
	rng := rand.New(rand.NewSource(4))
	for _, s := range []struct{ M, N, K int }{{1, 8, 16}, {16, 32, 32}, {17, 33, 65}} {
		A := make([]int8, s.M*s.K)
		B := make([]int8, s.N*s.K)
		for i := range A {
			A[i] = int8(rng.Intn(255) - 127)
		}
		for i := range B {
			B[i] = int8(rng.Intn(255) - 127)
		}
		as := randF32M(rng, s.M, 0.01)
		bs := randF32M(rng, s.N, 0.01)
		want := make([]float32, s.M*s.N)
		for m := range s.M {
			for n := range s.N {
				var acc int32
				for k := range s.K {
					acc += int32(A[m*s.K+k]) * int32(B[n*s.K+k])
				}
				want[m*s.N+n] = float32(acc) * as[m] * bs[n]
			}
		}
		dC := d.NewBufferLen(s.M * s.N)
		run1d(q, v.GEMMW8A8, s.M*s.N, mtg(s.M*s.N),
			NewBufferOf(d, A), NewBufferOf(d, as), NewBufferOf(d, B), NewBufferOf(d, bs), dC,
			i32b(d, s.M), i32b(d, s.N), i32b(d, s.K))
		if dmax := maxAbsDiffM(dC.Floats(), want); dmax > 1e-4 {
			t.Fatalf("W8A8 %dx%dx%d worst Δ %.3g", s.M, s.N, s.K, dmax)
		}
		Af := randF32M(rng, s.M*s.K, 1)
		Bf := randF32M(rng, s.N*s.K, 1)
		wantF := make([]float32, s.M*s.N)
		for m := range s.M {
			for n := range s.N {
				var acc float32
				for k := range s.K {
					acc += Af[m*s.K+k] * Bf[n*s.K+k]
				}
				wantF[m*s.N+n] = acc
			}
		}
		dCf := d.NewBufferLen(s.M * s.N)
		run1d(q, v.GEMMF32, s.M*s.N, mtg(s.M*s.N),
			NewBufferOf(d, Af), NewBufferOf(d, Bf), dCf, i32b(d, s.M), i32b(d, s.N), i32b(d, s.K))
		if dmax := maxAbsDiffM(dCf.Floats(), wantF); dmax > 1e-3 {
			t.Fatalf("f32 %dx%dx%d worst Δ %.3g", s.M, s.N, s.K, dmax)
		}
	}
	t.Log("gemm_w8a8 ≡ exact int32; gemm_f32 ≡ f32 reference (3 shapes each)")
}

func TestMetal_vitAttention(t *testing.T) {
	d, q, v := vitSetupM(t)
	rng := rand.New(rand.NewSource(5))
	worstAll := 0.0
	for _, s := range []struct{ np, nH, hd int }{{16, 2, 16}, {5, 1, 8}, {64, 4, 24}} {
		hidden := s.nH * s.hd
		qv := randF32M(rng, s.np*hidden, 1)
		kv := randF32M(rng, s.np*hidden, 1)
		vv := randF32M(rng, s.np*hidden, 1)
		scale := float32(1.0 / math.Sqrt(float64(s.hd)))
		want := make([]float32, s.np*hidden)
		scores := make([]float64, s.np)
		for h := range s.nH {
			off := h * s.hd
			for i := range s.np {
				maxv := math.Inf(-1)
				for j := range s.np {
					var acc float32
					for dd := range s.hd {
						acc += qv[i*hidden+off+dd] * kv[j*hidden+off+dd]
					}
					sv := float64(acc * scale)
					scores[j] = sv
					if sv > maxv {
						maxv = sv
					}
				}
				var sum float64
				for j := range s.np {
					e := math.Exp(scores[j] - maxv)
					scores[j] = e
					sum += e
				}
				for dd := range s.hd {
					var acc float64
					for j := range s.np {
						acc += (scores[j] / sum) * float64(vv[j*hidden+off+dd])
					}
					want[i*hidden+off+dd] = float32(acc)
				}
			}
		}
		out := d.NewBufferLen(s.np * hidden)
		run1dTG(q, v.Attention, s.np*s.nH*ViTBlock, ViTBlock, s.np*4,
			NewBufferOf(d, qv), NewBufferOf(d, kv), NewBufferOf(d, vv), out,
			i32b(d, s.np), i32b(d, s.nH), i32b(d, s.hd), f32b(d, scale))
		if dmax := maxAbsDiffM(out.Floats(), want); dmax > worstAll {
			worstAll = dmax
		}
	}
	t.Logf("attention f32-softmax worst Δ %.3g vs double CPU (CUDA bar was 1e-5)", worstAll)
	if worstAll > 5e-5 {
		t.Errorf("attention worst Δ %.3g exceeds the stated f32 bound 5e-5", worstAll)
	}
}

func TestMetal_vitAddOps(t *testing.T) {
	d, q, v := vitSetupM(t)
	rng := rand.New(rand.NewSource(6))
	const rows, dim = 9, 40
	x := randF32M(rng, rows*dim, 1)
	bias := randF32M(rng, dim, 1)
	vec := randF32M(rng, rows*dim, 1)
	wantBias := make([]float32, len(x))
	for r := range rows {
		for i := range dim {
			wantBias[r*dim+i] = x[r*dim+i] + bias[i]
		}
	}
	wantVec := make([]float32, len(x))
	for i := range x {
		wantVec[i] = x[i] + vec[i]
	}
	dx := NewBufferOf(d, x)
	run1d(q, v.AddBias, rows*dim, mtg(rows*dim), dx, NewBufferOf(d, bias), i32b(d, rows), i32b(d, dim))
	if dmax := maxAbsDiffM(dx.Floats(), wantBias); dmax > 1e-6 {
		t.Fatalf("add_bias worst Δ %.3g", dmax)
	}
	dx2 := NewBufferOf(d, x)
	run1d(q, v.AddVec, rows*dim, mtg(rows*dim), dx2, NewBufferOf(d, vec), i32b(d, rows*dim))
	if dmax := maxAbsDiffM(dx2.Floats(), wantVec); dmax > 1e-6 {
		t.Fatalf("add_vec worst Δ %.3g", dmax)
	}
	t.Log("add_bias / add_vec ≡ CPU references")
}

// --- Qwen2.5-VL kernel gates (Metal mirror of cuda_vit_test.go's Qwen section) ---

// segb binds int32 per-patch segment bounds as a device buffer. Non-negative bounds are
// bit-identical as uint32, and the kernel reads them as `device const int*`.
func segb(d *Device, s []int32) Buffer {
	u := make([]uint32, len(s))
	for i, v := range s {
		u[i] = uint32(v)
	}
	return NewBufferOf(d, u)
}

// TestMetal_vitRMSNorm gates weight-only RMSNorm against vision/qwen_encoder.go's
// rmsNorm: no mean subtraction, no bias, eps INSIDE the mean-square. The mean-square
// reduces in f32 (no double in MSL), so the bar is looser than the CUDA 1e-5, stated.
func TestMetal_vitRMSNorm(t *testing.T) {
	d, q, v := vitSetupM(t)
	rng := rand.New(rand.NewSource(21))
	const eps = 1e-6
	worstAll := 0.0
	for _, shape := range []struct{ rows, dim int }{{1, 32}, {12, 128}, {7, 257}, {64, 1152}} {
		x := randF32M(rng, shape.rows*shape.dim, 2)
		w := randF32M(rng, shape.dim, 1)
		want := make([]float32, len(x))
		for r := range shape.rows {
			xr := x[r*shape.dim : (r+1)*shape.dim]
			var ss float64
			for _, val := range xr {
				ss += float64(val) * float64(val)
			}
			inv := 1.0 / math.Sqrt(ss/float64(shape.dim)+eps)
			for i := range shape.dim {
				want[r*shape.dim+i] = float32(float64(xr[i])*inv) * w[i]
			}
		}
		out := d.NewBufferLen(len(x))
		run1d(q, v.RMSNorm, shape.rows*ViTBlock, ViTBlock,
			NewBufferOf(d, x), NewBufferOf(d, w), out, i32b(d, shape.rows), i32b(d, shape.dim), f32b(d, eps))
		if dmax := maxAbsDiffM(out.Floats(), want); dmax > worstAll {
			worstAll = dmax
		}
	}
	t.Logf("rmsnorm f32-reduction worst Δ %.3g vs double CPU (CUDA bar was 1e-5)", worstAll)
	if worstAll > 5e-5 {
		t.Errorf("rmsnorm worst Δ %.3g exceeds the stated f32 bound 5e-5", worstAll)
	}
}

// TestMetal_vitGELUErf gates the EXACT (erf) GELU and — the point of a separate kernel —
// proves it is measurably DIFFERENT from gelu_tanh. Qwen's merger uses erf; SigLIP uses
// tanh. MSL has no stdlib erf, so gelu_erf carries an A&S approximation; the bar vs the
// float64 reference is a stated f32 bound, but the distinctness from tanh (~5e-4) is what
// makes shipping the right one non-vacuous.
func TestMetal_vitGELUErf(t *testing.T) {
	d, q, v := vitSetupM(t)
	rng := rand.New(rand.NewSource(22))
	x := randF32M(rng, 4096, 3)
	want := make([]float32, len(x))
	for i, val := range x {
		vv := float64(val)
		want[i] = float32(0.5 * vv * (1.0 + math.Erf(vv/math.Sqrt2)))
	}
	dx := NewBufferOf(d, x)
	run1d(q, v.GELUErf, len(x), mtg(len(x)), dx, i32b(d, len(x)))
	got := append([]float32(nil), dx.Floats()...)
	dm := maxAbsDiffM(got, want)
	t.Logf("gelu_erf f32-approx worst Δ %.3g vs double math.Erf (CUDA bar was 1e-6)", dm)
	if dm > 5e-5 {
		t.Fatalf("gelu_erf worst Δ %.3g exceeds the stated f32 bound 5e-5", dm)
	}
	dx2 := NewBufferOf(d, x)
	run1d(q, v.GELUTanh, len(x), mtg(len(x)), dx2, i32b(d, len(x)))
	sep := maxAbsDiffM(dx2.Floats(), got)
	if sep <= 1e-6 {
		t.Errorf("gelu_erf and gelu_tanh are indistinguishable (Δ %.3g) — one of them is wrong", sep)
	}
	t.Logf("gelu_erf distinct from gelu_tanh by Δ %.3g", sep)
}

// TestMetal_vitSiLUMul gates gate = silu(gate) * up. silu runs in f32 here (no double),
// so the bar is a stated f32 bound rather than the CUDA 1e-6.
func TestMetal_vitSiLUMul(t *testing.T) {
	d, q, v := vitSetupM(t)
	rng := rand.New(rand.NewSource(23))
	gate := randF32M(rng, 2048, 2)
	up := randF32M(rng, 2048, 2)
	want := make([]float32, len(gate))
	for i := range gate {
		vv := float64(gate[i])
		want[i] = float32(vv/(1.0+math.Exp(-vv))) * up[i]
	}
	dg := NewBufferOf(d, gate)
	run1d(q, v.SiLUMul, len(gate), mtg(len(gate)), dg, NewBufferOf(d, up), i32b(d, len(gate)))
	dm := maxAbsDiffM(dg.Floats(), want)
	t.Logf("silu_mul f32 worst Δ %.3g vs double CPU (CUDA bar was 1e-6)", dm)
	if dm > 5e-5 {
		t.Fatalf("silu_mul worst Δ %.3g exceeds the stated f32 bound 5e-5", dm)
	}
}

// TestMetal_vitRopeQK gates the in-place NeoX rotary on the q and k thirds of a fused
// qkv buffer, and that the v third is left ALONE — an off-by-one third would be invisible
// to a q/k-only check. Pure elementwise f32, so the tight bound holds.
func TestMetal_vitRopeQK(t *testing.T) {
	d, q, v := vitSetupM(t)
	rng := rand.New(rand.NewSource(24))
	const seq, nH, hd = 9, 3, 16
	hidden := nH * hd
	qkv := randF32M(rng, seq*3*hidden, 1)
	cos := randF32M(rng, seq*hd, 1)
	sin := randF32M(rng, seq*hd, 1)

	want := append([]float32(nil), qkv...)
	hf := hd / 2
	for i := range seq {
		co, si := cos[i*hd:(i+1)*hd], sin[i*hd:(i+1)*hd]
		for head := range nH {
			for _, base := range []int{i*3*hidden + head*hd, i*3*hidden + hidden + head*hd} {
				for dd := range hf {
					x, y := want[base+dd], want[base+dd+hf]
					want[base+dd] = x*co[dd] - y*si[dd]
					want[base+dd+hf] = y*co[dd+hf] + x*si[dd+hf]
				}
			}
		}
	}

	dq := NewBufferOf(d, qkv)
	run1d(q, v.RopeQK, seq*nH*hf, mtg(seq*nH*hf),
		dq, NewBufferOf(d, cos), NewBufferOf(d, sin), i32b(d, seq), i32b(d, nH), i32b(d, hd))
	got := dq.Floats()
	if dm := maxAbsDiffM(got, want); dm > 1e-5 {
		t.Fatalf("rope_qk worst Δ %.3g", dm)
	}
	for i := range seq {
		vb := i*3*hidden + 2*hidden
		for j := range hidden {
			if got[vb+j] != qkv[vb+j] {
				t.Fatalf("rope_qk modified v at patch %d dim %d", i, j)
			}
		}
	}
	t.Log("rope_qk ≡ CPU applyRotaryVision on q,k; v untouched")
}

// TestMetal_vitAttentionSeg gates segmented attention: each query attends only within
// its own segment, with a structural break-it-first — a single whole-sequence segment
// must give a different answer, or the bounds are being ignored.
func TestMetal_vitAttentionSeg(t *testing.T) {
	d, q, v := vitSetupM(t)
	rng := rand.New(rand.NewSource(25))
	const seq, nH, hd = 12, 2, 8
	hidden := nH * hd
	qkv := randF32M(rng, seq*3*hidden, 1)
	segStart := make([]int32, seq)
	segEnd := make([]int32, seq)
	maxSeg := 0
	for _, b := range [][2]int{{0, 5}, {5, 12}} {
		if b[1]-b[0] > maxSeg {
			maxSeg = b[1] - b[0]
		}
		for i := b[0]; i < b[1]; i++ {
			segStart[i], segEnd[i] = int32(b[0]), int32(b[1])
		}
	}
	scale := float32(1.0 / math.Sqrt(float64(hd)))

	want := make([]float32, seq*hidden)
	for h := range nH {
		off := h * hd
		for i := range seq {
			s0, s1 := int(segStart[i]), int(segEnd[i])
			sc := make([]float64, s1-s0)
			maxv := math.Inf(-1)
			for tt := s0; tt < s1; tt++ {
				var acc float32
				for dd := range hd {
					acc += qkv[i*3*hidden+off+dd] * qkv[tt*3*hidden+hidden+off+dd]
				}
				sc[tt-s0] = float64(acc * scale)
				if sc[tt-s0] > maxv {
					maxv = sc[tt-s0]
				}
			}
			var sum float64
			for j := range sc {
				sc[j] = math.Exp(sc[j] - maxv)
				sum += sc[j]
			}
			for dd := range hd {
				var acc float64
				for tt := s0; tt < s1; tt++ {
					acc += (sc[tt-s0] / sum) * float64(qkv[tt*3*hidden+2*hidden+off+dd])
				}
				want[i*hidden+off+dd] = float32(acc)
			}
		}
	}

	dq := NewBufferOf(d, qkv)
	out := d.NewBufferLen(seq * hidden)
	run1dTG(q, v.AttentionSeg, seq*nH*ViTBlock, ViTBlock, maxSeg*4,
		dq, out, segb(d, segStart), segb(d, segEnd), i32b(d, seq), i32b(d, nH), i32b(d, hd), f32b(d, scale))
	got := append([]float32(nil), out.Floats()...)
	dm := maxAbsDiffM(got, want)
	t.Logf("attention_seg f32-softmax worst Δ %.3g vs double CPU (CUDA bar was 1e-5)", dm)
	if dm > 5e-5 {
		t.Fatalf("attention_seg worst Δ %.3g exceeds the stated f32 bound 5e-5", dm)
	}

	one := make([]int32, seq)
	oneEnd := make([]int32, seq)
	for i := range seq {
		one[i], oneEnd[i] = 0, seq
	}
	out2 := d.NewBufferLen(seq * hidden)
	run1dTG(q, v.AttentionSeg, seq*nH*ViTBlock, ViTBlock, seq*4,
		dq, out2, segb(d, one), segb(d, oneEnd), i32b(d, seq), i32b(d, nH), i32b(d, hd), f32b(d, scale))
	if maxAbsDiffM(out2.Floats(), got) < 1e-4 {
		t.Error("break-it-first vacuous: windowed and full attention agree — bounds ignored?")
	}
	t.Log("attention_seg ≡ CPU per-segment MHA; full-attention differs (bounds enforced)")
}

// TestMetal_gemmTiledMatchesUntiled gates the tiled GEMMs asymmetrically, mirroring the
// CUDA gate.
//
// W8A8 must be BIT-IDENTICAL to the untiled kernel: the accumulator is int32 and integer
// addition is associative, so staging K through threadgroup memory in TILE-sized chunks
// cannot change the sum. Asserting equality rather than a tolerance is what makes this
// able to catch a real tiling bug — a mis-indexed shared tile shifts a handful of outputs,
// which a loose bound would wave through.
//
// The f32 kernel is gated against a FLOAT64 CPU reference, not against its untiled
// counterpart: comparing two f32 kernels to each other pins whichever way they happen to
// agree as the expectation, so a shared bug would pass. The tiled kernel is what the
// consumers run; the float64 reference is what it has to equal.
func TestMetal_gemmTiledMatchesUntiled(t *testing.T) {
	d, q, v := vitSetupM(t)
	rng := rand.New(rand.NewSource(31))
	for _, sh := range []struct{ M, N, K int }{
		{1, 1, 1},       // degenerate
		{16, 16, 16},    // exactly one tile
		{17, 33, 47},    // overhang in all three dims + K tail
		{64, 96, 129},   // K tail
		{129, 65, 1152}, // realistic ViT projection
	} {
		A := make([]int8, sh.M*sh.K)
		B := make([]int8, sh.N*sh.K)
		for i := range A {
			A[i] = int8(rng.Intn(255) - 127)
		}
		for i := range B {
			B[i] = int8(rng.Intn(255) - 127)
		}
		as, bs := randF32M(rng, sh.M, 0.01), randF32M(rng, sh.N, 0.01)
		dA, dB := NewBufferOf(d, A), NewBufferOf(d, B)
		dAs, dBs := NewBufferOf(d, as), NewBufferOf(d, bs)
		c1 := d.NewBufferLen(sh.M * sh.N) // untiled
		c2 := d.NewBufferLen(sh.M * sh.N) // tiled
		run1d(q, v.GEMMW8A8, sh.M*sh.N, mtg(sh.M*sh.N),
			dA, dAs, dB, dBs, c1, i32b(d, sh.M), i32b(d, sh.N), i32b(d, sh.K))
		gx, gy, tgx, tgy := TileDims(sh.M, sh.N)
		run2d(q, v.GEMMW8A8Tiled, gx, gy, tgx, tgy,
			dA, dAs, dB, dBs, c2, i32b(d, sh.M), i32b(d, sh.N), i32b(d, sh.K))
		g1, g2 := c1.Floats(), c2.Floats()
		for i := range g1 {
			if g1[i] != g2[i] {
				t.Fatalf("W8A8 M=%d N=%d K=%d: tiled[%d]=%v != untiled %v — int32 accumulation is order-independent, so this is a real tiling bug",
					sh.M, sh.N, sh.K, i, g2[i], g1[i])
			}
		}

		Af, Bf := randF32M(rng, sh.M*sh.K, 1), randF32M(rng, sh.N*sh.K, 1)
		f2 := d.NewBufferLen(sh.M * sh.N)
		run2d(q, v.GEMMF32Tiled, gx, gy, tgx, tgy, NewBufferOf(d, Af), NewBufferOf(d, Bf), f2,
			i32b(d, sh.M), i32b(d, sh.N), i32b(d, sh.K))
		h2 := f2.Floats()
		worst := 0.0
		for m := range sh.M {
			for n := range sh.N {
				var acc float64
				for k := range sh.K {
					acc += float64(Af[m*sh.K+k]) * float64(Bf[n*sh.K+k])
				}
				den := math.Abs(acc)
				if den < 1 {
					den = 1
				}
				if r := math.Abs(float64(h2[m*sh.N+n])-acc) / den; r > worst {
					worst = r
				}
			}
		}
		// Same analytic bound as the CUDA gate: a K-term f32 dot accumulates ~sqrt(K)*eps
		// against a float64 reference, ~5e-5 relative at K=1152; 2e-4 sits just above and
		// still rejects anything structural. The SHARP gate is the W8A8 bit-identity above.
		if worst > 2e-4 {
			t.Fatalf("f32 tiled M=%d N=%d K=%d: worst relative Δ %.3g vs float64 CPU", sh.M, sh.N, sh.K, worst)
		}
	}
	t.Log("gemm_w8a8_tiled BIT-IDENTICAL to untiled across 5 shapes; gemm_f32_tiled ≡ float64 CPU reference")
}

// TestMetal_gemmF32SG gates the production simdgroup_matrix GEMM against a FLOAT64
// reference (not against gemm_f32_tiled — comparing two f32 kernels pins whichever way they
// agree). The odd shapes are the point: {17,33,47} overhangs the 32×32 tile in M and N and
// the 8-wide K chunk, so the zero-padded edge staging that keeps simdgroup_load in bounds is
// actually exercised — a mis-set pad would corrupt exactly those tiles.
func TestMetal_gemmF32SG(t *testing.T) {
	d, q, v := vitSetupM(t)
	rng := rand.New(rand.NewSource(41))
	for _, sh := range []struct{ M, N, K int }{
		{1, 1, 1},        // degenerate
		{8, 8, 8},        // one simdgroup fragment
		{32, 32, 32},     // exactly one threadgroup tile
		{17, 33, 47},     // overhang in all three dims + K tail
		{64, 96, 129},    // K tail
		{129, 65, 1152},  // realistic ViT projection
		{512, 768, 3072}, // encoder batch shape
	} {
		Af, Bf := randF32M(rng, sh.M*sh.K, 1), randF32M(rng, sh.N*sh.K, 1)
		// GEMMF32Plan routes aligned shapes to gemm_f32_sg_big and the rest to gemm_f32_sg,
		// so gating through it covers BOTH kernels on the shapes each actually serves.
		p, gx, gy, tgx, tgy := v.GEMMF32Plan(sh.M, sh.N, sh.K)
		out := d.NewBufferLen(sh.M * sh.N)
		run2d(q, p, gx, gy, tgx, tgy, NewBufferOf(d, Af), NewBufferOf(d, Bf), out,
			i32b(d, sh.M), i32b(d, sh.N), i32b(d, sh.K))
		h := out.Floats()
		worst := 0.0
		for m := range sh.M {
			for n := range sh.N {
				var acc float64
				for k := range sh.K {
					acc += float64(Af[m*sh.K+k]) * float64(Bf[n*sh.K+k])
				}
				den := math.Abs(acc)
				if den < 1 {
					den = 1
				}
				if r := math.Abs(float64(h[m*sh.N+n])-acc) / den; r > worst {
					worst = r
				}
			}
		}
		// Same analytic f32 bound as the tiled f32 gate (~sqrt(K)*eps, ~5e-5 at K=1152).
		if worst > 2e-4 {
			t.Fatalf("gemm_f32 M=%d N=%d K=%d: worst relative Δ %.3g vs float64 CPU", sh.M, sh.N, sh.K, worst)
		}
	}
	t.Log("gemm_f32_sg + gemm_f32_sg_big (via GEMMF32Plan) ≡ float64 CPU across 7 shapes (incl. overhang + aligned 512×768×3072)")
}

// BenchmarkMetalGEMMF32 compares the scalar-tiled GEMM to the simdgroup_matrix one at ViT /
// encoder shapes, resident and warm (no host transfer) — the isolated kernel speedup that
// motivates shipping gemm_f32_sg. GFLOP/s in the metric.
func BenchmarkMetalGEMMF32(b *testing.B) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		b.Skipf("no Metal device: %v", err)
	}
	b.Cleanup(func() { d.ReleaseObjects(); d.ReleaseAll() })
	v, err := d.NewViT()
	if err != nil {
		b.Fatalf("NewViT: %v", err)
	}
	q := d.NewCommandQueue()
	rng := rand.New(rand.NewSource(42))
	for _, s := range []struct{ M, N, K int }{
		{128, 768, 768},
		{128, 768, 3072},
		{512, 768, 3072},
		{1024, 1024, 4096},
	} {
		dA := NewBufferOf(d, randF32M(rng, s.M*s.K, 1))
		dB := NewBufferOf(d, randF32M(rng, s.N*s.K, 1))
		dC := d.NewBufferLen(s.M * s.N)
		mf := 2.0 * float64(s.M) * float64(s.K) * float64(s.N) / 1e6
		name := fmt.Sprintf("M%d_K%d_N%d_%.0fMFLOP", s.M, s.K, s.N, mf)
		b.Run(name+"/tiled", func(b *testing.B) {
			gx, gy, tgx, tgy := TileDims(s.M, s.N)
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()
			for range b.N {
				q.Run2D(v.GEMMF32Tiled, gx, gy, tgx, tgy, dA, dB, dC, i32b(d, s.M), i32b(d, s.N), i32b(d, s.K))
			}
			b.ReportMetric(mf/1e3/(b.Elapsed().Seconds()/float64(b.N)), "GFLOP/s")
		})
		b.Run(name+"/simdgroup", func(b *testing.B) {
			gx, gy, tgx, tgy := SGDims(s.M, s.N)
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()
			for range b.N {
				q.Run2D(v.GEMMF32SG, gx, gy, tgx, tgy, dA, dB, dC, i32b(d, s.M), i32b(d, s.N), i32b(d, s.K))
			}
			b.ReportMetric(mf/1e3/(b.Elapsed().Seconds()/float64(b.N)), "GFLOP/s")
		})
		b.Run(name+"/simdgroup_big", func(b *testing.B) {
			p, gx, gy, tgx, tgy := v.GEMMF32Plan(s.M, s.N, s.K)
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()
			for range b.N {
				q.Run2D(p, gx, gy, tgx, tgy, dA, dB, dC, i32b(d, s.M), i32b(d, s.N), i32b(d, s.K))
			}
			b.ReportMetric(mf/1e3/(b.Elapsed().Seconds()/float64(b.N)), "GFLOP/s")
		})
	}
}
