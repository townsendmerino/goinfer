//go:build linux

package gpudev

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// cuda_vit_test.go gates each encoder kernel INDIVIDUALLY against a Go reference that
// reimplements vision/encoder.go's arithmetic. Gating the kernels one at a time is the
// point: a whole-tower cosine can stay high while a single op is subtly wrong (a
// softmax that forgot the max-subtract still normalizes; an erf-GELU still looks like a
// GELU), and then the bug only shows on some other checkpoint. Each kernel here is
// compared against the exact formulation the CPU tower uses, at f32-vs-double
// tolerances tight enough that a wrong FORMULA cannot pass.

func vitSetup(t *testing.T) (*Device, Queue, ViT) {
	t.Helper()
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	t.Cleanup(d.ReleaseObjects)
	v, err := d.NewViT()
	if err != nil {
		t.Fatalf("NewViT: %v", err)
	}
	return d, d.NewCommandQueue(), v
}

func randF32(rng *rand.Rand, n int, scale float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(rng.NormFloat64()) * scale
	}
	return out
}

// maxAbsDiff reports the worst absolute deviation between two vectors.
func maxAbsDiff(a, b []float32) float64 {
	worst := 0.0
	for i := range a {
		if d := math.Abs(float64(a[i]) - float64(b[i])); d > worst {
			worst = d
		}
	}
	return worst
}

// TestCUDA_vitLayerNorm gates LayerNorm against layerNormInto's exact formulation:
// double-accumulated mean and variance, eps added to the VARIANCE, then scale+shift.
func TestCUDA_vitLayerNorm(t *testing.T) {
	d, q, v := vitSetup(t)
	rng := rand.New(rand.NewSource(1))
	for _, shape := range []struct{ rows, dim int }{{1, 32}, {16, 32}, {7, 257}, {64, 1152}} {
		x := randF32(rng, shape.rows*shape.dim, 2)
		w := randF32(rng, shape.dim, 1)
		b := randF32(rng, shape.dim, 1)
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
		out := NewBufferLenOf[float32](d, len(x))
		if err := q.Launch(v.LayerNorm, RowGrid(shape.rows),
			Arg(dx), Arg(dw), Arg(db), Arg(out),
			ArgValue(int32(shape.rows)), ArgValue(int32(shape.dim)), ArgValue(float32(eps))); err != nil {
			t.Fatalf("Launch: %v", err)
		}
		if err := q.Sync(); err != nil {
			t.Fatalf("Sync: %v", err)
		}
		got := make([]float32, len(x))
		if err := Download(out, got); err != nil {
			t.Fatalf("Download: %v", err)
		}
		if dmax := maxAbsDiff(got, want); dmax > 1e-5 {
			t.Fatalf("rows=%d dim=%d: worst Δ %.3g", shape.rows, shape.dim, dmax)
		}
		for _, bb := range []Buffer{dx, dw, db, out} {
			d.ReleaseBuf(bb)
		}
	}
	t.Log("layernorm ≡ CPU layerNormInto across 4 shapes")
}

// TestCUDA_vitGELU gates the TANH-approximation GELU. The break-it-first here is
// implicit but real: erf-GELU agrees with tanh-GELU to ~1e-3, so the 1e-6 bound below
// would reject it — this test distinguishes the two formulations, which is the whole
// reason to pin it.
func TestCUDA_vitGELU(t *testing.T) {
	d, q, v := vitSetup(t)
	rng := rand.New(rand.NewSource(2))
	x := randF32(rng, 4096, 3)
	const c = 0.7978845608028654
	want := make([]float32, len(x))
	for i, val := range x {
		vv := float64(val)
		want[i] = float32(0.5 * vv * (1.0 + math.Tanh(c*(vv+0.044715*vv*vv*vv))))
	}
	dx := NewBufferOf(d, x)
	if err := q.Launch(v.GELUTanh, Grid1D(len(x), 256), Arg(dx), ArgValue(int32(len(x)))); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	if err := q.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	got := make([]float32, len(x))
	if err := Download(dx, got); err != nil {
		t.Fatalf("Download: %v", err)
	}
	if dmax := maxAbsDiff(got, want); dmax > 1e-6 {
		t.Fatalf("worst Δ %.3g — wrong GELU formulation?", dmax)
	}

	// erf-GELU must NOT pass this bound, proving the test discriminates between the
	// two formulations rather than just "looks like a GELU".
	erf := make([]float32, len(x))
	for i, val := range x {
		vv := float64(val)
		erf[i] = float32(0.5 * vv * (1.0 + math.Erf(vv/math.Sqrt2)))
	}
	if dmax := maxAbsDiff(erf, want); dmax <= 1e-6 {
		t.Error("break-it-first vacuous: erf-GELU is indistinguishable at this tolerance")
	} else {
		t.Logf("gelu_tanh ≡ CPU geluTanh; erf-GELU rejected at Δ %.3g", dmax)
	}
}

// TestCUDA_vitQuantRows gates the per-row int8 quantizer against linalg's
// quantizeRowInt8, including the all-zero row (scale 0) edge case. Quantization must
// be byte-exact — it is integer output, so any deviation is a real bug.
func TestCUDA_vitQuantRows(t *testing.T) {
	d, q, v := vitSetup(t)
	rng := rand.New(rand.NewSource(3))
	const rows, dim = 12, 97
	x := randF32(rng, rows*dim, 2)
	for i := range dim { // row 5 is all zeros
		x[5*dim+i] = 0
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
	dq := NewBufferLenOf[int8](d, rows*dim)
	ds := NewBufferLenOf[float32](d, rows)
	if err := q.Launch(v.QuantRows, RowGrid(rows),
		Arg(dx), Arg(dq), Arg(ds), ArgValue(int32(rows)), ArgValue(int32(dim))); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	if err := q.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	gotQ := make([]int8, rows*dim)
	gotS := make([]float32, rows)
	if err := Download(dq, gotQ); err != nil {
		t.Fatalf("Download q: %v", err)
	}
	if err := Download(ds, gotS); err != nil {
		t.Fatalf("Download s: %v", err)
	}
	for i := range wantQ {
		if gotQ[i] != wantQ[i] {
			t.Fatalf("q[%d] = %d, want %d (row %d)", i, gotQ[i], wantQ[i], i/dim)
		}
	}
	for r := range rows {
		if gotS[r] != wantS[r] {
			t.Fatalf("scale[%d] = %v, want %v", r, gotS[r], wantS[r])
		}
	}
	if gotS[5] != 0 {
		t.Errorf("all-zero row scale = %v, want 0", gotS[5])
	}
	t.Log("quant_rows byte-exact vs linalg quantizeRowInt8, incl. the all-zero row")
}

// TestCUDA_vitGEMMs gates both GEMMs against exact references. The W8A8 one is sharp:
// the int8 dot is exact integer arithmetic, so only the final rescale rounds.
func TestCUDA_vitGEMMs(t *testing.T) {
	d, q, v := vitSetup(t)
	rng := rand.New(rand.NewSource(4))
	for _, shape := range []struct{ M, N, K int }{{1, 8, 16}, {16, 32, 32}, {17, 33, 65}} {
		// --- W8A8 ---
		A := make([]int8, shape.M*shape.K)
		B := make([]int8, shape.N*shape.K)
		for i := range A {
			A[i] = int8(rng.Intn(255) - 127)
		}
		for i := range B {
			B[i] = int8(rng.Intn(255) - 127)
		}
		as := randF32(rng, shape.M, 0.01)
		bs := randF32(rng, shape.N, 0.01)
		want := make([]float32, shape.M*shape.N)
		for m := range shape.M {
			for n := range shape.N {
				var acc int32
				for k := range shape.K {
					acc += int32(A[m*shape.K+k]) * int32(B[n*shape.K+k])
				}
				want[m*shape.N+n] = float32(acc) * as[m] * bs[n]
			}
		}
		dA, dB := NewBufferOf(d, A), NewBufferOf(d, B)
		dAs, dBs := NewBufferOf(d, as), NewBufferOf(d, bs)
		dC := NewBufferLenOf[float32](d, shape.M*shape.N)
		if err := q.Launch(v.GEMMW8A8, Grid1D(shape.M*shape.N, 256),
			Arg(dA), Arg(dAs), Arg(dB), Arg(dBs), Arg(dC),
			ArgValue(int32(shape.M)), ArgValue(int32(shape.N)), ArgValue(int32(shape.K))); err != nil {
			t.Fatalf("W8A8 Launch: %v", err)
		}
		if err := q.Sync(); err != nil {
			t.Fatalf("Sync: %v", err)
		}
		got := make([]float32, shape.M*shape.N)
		if err := Download(dC, got); err != nil {
			t.Fatalf("Download: %v", err)
		}
		if dmax := maxAbsDiff(got, want); dmax > 1e-4 {
			t.Fatalf("W8A8 M=%d N=%d K=%d: worst Δ %.3g", shape.M, shape.N, shape.K, dmax)
		}

		// --- f32 ---
		Af := randF32(rng, shape.M*shape.K, 1)
		Bf := randF32(rng, shape.N*shape.K, 1)
		wantF := make([]float32, shape.M*shape.N)
		for m := range shape.M {
			for n := range shape.N {
				var acc float32
				for k := range shape.K {
					acc += Af[m*shape.K+k] * Bf[n*shape.K+k]
				}
				wantF[m*shape.N+n] = acc
			}
		}
		dAf, dBf := NewBufferOf(d, Af), NewBufferOf(d, Bf)
		dCf := NewBufferLenOf[float32](d, shape.M*shape.N)
		if err := q.Launch(v.GEMMF32, Grid1D(shape.M*shape.N, 256),
			Arg(dAf), Arg(dBf), Arg(dCf),
			ArgValue(int32(shape.M)), ArgValue(int32(shape.N)), ArgValue(int32(shape.K))); err != nil {
			t.Fatalf("f32 Launch: %v", err)
		}
		if err := q.Sync(); err != nil {
			t.Fatalf("Sync: %v", err)
		}
		gotF := make([]float32, shape.M*shape.N)
		if err := Download(dCf, gotF); err != nil {
			t.Fatalf("Download: %v", err)
		}
		if dmax := maxAbsDiff(gotF, wantF); dmax > 1e-3 {
			t.Fatalf("f32 M=%d N=%d K=%d: worst Δ %.3g", shape.M, shape.N, shape.K, dmax)
		}
		for _, bb := range []Buffer{dA, dB, dAs, dBs, dC, dAf, dBf, dCf} {
			d.ReleaseBuf(bb)
		}
	}
	t.Log("gemm_w8a8 ≡ exact int32 reference; gemm_f32 ≡ f32 reference (3 shapes each)")
}

// TestCUDA_vitAttention gates bidirectional multi-head attention against a Go
// reference using the CPU tower's exact softmax (max-subtract, double sum) and its
// 1/sqrt(hd) pre-softmax scaling.
func TestCUDA_vitAttention(t *testing.T) {
	d, q, v := vitSetup(t)
	rng := rand.New(rand.NewSource(5))
	for _, shape := range []struct{ np, nH, hd int }{{16, 2, 16}, {5, 1, 8}, {64, 4, 24}} {
		hidden := shape.nH * shape.hd
		qv := randF32(rng, shape.np*hidden, 1)
		kv := randF32(rng, shape.np*hidden, 1)
		vv := randF32(rng, shape.np*hidden, 1)
		scale := float32(1.0 / math.Sqrt(float64(shape.hd)))

		want := make([]float32, shape.np*hidden)
		scores := make([]float64, shape.np)
		for h := range shape.nH {
			off := h * shape.hd
			for i := range shape.np {
				maxv := math.Inf(-1)
				for j := range shape.np {
					var acc float32
					for dd := range shape.hd {
						acc += qv[i*hidden+off+dd] * kv[j*hidden+off+dd]
					}
					s := float64(acc * scale)
					scores[j] = s
					if s > maxv {
						maxv = s
					}
				}
				var sum float64
				for j := range shape.np {
					e := math.Exp(scores[j] - maxv)
					scores[j] = e
					sum += e
				}
				for dd := range shape.hd {
					var acc float64
					for j := range shape.np {
						acc += (scores[j] / sum) * float64(vv[j*hidden+off+dd])
					}
					want[i*hidden+off+dd] = float32(acc)
				}
			}
		}

		dq, dk, dv := NewBufferOf(d, qv), NewBufferOf(d, kv), NewBufferOf(d, vv)
		out := NewBufferLenOf[float32](d, shape.np*hidden)
		if err := q.Launch(v.Attention, AttentionGrid(shape.np, shape.nH),
			Arg(dq), Arg(dk), Arg(dv), Arg(out),
			ArgValue(int32(shape.np)), ArgValue(int32(shape.nH)), ArgValue(int32(shape.hd)),
			ArgValue(scale)); err != nil {
			t.Fatalf("Launch: %v", err)
		}
		if err := q.Sync(); err != nil {
			t.Fatalf("Sync: %v", err)
		}
		got := make([]float32, shape.np*hidden)
		if err := Download(out, got); err != nil {
			t.Fatalf("Download: %v", err)
		}
		if dmax := maxAbsDiff(got, want); dmax > 1e-5 {
			t.Fatalf("np=%d nH=%d hd=%d: worst Δ %.3g", shape.np, shape.nH, shape.hd, dmax)
		}
		for _, bb := range []Buffer{dq, dk, dv, out} {
			d.ReleaseBuf(bb)
		}
	}
	t.Log("attention ≡ CPU bidirectional MHA across 3 shapes")
}

// TestCUDA_vitAddOps gates the two broadcast/elementwise adds.
func TestCUDA_vitAddOps(t *testing.T) {
	d, q, v := vitSetup(t)
	rng := rand.New(rand.NewSource(6))
	const rows, dim = 9, 40
	x := randF32(rng, rows*dim, 1)
	bias := randF32(rng, dim, 1)
	vec := randF32(rng, rows*dim, 1)

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
	if err := q.Launch(v.AddBias, Grid1D(rows*dim, 256), Arg(dx), Arg(NewBufferOf(d, bias)),
		ArgValue(int32(rows)), ArgValue(int32(dim))); err != nil {
		t.Fatalf("add_bias Launch: %v", err)
	}
	if err := q.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	got := make([]float32, len(x))
	if err := Download(dx, got); err != nil {
		t.Fatalf("Download: %v", err)
	}
	if dmax := maxAbsDiff(got, wantBias); dmax > 1e-6 {
		t.Fatalf("add_bias worst Δ %.3g", dmax)
	}

	dx2 := NewBufferOf(d, x)
	if err := q.Launch(v.AddVec, Grid1D(rows*dim, 256), Arg(dx2), Arg(NewBufferOf(d, vec)),
		ArgValue(int32(rows*dim))); err != nil {
		t.Fatalf("add_vec Launch: %v", err)
	}
	if err := q.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	if err := Download(dx2, got); err != nil {
		t.Fatalf("Download: %v", err)
	}
	if dmax := maxAbsDiff(got, wantVec); dmax > 1e-6 {
		t.Fatalf("add_vec worst Δ %.3g", dmax)
	}
	t.Log("add_bias / add_vec ≡ CPU references")
}

// --- Qwen2.5-VL kernel gates ---

// TestCUDA_vitRMSNorm gates weight-only RMSNorm against vision/qwen_encoder.go's
// rmsNorm: no mean subtraction, no bias, eps INSIDE the mean-square.
func TestCUDA_vitRMSNorm(t *testing.T) {
	d, q, v := vitSetup(t)
	rng := rand.New(rand.NewSource(11))
	const eps = 1e-6
	for _, shape := range []struct{ rows, dim int }{{1, 32}, {12, 128}, {7, 257}} {
		x := randF32(rng, shape.rows*shape.dim, 2)
		w := randF32(rng, shape.dim, 1)
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
		dx, dw := NewBufferOf(d, x), NewBufferOf(d, w)
		out := NewBufferLenOf[float32](d, len(x))
		if err := q.Launch(v.RMSNorm, RowGrid(shape.rows), Arg(dx), Arg(dw), Arg(out),
			ArgValue(int32(shape.rows)), ArgValue(int32(shape.dim)), ArgValue(float32(eps))); err != nil {
			t.Fatalf("Launch: %v", err)
		}
		if err := q.Sync(); err != nil {
			t.Fatalf("Sync: %v", err)
		}
		got := make([]float32, len(x))
		if err := Download(out, got); err != nil {
			t.Fatalf("Download: %v", err)
		}
		if dm := maxAbsDiff(got, want); dm > 1e-5 {
			t.Fatalf("rows=%d dim=%d worst Δ %.3g", shape.rows, shape.dim, dm)
		}
		for _, b := range []Buffer{dx, dw, out} {
			d.ReleaseBuf(b)
		}
	}
	t.Log("rmsnorm ≡ CPU rmsNorm across 3 shapes")
}

// TestCUDA_vitGELUErf gates the EXACT erf-GELU and — the point of a separate kernel —
// proves it is measurably DIFFERENT from gelu_tanh. Qwen's merger uses erf; SigLIP
// uses tanh. Shipping one for both would be a silent numeric bug.
func TestCUDA_vitGELUErf(t *testing.T) {
	d, q, v := vitSetup(t)
	rng := rand.New(rand.NewSource(12))
	x := randF32(rng, 4096, 3)
	want := make([]float32, len(x))
	for i, val := range x {
		vv := float64(val)
		want[i] = float32(0.5 * vv * (1.0 + math.Erf(vv/math.Sqrt2)))
	}
	dx := NewBufferOf(d, x)
	if err := q.Launch(v.GELUErf, Grid1D(len(x), 256), Arg(dx), ArgValue(int32(len(x)))); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	if err := q.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	got := make([]float32, len(x))
	if err := Download(dx, got); err != nil {
		t.Fatalf("Download: %v", err)
	}
	if dm := maxAbsDiff(got, want); dm > 1e-6 {
		t.Fatalf("gelu_erf worst Δ %.3g", dm)
	}
	dx2 := NewBufferOf(d, x)
	if err := q.Launch(v.GELUTanh, Grid1D(len(x), 256), Arg(dx2), ArgValue(int32(len(x)))); err != nil {
		t.Fatalf("tanh Launch: %v", err)
	}
	if err := q.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	tanhOut := make([]float32, len(x))
	if err := Download(dx2, tanhOut); err != nil {
		t.Fatalf("Download: %v", err)
	}
	sep := maxAbsDiff(tanhOut, got)
	if sep <= 1e-6 {
		t.Errorf("gelu_erf and gelu_tanh are indistinguishable (Δ %.3g) — one of them is wrong", sep)
	}
	t.Logf("gelu_erf ≡ CPU geluErf; distinct from gelu_tanh by Δ %.3g", sep)
}

// TestCUDA_vitSiLUMul gates gate = silu(gate) * up.
func TestCUDA_vitSiLUMul(t *testing.T) {
	d, q, v := vitSetup(t)
	rng := rand.New(rand.NewSource(13))
	gate := randF32(rng, 2048, 2)
	up := randF32(rng, 2048, 2)
	want := make([]float32, len(gate))
	for i := range gate {
		vv := float64(gate[i])
		want[i] = float32(vv/(1.0+math.Exp(-vv))) * up[i]
	}
	dg, du := NewBufferOf(d, gate), NewBufferOf(d, up)
	if err := q.Launch(v.SiLUMul, Grid1D(len(gate), 256), Arg(dg), Arg(du), ArgValue(int32(len(gate)))); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	if err := q.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	got := make([]float32, len(gate))
	if err := Download(dg, got); err != nil {
		t.Fatalf("Download: %v", err)
	}
	if dm := maxAbsDiff(got, want); dm > 1e-6 {
		t.Fatalf("silu_mul worst Δ %.3g", dm)
	}
	t.Log("silu_mul ≡ CPU silu+multiply")
}

// TestCUDA_vitRopeQK gates the in-place NeoX rotary on the q and k thirds of a fused
// qkv buffer, and that the v third is left ALONE — an off-by-one third would be
// invisible to a q/k-only check.
func TestCUDA_vitRopeQK(t *testing.T) {
	d, q, v := vitSetup(t)
	rng := rand.New(rand.NewSource(14))
	const seq, nH, hd = 9, 3, 16
	hidden := nH * hd
	qkv := randF32(rng, seq*3*hidden, 1)
	cos := randF32(rng, seq*hd, 1)
	sin := randF32(rng, seq*hd, 1)

	want := append([]float32(nil), qkv...)
	half := hd / 2
	for i := range seq {
		co, si := cos[i*hd:(i+1)*hd], sin[i*hd:(i+1)*hd]
		for head := range nH {
			for _, base := range []int{i*3*hidden + head*hd, i*3*hidden + hidden + head*hd} {
				for dd := range half {
					x, y := want[base+dd], want[base+dd+half]
					want[base+dd] = x*co[dd] - y*si[dd]
					want[base+dd+half] = y*co[dd+half] + x*si[dd+half]
				}
			}
		}
	}

	dq := NewBufferOf(d, qkv)
	dc, ds := NewBufferOf(d, cos), NewBufferOf(d, sin)
	if err := q.Launch(v.RopeQK, Grid1D(seq*nH*half, 256), Arg(dq), Arg(dc), Arg(ds),
		ArgValue(int32(seq)), ArgValue(int32(nH)), ArgValue(int32(hd))); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	if err := q.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	got := make([]float32, len(qkv))
	if err := Download(dq, got); err != nil {
		t.Fatalf("Download: %v", err)
	}
	if dm := maxAbsDiff(got, want); dm > 1e-6 {
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

// TestCUDA_vitAttentionSeg gates segmented attention: each query attends only within
// its own segment, with a structural break-it-first (a single whole-sequence segment
// must give a different answer, or the bounds are being ignored).
func TestCUDA_vitAttentionSeg(t *testing.T) {
	d, q, v := vitSetup(t)
	rng := rand.New(rand.NewSource(15))
	const seq, nH, hd = 12, 2, 8
	hidden := nH * hd
	qkv := randF32(rng, seq*3*hidden, 1)
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
	dss, dse := NewBufferOf(d, segStart), NewBufferOf(d, segEnd)
	out := NewBufferLenOf[float32](d, seq*hidden)
	if err := q.Launch(v.AttentionSeg, SegAttentionGrid(seq, nH, maxSeg),
		Arg(dq), Arg(out), Arg(dss), Arg(dse),
		ArgValue(int32(seq)), ArgValue(int32(nH)), ArgValue(int32(hd)), ArgValue(scale)); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	if err := q.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	got := make([]float32, seq*hidden)
	if err := Download(out, got); err != nil {
		t.Fatalf("Download: %v", err)
	}
	if dm := maxAbsDiff(got, want); dm > 1e-5 {
		t.Fatalf("attention_seg worst Δ %.3g", dm)
	}

	outTiled := NewBufferLenOf[float32](d, seq*hidden)
	if err := q.Launch(v.AttentionSegTiled, AttentionSegTiledLaunchConfig(seq, nH),
		Arg(dq), Arg(outTiled), Arg(dss), Arg(dse),
		ArgValue(int32(seq)), ArgValue(int32(nH)), ArgValue(int32(hd)), ArgValue(scale)); err != nil {
		t.Fatalf("Launch AttentionSegTiled: %v", err)
	}
	if err := q.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	gotTiled := make([]float32, seq*hidden)
	if err := Download(outTiled, gotTiled); err != nil {
		t.Fatalf("Download: %v", err)
	}
	if dm := maxAbsDiff(gotTiled, want); dm > 1e-4 {
		t.Fatalf("attention_seg_tiled worst Δ %.3g", dm)
	}

	one := make([]int32, seq)
	oneEnd := make([]int32, seq)
	for i := range seq {
		one[i], oneEnd[i] = 0, seq
	}
	out2 := NewBufferLenOf[float32](d, seq*hidden)
	if err := q.Launch(v.AttentionSeg, SegAttentionGrid(seq, nH, seq),
		Arg(dq), Arg(out2), Arg(NewBufferOf(d, one)), Arg(NewBufferOf(d, oneEnd)),
		ArgValue(int32(seq)), ArgValue(int32(nH)), ArgValue(int32(hd)), ArgValue(scale)); err != nil {
		t.Fatalf("full Launch: %v", err)
	}
	if err := q.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	full := make([]float32, seq*hidden)
	if err := Download(out2, full); err != nil {
		t.Fatalf("Download: %v", err)
	}
	if maxAbsDiff(full, got) < 1e-4 {
		t.Error("break-it-first vacuous: windowed and full attention agree — bounds ignored?")
	}
	t.Log("attention_seg ≡ CPU per-segment MHA; full-attention differs (bounds enforced)")
}

// --- tiled GEMMs ---

// TestCUDA_gemmTiledMatchesUntiled gates the tiled GEMMs, asymmetrically, because the
// two admit different claims.
//
// W8A8 must be BIT-IDENTICAL to the untiled kernel. Its accumulator is int32 and
// integer addition is associative, so staging K through shared memory in TILE-sized
// chunks cannot change the sum. Asserting equality rather than a tolerance is what
// makes this able to catch a real tiling bug — a mis-indexed shared tile typically
// shifts a handful of outputs, which a loose bound would wave through.
//
// The f32 kernel is gated against a float64 CPU reference, NOT against its untiled
// counterpart. Comparing the two GPU kernels to each other is the obvious thing and it
// was wrong: it made this test flaky at 3-8 failures per 25 runs, and in EVERY failure
// the tiled result matched CPU while the untiled one drifted (see the note on
// gemm_f32 in vit.cu). Encoding "these two agree" would have pinned a defect as the
// expectation. The tiled kernel is what the consumers run; the reference is what it
// has to equal.
func TestCUDA_gemmTiledMatchesUntiled(t *testing.T) {
	d, q, v := vitSetup(t)
	rng := rand.New(rand.NewSource(21))
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
		as, bs := randF32(rng, sh.M, 0.01), randF32(rng, sh.N, 0.01)
		dA, dB := NewBufferOf(d, A), NewBufferOf(d, B)
		dAs, dBs := NewBufferOf(d, as), NewBufferOf(d, bs)
		c1 := NewBufferLenOf[float32](d, sh.M*sh.N)
		c2 := NewBufferLenOf[float32](d, sh.M*sh.N)
		if err := q.Launch(v.GEMMW8A8, Grid1D(sh.M*sh.N, 256),
			Arg(dA), Arg(dAs), Arg(dB), Arg(dBs), Arg(c1),
			ArgValue(int32(sh.M)), ArgValue(int32(sh.N)), ArgValue(int32(sh.K))); err != nil {
			t.Fatalf("untiled: %v", err)
		}
		if err := q.Launch(v.GEMMW8A8Tiled, TileGrid(sh.M, sh.N),
			Arg(dA), Arg(dAs), Arg(dB), Arg(dBs), Arg(c2),
			ArgValue(int32(sh.M)), ArgValue(int32(sh.N)), ArgValue(int32(sh.K))); err != nil {
			t.Fatalf("tiled: %v", err)
		}
		if err := q.Sync(); err != nil {
			t.Fatalf("Sync: %v", err)
		}
		g1, g2 := make([]float32, sh.M*sh.N), make([]float32, sh.M*sh.N)
		if err := Download(c1, g1); err != nil {
			t.Fatalf("Download: %v", err)
		}
		if err := Download(c2, g2); err != nil {
			t.Fatalf("Download: %v", err)
		}
		for i := range g1 {
			if g1[i] != g2[i] {
				t.Fatalf("W8A8 M=%d N=%d K=%d: tiled[%d]=%v != untiled %v — int32 accumulation is order-independent, so this is a real tiling bug",
					sh.M, sh.N, sh.K, i, g2[i], g1[i])
			}
		}

		Af, Bf := randF32(rng, sh.M*sh.K, 1), randF32(rng, sh.N*sh.K, 1)
		dAf, dBf := NewBufferOf(d, Af), NewBufferOf(d, Bf)
		f2 := NewBufferLenOf[float32](d, sh.M*sh.N)
		if err := q.Launch(v.GEMMF32Tiled, TileGrid(sh.M, sh.N), Arg(dAf), Arg(dBf), Arg(f2),
			ArgValue(int32(sh.M)), ArgValue(int32(sh.N)), ArgValue(int32(sh.K))); err != nil {
			t.Fatalf("f32 tiled: %v", err)
		}
		if err := q.Sync(); err != nil {
			t.Fatalf("Sync: %v", err)
		}
		h2 := make([]float32, sh.M*sh.N)
		if err := Download(f2, h2); err != nil {
			t.Fatalf("Download: %v", err)
		}
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
		// Bound set from the analytic f32 error, not picked to fit: a K-term f32 dot
		// accumulates roughly sqrt(K)*eps*sum|terms| against a float64 reference, which
		// at K=1152 with unit-normal operands is ~5e-5 relative. 2e-4 sits just above
		// that and still rejects anything structural — the earlier upload race showed
		// up here as 6.6, five orders of magnitude clear of this line. The SHARP gate
		// is the W8A8 bit-identity check above; f32 cannot offer one.
		if worst > 2e-4 {
			t.Fatalf("f32 tiled M=%d N=%d K=%d: worst relative Δ %.3g vs float64 CPU", sh.M, sh.N, sh.K, worst)
		}
		for _, b := range []Buffer{dA, dB, dAs, dBs, c1, c2, dAf, dBf, f2} {
			d.ReleaseBuf(b)
		}
	}
	t.Log("gemm_w8a8_tiled BIT-IDENTICAL to untiled across 5 shapes; gemm_f32_tiled ≡ float64 CPU reference")
}

// TestCUDA_gemmF32Reg gates the register-blocked f32 GEMM against a float64 CPU
// reference at the 2e-4 bound the encoder's parity depends on, and — separately —
// checks that GEMMF32Plan routes correctly.
//
// The routing check is not ceremony. gemm_f32_reg carries NO bounds checks (that is
// where its speed comes from), so a plan that handed it a misaligned shape would read
// and write out of range. The aligned/unaligned split is the only thing standing
// between that kernel and memory corruption.
func TestCUDA_gemmF32Reg(t *testing.T) {
	d, q, v := vitSetup(t)
	rng := rand.New(rand.NewSource(31))

	for _, c := range []struct {
		M, N, K int
		wantReg bool
	}{
		{64, 64, 16, true},
		{128, 256, 768, true}, // a real encoder projection
		{63, 64, 16, false},   // M unaligned
		{64, 65, 16, false},   // N unaligned
		{64, 64, 17, false},   // K unaligned
		{0, 64, 16, false},    // degenerate
	} {
		p, _ := v.GEMMF32Plan(c.M, c.N, c.K)
		gotReg := p == v.GEMMF32Reg
		if gotReg != c.wantReg {
			t.Fatalf("GEMMF32Plan(%d,%d,%d) picked reg=%v, want %v — a misaligned shape on the unchecked kernel is out-of-bounds access",
				c.M, c.N, c.K, gotReg, c.wantReg)
		}
	}

	for _, s := range []struct{ M, N, K int }{
		{64, 64, 16},
		{128, 128, 256},
		{256, 512, 768},
		{512, 3072, 768},
	} {
		A, B := randF32(rng, s.M*s.K, 1), randF32(rng, s.N*s.K, 1)
		dA, dB := NewBufferOf(d, A), NewBufferOf(d, B)
		out := NewBufferLenOf[float32](d, s.M*s.N)
		p, cfg := v.GEMMF32Plan(s.M, s.N, s.K)
		if p != v.GEMMF32Reg {
			t.Fatalf("shape %dx%dx%d should be on the register kernel", s.M, s.N, s.K)
		}
		if err := q.Launch(p, cfg, Arg(dA), Arg(dB), Arg(out),
			ArgValue(int32(s.M)), ArgValue(int32(s.N)), ArgValue(int32(s.K))); err != nil {
			t.Fatalf("Launch: %v", err)
		}
		if err := q.Sync(); err != nil {
			t.Fatalf("Sync: %v", err)
		}
		got := make([]float32, s.M*s.N)
		if err := Download(out, got); err != nil {
			t.Fatalf("Download: %v", err)
		}
		worst := 0.0
		for m := range s.M {
			for n := range s.N {
				var acc float64
				for k := range s.K {
					acc += float64(A[m*s.K+k]) * float64(B[n*s.K+k])
				}
				den := math.Abs(acc)
				if den < 1 {
					den = 1
				}
				if r := math.Abs(float64(got[m*s.N+n])-acc) / den; r > worst {
					worst = r
				}
			}
		}
		if worst > 2e-4 {
			t.Fatalf("M=%d N=%d K=%d: worst relative Δ %.3g vs float64 CPU (encoder bound is 2e-4)", s.M, s.N, s.K, worst)
		}
		for _, b := range []Buffer{dA, dB, out} {
			d.ReleaseBuf(b)
		}
	}
	t.Log("gemm_f32_reg ≡ float64 CPU within 2e-4 across 4 aligned shapes; GEMMF32Plan routes correctly")
}

// BenchmarkGEMMF32 compares the tiled and register-blocked f32 kernels at real encoder
// / ViT shapes. Steady compute only (operands resident, warm, one Sync per iteration).
func BenchmarkGEMMF32(b *testing.B) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		b.Skipf("no CUDA device: %v", err)
	}
	defer d.ReleaseObjects()
	v, err := d.NewViT()
	if err != nil {
		b.Fatalf("NewViT: %v", err)
	}
	q := d.NewCommandQueue()
	rng := rand.New(rand.NewSource(41))
	for _, s := range []struct{ M, N, K int }{
		{512, 3072, 768},
		{1024, 1024, 1024},
		{4096, 1152, 1152},
		{4096, 4352, 1152},
	} {
		A, B := randF32(rng, s.M*s.K, 1), randF32(rng, s.N*s.K, 1)
		dA, dB := NewBufferOf(d, A), NewBufferOf(d, B)
		out := NewBufferLenOf[float32](d, s.M*s.N)
		gflop := 2.0 * float64(s.M) * float64(s.N) * float64(s.K) / 1e9
		run := func(p Pipeline, cfg LaunchConfig) {
			if err := q.Launch(p, cfg, Arg(dA), Arg(dB), Arg(out),
				ArgValue(int32(s.M)), ArgValue(int32(s.N)), ArgValue(int32(s.K))); err != nil {
				b.Fatalf("Launch: %v", err)
			}
			if err := q.Sync(); err != nil {
				b.Fatalf("Sync: %v", err)
			}
		}
		name := fmt.Sprintf("M%d_N%d_K%d", s.M, s.N, s.K)
		b.Run(name+"/tiled", func(b *testing.B) {
			for range 3 {
				run(v.GEMMF32Tiled, TileGrid(s.M, s.N))
			}
			b.ResetTimer()
			for range b.N {
				run(v.GEMMF32Tiled, TileGrid(s.M, s.N))
			}
			b.ReportMetric(gflop/(b.Elapsed().Seconds()/float64(b.N)), "GFLOP/s")
		})
		b.Run(name+"/reg", func(b *testing.B) {
			p, cfg := v.GEMMF32Plan(s.M, s.N, s.K)
			for range 3 {
				run(p, cfg)
			}
			b.ResetTimer()
			for range b.N {
				run(p, cfg)
			}
			b.ReportMetric(gflop/(b.Elapsed().Seconds()/float64(b.N)), "GFLOP/s")
		})
		for _, bb := range []Buffer{dA, dB, out} {
			d.ReleaseBuf(bb)
		}
	}
}
