//go:build darwin

package gpudev

import (
	"math"
	"math/rand"
	"testing"
)

// TestMetal_gemmW8A8Fused pins the fused register kernels Metal's SigLIP tower uses (gpu/visionmetal: GEMMW8A8Bias for
// fc1, GEMMW8A8BiasAdd for o-proj and fc2) against the plain register kernel (TestMetal_gemmW8A8Reg checks it against the
// bounds-checked tiled one) plus the bias, and the residual, added on the host. Only the CUDA twins were tested; goinfer's
// S3 found visionmetal right at the tiny tower's aligned shapes and wrong at SigLIP-so400m's (hidden 1152, MLP 4304,
// 4096 patches), and the shapes here include those.
func TestMetal_gemmW8A8Fused(t *testing.T) {
	d, q, v := vitSetupM(t)
	rng := rand.New(rand.NewSource(7))
	for _, sh := range []struct{ M, N, K int }{
		{64, 64, 16},
		{128, 192, 512},
		{100, 4304, 1152}, // fc1: N ragged
		{256, 1152, 4304}, // fc2: K = the so400m intermediate
		{63, 65, 512},     // both edges ragged
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
		bias, res := randF32M(rng, sh.N, 1), randF32M(rng, sh.M*sh.N, 1)
		dA, dB, dAs, dBs := NewBufferOf(d, A), NewBufferOf(d, B), NewBufferOf(d, as), NewBufferOf(d, bs)
		dBias := NewBufferOf(d, bias)

		ref := d.NewBufferLen(sh.M * sh.N)
		p, gx, gy, tgx, tgy := v.GEMMW8A8Plan(sh.M, sh.N, sh.K)
		run2d(q, p, gx, gy, tgx, tgy, dA, dAs, dB, dBs, ref, i32b(d, sh.M), i32b(d, sh.N), i32b(d, sh.K))
		want := append([]float32(nil), ref.Floats()[:sh.M*sh.N]...)
		for r := range sh.M {
			for c := range sh.N {
				want[r*sh.N+c] += bias[c]
			}
		}

		pb, gx, gy, tgx, tgy := v.GEMMW8A8BiasPlan(sh.M, sh.N, sh.K)
		if pb != v.GEMMW8A8Bias {
			t.Fatalf("shape %v did not route to the fused bias kernel", sh)
		}
		got := d.NewBufferLen(sh.M * sh.N)
		run2d(q, pb, gx, gy, tgx, tgy, dA, dAs, dB, dBs, dBias, got, i32b(d, sh.M), i32b(d, sh.N), i32b(d, sh.K))
		checkClose(t, "bias", sh, got.Floats()[:sh.M*sh.N], want)

		pba, gx, gy, tgx, tgy := v.GEMMW8A8BiasAddPlan(sh.M, sh.N, sh.K)
		if pba != v.GEMMW8A8BiasAdd {
			t.Fatalf("shape %v did not route to the fused bias-add kernel", sh)
		}
		dRes := NewBufferOf(d, res)
		run2d(q, pba, gx, gy, tgx, tgy, dA, dAs, dB, dBs, dBias, dRes, i32b(d, sh.M), i32b(d, sh.N), i32b(d, sh.K))
		wantAdd := make([]float32, len(want))
		for i := range want {
			wantAdd[i] = want[i] + res[i]
		}
		checkClose(t, "bias-add", sh, dRes.Floats()[:sh.M*sh.N], wantAdd)
	}
}

func checkClose(t *testing.T, what string, sh struct{ M, N, K int }, got, want []float32) {
	t.Helper()
	bad, worst := 0, 0.0
	for i := range want {
		d := math.Abs(float64(got[i] - want[i]))
		if d > 1e-4*math.Max(1, math.Abs(float64(want[i]))) {
			bad++
		}
		worst = math.Max(worst, d)
	}
	if bad > 0 {
		t.Errorf("%s %v: %d of %d outputs differ from the plain kernel plus the host's add (worst |diff| %g)", what, sh, bad, len(want), worst)
	}
}
