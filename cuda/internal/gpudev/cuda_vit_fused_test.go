//go:build linux

package gpudev

import (
	"math"
	"math/rand"
	"testing"
)

// TestCUDA_gemmW8A8Bias tests the fused GEMM + bias and GEMM + bias + residual add kernels.
func TestCUDA_gemmW8A8Bias(t *testing.T) {
	d, q, v := vitSetup(t)

	// Routing checks.
	for _, c := range []struct {
		M, N, K  int
		wantPlan bool
	}{
		{64, 64, 16, true},
		{128, 256, 1152, true},
		{64, 4304, 1152, true},
		{63, 65, 16, true},
		{64, 64, 15, false},
		{0, 64, 16, false},
	} {
		pb, _ := v.GEMMW8A8BiasPlan(c.M, c.N, c.K)
		pba, _ := v.GEMMW8A8BiasAddPlan(c.M, c.N, c.K)
		if (pb == v.GEMMW8A8Bias) != c.wantPlan {
			t.Fatalf("GEMMW8A8BiasPlan(%d,%d,%d) got %v, want %v", c.M, c.N, c.K, pb == v.GEMMW8A8Bias, c.wantPlan)
		}
		if (pba == v.GEMMW8A8BiasAdd) != c.wantPlan {
			t.Fatalf("GEMMW8A8BiasAddPlan(%d,%d,%d) got %v, want %v", c.M, c.N, c.K, pba == v.GEMMW8A8BiasAdd, c.wantPlan)
		}
	}

	// Parity checks against un-fused sequence (gemm_w8a8_reg + add_bias / add_vec).
	rng := rand.New(rand.NewSource(1234))
	for _, sh := range []struct{ M, N, K int }{
		{64, 64, 16},
		{64, 64, 1152},
		{128, 192, 512},
		{63, 65, 512},
		{100, 4304, 1152}, // SigLIP MLP shape
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
		bias := randF32(rng, sh.N, 0.01)
		resInit := randF32(rng, sh.M*sh.N, 0.1)

		dA, dB := NewBufferOf(d, A), NewBufferOf(d, B)
		dAs, dBs := NewBufferOf(d, as), NewBufferOf(d, bs)
		dBias := NewBufferOf(d, bias)

		// 1. Test gemm_w8a8_bias vs unfused (gemm_w8a8_reg + add_bias)
		cFused := NewBufferLenOf[float32](d, sh.M*sh.N)
		cUnfused := NewBufferLenOf[float32](d, sh.M*sh.N)

		pb, cfgb := v.GEMMW8A8BiasPlan(sh.M, sh.N, sh.K)
		if err := q.Launch(pb, cfgb,
			Arg(dA), Arg(dAs), Arg(dB), Arg(dBs), Arg(dBias), Arg(cFused),
			ArgValue(int32(sh.M)), ArgValue(int32(sh.N)), ArgValue(int32(sh.K))); err != nil {
			t.Fatalf("fused bias launch: %v", err)
		}

		preg, cfgreg := v.GEMMW8A8Plan(sh.M, sh.N, sh.K)
		if err := q.Launch(preg, cfgreg,
			Arg(dA), Arg(dAs), Arg(dB), Arg(dBs), Arg(cUnfused),
			ArgValue(int32(sh.M)), ArgValue(int32(sh.N)), ArgValue(int32(sh.K))); err != nil {
			t.Fatalf("unfused gemm launch: %v", err)
		}
		if err := q.Launch(v.AddBias, Grid1D(sh.M*sh.N, 256),
			Arg(cUnfused), Arg(dBias), ArgValue(int32(sh.M)), ArgValue(int32(sh.N))); err != nil {
			t.Fatalf("add_bias launch: %v", err)
		}
		if err := q.Sync(); err != nil {
			t.Fatalf("Sync: %v", err)
		}

		gotFused := make([]float32, sh.M*sh.N)
		gotUnfused := make([]float32, sh.M*sh.N)
		if err := Download(cFused, gotFused); err != nil {
			t.Fatalf("Download: %v", err)
		}
		if err := Download(cUnfused, gotUnfused); err != nil {
			t.Fatalf("Download: %v", err)
		}

		for i := range gotFused {
			diff := math.Abs(float64(gotFused[i]) - float64(gotUnfused[i]))
			denom := math.Max(1.0, math.Abs(float64(gotFused[i])))
			if diff/denom > 1e-6 {
				t.Fatalf("shape %v elem %d: fused bias %v != unfused %v (diff %g, rel %g)", sh, i, gotFused[i], gotUnfused[i], diff, diff/denom)
			}
		}

		// 2. Test gemm_w8a8_bias_add vs unfused (cUnfused + add_vec into residual)
		dResFused := NewBufferOf(d, resInit)
		dResUnfused := NewBufferOf(d, resInit)

		pba, cba := v.GEMMW8A8BiasAddPlan(sh.M, sh.N, sh.K)
		if err := q.Launch(pba, cba,
			Arg(dA), Arg(dAs), Arg(dB), Arg(dBs), Arg(dBias), Arg(dResFused),
			ArgValue(int32(sh.M)), ArgValue(int32(sh.N)), ArgValue(int32(sh.K))); err != nil {
			t.Fatalf("fused bias_add launch: %v", err)
		}
		if err := q.Launch(v.AddVec, Grid1D(sh.M*sh.N, 256),
			Arg(dResUnfused), Arg(cUnfused), ArgValue(int32(sh.M*sh.N))); err != nil {
			t.Fatalf("add_vec launch: %v", err)
		}
		if err := q.Sync(); err != nil {
			t.Fatalf("Sync: %v", err)
		}

		gotResFused := make([]float32, sh.M*sh.N)
		gotResUnfused := make([]float32, sh.M*sh.N)
		if err := Download(dResFused, gotResFused); err != nil {
			t.Fatalf("Download: %v", err)
		}
		if err := Download(dResUnfused, gotResUnfused); err != nil {
			t.Fatalf("Download: %v", err)
		}

		for i := range gotResFused {
			diff := math.Abs(float64(gotResFused[i]) - float64(gotResUnfused[i]))
			denom := math.Max(1.0, math.Abs(float64(gotResFused[i])))
			if diff/denom > 1e-6 {
				t.Fatalf("shape %v elem %d: fused bias_add %v != unfused %v (diff %g, rel %g)", sh, i, gotResFused[i], gotResUnfused[i], diff, diff/denom)
			}
		}

		for _, buf := range []Buffer{dA, dB, dAs, dBs, dBias, cFused, cUnfused, dResFused, dResUnfused} {
			d.ReleaseBuf(buf)
		}
	}
	t.Log("gemm_w8a8_bias and gemm_w8a8_bias_add BIT-IDENTICAL to unfused gemm+bias and gemm+bias+add across all shapes")
}

// TestCUDA_gemmF32Bias tests the fused FP32 GEMM + bias and GEMM + bias + residual add kernels.
func TestCUDA_gemmF32Bias(t *testing.T) {
	d, q, v := vitSetup(t)

	// Routing checks.
	for _, c := range []struct {
		M, N, K  int
		wantPlan bool
	}{
		{64, 64, 16, true},
		{128, 256, 1152, true},
		{64, 4304, 1152, true},
		{63, 65, 16, true},
		{64, 64, 15, false},
		{0, 64, 16, false},
	} {
		pb, _ := v.GEMMF32BiasPlan(c.M, c.N, c.K)
		pba, _ := v.GEMMF32BiasAddPlan(c.M, c.N, c.K)
		if (pb == v.GEMMF32Bias) != c.wantPlan {
			t.Fatalf("GEMMF32BiasPlan(%d,%d,%d) got %v, want %v", c.M, c.N, c.K, pb == v.GEMMF32Bias, c.wantPlan)
		}
		if (pba == v.GEMMF32BiasAdd) != c.wantPlan {
			t.Fatalf("GEMMF32BiasAddPlan(%d,%d,%d) got %v, want %v", c.M, c.N, c.K, pba == v.GEMMF32BiasAdd, c.wantPlan)
		}
	}

	// Parity checks against un-fused sequence (gemm_f32_reg + add_bias / add_vec).
	rng := rand.New(rand.NewSource(5678))
	for _, sh := range []struct{ M, N, K int }{
		{64, 64, 16},
		{64, 64, 1152},
		{128, 192, 512},
		{63, 65, 512},
		{100, 4304, 1152},
	} {
		A := randF32(rng, sh.M*sh.K, 0.5)
		B := randF32(rng, sh.N*sh.K, 0.5)
		bias := randF32(rng, sh.N, 0.1)
		resInit := randF32(rng, sh.M*sh.N, 0.1)

		dA, dB := NewBufferOf(d, A), NewBufferOf(d, B)
		dBias := NewBufferOf(d, bias)
		cFused := NewBufferLenOf[float32](d, sh.M*sh.N)
		cUnfused := NewBufferLenOf[float32](d, sh.M*sh.N)

		// 1. Test gemm_f32_bias vs unfused (gemm_f32_reg + add_bias)
		pb, cb := v.GEMMF32BiasPlan(sh.M, sh.N, sh.K)
		if err := q.Launch(pb, cb,
			Arg(dA), Arg(dB), Arg(dBias), Arg(cFused),
			ArgValue(int32(sh.M)), ArgValue(int32(sh.N)), ArgValue(int32(sh.K))); err != nil {
			t.Fatalf("fused f32 bias launch: %v", err)
		}

		// Unfused reference
		preg, creg := v.GEMMF32Plan(sh.M, sh.N, sh.K)
		if err := q.Launch(preg, creg,
			Arg(dA), Arg(dB), Arg(cUnfused),
			ArgValue(int32(sh.M)), ArgValue(int32(sh.N)), ArgValue(int32(sh.K))); err != nil {
			t.Fatalf("unfused gemm_f32 launch: %v", err)
		}
		if err := q.Launch(v.AddBias, Grid1D(sh.M*sh.N, 256),
			Arg(cUnfused), Arg(dBias), ArgValue(int32(sh.M)), ArgValue(int32(sh.N))); err != nil {
			t.Fatalf("add_bias launch: %v", err)
		}
		if err := q.Sync(); err != nil {
			t.Fatalf("Sync: %v", err)
		}

		gotFused := make([]float32, sh.M*sh.N)
		gotUnfused := make([]float32, sh.M*sh.N)
		if err := Download(cFused, gotFused); err != nil {
			t.Fatalf("Download: %v", err)
		}
		if err := Download(cUnfused, gotUnfused); err != nil {
			t.Fatalf("Download: %v", err)
		}

		for i := range gotFused {
			diff := math.Abs(float64(gotFused[i]) - float64(gotUnfused[i]))
			denom := math.Max(1.0, math.Abs(float64(gotFused[i])))
			if diff/denom > 1e-5 {
				t.Fatalf("f32 shape %v elem %d: fused bias %v != unfused %v (diff %g, rel %g)", sh, i, gotFused[i], gotUnfused[i], diff, diff/denom)
			}
		}

		// 2. Test gemm_f32_bias_add vs unfused (cUnfused + add_vec into residual)
		dResFused := NewBufferOf(d, resInit)
		dResUnfused := NewBufferOf(d, resInit)

		pba, cba := v.GEMMF32BiasAddPlan(sh.M, sh.N, sh.K)
		if err := q.Launch(pba, cba,
			Arg(dA), Arg(dB), Arg(dBias), Arg(dResFused),
			ArgValue(int32(sh.M)), ArgValue(int32(sh.N)), ArgValue(int32(sh.K))); err != nil {
			t.Fatalf("fused f32 bias_add launch: %v", err)
		}
		if err := q.Launch(v.AddVec, Grid1D(sh.M*sh.N, 256),
			Arg(dResUnfused), Arg(cUnfused), ArgValue(int32(sh.M*sh.N))); err != nil {
			t.Fatalf("add_vec launch: %v", err)
		}
		if err := q.Sync(); err != nil {
			t.Fatalf("Sync: %v", err)
		}

		gotResFused := make([]float32, sh.M*sh.N)
		gotResUnfused := make([]float32, sh.M*sh.N)
		if err := Download(dResFused, gotResFused); err != nil {
			t.Fatalf("Download: %v", err)
		}
		if err := Download(dResUnfused, gotResUnfused); err != nil {
			t.Fatalf("Download: %v", err)
		}

		for i := range gotResFused {
			diff := math.Abs(float64(gotResFused[i]) - float64(gotResUnfused[i]))
			denom := math.Max(1.0, math.Abs(float64(gotResFused[i])))
			if diff/denom > 1e-5 {
				t.Fatalf("f32 shape %v elem %d: fused bias_add %v != unfused %v (diff %g, rel %g)", sh, i, gotResFused[i], gotResUnfused[i], diff, diff/denom)
			}
		}

		for _, buf := range []Buffer{dA, dB, dBias, cFused, cUnfused, dResFused, dResUnfused} {
			d.ReleaseBuf(buf)
		}
	}
	t.Log("gemm_f32_bias and gemm_f32_bias_add match unfused gemm+bias and gemm+bias+add across all shapes")
}
