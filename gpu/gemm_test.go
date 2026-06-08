//go:build gpu

package gpu

import (
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

// TestTiled_parity checks the shared-memory tiled GEMM against the integer
// reference at odd dims (exercises M/N/K tile tails).
func TestTiled_parity(t *testing.T) {
	ctx, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	defer ctx.Close()

	const M, K, N = 37, 517, 131
	weight := randMat(N*K, 1)
	act := randMat(M*K, 2)
	bq, bScales := linalg.QuantizeRowsInt8(weight, N, K)
	aq, aScales := linalg.QuantizeRowsInt8(act, M, K)

	ref := make([]float32, M*N)
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			var acc int32
			for k := 0; k < K; k++ {
				acc += int32(aq[m*K+k]) * int32(bq[n*K+k])
			}
			ref[m*N+n] = float32(acc) * aScales[m] * bScales[n]
		}
	}
	rm, err := ctx.UploadW8A8(bq, bScales, N, K)
	if err != nil {
		t.Fatalf("UploadW8A8: %v", err)
	}
	defer rm.Release()
	got, err := ctx.MatmulW8A8Tiled(aq, aScales, rm, M)
	if err != nil {
		t.Fatalf("MatmulW8A8Tiled: %v", err)
	}
	cos, maxAbs := cosine(got, ref)
	t.Logf("tiled parity: cosine=%.8f maxAbs=%.3e", cos, maxAbs)
	if cos < 0.99999 || maxAbs > 1e-3 {
		t.Errorf("tiled diverges: cosine=%.8f maxAbs=%.3e", cos, maxAbs)
	}
}

// TestTiled_microbench compares the tiled GEMM to the naive matmul and CPU at
// prefill shapes. Logs; run -v.
func TestTiled_microbench(t *testing.T) {
	if testing.Short() {
		t.Skip("microbench")
	}
	ctx := newOrSkipHW(t)
	defer ctx.Close()
	t.Logf("backend: %s", ctx.Backend())

	const K, N = 4096, 4096
	weight := randMat(N*K, 1)
	bq, bScales := linalg.QuantizeRowsInt8(weight, N, K)
	rm, err := ctx.UploadW8A8(bq, bScales, N, K)
	if err != nil {
		t.Fatalf("UploadW8A8: %v", err)
	}
	defer rm.Release()

	bench := func(M, iters int) {
		act := randMat(M*K, uint64(M)+7)
		aq, aScales := linalg.QuantizeRowsInt8(act, M, K)
		cpuDst := make([]float32, M*N)
		linalg.MatmulBTW8A8(act, bq, bScales, cpuDst, M, K, N)
		if _, err := ctx.MatmulW8A8(aq, aScales, rm, M); err != nil {
			t.Fatalf("naive: %v", err)
		}
		if _, err := ctx.MatmulW8A8Tiled(aq, aScales, rm, M); err != nil {
			t.Fatalf("tiled: %v", err)
		}

		t0 := time.Now()
		for i := 0; i < iters; i++ {
			linalg.MatmulBTW8A8(act, bq, bScales, cpuDst, M, K, N)
		}
		cpu := time.Since(t0) / time.Duration(iters)
		t1 := time.Now()
		for i := 0; i < iters; i++ {
			if _, err := ctx.MatmulW8A8(aq, aScales, rm, M); err != nil {
				t.Fatalf("naive: %v", err)
			}
		}
		naive := time.Since(t1) / time.Duration(iters)
		t2 := time.Now()
		for i := 0; i < iters; i++ {
			if _, err := ctx.MatmulW8A8Tiled(aq, aScales, rm, M); err != nil {
				t.Fatalf("tiled: %v", err)
			}
		}
		tiled := time.Since(t2) / time.Duration(iters)

		gf := func(d time.Duration) float64 { return float64(2*M*N*K) / d.Seconds() / 1e9 }
		t.Logf("M=%-4d  CPU %7.2f GFLOP/s  |  naive %7.2f  |  tiled %7.2f  (tiled vs CPU %.2f×, vs naive %.2f×)",
			M, gf(cpu), gf(naive), gf(tiled), float64(cpu)/float64(tiled), float64(naive)/float64(tiled))
	}
	bench(64, 30)
	bench(256, 20)
	bench(512, 10)
}
