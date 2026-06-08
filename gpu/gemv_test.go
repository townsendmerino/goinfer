//go:build gpu

package gpu

import (
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

// TestGEMV_parity checks the coalesced GEMV kernel against the integer reference
// (it must equal the naive kernel at M=1).
func TestGEMV_parity(t *testing.T) {
	ctx, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	defer ctx.Close()

	const K, N = 517, 131
	weight := randMat(N*K, 1)
	act := randMat(K, 2)
	bq, bScales := linalg.QuantizeRowsInt8(weight, N, K)
	aq, aScales := linalg.QuantizeRowsInt8(act, 1, K)

	ref := make([]float32, N)
	for n := 0; n < N; n++ {
		var acc int32
		for k := 0; k < K; k++ {
			acc += int32(aq[k]) * int32(bq[n*K+k])
		}
		ref[n] = float32(acc) * aScales[0] * bScales[n]
	}
	rm, err := ctx.UploadW8A8(bq, bScales, N, K)
	if err != nil {
		t.Fatalf("UploadW8A8: %v", err)
	}
	defer rm.Release()
	got, err := ctx.MatmulW8A8GEMV(aq, aScales[0], rm)
	if err != nil {
		t.Fatalf("MatmulW8A8GEMV: %v", err)
	}
	cos, maxAbs := cosine(got, ref)
	t.Logf("GEMV parity: cosine=%.8f maxAbs=%.3e", cos, maxAbs)
	if cos < 0.99999 || maxAbs > 1e-3 {
		t.Errorf("GEMV diverges: cosine=%.8f maxAbs=%.3e", cos, maxAbs)
	}
}

// TestGEMV_microbench is the decode-unlock test: compare the coalesced GEMV
// kernel to the naive matmul and the CPU at M=1 on a realistic projection. The
// hypothesis (from the Stage-2 finding) is that coalescing the weight stream —
// not removing per-call syncs — is what beats the CPU at decode. Logs; run -v.
func TestGEMV_microbench(t *testing.T) {
	if testing.Short() {
		t.Skip("microbench")
	}
	ctx, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	defer ctx.Close()
	t.Logf("backend: %s", ctx.Backend())

	const K, N, iters = 4096, 4096, 100
	weight := randMat(N*K, 1)
	act := randMat(K, 2)
	bq, bScales := linalg.QuantizeRowsInt8(weight, N, K)
	aq, aScale := linalg.QuantizeRowsInt8(act, 1, K)
	rm, err := ctx.UploadW8A8(bq, bScales, N, K)
	if err != nil {
		t.Fatalf("UploadW8A8: %v", err)
	}
	defer rm.Release()
	cpuDst := make([]float32, N)

	// warm up
	linalg.MatmulBTW8A8(act, bq, bScales, cpuDst, 1, K, N)
	if _, err := ctx.MatmulW8A8(aq, aScale, rm, 1); err != nil {
		t.Fatalf("naive: %v", err)
	}
	if _, err := ctx.MatmulW8A8GEMV(aq, aScale[0], rm); err != nil {
		t.Fatalf("gemv: %v", err)
	}

	t0 := time.Now()
	for i := 0; i < iters; i++ {
		linalg.MatmulBTW8A8(act, bq, bScales, cpuDst, 1, K, N)
	}
	cpu := time.Since(t0) / iters

	t1 := time.Now()
	for i := 0; i < iters; i++ {
		if _, err := ctx.MatmulW8A8(aq, aScale, rm, 1); err != nil {
			t.Fatalf("naive: %v", err)
		}
	}
	naive := time.Since(t1) / iters

	t2 := time.Now()
	for i := 0; i < iters; i++ {
		if _, err := ctx.MatmulW8A8GEMV(aq, aScale[0], rm); err != nil {
			t.Fatalf("gemv: %v", err)
		}
	}
	gemv := time.Since(t2) / iters

	// Reusable runner (steady-state decode: buffers allocated once, WriteBuffer
	// the activation per call) — the path the decoder caches per weight.
	runner, err := ctx.NewGEMVRunner(rm)
	if err != nil {
		t.Fatalf("NewGEMVRunner: %v", err)
	}
	defer runner.Release()
	if _, err := runner.Run(aq, aScale[0]); err != nil {
		t.Fatalf("runner warmup: %v", err)
	}
	t3 := time.Now()
	for i := 0; i < iters; i++ {
		if _, err := runner.Run(aq, aScale[0]); err != nil {
			t.Fatalf("runner: %v", err)
		}
	}
	runr := time.Since(t3) / iters

	bw := func(d time.Duration) float64 { return float64(N*K) / d.Seconds() / 1e9 } // GB/s (1B/weight)
	t.Logf("M=1 K=N=%d  |  CPU %6.3f ms (%5.1f GB/s)  |  GPU naive %6.3f ms (%5.1f GB/s)  |  GPU GEMV one-shot %6.3f ms (%5.1f GB/s)  |  GPU GEMV runner %6.3f ms (%5.1f GB/s)",
		K, ms(cpu), bw(cpu), ms(naive), bw(naive), ms(gemv), bw(gemv), ms(runr), bw(runr))
	t.Logf("  → runner vs CPU %.2f×  |  runner vs one-shot %.2f×  |  one-shot vs CPU %.2f×",
		float64(cpu)/float64(runr), float64(gemv)/float64(runr), float64(cpu)/float64(gemv))
}
