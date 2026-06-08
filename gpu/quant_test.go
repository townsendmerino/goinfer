//go:build gpu

package gpu

import (
	"math"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

// randMat fills a deterministic pseudo-random [-1,1] matrix (no Math.rand dep
// concerns; just a cheap LCG so the test is reproducible).
func randMat(n int, seed uint64) []float32 {
	out := make([]float32, n)
	s := seed
	for i := range out {
		s = s*6364136223846793005 + 1442695040888963407
		out[i] = (float32(s>>40)/float32(1<<24))*2 - 1
	}
	return out
}

// TestW8A8_parity gates the WGSL W8A8 kernel against the exact integer reference
// (int dot × per-row scales) — the same math linalg.MatmulBTW8A8 computes. The
// kernel must match to fp32-dequant tolerance.
func TestW8A8_parity(t *testing.T) {
	ctx, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	defer ctx.Close()

	const M, K, N = 5, 517, 131 // odd dims to exercise K-padding + tail tiles
	weight := randMat(N*K, 1)
	act := randMat(M*K, 2)
	bq, bScales := linalg.QuantizeRowsInt8(weight, N, K)
	aq, aScales := linalg.QuantizeRowsInt8(act, M, K)

	// Exact integer reference (what the kernel computes, bit-for-bit on the dot).
	ref := make([]float32, M*N)
	for m := range M {
		for n := range N {
			var acc int32
			for k := range K {
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
	got, err := ctx.MatmulW8A8(aq, aScales, rm, M)
	if err != nil {
		t.Fatalf("MatmulW8A8: %v", err)
	}

	var maxAbs, dot, na, nb float64
	for i := range ref {
		d := math.Abs(float64(got[i] - ref[i]))
		if d > maxAbs {
			maxAbs = d
		}
		dot += float64(got[i]) * float64(ref[i])
		na += float64(got[i]) * float64(got[i])
		nb += float64(ref[i]) * float64(ref[i])
	}
	cos := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-12)
	t.Logf("W8A8 kernel parity: maxAbs=%.3e cosine=%.8f", maxAbs, cos)
	if maxAbs > 1e-3 || cos < 0.99999 {
		t.Errorf("W8A8 kernel diverges: maxAbs=%.3e cosine=%.8f", maxAbs, cos)
	}
}

// TestW8A8_microbench is the Stage-1 gate's performance half: time the GPU W8A8
// matmul (resident weight; per-call activation quant + upload + readback — the
// honest Stage-1 cost) against the production CPU kernel, at decode (M=1) and
// prefill (M=64) on a realistic layer shape. Stage 1 keeps a per-call readback,
// so M=1 is expected to be overhead-bound (Stage 2 removes it) while the larger
// prefill matmul should already show the GPU win. Informational (logs), gated
// only on correctness; run with -v.
func TestW8A8_microbench(t *testing.T) {
	if testing.Short() {
		t.Skip("microbench")
	}
	ctx := newOrSkipHW(t)
	defer ctx.Close()
	t.Logf("backend: %s", ctx.Backend())

	const K, N = 4096, 4096 // a representative attention/MLP projection
	weight := randMat(N*K, 1)
	bq, bScales := linalg.QuantizeRowsInt8(weight, N, K)
	rm, err := ctx.UploadW8A8(bq, bScales, N, K)
	if err != nil {
		t.Fatalf("UploadW8A8: %v", err)
	}
	defer rm.Release()

	bench := func(M, iters int) {
		act := randMat(M*K, uint64(M)+7)
		cpuDst := make([]float32, M*N)
		// warm up both paths.
		linalg.MatmulBTW8A8(act, bq, bScales, cpuDst, M, K, N)
		aq, aScales := linalg.QuantizeRowsInt8(act, M, K)
		if _, err := ctx.MatmulW8A8(aq, aScales, rm, M); err != nil {
			t.Fatalf("MatmulW8A8(M=%d): %v", M, err)
		}

		t0 := time.Now()
		for range iters {
			linalg.MatmulBTW8A8(act, bq, bScales, cpuDst, M, K, N)
		}
		cpu := time.Since(t0) / time.Duration(iters)

		t1 := time.Now()
		for range iters {
			aq, aScales := linalg.QuantizeRowsInt8(act, M, K) // production would quantize each call
			if _, err := ctx.MatmulW8A8(aq, aScales, rm, M); err != nil {
				t.Fatalf("MatmulW8A8(M=%d): %v", M, err)
			}
		}
		gpu := time.Since(t1) / time.Duration(iters)

		flops := float64(2 * M * N * K)
		t.Logf("M=%-3d  CPU %8.3f ms (%5.1f GFLOP/s)  |  GPU %8.3f ms (%5.1f GFLOP/s)  |  speedup %.2fx",
			M, float64(cpu.Microseconds())/1000, flops/cpu.Seconds()/1e9,
			float64(gpu.Microseconds())/1000, flops/gpu.Seconds()/1e9, float64(cpu)/float64(gpu))
	}
	bench(1, 50)   // decode (memory-bound; Stage-1 per-call readback dominates)
	bench(64, 30)  // prefill chunk (compute amortizes the round-trip)
	bench(256, 20) // larger prefill
}
