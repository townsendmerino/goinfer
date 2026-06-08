//go:build gpu

package gpu

import (
	"math"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

// cpuChainW8A8 is the CPU reference for ChainW8A8: each step quantizes the
// running activation (internally, in MatmulBTW8A8) and matmuls. Square weights.
func cpuChainW8A8(act []float32, M, K int, bq []int8, bScales []float32, depth int) []float32 {
	cur := append([]float32(nil), act...)
	dst := make([]float32, M*K)
	for d := 0; d < depth; d++ {
		linalg.MatmulBTW8A8(cur, bq, bScales, dst, M, K, K) // N==K (square)
		copy(cur, dst)
	}
	return cur
}

func cosine(a, b []float32) (cos, maxAbs float64) {
	var dot, na, nb float64
	for i := range a {
		d := math.Abs(float64(a[i] - b[i]))
		if d > maxAbs {
			maxAbs = d
		}
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-12), maxAbs
}

// TestChainW8A8_parity checks the device-resident chain (on-GPU requantize +
// matmul, synced once) against the CPU chain. The only divergence is GPU vs CPU
// quantization rounding, which stays high-cosine across the chain depth.
func TestChainW8A8_parity(t *testing.T) {
	ctx, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	defer ctx.Close()

	const M, K, depth = 3, 512, 4 // square K=N so each step feeds the next
	weight := randMat(K*K, 1)
	act := randMat(M*K, 2)
	bq, bScales := linalg.QuantizeRowsInt8(weight, K, K)

	rm, err := ctx.UploadW8A8(bq, bScales, K, K)
	if err != nil {
		t.Fatalf("UploadW8A8: %v", err)
	}
	defer rm.Release()
	chain := make([]*ResidentW8A8, depth)
	for i := range chain {
		chain[i] = rm
	}
	got, err := ctx.ChainW8A8(act, M, chain)
	if err != nil {
		t.Fatalf("ChainW8A8: %v", err)
	}
	ref := cpuChainW8A8(act, M, K, bq, bScales, depth)
	cos, maxAbs := cosine(got, ref)
	t.Logf("chain depth=%d parity: cosine=%.6f maxAbs=%.3e", depth, cos, maxAbs)
	if cos < 0.999 {
		t.Errorf("device chain diverges from CPU chain: cosine=%.6f", cos)
	}
}

// TestChainW8A8_microbench is the Stage-2 thesis test: at M=1 decode the cost is
// per-call sync latency, not data movement. It times a depth-D chain three ways —
// GPU device-resident (ONE Poll for the whole chain), GPU per-call (Stage-1
// style: a Poll + CPU requant per matmul), and CPU — to show the device chain
// collapses the round-trip latency the per-call path pays D times. Logs; run -v.
func TestChainW8A8_microbench(t *testing.T) {
	if testing.Short() {
		t.Skip("microbench")
	}
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	const M, K, depth, iters = 1, 4096, 16, 20
	weight := randMat(K*K, 1)
	act := randMat(M*K, 2)
	bq, bScales := linalg.QuantizeRowsInt8(weight, K, K)
	rm, err := ctx.UploadW8A8(bq, bScales, K, K)
	if err != nil {
		t.Fatalf("UploadW8A8: %v", err)
	}
	defer rm.Release()
	chain := make([]*ResidentW8A8, depth)
	for i := range chain {
		chain[i] = rm
	}

	// GPU per-call (Stage-1): CPU requant + a synced MatmulW8A8 per step.
	perCall := func() {
		cur := append([]float32(nil), act...)
		for d := 0; d < depth; d++ {
			aq, aScales := linalg.QuantizeRowsInt8(cur, M, K)
			out, err := ctx.MatmulW8A8(aq, aScales, rm, M)
			if err != nil {
				t.Fatalf("MatmulW8A8: %v", err)
			}
			cur = out
		}
	}
	// warm up
	if _, err := ctx.ChainW8A8(act, M, chain); err != nil {
		t.Fatalf("ChainW8A8: %v", err)
	}
	perCall()

	t0 := time.Now()
	for i := 0; i < iters; i++ {
		if _, err := ctx.ChainW8A8(act, M, chain); err != nil {
			t.Fatalf("ChainW8A8: %v", err)
		}
	}
	chained := time.Since(t0) / iters

	t1 := time.Now()
	for i := 0; i < iters; i++ {
		perCall()
	}
	percall := time.Since(t1) / iters

	dst := make([]float32, M*K)
	t2 := time.Now()
	for i := 0; i < iters; i++ {
		cur := append([]float32(nil), act...)
		for d := 0; d < depth; d++ {
			linalg.MatmulBTW8A8(cur, bq, bScales, dst, M, K, K)
			copy(cur, dst)
		}
	}
	cpu := time.Since(t2) / iters

	t.Logf("depth=%d M=%d  |  GPU chained (1 sync) %7.3f ms  |  GPU per-call (%d syncs) %7.3f ms  |  CPU %7.3f ms",
		depth, M, ms(chained), depth, ms(percall), ms(cpu))
	t.Logf("  → device-resident chain is %.2f× faster than per-call readback; %.2f× vs CPU",
		float64(percall)/float64(chained), float64(cpu)/float64(chained))
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }
