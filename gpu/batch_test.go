//go:build gpu

package gpu

import (
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

// qkvShapes mimics a GQA qkv projection: q full width, k/v narrower, shared K.
var qkvShapes = []struct{ N int }{{1536}, {256}, {256}}

// TestBatchGEMV_parity checks the one-submit batch against the integer reference.
func TestBatchGEMV_parity(t *testing.T) {
	ctx, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	defer ctx.Close()

	const K = 1536
	act := randMat(K, 7)
	aq, aScales := linalg.QuantizeRowsInt8(act, 1, K)
	rms := make([]*ResidentW8A8, len(qkvShapes))
	refs := make([][]float32, len(qkvShapes))
	for i, sh := range qkvShapes {
		w := randMat(sh.N*K, uint64(i)+1)
		bq, bScales := linalg.QuantizeRowsInt8(w, sh.N, K)
		rm, err := ctx.UploadW8A8(bq, bScales, sh.N, K)
		if err != nil {
			t.Fatalf("UploadW8A8: %v", err)
		}
		defer rm.Release()
		rms[i] = rm
		ref := make([]float32, sh.N)
		for n := 0; n < sh.N; n++ {
			var acc int32
			for k := 0; k < K; k++ {
				acc += int32(aq[k]) * int32(bq[n*K+k])
			}
			ref[n] = float32(acc) * aScales[0] * bScales[n]
		}
		refs[i] = ref
	}
	outs, err := ctx.BatchGEMV(aq, aScales[0], rms)
	if err != nil {
		t.Fatalf("BatchGEMV: %v", err)
	}
	for i := range outs {
		cos, maxAbs := cosine(outs[i], refs[i])
		if cos < 0.99999 || maxAbs > 1e-3 {
			t.Errorf("op %d diverges: cosine=%.8f maxAbs=%.3e", i, cos, maxAbs)
		}
	}
	t.Logf("BatchGEMV parity OK across %d ops", len(outs))
}

// TestBatchTiled_parity checks the M>1 (prefill) shared-activation batch against
// the integer reference.
func TestBatchTiled_parity(t *testing.T) {
	ctx, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	defer ctx.Close()

	const M, K = 37, 1536
	act := randMat(M*K, 7)
	aq, aScales := linalg.QuantizeRowsInt8(act, M, K)
	rms := make([]*ResidentW8A8, len(qkvShapes))
	refs := make([][]float32, len(qkvShapes))
	for i, sh := range qkvShapes {
		w := randMat(sh.N*K, uint64(i)+1)
		bq, bScales := linalg.QuantizeRowsInt8(w, sh.N, K)
		rm, err := ctx.UploadW8A8(bq, bScales, sh.N, K)
		if err != nil {
			t.Fatalf("UploadW8A8: %v", err)
		}
		defer rm.Release()
		rms[i] = rm
		ref := make([]float32, M*sh.N)
		for m := 0; m < M; m++ {
			for n := 0; n < sh.N; n++ {
				var acc int32
				for kk := 0; kk < K; kk++ {
					acc += int32(aq[m*K+kk]) * int32(bq[n*K+kk])
				}
				ref[m*sh.N+n] = float32(acc) * aScales[m] * bScales[n]
			}
		}
		refs[i] = ref
	}
	outs, err := ctx.BatchTiled(aq, aScales, M, rms)
	if err != nil {
		t.Fatalf("BatchTiled: %v", err)
	}
	for i := range outs {
		cos, maxAbs := cosine(outs[i], refs[i])
		if cos < 0.99999 || maxAbs > 1e-3 {
			t.Errorf("op %d diverges: cosine=%.8f maxAbs=%.3e", i, cos, maxAbs)
		}
	}
	t.Logf("BatchTiled parity OK across %d ops (M=%d)", len(outs), M)
}

// TestBatchGEMV_microbench measures the sync-floor cut: the fused batch (one
// Poll for all ops) vs the same ops as separate synced GEMVRunner calls (one Poll
// each) vs the CPU batch kernel. Logs; run -v.
func TestBatchGEMV_microbench(t *testing.T) {
	if testing.Short() {
		t.Skip("microbench")
	}
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	const K, iters = 1536, 100
	act := randMat(K, 7)
	aq, aScales := linalg.QuantizeRowsInt8(act, 1, K)
	rms := make([]*ResidentW8A8, len(qkvShapes))
	runners := make([]*GEMVRunner, len(qkvShapes))
	ops := make([]linalg.W8A8Op, len(qkvShapes))
	bqs := make([][]int8, len(qkvShapes))
	bss := make([][]float32, len(qkvShapes))
	for i, sh := range qkvShapes {
		w := randMat(sh.N*K, uint64(i)+1)
		bq, bScales := linalg.QuantizeRowsInt8(w, sh.N, K)
		bqs[i], bss[i] = bq, bScales
		rm, _ := ctx.UploadW8A8(bq, bScales, sh.N, K)
		defer rm.Release()
		rms[i] = rm
		r, _ := ctx.NewGEMVRunner(rm)
		defer r.Release()
		runners[i] = r
		ops[i] = linalg.W8A8Op{BQ: bq, Scales: bScales, Dst: make([]float32, sh.N), N: sh.N}
	}

	// warm up
	ctx.BatchGEMV(aq, aScales[0], rms)
	for _, r := range runners {
		r.Run(aq, aScales[0])
	}

	t0 := time.Now()
	for it := 0; it < iters; it++ {
		if _, err := ctx.BatchGEMV(aq, aScales[0], rms); err != nil {
			t.Fatalf("BatchGEMV: %v", err)
		}
	}
	batch := time.Since(t0) / iters

	t1 := time.Now()
	for it := 0; it < iters; it++ {
		for _, r := range runners {
			if _, err := r.Run(aq, aScales[0]); err != nil {
				t.Fatalf("runner: %v", err)
			}
		}
	}
	separate := time.Since(t1) / iters

	var ws linalg.Workspace
	t2 := time.Now()
	for it := 0; it < iters; it++ {
		linalg.MatmulBTW8A8Batch(&ws, act, 1, K, ops)
	}
	cpu := time.Since(t2) / iters

	t.Logf("qkv batch (3 ops, K=%d)  |  GPU fused (1 sync) %.3f ms  |  GPU separate (3 syncs) %.3f ms  |  CPU %.3f ms",
		K, ms(batch), ms(separate), ms(cpu))
	t.Logf("  → fused vs separate %.2f×  |  fused vs CPU %.2f×", float64(separate)/float64(batch), float64(cpu)/float64(batch))
}
