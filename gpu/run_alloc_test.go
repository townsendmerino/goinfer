//go:build gpu

package gpu

import (
	"math"
	"os"
	"testing"
	"time"
)

// TestZZ_decodeRunAllocs reports allocs/op and B/op for DecodeRunner.Run via testing.Benchmark,
// isolating the per-token host logits allocation. Uses realistic qwen3-class vocab (151936) with
// only 2 layers so setup is fast and Run is cheap — the vocab-sized readback slice dominates the
// alloc delta this change targets (make([]float32, vocab) → reused r.logitsHost). Compare the
// numbers on main (before) vs this branch (after): B/op should drop by ~vocab*4 = 607744 and
// allocs/op by ~1. Opt-in diagnostic, not a gate.
func TestZZ_decodeRunAllocs(t *testing.T) {
	if os.Getenv("GOINFER_GPU_ALLOC_BENCH") == "" {
		t.Skip("Run alloc/op diagnostic; set GOINFER_GPU_ALLOC_BENCH=1")
	}
	ctx := newOrSkipHW(t)

	const hidden, nH, nKV, hd, inter, vocab, L = 1536, 12, 2, 128, 8960, 151936, 2
	const pos, maxLen = 100, 256
	qDim, kvDim := nH*hd, nKV*hd
	half := hd / 2
	eps := float32(1e-6)
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	x0 := randMat(hidden, 100)
	invFreq := make([]float32, half)
	for d := range invFreq {
		invFreq[d] = float32(1.0 / math.Pow(1e6, float64(2*d)/float64(hd)))
	}
	mk := func(N, K int, seed uint64) *ResidentW8A8 {
		bq, s := quantW(N, K, seed)
		rm, e := ctx.UploadW8A8(bq, s, N, K)
		if e != nil {
			t.Fatal(e)
		}
		return rm
	}
	up32 := func(v []float32) *DeviceBuffer { d, _ := ctx.UploadF32(v); return d }
	invD := up32(invFreq)

	mw := ModelW{FinalNorm: up32(randMat(hidden, 1)), LMHead: mk(vocab, hidden, 7)}
	var sd uint64 = 10
	for l := range L {
		prior := randMat(pos*kvDim, uint64(l))
		kc, _ := ctx.NewKVCache(prior, maxLen*kvDim)
		vc, _ := ctx.NewKVCache(prior, maxLen*kvDim)
		sd += 7
		mw.Layers = append(mw.Layers, LayerW{
			Attn: AttnWeights{
				Norm: up32(randMat(hidden, sd)), QProj: mk(qDim, hidden, sd+1), KProj: mk(kvDim, hidden, sd+2),
				VProj: mk(kvDim, hidden, sd+3), OProj: mk(hidden, qDim, sd+4), InvFreq: invD, KCache: kc, VCache: vc,
			},
			MLPNorm: up32(randMat(hidden, sd+5)), Gate: mk(inter, hidden, sd+6), Up: mk(inter, hidden, sd+7), Down: mk(hidden, inter, sd+8),
		})
	}

	runner, err := ctx.NewDecodeRunner(mw, hidden, nH, nKV, hd, inter, 0, eps, scale, false)
	if err != nil {
		t.Fatalf("NewDecodeRunner: %v", err)
	}
	defer runner.Release()
	if _, err := runner.Run(x0, pos); err != nil { // warm
		t.Fatalf("Run warm: %v", err)
	}

	res := testing.Benchmark(func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := runner.Run(x0, pos); err != nil {
				b.Fatalf("Run: %v", err)
			}
		}
	})
	t.Logf("DecodeRunner.Run (vocab=%d, %d layers): %d allocs/op, %d B/op, %v/op",
		vocab, L, res.AllocsPerOp(), res.AllocedBytesPerOp(), time.Duration(res.NsPerOp()))
	t.Logf("(the per-token make([]float32, vocab) this change removes is 1 alloc / %d B)", vocab*4)
}
