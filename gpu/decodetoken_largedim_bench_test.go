//go:build gpu

package gpu

import (
	"math"
	"strconv"
	"testing"
	"time"
)

// TestDecodeTokenFusedBatched_largedim is the docs/spec/07 follow-up kill-gate: the
// Stage-B batched verify is ~0.98× (no win) at qwen2.5-0.5b dims because at M=8 the
// resident decode is COMPUTE-bound, so batching's weight-reuse saving recovers nothing.
// The open hypothesis: on LARGE models (big hidden/inter) decode becomes
// BANDWIDTH-bound on streaming the projection weights, so batching M rows reads each
// weight once for all M → real amortization → batched beats per-row×M. This sweeps a
// dim ladder and prints the batched-vs-sequential speedup at each. ≥~1.3× at large dims
// would justify the (high-risk) Stage-B runner surgery; ~1× confirms it's dead. Run -v.
func TestDecodeTokenFusedBatched_largedim(t *testing.T) {
	if testing.Short() {
		t.Skip("microbench")
	}
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	type cfg struct {
		name                          string
		hidden, nH, nKV, hd, inter, L int
	}
	// vocab fixed small (32k) so the LM head doesn't dominate/OOM — we are measuring the
	// PROJECTION amortization (q/k/v/o + gate/up/down), which is where a big model's
	// decode bandwidth goes. L kept small at the biggest dims to fit 8 GB.
	const vocab = 32000
	configs := []cfg{
		{"qwen0.5b", 896, 14, 2, 64, 4864, 8},
		{"llama7b", 4096, 32, 8, 128, 14336, 2},
		{"llama13b", 5120, 40, 40, 128, 13824, 2},
		{"llama70b-layer", 8192, 64, 8, 128, 28672, 1},
	}
	const start, M, iters = 64, 8, 12

	for _, c := range configs {
		runLargeDim(t, ctx, c.name, c.hidden, c.nH, c.nKV, c.hd, c.inter, vocab, c.L, start, M, iters)
	}

	// M-sweep at 70B dims: does the amortization GROW with the batch (more rows reuse
	// each weight load)? EAGLE trees produce many nodes, so a rising curve toward a high
	// ceiling is what would make Stage-B + a model drafter viable end-to-end.
	t.Log("--- M-sweep @ llama70b-layer dims ---")
	for _, mm := range []int{4, 8, 16, 32, 48} {
		// gemmRowMaxM is a HARD kernel limit, not a tuning knob: the thin-M gemmRow kernel
		// accumulates into a private array<i32, gemmRowMaxM> (gemm_rows.go:19), and
		// DecodeTokenFusedBatched returns a clean error above it. This sweep used to request
		// M=32/48 anyway and t.Fatalf on that error, so the whole test reported FAIL for asking
		// the API to do something it documents it cannot — charging every unrelated leg an
		// investigation. (Proven unrelated to the MoE cap work in 0018114 by stashing that
		// commit's only gpu/ change and re-running: identical failure.) The measurable part of
		// the sweep is M <= gemmRowMaxM; beyond it the answer is "chunk the block", which is a
		// Stage-B design question, not a measurement.
		if mm > gemmRowMaxM {
			t.Logf("70b-M%d: SKIPPED — exceeds gemmRowMaxM=%d (kernel limit; batching past it needs "+
				"block chunking, tracked as Stage-B runner surgery in docs/spec/07)", mm, gemmRowMaxM)
			continue
		}
		runLargeDim(t, ctx, "70b-M"+strconv.Itoa(mm), 8192, 64, 8, 128, 28672, vocab, 1, start, mm, iters)
	}
}

func runLargeDim(t *testing.T, ctx *Context, name string, hidden, nH, nKV, hd, inter, vocab, L, start, M, iters int) {
	qDim, kvDim := nH*hd, nKV*hd
	half := hd / 2
	eps := float32(1e-6)
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	capKV := (start + M) * kvDim

	invFreq := make([]float32, half)
	for d := range invFreq {
		invFreq[d] = float32(1.0 / math.Pow(1e6, float64(2*d)/float64(hd)))
	}
	var bufs []*DeviceBuffer
	var rms []*ResidentW8A8
	up32 := func(v []float32) *DeviceBuffer {
		d, _ := ctx.UploadF32(v)
		bufs = append(bufs, d)
		return d
	}
	mk := func(N, K int, seed uint64) *ResidentW8A8 {
		bq, s := quantW(N, K, seed)
		rm, e := ctx.UploadW8A8(bq, s, N, K)
		if e != nil {
			t.Fatal(e)
		}
		rms = append(rms, rm)
		return rm
	}
	invD := up32(invFreq)
	seed := uint64(1)
	W := func(N, K int) *ResidentW8A8 { seed++; return mk(N, K, seed) }
	mw := ModelW{FinalNorm: up32(randMat(hidden, 600)), LMHead: mk(vocab, hidden, 999)}
	for l := range L {
		kc, _ := ctx.NewKVCache(randMat(start*kvDim, uint64(400+l)), capKV)
		vc, _ := ctx.NewKVCache(randMat(start*kvDim, uint64(500+l)), capKV)
		bufs = append(bufs, kc, vc)
		mw.Layers = append(mw.Layers, LayerW{
			Attn: AttnWeights{
				Norm: up32(randMat(hidden, uint64(200+l))), QProj: W(qDim, hidden), KProj: W(kvDim, hidden),
				VProj: W(kvDim, hidden), OProj: W(hidden, qDim), InvFreq: invD, KCache: kc, VCache: vc,
			},
			MLPNorm: up32(randMat(hidden, uint64(300+l))), Gate: W(inter, hidden), Up: W(inter, hidden), Down: W(hidden, inter),
		})
	}
	xs := make([][]float32, M)
	for r := range xs {
		xs[r] = randMat(hidden, uint64(1000+r))
	}
	positions := make([]int, M)
	for r := range positions {
		positions[r] = start + r
	}

	seqRun := func() {
		for r := range xs {
			if _, err := ctx.DecodeTokenFused(xs[r], mw, hidden, nH, nKV, hd, inter, start+r, 0, eps, scale, false); err != nil {
				t.Fatalf("%s seq: %v", name, err)
			}
		}
	}
	batRun := func() {
		if _, err := ctx.DecodeTokenFusedBatched(xs, mw, hidden, nH, nKV, hd, inter, positions, 0, eps, scale, false); err != nil {
			t.Fatalf("%s batched: %v", name, err)
		}
	}
	seqRun()
	batRun() // warm

	t0 := time.Now()
	for range iters {
		seqRun()
	}
	seq := time.Since(t0) / time.Duration(iters)
	t1 := time.Now()
	for range iters {
		batRun()
	}
	bat := time.Since(t1) / time.Duration(iters)

	ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }
	t.Logf("%-16s (h=%d inter=%d L=%d M=%d): per-row×M %7.2f ms | batched %7.2f ms | speedup %.2f×",
		name, hidden, inter, L, M, ms(seq), ms(bat), float64(seq)/float64(bat))

	for _, r := range rms {
		r.Release()
	}
	for _, b := range bufs {
		b.Release()
	}
}
