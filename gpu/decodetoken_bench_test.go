//go:build gpu

package gpu

import (
	"math"
	"testing"
	"time"

	"github.com/cogentcore/webgpu/wgpu"
)

// TestDecodeToken_throughput times the one-fence DecodeToken on the real Qwen-1.5B
// shapes (28 layers, hidden 1536, GQA 12/2, FFN 8960, vocab 151936) with resident
// random weights — the shapes drive the timing. This is the practical payoff: the
// per-token decode rate of the one-command-buffer forward, to compare against the
// staged W8A8 path (25.6 tok/s E2E, 3.07× CPU). Logs; run -v.
func TestDecodeToken_throughput(t *testing.T) {
	if testing.Short() {
		t.Skip("throughput")
	}
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	const hidden, nH, nKV, hd, inter, vocab, L = 1536, 12, 2, 128, 8960, 151936, 28
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

	t.Logf("uploading %d-layer resident model (~1.7 GB int8)…", L)
	mw := ModelW{FinalNorm: up32(randMat(hidden, 1)), LMHead: mk(vocab, hidden, 7)}
	defer mw.Release() // resident weights are caller-owned; Context.Close does not free them
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

	// warm up both
	if _, err := ctx.DecodeToken(x0, mw, hidden, nH, nKV, hd, inter, pos, 0, eps, scale, false); err != nil {
		t.Fatalf("DecodeToken: %v", err)
	}
	if _, err := ctx.DecodeTokenFused(x0, mw, hidden, nH, nKV, hd, inter, pos, 0, eps, scale, false); err != nil {
		t.Fatalf("DecodeTokenFused: %v", err)
	}
	const iters = 30
	t0 := time.Now()
	for range iters {
		ctx.DecodeToken(x0, mw, hidden, nH, nKV, hd, inter, pos, 0, eps, scale, false)
	}
	perMulti := time.Since(t0) / iters
	t1 := time.Now()
	for range iters {
		ctx.DecodeTokenFused(x0, mw, hidden, nH, nKV, hd, inter, pos, 0, eps, scale, false)
	}
	perFused := time.Since(t1) / iters
	// persistent runner: build once, time Run
	runner, err := ctx.NewDecodeRunner(mw, hidden, nH, nKV, hd, inter, 0, eps, scale, false)
	if err != nil {
		t.Fatalf("NewDecodeRunner: %v", err)
	}
	defer runner.Release()
	runner.Run(x0, pos)
	t2 := time.Now()
	for range iters {
		runner.Run(x0, pos)
	}
	perRun := time.Since(t2) / iters

	// §5 attribution: time the built plan with only the gemv dispatches vs only
	// the non-gemv (glue) dispatches, each one pass + submit + blocking Poll, min
	// over reps. Correctness is irrelevant here — this splits the 42 ms of GPU
	// execution into matmul-kernel time vs glue-kernel time directly.
	timeSteps := func(keep func(s runStep) bool) time.Duration {
		best := time.Hour
		for range 30 {
			enc, _ := ctx.device.CreateCommandEncoder(nil)
			pass := enc.BeginComputePass(nil)
			for _, s := range runner.steps {
				if !keep(s) {
					continue
				}
				pass.SetPipeline(s.pl)
				pass.SetBindGroup(0, s.bg, nil)
				pass.DispatchWorkgroups(s.gx, s.gy, 1)
			}
			pass.End()
			pass.Release()
			cmd, _ := enc.Finish(nil)
			t0 := time.Now()
			ctx.queue.Submit(cmd)
			ctx.device.Poll(true, nil)
			d := time.Since(t0)
			cmd.Release()
			enc.Release()
			if d < best {
				best = d
			}
		}
		return best
	}
	mss := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }
	isGemv := func(s runStep) bool { return s.pl == ctx.gemvPipeline }
	byPipe := func(pl *wgpu.ComputePipeline) time.Duration {
		return timeSteps(func(s runStep) bool { return s.pl == pl })
	}
	allSteps := timeSteps(func(s runStep) bool { return true })
	t.Logf("  step attribution (GPU exec, min/30): all %.1f ms | gemv %.1f ms | glue %.1f ms",
		mss(allSteps), mss(timeSteps(isGemv)), mss(timeSteps(func(s runStep) bool { return !isGemv(s) })))
	t.Logf("  per-kernel: quantize %.1f | rmsnorm %.1f | attn %.1f | rope %.1f | ropeStore %.1f | kvStore %.1f | swiglu %.1f | residual %.1f ms",
		mss(byPipe(ctx.quantizePipeline)), mss(byPipe(ctx.rmsnormPipeline)), mss(byPipe(ctx.attnPipeline)),
		mss(byPipe(ctx.ropePipeline)), mss(byPipe(ctx.ropeStorePipeline)), mss(byPipe(ctx.kvStorePipeline)),
		mss(byPipe(ctx.swigluPipeline)), mss(byPipe(ctx.residualPipeline)))

	// Resident int8 weight bytes streamed once per token (the decode roofline
	// denominator): per-layer qkv+o+gate+up+down, plus the LM head, plus the f32
	// per-row scales. effective GB/s = weightBytes × tok/s, vs ~448 peak / ~350
	// streaming on this card.
	perLayer := qDim*hidden + 2*kvDim*hidden + hidden*qDim + 3*inter*hidden
	scales := (qDim + 2*kvDim + hidden + 3*inter) * 4 // f32 scale per row, per layer
	weightBytes := float64(L*(perLayer+scales) + vocab*hidden + vocab*4)
	ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }
	tps := func(d time.Duration) float64 { return 1e6 / float64(d.Microseconds()) }
	gbs := func(d time.Duration) float64 { return weightBytes * tps(d) / 1e9 }
	t.Logf("resident weight bytes/token: %.2f GB", weightBytes/1e9)
	t.Logf("per-op-submit DecodeToken:   %.1f ms = %5.1f tok/s = %5.1f GB/s", ms(perMulti), tps(perMulti), gbs(perMulti))
	t.Logf("one-buffer DecodeTokenFused: %.1f ms = %5.1f tok/s = %5.1f GB/s", ms(perFused), tps(perFused), gbs(perFused))
	t.Logf("persistent DecodeRunner:     %.1f ms = %5.1f tok/s = %5.1f GB/s (%.0f%% of 350 GB/s roofline)",
		ms(perRun), tps(perRun), gbs(perRun), gbs(perRun)/350*100)
	// §5 phase split of the last Run: where the per-token wall actually goes.
	t.Logf("  phase split: write %.1f ms | encode %.1f ms | submit+poll %.1f ms",
		ms(runner.TWrite), ms(runner.TEncode), ms(runner.TSync))
	t.Logf("  → runner vs staged-E2E(25.6) %.2fx | vs CPU(8.5) %.2fx | %.0f%% of CUDA ceiling(171)",
		tps(perRun)/25.6, tps(perRun)/8.5, tps(perRun)/171*100)
}
