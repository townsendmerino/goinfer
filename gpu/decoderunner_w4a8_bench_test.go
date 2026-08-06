//go:build gpu

package gpu

import (
	"math"
	"testing"
	"time"
)

// TestDecodeRunnerW4A8_7B_fit is the W4A8 footprint gate: it builds the FULL
// Qwen2.5-7B shape as a resident int4 model on the GPU with a 16k-context f32 KV
// cache (F2), confirms it FITS the 8 GB card (no OOM, resident bytes logged vs
// budget), and measures decode throughput. The footprint is shape-determined
// (weight values don't affect resident bytes or bandwidth), so synthetic int4
// weights of the exact 7B shape give the real fit + tok/s — the same way the
// 1.5B int8 throughput bench works; bit-exact correctness is gated separately on
// the 1.5B (TestDecodeRunnerW4A8_parity). The claim under test: a model that does
// NOT fit at int8 (7.07 GB weights) runs at int4.
func TestDecodeRunnerW4A8_7B_fit(t *testing.T) {
	if testing.Short() {
		t.Skip("7B fit")
	}
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	// Qwen2.5-7B-Instruct shape.
	const hidden, nH, nKV, hd, inter, vocab, L = 3584, 28, 4, 128, 18944, 152064, 28
	const ctxLen = 16384 // F2: f32 KV, cap at 16k
	const pos, group = 100, w4a8GroupSize
	qDim, kvDim := nH*hd, nKV*hd
	half := hd / 2
	eps := float32(1e-6)
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	x0 := randMat(hidden, 100)
	invFreq := make([]float32, half)
	for d := range invFreq {
		invFreq[d] = float32(1.0 / math.Pow(1e6, float64(2*d)/float64(hd)))
	}

	// resident byte accounting (what we expect to allocate)
	var wBytes int64
	w4 := func(N, K int) *ResidentW4A8 {
		kp := (K + 31) &^ 31
		nGroups := kp / group
		nib := make([]uint8, N*K)
		for i := range nib {
			nib[i] = uint8(i & 0xF)
		}
		sc := make([]float32, N*nGroups)
		for i := range sc {
			sc[i] = 0.02
		}
		rm, e := ctx.UploadW4A8(nib, sc, N, K)
		if e != nil {
			t.Fatalf("UploadW4A8 [%d×%d]: %v (likely VRAM)", N, K, e)
		}
		wBytes += int64(N*kp/2) + int64(N*nGroups*2) // nibbles + f16 scales
		return rm
	}
	up32 := func(v []float32) *DeviceBuffer { d, _ := ctx.UploadF32(v); return d }

	t.Logf("building resident Qwen2.5-7B int4 (%d layers, 16k KV)…", L)
	invD := up32(invFreq)
	rm := runModel{finalNorm: up32(randMat(hidden, 1)).buf, lmHead: w4(vocab, hidden)}
	var kvBytes int64
	prior := randMat(pos*kvDim, 1) // small prior; cache capacity is the 16k allocation
	for l := range L {
		kc, e1 := ctx.NewKVCache(prior, ctxLen*kvDim)
		vc, e2 := ctx.NewKVCache(prior, ctxLen*kvDim)
		if e1 != nil || e2 != nil {
			t.Fatalf("NewKVCache (layer %d): %v %v (VRAM exhausted — 7B+16k KV does not fit)", l, e1, e2)
		}
		kvBytes += 2 * int64(ctxLen*kvDim*4)
		rm.layers = append(rm.layers, runLayer{
			attnNorm: up32(randMat(hidden, uint64(10+l))).buf, invFreq: invD.buf, kCache: kc.buf, vCache: vc.buf,
			mlpNorm: up32(randMat(hidden, uint64(20+l))).buf,
			q:       w4(qDim, hidden), k: w4(kvDim, hidden), v: w4(kvDim, hidden), o: w4(hidden, qDim),
			gate: w4(inter, hidden), up: w4(inter, hidden), down: w4(hidden, inter),
		})
	}
	runner, err := ctx.newDecodeRunner(rm, hidden, nH, nKV, hd, inter, 0, eps, scale, false)
	if err != nil {
		t.Fatalf("newDecodeRunner(7B W4A8): %v", err)
	}
	defer runner.Release()

	gb := func(b int64) float64 { return float64(b) / 1e9 }
	t.Logf("FIT: weights %.2f GB (int4) + KV %.2f GB (f32 @16k) = %.2f GB resident (vs int8 weights would be ~7.07 GB — won't fit)",
		gb(wBytes), gb(kvBytes), gb(wBytes+kvBytes))

	// throughput: warm, then time decode Run().
	if _, err := runner.Run(x0, pos); err != nil {
		t.Fatalf("7B W4A8 Run: %v (allocated but failed to decode)", err)
	}
	const iters = 20
	t0 := time.Now()
	for range iters {
		if _, err := runner.Run(x0, pos); err != nil {
			t.Fatalf("Run: %v", err)
		}
	}
	per := time.Since(t0) / iters
	tps := 1e6 / float64(per.Microseconds())
	effGBs := gb(wBytes) * tps // weight bytes streamed per token × tok/s
	t.Logf("THROUGHPUT: %.1f ms/token = %.1f tok/s | effective %.1f GB/s = %.0f%% of ~350 roofline (weights %.2f GB/token)",
		float64(per.Microseconds())/1000, tps, effGBs, effGBs/350*100, gb(wBytes))
	t.Logf("  → full-token effective GB/s vs the 1.5B int4 (~0.87 GB/token × ~96 tok/s ≈ 84 GB/s, ~24%% roofline):")
	t.Logf("    7B amortizes the fixed per-token overhead (encode/glue/barriers) over more matmul → higher %% roofline")
}
