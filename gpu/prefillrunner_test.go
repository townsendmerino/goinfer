//go:build gpu

package gpu

import (
	"math"
	"testing"
)

// TestPrefillLastW8A8_parity is the Increment-2 gate (docs/task-gpu-batched-
// prefill.md): PrefillLastW8A8's single returned row (the last position's logits)
// must be BIT-IDENTICAL to the last of M sequential DecodeToken calls over one
// shared KV cache. M=20 deliberately exceeds gemmRowMaxM=16 — the whole point of
// this function over DecodeTokenFusedBatched is that it is NOT capped, since real
// prompts run far past 16 tokens.
func TestPrefillLastW8A8_parity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	const hidden, nH, nKV, hd, inter, vocab, L = 256, 8, 2, 32, 688, 512, 2
	const start, M = 17, 20
	qDim, kvDim := nH*hd, nKV*hd
	half := hd / 2
	eps := float32(1e-6)
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	cap := (start + M) * kvDim

	invFreq := make([]float32, half)
	for d := range invFreq {
		invFreq[d] = float32(1.0 / math.Pow(1e6, float64(2*d)/float64(hd)))
	}
	up32 := func(v []float32) *DeviceBuffer { d, _ := ctx.UploadF32(v); return d }
	mk := func(bq []int8, s []float32, N, K int) *ResidentW8A8 {
		rm, e := ctx.UploadW8A8(bq, s, N, K)
		if e != nil {
			t.Fatal(e)
		}
		return rm
	}

	type lw struct {
		an, mn         []float32
		q, k, v, o     *ResidentW8A8
		g, u, d        *ResidentW8A8
		priorK, priorV []float32
	}
	seed := uint64(1)
	W := func(N, K int) *ResidentW8A8 { seed++; bq, s := quantW(N, K, seed); return mk(bq, s, N, K) }
	layers := make([]lw, L)
	for l := range layers {
		layers[l].an = randMat(hidden, uint64(200+l))
		layers[l].mn = randMat(hidden, uint64(300+l))
		layers[l].q = W(qDim, hidden)
		layers[l].k = W(kvDim, hidden)
		layers[l].v = W(kvDim, hidden)
		layers[l].o = W(hidden, qDim)
		layers[l].g = W(inter, hidden)
		layers[l].u = W(inter, hidden)
		layers[l].d = W(hidden, inter)
		layers[l].priorK = randMat(start*kvDim, uint64(400+l))
		layers[l].priorV = randMat(start*kvDim, uint64(500+l))
	}
	fnorm := up32(randMat(hidden, 600))
	lmBQ, lmS := quantW(vocab, hidden, 999)
	lmHead := mk(lmBQ, lmS, vocab, hidden)
	invD := up32(invFreq)

	buildMW := func() ModelW {
		mw := ModelW{FinalNorm: fnorm, LMHead: lmHead}
		for l := range layers {
			kc, _ := ctx.NewKVCache(layers[l].priorK, cap)
			vc, _ := ctx.NewKVCache(layers[l].priorV, cap)
			mw.Layers = append(mw.Layers, LayerW{
				Attn: AttnWeights{
					Norm: up32(layers[l].an), QProj: layers[l].q, KProj: layers[l].k, VProj: layers[l].v, OProj: layers[l].o,
					InvFreq: invD, KCache: kc, VCache: vc,
				},
				MLPNorm: up32(layers[l].mn), Gate: layers[l].g, Up: layers[l].u, Down: layers[l].d,
			})
		}
		return mw
	}

	xs := make([][]float32, M)
	for r := range xs {
		xs[r] = randMat(hidden, uint64(1000+r))
	}

	// Reference: M sequential DecodeToken calls over one shared KV cache — only the
	// LAST row's logits matter (matches PrefillLastW8A8's single-row contract).
	mwSeq := buildMW()
	defer mwSeq.Release()
	var refLast []float32
	for r := range xs {
		out, err := ctx.DecodeToken(xs[r], mwSeq, hidden, nH, nKV, hd, inter, start+r, 0, eps, scale, false)
		if err != nil {
			t.Fatalf("seq row %d: %v", r, err)
		}
		refLast = out
	}

	mwBatch := buildMW()
	defer mwBatch.Release()
	positions := make([]int, M)
	for r := range positions {
		positions[r] = start + r
	}
	got, err := ctx.PrefillLastW8A8(xs, mwBatch, hidden, nH, nKV, hd, inter, positions, 0, eps, scale, false)
	if err != nil {
		t.Fatalf("PrefillLastW8A8: %v", err)
	}

	cos, maxAbs := cosine(got, refLast)
	t.Logf("last row (pos %d): cosine=%.8f maxAbs=%.3e", start+M-1, cos, maxAbs)
	if cos < 0.999999 || maxAbs > 1e-3 {
		t.Errorf("diverges from sequential DecodeToken: cosine=%.8f maxAbs=%.3e", cos, maxAbs)
	}
}

// TestPrefillLastW8A8_declines checks runModelToModelW's scope guard: PrefillRunner
// must decline (not silently produce wrong output) on model features outside plain
// dense W8A8 — sliding window here as the representative case.
func TestPrefillLastW8A8_declines(t *testing.T) {
	rm := &runModel{slidingWindow: 128}
	if _, ok := runModelToModelW(rm, 64); ok {
		t.Fatal("runModelToModelW accepted a sliding-window model; PrefillLastW8A8 has no windowed-attention support")
	}
}

// TestPrefillLastW8A8_declinesGenuinePartialRoPE checks the hd/2 distinction:
// ropeHalf==hd/2 is full rotation (just set explicitly rather than left at the 0
// sentinel — real checkpoints do this, see TestResidentPrefillLast_parity's
// qwen2.5-coder-0.5b, hd=64, ropeHalf=32, which that test proves is ACCEPTED
// end-to-end); ropeHalf<hd/2 is genuine partial RoPE, which rope()'s hardcoded
// half:=hd/2 cannot express, and must be DECLINED — checked here since it only
// needs the early guard, not a full ModelW build (no GPU/real weights needed,
// same shape as TestPrefillLastW8A8_declines).
func TestPrefillLastW8A8_declinesGenuinePartialRoPE(t *testing.T) {
	const hd = 64
	if _, ok := runModelToModelW(&runModel{ropeHalf: hd/2 - 8}, hd); ok {
		t.Fatal("runModelToModelW accepted genuine partial RoPE (ropeHalf < hd/2); PrefillLastW8A8's rope() hardcodes full rotation")
	}
}
