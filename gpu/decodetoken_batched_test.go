//go:build gpu

package gpu

import (
	"math"
	"testing"
)

// TestDecodeTokenFusedBatched_parity is the Stage-B (docs/spec/07) increment-1 gate:
// the batched M=K verify forward (projections as one tiled GEMM, attention per-row)
// must be BIT-IDENTICAL to M sequential DecodeTokenFused calls at consecutive
// positions sharing one KV cache — the per-row path it will replace. Same int8
// inputs + int32 accumulation ⇒ exact equality, including each row attending to the
// earlier rows of the block via the shared cache.
func TestDecodeTokenFusedBatched_parity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	const hidden, nH, nKV, hd, inter, vocab, L = 256, 8, 2, 32, 688, 512, 2
	const start, M = 17, 5
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

	// Per-layer weights (shared read-only across the two ModelWs); the KV caches are
	// rebuilt per ModelW from the same prefix so both runs start identical.
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

	// Reference: M sequential DecodeTokenFused over one shared KV cache.
	mwSeq := buildMW()
	ref := make([][]float32, M)
	for r := range xs {
		out, err := ctx.DecodeTokenFused(xs[r], mwSeq, hidden, nH, nKV, hd, inter, start+r, 0, eps, scale, false)
		if err != nil {
			t.Fatalf("seq row %d: %v", r, err)
		}
		ref[r] = out
	}

	// Batched: one call, fresh KV cache from the same prefix.
	mwBatch := buildMW()
	positions := make([]int, M)
	for r := range positions {
		positions[r] = start + r
	}
	got, err := ctx.DecodeTokenFusedBatched(xs, mwBatch, hidden, nH, nKV, hd, inter, positions, 0, eps, scale, false)
	if err != nil {
		t.Fatalf("batched: %v", err)
	}

	for r := range ref {
		cos, maxAbs := cosine(got[r], ref[r])
		t.Logf("row %d (pos %d): cosine=%.8f maxAbs=%.3e", r, start+r, cos, maxAbs)
		if cos < 0.999999 || maxAbs > 1e-3 {
			t.Errorf("row %d diverges: cosine=%.8f maxAbs=%.3e", r, cos, maxAbs)
		}
	}
}
