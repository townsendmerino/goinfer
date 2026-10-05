//go:build gpu

package gpu

import (
	"math"
	"testing"
)

// TestPrefillLastW8A8_sharedQuantBitIdentical is R-25's gate (docs/tasks/task-recompute-audit.md): the batched prefill
// quantizes xn once for q, k and v and xn2 once for gate and up (sharedQ) instead of once per projection. Against the
// same prefill with a quantization per projection (prefillQuantPerProj), the last row's logits and every element of
// every layer's K and V cache must be equal bit for bit, with and without Qwen2's q/k/v biases (the bias-epilogue GEMM),
// and the shared path must have run: 2 shared quantizations per layer.
func TestPrefillLastW8A8_sharedQuantBitIdentical(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()
	defer func() { prefillQuantPerProj = false }()

	const hidden, nH, nKV, hd, inter, vocab, L = 256, 8, 2, 32, 688, 512, 2
	const start, M = 5, 24
	qDim, kvDim := nH*hd, nKV*hd
	half := hd / 2
	eps := float32(1e-6)
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	capElems := (start + M) * kvDim
	invFreq := make([]float32, half)
	for d := range invFreq {
		invFreq[d] = float32(1.0 / math.Pow(1e6, float64(2*d)/float64(hd)))
	}
	up32 := func(v []float32) *DeviceBuffer { d, _ := ctx.UploadF32(v); return d }
	seed := uint64(7)
	W := func(N, K int) *ResidentW8A8 {
		seed++
		bq, s := quantW(N, K, seed)
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
		qb, kb, vb     []float32
		priorK, priorV []float32
	}
	layers := make([]lw, L)
	for l := range layers {
		layers[l] = lw{an: randMat(hidden, uint64(200+l)), mn: randMat(hidden, uint64(300+l)),
			q: W(qDim, hidden), k: W(kvDim, hidden), v: W(kvDim, hidden), o: W(hidden, qDim),
			g: W(inter, hidden), u: W(inter, hidden), d: W(hidden, inter),
			qb: randMat(qDim, uint64(700+l)), kb: randMat(kvDim, uint64(710+l)), vb: randMat(kvDim, uint64(720+l)),
			priorK: randMat(start*kvDim, uint64(400+l)), priorV: randMat(start*kvDim, uint64(500+l))}
	}
	fnorm := up32(randMat(hidden, 600))
	lmBQ, lmS := quantW(vocab, hidden, 999)
	lmHead, err := ctx.UploadW8A8(lmBQ, lmS, vocab, hidden)
	if err != nil {
		t.Fatal(err)
	}
	invD := up32(invFreq)
	xs := make([][]float32, M)
	for r := range xs {
		xs[r] = randMat(hidden, uint64(1000+r))
	}
	positions := make([]int, M)
	for r := range positions {
		positions[r] = start + r
	}
	for _, withBias := range []bool{false, true} {
		run := func(perProj bool) (logits []float32, caches [][]float32, shared int64) {
			mw := ModelW{FinalNorm: fnorm, LMHead: lmHead}
			for l := range layers {
				kc, _ := ctx.NewKVCache(layers[l].priorK, capElems)
				vc, _ := ctx.NewKVCache(layers[l].priorV, capElems)
				a := AttnWeights{Norm: up32(layers[l].an), QProj: layers[l].q, KProj: layers[l].k, VProj: layers[l].v,
					OProj: layers[l].o, InvFreq: invD, KCache: kc, VCache: vc}
				if withBias {
					a.QBias, a.KBias, a.VBias = up32(layers[l].qb), up32(layers[l].kb), up32(layers[l].vb)
				}
				mw.Layers = append(mw.Layers, LayerW{Attn: a, MLPNorm: up32(layers[l].mn), Gate: layers[l].g, Up: layers[l].u, Down: layers[l].d})
			}
			prefillQuantPerProj = perProj
			s0 := prefillSharedQuants.Load()
			out, err := ctx.PrefillLastW8A8(xs, mw, hidden, nH, nKV, hd, inter, positions, 0, eps, scale, false)
			if err != nil {
				t.Fatalf("bias %v, per-projection %v: %v", withBias, perProj, err)
			}
			shared = prefillSharedQuants.Load() - s0
			for l := range mw.Layers {
				for _, c := range []*DeviceBuffer{mw.Layers[l].Attn.KCache, mw.Layers[l].Attn.VCache} {
					v, err := ctx.Readback(c)
					if err != nil {
						t.Fatal(err)
					}
					caches = append(caches, v)
				}
			}
			return append([]float32(nil), out...), caches, shared
		}
		refL, refC, refS := run(true)
		gotL, gotC, gotS := run(false)
		if refS != 0 || gotS != 2*L {
			t.Fatalf("bias %v: shared quantizations %d per-projection, %d shared; want 0 and %d", withBias, refS, gotS, 2*L)
		}
		for j := range refL {
			if math.Float32bits(refL[j]) != math.Float32bits(gotL[j]) {
				t.Fatalf("bias %v: logit %d is %v shared, %v per projection", withBias, j, gotL[j], refL[j])
			}
		}
		for i := range refC {
			for j := range refC[i][:(start+M)*kvDim] {
				if math.Float32bits(refC[i][j]) != math.Float32bits(gotC[i][j]) {
					t.Fatalf("bias %v: cache %d element %d differs", withBias, i, j)
				}
			}
		}
		t.Logf("bias %v: %d rows, logits and K/V bit-identical; %d quantizations shared", withBias, M, gotS)
	}
}
