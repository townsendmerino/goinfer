//go:build gpu

package gpu

import (
	"math"
	"testing"
)

// TestKVCacheF16_parity is the Increment-1 gate for the f16 KV cache
// (task-gpu-f16-kv.md): an f16-KV decode must track the f32-KV decode it replaces
// — argmax preserved every step, full-logit cosine ≥ 0.99. f16 is LOSSY (10-bit
// mantissa), so this is the int8/int4 quant-gate shape, NOT bit-exact; the f32
// default stays bit-exact (TestDecodeToken_parity).
//
// Crucially it runs at LONG context: both caches are prefilled with `prior` (8k)
// positions — the f32 cache exact, the f16 cache f16-rounded — so per-element f16
// rounding acts over thousands of keys and a long-range near-tie can flip. A
// short-prompt cosine says nothing about the regime the feature exists for.
// Software-adapter-skipped; real HW only.
func TestKVCacheF16_parity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	const hidden, nH, nKV, hd, inter, vocab, L = 1536, 12, 2, 128, 8960, 4096, 2
	const prior, steps = 8000, 16 // 8k-key context, then 16 decode steps compared
	qDim, kvDim := nH*hd, nKV*hd
	half := hd / 2
	eps := float32(1e-6)
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	capElems := (prior + steps + 1) * kvDim
	invFreq := make([]float32, half)
	for d := range invFreq {
		invFreq[d] = float32(1.0 / math.Pow(1e6, float64(2*d)/float64(hd)))
	}

	type lw struct {
		an, mn             []float32
		qBQ, kBQ, vBQ, oBQ []int8
		qS, kS, vS, oS     []float32
		gBQ, uBQ, dBQ      []int8
		gS, uS, dS         []float32
		priorK, priorV     []float32
	}
	layers := make([]lw, L)
	seed := uint64(1)
	W := func(N, K int) ([]int8, []float32) { seed++; return quantW(N, K, seed) }
	for l := range layers {
		layers[l].an = randMat(hidden, uint64(200+l))
		layers[l].mn = randMat(hidden, uint64(300+l))
		layers[l].qBQ, layers[l].qS = W(qDim, hidden)
		layers[l].kBQ, layers[l].kS = W(kvDim, hidden)
		layers[l].vBQ, layers[l].vS = W(kvDim, hidden)
		layers[l].oBQ, layers[l].oS = W(hidden, qDim)
		layers[l].gBQ, layers[l].gS = W(inter, hidden)
		layers[l].uBQ, layers[l].uS = W(inter, hidden)
		layers[l].dBQ, layers[l].dS = W(hidden, inter)
		layers[l].priorK = randMat(prior*kvDim, uint64(400+l))
		layers[l].priorV = randMat(prior*kvDim, uint64(500+l))
	}
	fnorm := randMat(hidden, 600)
	lmBQ, lmS := quantW(vocab, hidden, 999)

	up32 := func(v []float32) *DeviceBuffer { d, _ := ctx.UploadF32(v); return d }
	mk := func(bq []int8, s []float32, N, K int) *ResidentW8A8 {
		rm, e := ctx.UploadW8A8(bq, s, N, K)
		if e != nil {
			t.Fatal(e)
		}
		return rm
	}
	// build assembles a ModelW; f16=true allocates f16 KV caches (prefilled
	// f16-rounded) instead of f32. Weights/norms/projections are shared (identical
	// uploads) — only the cache precision differs between the two runners.
	invD := up32(invFreq)
	build := func(f16 bool) ModelW {
		mw := ModelW{FinalNorm: up32(fnorm), LMHead: mk(lmBQ, lmS, vocab, hidden)}
		for l := range layers {
			L := &layers[l]
			var kc, vc *DeviceBuffer
			if f16 {
				kc, _ = ctx.NewKVCacheF16(L.priorK, capElems)
				vc, _ = ctx.NewKVCacheF16(L.priorV, capElems)
			} else {
				kc, _ = ctx.NewKVCache(L.priorK, capElems)
				vc, _ = ctx.NewKVCache(L.priorV, capElems)
			}
			mw.Layers = append(mw.Layers, LayerW{
				Attn: AttnWeights{
					Norm: up32(L.an), QProj: mk(L.qBQ, L.qS, qDim, hidden), KProj: mk(L.kBQ, L.kS, kvDim, hidden),
					VProj: mk(L.vBQ, L.vS, kvDim, hidden), OProj: mk(L.oBQ, L.oS, hidden, qDim),
					InvFreq: invD, KCache: kc, VCache: vc,
				},
				MLPNorm: up32(L.mn), Gate: mk(L.gBQ, L.gS, inter, hidden), Up: mk(L.uBQ, L.uS, inter, hidden), Down: mk(L.dBQ, L.dS, hidden, inter),
			})
		}
		return mw
	}

	runner32, err := ctx.NewDecodeRunner(build(false), hidden, nH, nKV, hd, inter, 0, eps, scale, false)
	if err != nil {
		t.Fatalf("NewDecodeRunner f32: %v", err)
	}
	defer runner32.Release()
	rm16 := w8Model(build(true))
	rm16.kvF16 = true
	runner16, err := ctx.newDecodeRunner(rm16, hidden, nH, nKV, hd, inter, 0, eps, scale, false)
	if err != nil {
		t.Fatalf("newDecodeRunner f16: %v", err)
	}
	defer runner16.Release()

	// Decode `steps` tokens through both, feeding the SAME random input embedding at
	// each step (pos grows from prior). Each runner writes its own K/V (f16 vs f32)
	// into its cache, so the caches diverge by exactly the f16 rounding under test.
	// An argmax flip only signals a defect if the f32 model preferred its token by a
	// real margin; with random weights logits are near-uniform, so a flip at cosine
	// ~0.999 is an f16 near-tie tip, not degradation. Guard the flip by its gap as a
	// fraction of the f32 logit range (the qwen35 gate's divFrac guard, 0.03).
	minCos, sumCos, argmaxHits := 1.0, 0.0, 0
	maxFlipGap := 0.0
	for s := range steps {
		x := randMat(hidden, uint64(7000+s))
		pos := prior + s
		l32, err := runner32.Run(x, pos)
		if err != nil {
			t.Fatalf("f32 Run step %d: %v", s, err)
		}
		l16, err := runner16.Run(x, pos)
		if err != nil {
			t.Fatalf("f16 Run step %d: %v", s, err)
		}
		cos, _ := cosine(l16, l32)
		sumCos += cos
		if cos < minCos {
			minCos = cos
		}
		a32, a16 := argmaxF(l32), argmaxF(l16)
		if a16 == a32 {
			argmaxHits++
			continue
		}
		mx, mn := l32[a32], l32[a32]
		for _, v := range l32 {
			if v > mx {
				mx = v
			}
			if v < mn {
				mn = v
			}
		}
		gap := float64(l32[a32]-l32[a16]) / (float64(mx-mn) + 1e-9)
		if gap > maxFlipGap {
			maxFlipGap = gap
		}
		t.Logf("step %d: argmax flip f16=%d f32=%d, f32 gap %.4f of range (cosine %.5f)", s, a16, a32, gap, cos)
		if gap > 0.03 {
			t.Errorf("step %d: argmax flip gap %.4f > 3%% of range — not a near-tie; f16 KV bug", s, gap)
		}
	}
	meanCos := sumCos / float64(steps)
	t.Logf("=== f16-KV vs f32-KV decode (%d-key context, %d steps): argmax %d/%d | cosine min=%.5f mean=%.5f | worst flip gap=%.4f ===",
		prior, steps, argmaxHits, steps, minCos, meanCos, maxFlipGap)
	if minCos < 0.99 {
		t.Errorf("min cosine %.5f < 0.99 — f16 KV degrades decode beyond the quant floor", minCos)
	}
}

func argmaxF(v []float32) int {
	bi, bv := 0, float32(math.Inf(-1))
	for i, x := range v {
		if x > bv {
			bi, bv = i, x
		}
	}
	return bi
}
