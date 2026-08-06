//go:build gpu

package gpu

import (
	"math"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestDecodeRunnerMoE_parity gates Lever C3c: the integrated resident MoE forward
// (router top-k → indexed gate/up GEMVs → SwiGLU → weighted down-combine, all on the
// device in one command buffer) must match a CPU oracle running the SAME math. Both
// sides share the int8 expert/router weights and the int8 activation quantization, so
// this is a wiring/kernel gate (cosine ~1.0): it proves the on-GPU router selects the
// right experts and the stacked indexed GEMVs combine them into the residual exactly
// as moeMLP does. Mixtral-class shape: 8 experts, top-2, softmax + NormTopKProb, no
// shared expert, every layer MoE.
func TestDecodeRunnerMoE_parity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	const hidden, nH, nKV, hd, inter, pos, vocab, L = 256, 4, 2, 64, 128, 8, 512, 2
	const nE, topK = 8, 2
	qDim, kvDim := nH*hd, nKV*hd
	half := hd / 2
	eps := float32(1e-6)
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	x0 := randMat(hidden, 100)
	invFreq := make([]float32, half)
	for d := range invFreq {
		invFreq[d] = float32(1.0 / math.Pow(1e6, float64(2*d)/float64(hd)))
	}

	type lw struct {
		an, mn             []float32
		qBQ, kBQ, vBQ, oBQ []int8
		qS, kS, vS, oS     []float32
		rBQ                []int8    // router [nE, hidden]
		rS                 []float32 // router scales [nE]
		gBQ, uBQ, dBQ      [][]int8  // per-expert gate/up/down int8 weights
		gS, uS, dS         [][]float32
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
		layers[l].rBQ, layers[l].rS = W(nE, hidden)
		layers[l].gBQ = make([][]int8, nE)
		layers[l].uBQ = make([][]int8, nE)
		layers[l].dBQ = make([][]int8, nE)
		layers[l].gS = make([][]float32, nE)
		layers[l].uS = make([][]float32, nE)
		layers[l].dS = make([][]float32, nE)
		for e := range nE {
			layers[l].gBQ[e], layers[l].gS[e] = W(inter, hidden)
			layers[l].uBQ[e], layers[l].uS[e] = W(inter, hidden)
			layers[l].dBQ[e], layers[l].dS[e] = W(hidden, inter)
		}
		layers[l].priorK = randMat(pos*kvDim, uint64(400+l))
		layers[l].priorV = randMat(pos*kvDim, uint64(500+l))
	}
	fnorm := randMat(hidden, 600)
	lmBQ, lmS := quantW(vocab, hidden, 999)

	silu := func(g float32) float32 { return g / (1 + float32(math.Exp(float64(-g)))) }

	// routeRef mirrors decoder.routeExperts for the Mixtral path AND the on-GPU
	// moeRoute kernel: softmax over experts, top-k by probability (strict-greater
	// ascending scan = lowest index on ties, exactly the kernel), weights = those
	// probabilities renormalized to sum 1.
	routeRef := func(logits []float32) (idx []int, wts []float32) {
		var mx float32 = logits[0]
		for _, v := range logits[1:] {
			if v > mx {
				mx = v
			}
		}
		probs := make([]float32, nE)
		var sum float32
		for i, v := range logits {
			probs[i] = float32(math.Exp(float64(v - mx)))
			sum += probs[i]
		}
		for i := range probs {
			probs[i] /= sum
		}
		sel := append([]float32(nil), probs...)
		idx = make([]int, topK)
		wts = make([]float32, topK)
		var wsum float32
		for j := range topK {
			best, bestv := 0, float32(math.Inf(-1))
			for i, v := range sel {
				if v > bestv {
					best, bestv = i, v
				}
			}
			idx[j] = best
			wts[j] = probs[best]
			wsum += probs[best]
			sel[best] = float32(math.Inf(-1))
		}
		for j := range wts {
			wts[j] /= wsum // NormTopKProb
		}
		return idx, wts
	}

	// --- CPU oracle (same math as the GPU MoE forward) ---
	x := append([]float32(nil), x0...)
	for l := range layers {
		L := &layers[l]
		xn := refRMSNorm(x, L.an, hidden, eps, false)
		q := make([]float32, qDim)
		linalg.MatmulBTW8A8(xn, L.qBQ, L.qS, q, 1, hidden, qDim)
		k := make([]float32, kvDim)
		linalg.MatmulBTW8A8(xn, L.kBQ, L.kS, k, 1, hidden, kvDim)
		v := make([]float32, kvDim)
		linalg.MatmulBTW8A8(xn, L.vBQ, L.vS, v, 1, hidden, kvDim)
		refRoPE(q, nH, hd, pos, invFreq)
		refRoPE(k, nKV, hd, pos, invFreq)
		kFull := append(append([]float32(nil), L.priorK...), k...)
		vFull := append(append([]float32(nil), L.priorV...), v...)
		cv := refAttn(q, kFull, vFull, nH, nKV, hd, pos+1, scale)
		ao := make([]float32, hidden)
		linalg.MatmulBTW8A8(cv, L.oBQ, L.oS, ao, 1, qDim, hidden)
		for i := range x {
			x[i] += ao[i]
		}
		// MoE FFN.
		xn2 := refRMSNorm(x, L.mn, hidden, eps, false)
		logits := make([]float32, nE)
		linalg.MatmulBTW8A8(xn2, L.rBQ, L.rS, logits, 1, hidden, nE)
		idx, wts := routeRef(logits)
		for j, e := range idx {
			gate := make([]float32, inter)
			linalg.MatmulBTW8A8(xn2, L.gBQ[e], L.gS[e], gate, 1, hidden, inter)
			up := make([]float32, inter)
			linalg.MatmulBTW8A8(xn2, L.uBQ[e], L.uS[e], up, 1, hidden, inter)
			mid := make([]float32, inter)
			for i := range mid {
				mid[i] = silu(gate[i]) * up[i]
			}
			down := make([]float32, hidden)
			linalg.MatmulBTW8A8(mid, L.dBQ[e], L.dS[e], down, 1, inter, hidden)
			w := wts[j]
			for i := range x {
				x[i] += w * down[i]
			}
		}
	}
	xnf := refRMSNorm(x, fnorm, hidden, eps, false)
	refLogits := make([]float32, vocab)
	linalg.MatmulBTW8A8(xnf, lmBQ, lmS, refLogits, 1, hidden, vocab)

	// --- GPU MoE DecodeRunner ---
	mk := func(bq []int8, s []float32, N, K int) *ResidentW8A8 {
		rm, e := ctx.UploadW8A8(bq, s, N, K)
		if e != nil {
			t.Fatal(e)
		}
		return rm
	}
	stack := func(bq [][]int8, s [][]float32, N, K int) *ResidentStackedW8A8 {
		st, e := ctx.UploadStackedExperts(bq, s, nE, N, K)
		if e != nil {
			t.Fatal(e)
		}
		return st
	}
	up32 := func(v []float32) *DeviceBuffer { d, _ := ctx.UploadF32(v); return d }
	invD := up32(invFreq)
	rm := runModel{
		finalNorm: up32(fnorm).buf,
		lmHead:    mk(lmBQ, lmS, vocab, hidden),
		moe:       &moeRunParams{nE: nE, k: topK, inter: inter, sigmoid: false, norm: true, scale: 0},
	}
	for l := range layers {
		L := &layers[l]
		kc, _ := ctx.NewKVCache(L.priorK, (pos+1)*kvDim)
		vc, _ := ctx.NewKVCache(L.priorV, (pos+1)*kvDim)
		rm.layers = append(rm.layers, runLayer{
			attnNorm: up32(L.an).buf, invFreq: invD.buf, kCache: kc.buf, vCache: vc.buf, mlpNorm: up32(L.mn).buf,
			q: mk(L.qBQ, L.qS, qDim, hidden), k: mk(L.kBQ, L.kS, kvDim, hidden), v: mk(L.vBQ, L.vS, kvDim, hidden),
			o:       mk(L.oBQ, L.oS, hidden, qDim),
			isMoE:   true,
			router:  mk(L.rBQ, L.rS, nE, hidden),
			expGate: stack(L.gBQ, L.gS, inter, hidden),
			expUp:   stack(L.uBQ, L.uS, inter, hidden),
			expDown: stack(L.dBQ, L.dS, hidden, inter),
		})
	}
	runner, err := ctx.newDecodeRunner(rm, hidden, nH, nKV, hd, inter, 0, eps, scale, false)
	if err != nil {
		t.Fatalf("newDecodeRunner(MoE): %v", err)
	}
	defer runner.Release()
	got, err := runner.Run(x0, pos)
	if err != nil {
		t.Fatalf("MoE Run: %v", err)
	}
	cos, maxAbs := cosine(got, refLogits)
	t.Logf("MoE DecodeRunner parity (%d layers, %d experts top-%d): cosine=%.6f maxAbs=%.3e", L, nE, topK, cos, maxAbs)
	if cos < 0.9999 {
		t.Errorf("MoE DecodeRunner diverges: cosine=%.6f maxAbs=%.3e", cos, maxAbs)
	}
}
