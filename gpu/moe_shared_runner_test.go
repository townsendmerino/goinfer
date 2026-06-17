//go:build gpu

package gpu

import (
	"math"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestDecodeRunnerMoEShared_parity gates Lever C3d: the resident MoE forward with an
// always-on shared expert. Two flavors run through the integrated runner — GLM-style
// (sigmoid routing + selection bias, UNGATED shared expert) and qwen2_moe-style
// (softmax routing, sigmoid-GATED shared expert). Each must match a CPU oracle running
// the identical int8 math (cosine ~1.0): proves the shared-expert add (gated/ungated)
// and the sigmoid+bias routing land correctly on top of the C3c routed-expert combine.
func TestDecodeRunnerMoEShared_parity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	const hidden, nH, nKV, hd, inter, sInter, pos, vocab, L = 256, 4, 2, 64, 128, 192, 8, 512, 2
	const nE, topK = 8, 2
	qDim, kvDim := nH*hd, nKV*hd
	half := hd / 2
	eps := float32(1e-6)
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	silu := func(g float32) float32 { return g / (1 + float32(math.Exp(float64(-g)))) }

	cases := []struct {
		name          string
		sigmoid, bias bool // routing flavor
		ungated       bool // shared-expert combine
	}{
		{"glm_sigmoid_bias_ungated", true, true, true},
		{"qwen2moe_softmax_gated", false, false, false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			x0 := randMat(hidden, 100)
			invFreq := make([]float32, half)
			for d := range invFreq {
				invFreq[d] = float32(1.0 / math.Pow(1e6, float64(2*d)/float64(hd)))
			}
			type lw struct {
				an, mn             []float32
				qBQ, kBQ, vBQ, oBQ []int8
				qS, kS, vS, oS     []float32
				rBQ                []int8
				rS                 []float32
				rBias              []float32 // router selection bias [nE] (sigmoid path)
				gBQ, uBQ, dBQ      [][]int8
				gS, uS, dS         [][]float32
				sgBQ, suBQ, sdBQ   []int8 // shared expert gate/up/down
				sgS, suS, sdS      []float32
				sGateBQ            []int8 // shared sigmoid gate [1,hidden]
				sGateS             []float32
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
				if tc.bias {
					layers[l].rBias = randMat(nE, uint64(700+l)) // small biases shift selection
				}
				layers[l].gBQ = make([][]int8, nE)
				layers[l].uBQ = make([][]int8, nE)
				layers[l].dBQ = make([][]int8, nE)
				layers[l].gS = make([][]float32, nE)
				layers[l].uS = make([][]float32, nE)
				layers[l].dS = make([][]float32, nE)
				for e := 0; e < nE; e++ {
					layers[l].gBQ[e], layers[l].gS[e] = W(inter, hidden)
					layers[l].uBQ[e], layers[l].uS[e] = W(inter, hidden)
					layers[l].dBQ[e], layers[l].dS[e] = W(hidden, inter)
				}
				layers[l].sgBQ, layers[l].sgS = W(sInter, hidden)
				layers[l].suBQ, layers[l].suS = W(sInter, hidden)
				layers[l].sdBQ, layers[l].sdS = W(hidden, sInter)
				if !tc.ungated {
					layers[l].sGateBQ, layers[l].sGateS = W(1, hidden)
				}
				layers[l].priorK = randMat(pos*kvDim, uint64(400+l))
				layers[l].priorV = randMat(pos*kvDim, uint64(500+l))
			}
			fnorm := randMat(hidden, 600)
			lmBQ, lmS := quantW(vocab, hidden, 999)

			// routeRef mirrors decoder.routeExperts AND the moeRoute kernel: softmax or
			// per-expert sigmoid scores, +bias for the top-k SELECTION only, weights are
			// the un-biased scores renormalized to sum 1 (NormTopKProb).
			routeRef := func(logits, bias []float32) (idx []int, wts []float32) {
				score := make([]float32, nE)
				if tc.sigmoid {
					for i, v := range logits {
						score[i] = float32(1.0 / (1.0 + math.Exp(-float64(v))))
					}
				} else {
					var mx = logits[0]
					for _, v := range logits[1:] {
						if v > mx {
							mx = v
						}
					}
					var sum float32
					for i, v := range logits {
						score[i] = float32(math.Exp(float64(v - mx)))
						sum += score[i]
					}
					for i := range score {
						score[i] /= sum
					}
				}
				sel := append([]float32(nil), score...)
				if bias != nil {
					for i := range sel {
						sel[i] += bias[i]
					}
				}
				idx = make([]int, topK)
				wts = make([]float32, topK)
				var wsum float32
				for j := 0; j < topK; j++ {
					best, bestv := 0, float32(math.Inf(-1))
					for i, v := range sel {
						if v > bestv {
							best, bestv = i, v
						}
					}
					idx[j] = best
					wts[j] = score[best]
					wsum += score[best]
					sel[best] = float32(math.Inf(-1))
				}
				for j := range wts {
					wts[j] /= wsum
				}
				return idx, wts
			}

			// --- CPU oracle ---
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
				xn2 := refRMSNorm(x, L.mn, hidden, eps, false)
				logits := make([]float32, nE)
				linalg.MatmulBTW8A8(xn2, L.rBQ, L.rS, logits, 1, hidden, nE)
				idx, wts := routeRef(logits, L.rBias)
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
					for i := range x {
						x[i] += wts[j] * down[i]
					}
				}
				// shared expert.
				sgate := make([]float32, sInter)
				linalg.MatmulBTW8A8(xn2, L.sgBQ, L.sgS, sgate, 1, hidden, sInter)
				sup := make([]float32, sInter)
				linalg.MatmulBTW8A8(xn2, L.suBQ, L.suS, sup, 1, hidden, sInter)
				smid := make([]float32, sInter)
				for i := range smid {
					smid[i] = silu(sgate[i]) * sup[i]
				}
				sdown := make([]float32, hidden)
				linalg.MatmulBTW8A8(smid, L.sdBQ, L.sdS, sdown, 1, sInter, hidden)
				g := float32(1)
				if !tc.ungated {
					gl := make([]float32, 1)
					linalg.MatmulBTW8A8(xn2, L.sGateBQ, L.sGateS, gl, 1, hidden, 1)
					g = float32(1.0 / (1.0 + math.Exp(-float64(gl[0]))))
				}
				for i := range x {
					x[i] += g * sdown[i]
				}
			}
			xnf := refRMSNorm(x, fnorm, hidden, eps, false)
			refLogits := make([]float32, vocab)
			linalg.MatmulBTW8A8(xnf, lmBQ, lmS, refLogits, 1, hidden, vocab)

			// --- GPU resident MoE+shared runner ---
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
				moe: &moeRunParams{
					nE: nE, k: topK, inter: inter, sigmoid: tc.sigmoid, norm: true, scale: 0,
					sharedInter: sInter, sharedUngated: tc.ungated,
				},
			}
			for l := range layers {
				L := &layers[l]
				kc, _ := ctx.NewKVCache(L.priorK, (pos+1)*kvDim)
				vc, _ := ctx.NewKVCache(L.priorV, (pos+1)*kvDim)
				rl := runLayer{
					attnNorm: up32(L.an).buf, invFreq: invD.buf, kCache: kc.buf, vCache: vc.buf, mlpNorm: up32(L.mn).buf,
					q: mk(L.qBQ, L.qS, qDim, hidden), k: mk(L.kBQ, L.kS, kvDim, hidden), v: mk(L.vBQ, L.vS, kvDim, hidden),
					o:       mk(L.oBQ, L.oS, hidden, qDim),
					isMoE:   true,
					router:  mk(L.rBQ, L.rS, nE, hidden),
					expGate: stack(L.gBQ, L.gS, inter, hidden),
					expUp:   stack(L.uBQ, L.uS, inter, hidden),
					expDown: stack(L.dBQ, L.dS, hidden, inter),
					shGate:  mk(L.sgBQ, L.sgS, sInter, hidden),
					shUp:    mk(L.suBQ, L.suS, sInter, hidden),
					shDown:  mk(L.sdBQ, L.sdS, hidden, sInter),
				}
				if L.rBias != nil {
					rl.routerBias = up32(L.rBias).buf
				}
				if !tc.ungated {
					rl.shGateW = mk(L.sGateBQ, L.sGateS, 1, hidden)
				}
				rm.layers = append(rm.layers, rl)
			}
			runner, err := ctx.newDecodeRunner(rm, hidden, nH, nKV, hd, inter, 0, eps, scale, false)
			if err != nil {
				t.Fatalf("newDecodeRunner(MoE+shared): %v", err)
			}
			defer runner.Release()
			got, err := runner.Run(x0, pos)
			if err != nil {
				t.Fatalf("MoE+shared Run: %v", err)
			}
			cos, maxAbs := cosine(got, refLogits)
			t.Logf("%s: cosine=%.6f maxAbs=%.3e", tc.name, cos, maxAbs)
			if cos < 0.9999 {
				t.Errorf("MoE+shared diverges: cosine=%.6f maxAbs=%.3e", cos, maxAbs)
			}
		})
	}
}
