//go:build gpu

package gpu

import (
	"math"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// mlaRopeRef mirrors decoder.mlaRope: optional de-interleave (V3 GPT-J pairwise) then
// NeoX rotate_half on a single [ropeDim] vector at position pos.
func mlaRopeRef(vec []float32, ropeDim, pos int, invFreq []float32, interleave bool) []float32 {
	out := append([]float32(nil), vec...)
	half := ropeDim / 2
	if interleave {
		tmp := make([]float32, ropeDim)
		for i := range half {
			tmp[i] = out[2*i]
			tmp[half+i] = out[2*i+1]
		}
		copy(out, tmp)
	}
	rot := append([]float32(nil), out...)
	for d := range half {
		theta := float64(pos) * float64(invFreq[d])
		c, s := float32(math.Cos(theta)), float32(math.Sin(theta))
		rot[d] = out[d]*c - out[half+d]*s
		rot[half+d] = out[half+d]*c + out[d]*s
	}
	return rot
}

// TestDecodeRunnerMLA_parity gates Lever C4c: the full MLA latent-attention forward on
// the resident runner, end-to-end against a CPU int8 oracle that mirrors the same absorb
// math. The tiny model is DeepSeek-V3-shaped — q-LoRA bottleneck, compressed-KV latent
// cache, decoupled interleaved RoPE, and a DeepSeekMoE FFN (sigmoid routing + selection
// bias + group-limited top-k + ungated shared expert). Every int8 GEMV runs identical
// math both sides (W8A8); the absorb/lift/attention are f32. Cosine must be ~1.0, proving
// the latent store, W_UK absorb, qRope, rank-space attend, W_UV lift, group routing, and
// the shared-expert combine all land correctly on top of the C3 MoE machinery.
func TestDecodeRunnerMLA_parity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	const hidden, nH = 256, 4
	const qkNope, qkRope, vHead, rank, qLoRA = 32, 16, 32, 64, 96
	const nE, topK, nGroup, topkGroup = 8, 2, 2, 1
	const inter, sInter, pos, vocab, L = 64, 96, 6, 512, 2
	const interleave = true
	qkHead := qkNope + qkRope
	latDim := rank + qkRope
	eps := float32(1e-6)
	scale := float32(1.0 / math.Sqrt(float64(qkHead)))
	silu := func(g float32) float32 { return g / (1 + float32(math.Exp(float64(-g)))) }

	invFreq := make([]float32, qkRope/2)
	for d := range invFreq {
		invFreq[d] = float32(1.0 / math.Pow(1e4, float64(2*d)/float64(qkRope)))
	}

	type lw struct {
		an, mn                 []float32
		qaBQ, qbBQ, kvaBQ, oBQ []int8
		qaS, qbS, kvaS, oS     []float32
		qaNorm, kvaNorm        []float32
		wuk, wuv               []float32 // f32 absorb/lift, per head
		rBQ                    []int8
		rS, rBias              []float32
		gBQ, uBQ, dBQ          [][]int8
		gS, uS, dS             [][]float32
		sgBQ, suBQ, sdBQ       []int8
		sgS, suS, sdS          []float32
		priorLat               []float32 // pos already-stored latents [pos*latDim]
	}
	x0 := randMat(hidden, 100)
	layers := make([]lw, L)
	seed := uint64(1)
	W := func(N, K int) ([]int8, []float32) { seed++; return quantW(N, K, seed) }
	for l := range layers {
		layers[l].an = randMat(hidden, uint64(200+l))
		layers[l].mn = randMat(hidden, uint64(300+l))
		layers[l].qaBQ, layers[l].qaS = W(qLoRA, hidden)
		layers[l].qbBQ, layers[l].qbS = W(nH*qkHead, qLoRA)
		layers[l].kvaBQ, layers[l].kvaS = W(latDim, hidden)
		layers[l].oBQ, layers[l].oS = W(hidden, nH*vHead)
		layers[l].qaNorm = randMat(qLoRA, uint64(210+l))
		layers[l].kvaNorm = randMat(rank, uint64(220+l))
		layers[l].wuk = randMat(nH*rank*qkNope, uint64(230+l))
		layers[l].wuv = randMat(nH*vHead*rank, uint64(240+l))
		layers[l].rBQ, layers[l].rS = W(nE, hidden)
		layers[l].rBias = randMat(nE, uint64(700+l))
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
		layers[l].sgBQ, layers[l].sgS = W(sInter, hidden)
		layers[l].suBQ, layers[l].suS = W(sInter, hidden)
		layers[l].sdBQ, layers[l].sdS = W(hidden, sInter)
		layers[l].priorLat = randMat(pos*latDim, uint64(400+l))
	}
	fnorm := randMat(hidden, 600)
	lmBQ, lmS := quantW(vocab, hidden, 999)

	// routeRef mirrors decoder.routeExperts with group-limiting: sigmoid scores, +bias
	// for SELECTION, group-limit (top-2-sum group score, keep topkGroup groups), top-k,
	// weights are the un-biased scores renormalized to sum 1.
	routeRef := func(logits, bias []float32) (idx []int, wts []float32) {
		score := make([]float32, nE)
		for i, v := range logits {
			score[i] = float32(1.0 / (1.0 + math.Exp(-float64(v))))
		}
		sel := make([]float32, nE)
		for i := range score {
			sel[i] = score[i] + bias[i]
		}
		gsz := nE / nGroup
		type gs struct {
			g int
			v float32
		}
		groups := make([]gs, nGroup)
		for g := range nGroup {
			t1, t2 := float32(math.Inf(-1)), float32(math.Inf(-1))
			for i := g * gsz; i < (g+1)*gsz; i++ {
				if sel[i] > t1 {
					t1, t2 = sel[i], t1
				} else if sel[i] > t2 {
					t2 = sel[i]
				}
			}
			groups[g] = gs{g, t1 + t2}
		}
		keep := make([]bool, nGroup)
		for range topkGroup {
			best, bv := -1, float32(math.Inf(-1))
			for _, gg := range groups {
				if !keep[gg.g] && gg.v > bv {
					best, bv = gg.g, gg.v
				}
			}
			keep[best] = true
		}
		for g := range nGroup {
			if !keep[g] {
				for i := g * gsz; i < (g+1)*gsz; i++ {
					sel[i] = float32(math.Inf(-1))
				}
			}
		}
		idx = make([]int, topK)
		wts = make([]float32, topK)
		var wsum float32
		for j := range topK {
			best, bv := 0, float32(math.Inf(-1))
			for i, v := range sel {
				if v > bv {
					best, bv = i, v
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

	// --- CPU oracle (op-by-op mirror of the resident MLA forward) ---
	x := append([]float32(nil), x0...)
	for l := range layers {
		L := &layers[l]
		xn := refRMSNorm(x, L.an, hidden, eps, false)
		// q: q_a → norm → q_b LoRA bottleneck.
		qa := make([]float32, qLoRA)
		linalg.MatmulBTW8A8(xn, L.qaBQ, L.qaS, qa, 1, hidden, qLoRA)
		qan := refRMSNorm(qa, L.qaNorm, qLoRA, eps, false)
		q := make([]float32, nH*qkHead)
		linalg.MatmulBTW8A8(qan, L.qbBQ, L.qbS, q, 1, qLoRA, nH*qkHead)
		// kv-down → latent ‖ rope-key; store normalized+roped at pos.
		kvDown := make([]float32, latDim)
		linalg.MatmulBTW8A8(xn, L.kvaBQ, L.kvaS, kvDown, 1, hidden, latDim)
		curLat := make([]float32, latDim)
		cn := refRMSNorm(kvDown[:rank], L.kvaNorm, rank, eps, false)
		copy(curLat[:rank], cn)
		copy(curLat[rank:], mlaRopeRef(kvDown[rank:], qkRope, pos, invFreq, interleave))
		// full latent set = priors ‖ current.
		lat := append(append([]float32(nil), L.priorLat...), curLat...)
		nKeys := pos + 1
		// absorb W_UK + qRope → qAbs per head.
		qNopeAbs := make([]float32, nH*rank)
		for h := range nH {
			for c := range rank {
				var s float32
				for d := range qkNope {
					s += q[h*qkHead+d] * L.wuk[(h*rank+c)*qkNope+d]
				}
				qNopeAbs[h*rank+c] = s
			}
		}
		qRope := make([]float32, nH*qkRope)
		for h := range nH {
			copy(qRope[h*qkRope:], mlaRopeRef(q[h*qkHead+qkNope:h*qkHead+qkHead], qkRope, pos, invFreq, interleave))
		}
		// rank-space attention.
		wsum := make([]float32, nH*rank)
		for h := range nH {
			sc := make([]float64, nKeys)
			mx := math.Inf(-1)
			for j := range nKeys {
				var dot float64
				for c := range rank {
					dot += float64(qNopeAbs[h*rank+c]) * float64(lat[j*latDim+c])
				}
				for d := range qkRope {
					dot += float64(qRope[h*qkRope+d]) * float64(lat[j*latDim+rank+d])
				}
				sc[j] = dot * float64(scale)
				if sc[j] > mx {
					mx = sc[j]
				}
			}
			var sum float64
			for j := range sc {
				sc[j] = math.Exp(sc[j] - mx)
				sum += sc[j]
			}
			for j := range nKeys {
				w := sc[j] / sum
				for c := range rank {
					wsum[h*rank+c] += float32(w * float64(lat[j*latDim+c]))
				}
			}
		}
		// lift W_UV → ctx.
		cv := make([]float32, nH*vHead)
		for h := range nH {
			for e := range vHead {
				var s float32
				for c := range rank {
					s += L.wuv[(h*vHead+e)*rank+c] * wsum[h*rank+c]
				}
				cv[h*vHead+e] = s
			}
		}
		ao := make([]float32, hidden)
		linalg.MatmulBTW8A8(cv, L.oBQ, L.oS, ao, 1, nH*vHead, hidden)
		for i := range x {
			x[i] += ao[i]
		}
		// DeepSeekMoE FFN (group-limited routing + ungated shared expert).
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
		for i := range x {
			x[i] += sdown[i]
		}
	}
	xnf := refRMSNorm(x, fnorm, hidden, eps, false)
	refLogits := make([]float32, vocab)
	linalg.MatmulBTW8A8(xnf, lmBQ, lmS, refLogits, 1, hidden, vocab)

	// --- GPU resident MLA runner ---
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
			nE: nE, k: topK, inter: inter, sigmoid: true, norm: true, scale: 0,
			sharedInter: sInter, sharedUngated: true, nGroup: nGroup, topkGroup: topkGroup,
		},
		mla: &mlaRunParams{
			qLoRARank: qLoRA, kvLoRARank: rank, qkNope: qkNope, qkRope: qkRope,
			vHead: vHead, interleave: interleave, ropeScale: 1.0,
		},
	}
	for l := range layers {
		L := &layers[l]
		lc, _ := ctx.NewKVCache(L.priorLat, (pos+1)*latDim)
		rl := runLayer{
			attnNorm: up32(L.an).buf, invFreq: invD.buf, mlpNorm: up32(L.mn).buf,
			isMoE:      true,
			router:     mk(L.rBQ, L.rS, nE, hidden),
			routerBias: up32(L.rBias).buf,
			expGate:    stack(L.gBQ, L.gS, inter, hidden),
			expUp:      stack(L.uBQ, L.uS, inter, hidden),
			expDown:    stack(L.dBQ, L.dS, hidden, inter),
			shGate:     mk(L.sgBQ, L.sgS, sInter, hidden),
			shUp:       mk(L.suBQ, L.suS, sInter, hidden),
			shDown:     mk(L.sdBQ, L.sdS, hidden, sInter),
			mlaQA:      mk(L.qaBQ, L.qaS, qLoRA, hidden),
			mlaQB:      mk(L.qbBQ, L.qbS, nH*qkHead, qLoRA),
			mlaKVA:     mk(L.kvaBQ, L.kvaS, latDim, hidden),
			mlaO:       mk(L.oBQ, L.oS, hidden, nH*vHead),
			mlaQANorm:  up32(L.qaNorm).buf,
			mlaKVANorm: up32(L.kvaNorm).buf,
			mlaWUK:     up32(L.wuk).buf,
			mlaWUV:     up32(L.wuv).buf,
			latCache:   lc.buf,
		}
		rm.layers = append(rm.layers, rl)
	}
	runner, err := ctx.newDecodeRunner(rm, hidden, nH, 1, vHead, inter, 0, eps, scale, false)
	if err != nil {
		t.Fatalf("newDecodeRunner(MLA): %v", err)
	}
	defer runner.Release()
	got, err := runner.Run(x0, pos)
	if err != nil {
		t.Fatalf("MLA Run: %v", err)
	}
	cos, maxAbs := cosine(got, refLogits)
	t.Logf("MLA runner parity: cosine=%.6f maxAbs=%.3e", cos, maxAbs)
	if cos < 0.9999 {
		t.Errorf("MLA runner diverges: cosine=%.6f maxAbs=%.3e", cos, maxAbs)
	}
}

// TestMLAHeadMatvec_parity gates Lever C4b's per-head block-diagonal matvec — the W_UK

// TestMLALatentStore_parity gates Lever C4b's latent append: kvA-norm the rank latent +
// decoupled-RoPE the key, mirroring decoder.cache.AppendLatent's normalized/roped form
// (cn ‖ krj). Both V3 GPT-J interleave and plain NeoX are covered against a CPU f64
// reference (rmsNorm + mlaRope) at a nonzero position so the rope-at-pos path is real.
func TestMLALatentStore_parity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	const rank, qkRope, pos = 512, 64, 7
	eps := float32(1e-6)
	half := qkRope / 2
	invFreq := make([]float32, half)
	for d := range invFreq {
		invFreq[d] = float32(1.0 / math.Pow(1e4, float64(2*d)/float64(qkRope)))
	}
	for _, interleave := range []bool{true, false} {
		name := "neox"
		if interleave {
			name = "interleave"
		}
		t.Run(name, func(t *testing.T) {
			kvDown := randMat(rank+qkRope, 11)
			normW := randMat(rank, 12)

			// CPU reference: RMSNorm(kvDown[:rank], normW) ‖ mlaRope(kvDown[rank:], pos).
			ref := make([]float32, rank+qkRope)
			var ss float64
			for i := range rank {
				ss += float64(kvDown[i]) * float64(kvDown[i])
			}
			inv := float32(1.0 / math.Sqrt(ss/float64(rank)+float64(eps)))
			for i := range rank {
				ref[i] = kvDown[i] * inv * normW[i]
			}
			key := append([]float32(nil), kvDown[rank:]...)
			if interleave {
				tmp := make([]float32, qkRope)
				for i := range half {
					tmp[i] = key[2*i]
					tmp[half+i] = key[2*i+1]
				}
				copy(key, tmp)
			}
			for d := range half {
				theta := float64(pos) * float64(invFreq[d])
				c, s := float32(math.Cos(theta)), float32(math.Sin(theta))
				x1, x2 := key[d], key[half+d]
				ref[rank+d] = x1*c - x2*s
				ref[rank+half+d] = x2*c + x1*s
			}

			got, err := ctx.MLALatentStore(kvDown, normW, invFreq, rank, qkRope, pos, eps, 1.0, interleave)
			if err != nil {
				t.Fatalf("MLALatentStore: %v", err)
			}
			cos, maxAbs := cosine(got, ref)
			t.Logf("%s: cosine=%.8f maxAbs=%.3e", name, cos, maxAbs)
			if cos < 0.9999 || maxAbs > 1e-3 {
				t.Errorf("MLALatentStore diverges: cosine=%.8f maxAbs=%.3e", cos, maxAbs)
			}
		})
	}
}

// TestMLAHeadMatvec_parity gates Lever C4b's per-head block-diagonal matvec — the W_UK
// absorb (a=q with a wider stride than K, so the qk_rope tail is skipped) and the W_UV
// lift (aStride==K). A CPU f64 reference over random per-head a/w must match (cosine ~1.0).
func TestMLAHeadMatvec_parity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	cases := []struct {
		name              string
		nH, N, K, aStride int
	}{
		{"absorb_WUK", 8, 512, 128, 192}, // aStride=qkHead=qkNope(128)+qkRope(64); K=qkNope
		{"lift_WUV", 8, 128, 512, 512},   // aStride==K=rank
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a := randMat(tc.nH*tc.aStride, 21)
			w := randMat(tc.nH*tc.N*tc.K, 22)
			ref := make([]float32, tc.nH*tc.N)
			for h := 0; h < tc.nH; h++ {
				for n := 0; n < tc.N; n++ {
					var s float64
					for k := 0; k < tc.K; k++ {
						s += float64(a[h*tc.aStride+k]) * float64(w[(h*tc.N+n)*tc.K+k])
					}
					ref[h*tc.N+n] = float32(s)
				}
			}
			got, err := ctx.MLAHeadMatvec(a, w, tc.nH, tc.N, tc.K, tc.aStride)
			if err != nil {
				t.Fatalf("MLAHeadMatvec: %v", err)
			}
			cos, maxAbs := cosine(got, ref)
			t.Logf("%s: cosine=%.8f maxAbs=%.3e", tc.name, cos, maxAbs)
			if cos < 0.9999 || maxAbs > 1e-3 {
				t.Errorf("MLAHeadMatvec diverges: cosine=%.8f maxAbs=%.3e", cos, maxAbs)
			}
		})
	}
}

// TestMLAAttn_parity gates Lever C4a: the absorb-path MLA rank-space attention kernel.
// It mirrors decoder.mlaAttentionAbsorb steps 4b+5 — score each cached latent by the
// full latDim dot (qNopeAbs·cn + qRope·krj), per-head softmax, then collapse V to the
// rank-space weighted latent sum wsum[h] = Σ_j p_j·cn_j. A CPU f64 reference over the
// same random qAbs/latent must match the GPU online-softmax kernel (cosine ~1.0). Uses
// DeepSeek-V3-ish dims (rank 512 > the 128-lane width) so the strided score/value paths
// are exercised, plus a small-rank case so the rank ≤ WG path is covered too.
func TestMLAAttn_parity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	cases := []struct {
		name                    string
		nH, rank, qkRope, nKeys int
	}{
		{"v3_rank512", 8, 512, 64, 40},
		{"small_rank96", 6, 96, 32, 17},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			latDim := tc.rank + tc.qkRope
			scale := float32(1.0 / math.Sqrt(float64(tc.rank+tc.qkRope)))
			qAbs := randMat(tc.nH*latDim, 1)
			lat := randMat(tc.nKeys*latDim, 2)

			// CPU reference (f64): per head, softmax over the latDim dot, V = cn = lat[:rank].
			ref := make([]float32, tc.nH*tc.rank)
			for h := 0; h < tc.nH; h++ {
				sc := make([]float64, tc.nKeys)
				mx := math.Inf(-1)
				for j := 0; j < tc.nKeys; j++ {
					var dot float64
					for d := range latDim {
						dot += float64(qAbs[h*latDim+d]) * float64(lat[j*latDim+d])
					}
					sc[j] = dot * float64(scale)
					if sc[j] > mx {
						mx = sc[j]
					}
				}
				var sum float64
				for j := range sc {
					sc[j] = math.Exp(sc[j] - mx)
					sum += sc[j]
				}
				for j := 0; j < tc.nKeys; j++ {
					w := sc[j] / sum
					for c := 0; c < tc.rank; c++ {
						ref[h*tc.rank+c] += float32(w * float64(lat[j*latDim+c]))
					}
				}
			}

			got, err := ctx.MLAAttn(qAbs, lat, tc.nH, latDim, tc.rank, tc.nKeys, scale)
			if err != nil {
				t.Fatalf("MLAAttn: %v", err)
			}
			cos, maxAbs := cosine(got, ref)
			t.Logf("%s: cosine=%.8f maxAbs=%.3e", tc.name, cos, maxAbs)
			if cos < 0.9999 || maxAbs > 1e-3 {
				t.Errorf("MLAAttn diverges: cosine=%.8f maxAbs=%.3e", cos, maxAbs)
			}
		})
	}
}
