//go:build gpu

package gpu

import (
	"math"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestDecodeRunnerPerLayerRoPE_parity gates Lever C7: the resident runner binding a
// DIFFERENT RoPE table + cos/sin scale (mscale) per layer — Mellum's YaRN-on-global vs
// default-local interleave. Two dense layers use distinct (invFreq, ropeScale) pairs; a
// CPU int8 oracle ropes each layer with its own table+scale. A bug that shared one rope
// across layers (the pre-C7 behavior) would diverge here. Cosine must be ~1.0.
func TestDecodeRunnerPerLayerRoPE_parity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	const hidden, nH, nKV, hd, inter, pos, vocab, L = 256, 4, 2, 64, 128, 5, 512, 2
	qDim, kvDim := nH*hd, nKV*hd
	half := hd / 2
	eps := float32(1e-6)
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	silu := func(g float32) float32 { return g / (1 + float32(math.Exp(float64(-g)))) }

	// Two distinct rope tables + scales (mirrors global=YaRN vs local=default).
	invFreq := [L][]float32{make([]float32, half), make([]float32, half)}
	bases := [L]float64{1e6, 1e4}
	ropeScale := [L]float32{1.3, 1.0} // per-layer mscale
	for l := 0; l < L; l++ {
		for d := range invFreq[l] {
			invFreq[l][d] = float32(1.0 / math.Pow(bases[l], float64(2*d)/float64(hd)))
		}
	}
	// scaledRoPE applies decoder.applyRoPE with a cos/sin scale (the per-layer mscale).
	scaledRoPE := func(vec []float32, heads int, inv []float32, rs float32) {
		for d := 0; d < half; d++ {
			theta := float64(pos) * float64(inv[d])
			c := float32(math.Cos(theta)) * rs
			s := float32(math.Sin(theta)) * rs
			for h := 0; h < heads; h++ {
				off := h * hd
				x1, x2 := vec[off+d], vec[off+half+d]
				vec[off+d] = x1*c - x2*s
				vec[off+half+d] = x2*c + x1*s
			}
		}
	}

	type lw struct {
		an, mn             []float32
		qBQ, kBQ, vBQ, oBQ []int8
		qS, kS, vS, oS     []float32
		gBQ, uBQ, dBQ      []int8
		gS, uS, dS         []float32
		priorK, priorV     []float32
	}
	x0 := randMat(hidden, 100)
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
		layers[l].priorK = randMat(pos*kvDim, uint64(400+l))
		layers[l].priorV = randMat(pos*kvDim, uint64(500+l))
	}
	fnorm := randMat(hidden, 600)
	lmBQ, lmS := quantW(vocab, hidden, 999)

	// --- CPU oracle (per-layer rope) ---
	x := append([]float32(nil), x0...)
	for l := range layers {
		Lw := &layers[l]
		xn := refRMSNorm(x, Lw.an, hidden, eps, false)
		q := make([]float32, qDim)
		linalg.MatmulBTW8A8(xn, Lw.qBQ, Lw.qS, q, 1, hidden, qDim)
		k := make([]float32, kvDim)
		linalg.MatmulBTW8A8(xn, Lw.kBQ, Lw.kS, k, 1, hidden, kvDim)
		v := make([]float32, kvDim)
		linalg.MatmulBTW8A8(xn, Lw.vBQ, Lw.vS, v, 1, hidden, kvDim)
		scaledRoPE(q, nH, invFreq[l], ropeScale[l])
		scaledRoPE(k, nKV, invFreq[l], ropeScale[l])
		kFull := append(append([]float32(nil), Lw.priorK...), k...)
		vFull := append(append([]float32(nil), Lw.priorV...), v...)
		cv := refAttn(q, kFull, vFull, nH, nKV, hd, pos+1, scale)
		ao := make([]float32, hidden)
		linalg.MatmulBTW8A8(cv, Lw.oBQ, Lw.oS, ao, 1, qDim, hidden)
		for i := range x {
			x[i] += ao[i]
		}
		xn2 := refRMSNorm(x, Lw.mn, hidden, eps, false)
		gate := make([]float32, inter)
		linalg.MatmulBTW8A8(xn2, Lw.gBQ, Lw.gS, gate, 1, hidden, inter)
		up := make([]float32, inter)
		linalg.MatmulBTW8A8(xn2, Lw.uBQ, Lw.uS, up, 1, hidden, inter)
		mid := make([]float32, inter)
		for i := range mid {
			mid[i] = silu(gate[i]) * up[i]
		}
		down := make([]float32, hidden)
		linalg.MatmulBTW8A8(mid, Lw.dBQ, Lw.dS, down, 1, inter, hidden)
		for i := range x {
			x[i] += down[i]
		}
	}
	xnf := refRMSNorm(x, fnorm, hidden, eps, false)
	refLogits := make([]float32, vocab)
	linalg.MatmulBTW8A8(xnf, lmBQ, lmS, refLogits, 1, hidden, vocab)

	// --- GPU resident runner (per-layer rope) ---
	mk := func(bq []int8, s []float32, N, K int) *ResidentW8A8 {
		rm, e := ctx.UploadW8A8(bq, s, N, K)
		if e != nil {
			t.Fatal(e)
		}
		return rm
	}
	up32 := func(v []float32) *DeviceBuffer { d, _ := ctx.UploadF32(v); return d }
	invD := [L]*DeviceBuffer{up32(invFreq[0]), up32(invFreq[1])}
	rm := runModel{finalNorm: up32(fnorm).buf, lmHead: mk(lmBQ, lmS, vocab, hidden)}
	for l := range layers {
		Lw := &layers[l]
		kc, _ := ctx.NewKVCache(Lw.priorK, (pos+1)*kvDim)
		vc, _ := ctx.NewKVCache(Lw.priorV, (pos+1)*kvDim)
		rm.layers = append(rm.layers, runLayer{
			attnNorm: up32(Lw.an).buf, invFreq: invD[l].buf, kCache: kc.buf, vCache: vc.buf, mlpNorm: up32(Lw.mn).buf,
			q: mk(Lw.qBQ, Lw.qS, qDim, hidden), k: mk(Lw.kBQ, Lw.kS, kvDim, hidden), v: mk(Lw.vBQ, Lw.vS, kvDim, hidden),
			o:         mk(Lw.oBQ, Lw.oS, hidden, qDim),
			gate:      mk(Lw.gBQ, Lw.gS, inter, hidden),
			up:        mk(Lw.uBQ, Lw.uS, inter, hidden),
			down:      mk(Lw.dBQ, Lw.dS, hidden, inter),
			ropeScale: ropeScale[l],
		})
	}
	runner, err := ctx.newDecodeRunner(rm, hidden, nH, nKV, hd, inter, 0, eps, scale, false)
	if err != nil {
		t.Fatalf("newDecodeRunner(per-layer rope): %v", err)
	}
	defer runner.Release()
	got, err := runner.Run(x0, pos)
	if err != nil {
		t.Fatalf("per-layer rope Run: %v", err)
	}
	cos, maxAbs := cosine(got, refLogits)
	t.Logf("per-layer rope: cosine=%.6f maxAbs=%.3e", cos, maxAbs)
	if cos < 0.9999 {
		t.Errorf("per-layer rope runner diverges: cosine=%.6f maxAbs=%.3e", cos, maxAbs)
	}
}
