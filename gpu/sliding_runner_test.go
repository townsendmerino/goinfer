//go:build gpu

package gpu

import (
	"math"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestDecodeRunnerSlidingWindow_parity gates Lever C6: the resident dense forward with
// sliding-window (local) attention. All layers are local with a window W small enough to
// BITE at the test position (start = max(0, pos+1-W) > 0), so the runner must attend only
// the last W cached keys — not the full history. A CPU int8 oracle that windows the same
// way must match (cosine ~1.0). Mirrors Mistral's all-local attention; full-attention
// layers (isLocal=false) keep the unwindowed start and are covered by the other runner
// gates. The window biting is the point: at W≥nKeys it would be a no-op.
func TestDecodeRunnerSlidingWindow_parity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	const hidden, nH, nKV, hd, inter, pos, vocab, L, window = 256, 4, 2, 64, 128, 6, 512, 2, 3
	qDim, kvDim := nH*hd, nKV*hd
	half := hd / 2
	eps := float32(1e-6)
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	silu := func(g float32) float32 { return g / (1 + float32(math.Exp(float64(-g)))) }

	// window bites: nKeys = pos+1 = 7, start = max(0, 7-3) = 4 → attend keys [4,7).
	nKeys := pos + 1
	start := nKeys - window
	if start <= 0 {
		t.Fatalf("window must bite for this test (start=%d)", start)
	}

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

	// --- CPU oracle (windowed attention) ---
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
		// Window: attend only keys [start, nKeys) — slice off the masked prefix.
		cv := refAttn(q, kFull[start*kvDim:], vFull[start*kvDim:], nH, nKV, hd, nKeys-start, scale)
		ao := make([]float32, hidden)
		linalg.MatmulBTW8A8(cv, L.oBQ, L.oS, ao, 1, qDim, hidden)
		for i := range x {
			x[i] += ao[i]
		}
		xn2 := refRMSNorm(x, L.mn, hidden, eps, false)
		gate := make([]float32, inter)
		linalg.MatmulBTW8A8(xn2, L.gBQ, L.gS, gate, 1, hidden, inter)
		up := make([]float32, inter)
		linalg.MatmulBTW8A8(xn2, L.uBQ, L.uS, up, 1, hidden, inter)
		mid := make([]float32, inter)
		for i := range mid {
			mid[i] = silu(gate[i]) * up[i]
		}
		down := make([]float32, hidden)
		linalg.MatmulBTW8A8(mid, L.dBQ, L.dS, down, 1, inter, hidden)
		for i := range x {
			x[i] += down[i]
		}
	}
	xnf := refRMSNorm(x, fnorm, hidden, eps, false)
	refLogits := make([]float32, vocab)
	linalg.MatmulBTW8A8(xnf, lmBQ, lmS, refLogits, 1, hidden, vocab)

	// --- GPU resident runner (sliding window) ---
	mk := func(bq []int8, s []float32, N, K int) *ResidentW8A8 {
		rm, e := ctx.UploadW8A8(bq, s, N, K)
		if e != nil {
			t.Fatal(e)
		}
		return rm
	}
	up32 := func(v []float32) *DeviceBuffer { d, _ := ctx.UploadF32(v); return d }
	invD := up32(invFreq)
	rm := runModel{
		finalNorm:     up32(fnorm).buf,
		lmHead:        mk(lmBQ, lmS, vocab, hidden),
		slidingWindow: window,
	}
	for l := range layers {
		L := &layers[l]
		kc, _ := ctx.NewKVCache(L.priorK, (pos+1)*kvDim)
		vc, _ := ctx.NewKVCache(L.priorV, (pos+1)*kvDim)
		rm.layers = append(rm.layers, runLayer{
			attnNorm: up32(L.an).buf, invFreq: invD.buf, kCache: kc.buf, vCache: vc.buf, mlpNorm: up32(L.mn).buf,
			q: mk(L.qBQ, L.qS, qDim, hidden), k: mk(L.kBQ, L.kS, kvDim, hidden), v: mk(L.vBQ, L.vS, kvDim, hidden),
			o:       mk(L.oBQ, L.oS, hidden, qDim),
			gate:    mk(L.gBQ, L.gS, inter, hidden),
			up:      mk(L.uBQ, L.uS, inter, hidden),
			down:    mk(L.dBQ, L.dS, hidden, inter),
			isLocal: true,
		})
	}
	runner, err := ctx.newDecodeRunner(rm, hidden, nH, nKV, hd, inter, 0, eps, scale, false)
	if err != nil {
		t.Fatalf("newDecodeRunner(sliding): %v", err)
	}
	defer runner.Release()
	got, err := runner.Run(x0, pos)
	if err != nil {
		t.Fatalf("sliding Run: %v", err)
	}
	cos, maxAbs := cosine(got, refLogits)
	t.Logf("sliding-window (W=%d, start=%d): cosine=%.6f maxAbs=%.3e", window, start, cos, maxAbs)
	if cos < 0.9999 {
		t.Errorf("sliding-window runner diverges: cosine=%.6f maxAbs=%.3e", cos, maxAbs)
	}
}
