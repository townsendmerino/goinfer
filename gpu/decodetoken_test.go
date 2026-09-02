//go:build gpu

package gpu

import (
	"math"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestDecodeToken_parity validates the full one-fence decode token (all layers'
// attention + MLP chained on device, then final norm + LM head) against a Go
// reference of the same math. 2 layers exercises the inter-layer xd flow.
func TestDecodeToken_parity(t *testing.T) {
	ctx := newOrSkipHW(t) // bit-exact gate — skip on software (imprecise), run on real HW
	defer ctx.Close()

	const hidden, nH, nKV, hd, inter, pos, vocab, L = 1536, 12, 2, 128, 8960, 20, 4096, 2
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
		an, mn             []float32 // attn-norm, mlp-norm
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
		layers[l].priorK = randMat(pos*kvDim, uint64(400+l))
		layers[l].priorV = randMat(pos*kvDim, uint64(500+l))
	}
	fnorm := randMat(hidden, 600)
	lmBQ, lmS := quantW(vocab, hidden, 999)

	// --- Go reference ---
	silu := func(g float32) float32 { return g / (1 + float32(math.Exp(float64(-g)))) }
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

	// --- GPU DecodeToken ---
	mk := func(bq []int8, s []float32, N, K int) *ResidentW8A8 {
		rm, e := ctx.UploadW8A8(bq, s, N, K)
		if e != nil {
			t.Fatal(e)
		}
		return rm
	}
	up32 := func(v []float32) *DeviceBuffer { d, _ := ctx.UploadF32(v); return d }
	invD := up32(invFreq)
	mw := ModelW{FinalNorm: up32(fnorm), LMHead: mk(lmBQ, lmS, vocab, hidden)}
	defer mw.Release() // resident weights are caller-owned; Context.Close does not free them
	for l := range layers {
		L := &layers[l]
		kc, _ := ctx.NewKVCache(L.priorK, (pos+1)*kvDim)
		vc, _ := ctx.NewKVCache(L.priorV, (pos+1)*kvDim)
		mw.Layers = append(mw.Layers, LayerW{
			Attn: AttnWeights{
				Norm: up32(L.an), QProj: mk(L.qBQ, L.qS, qDim, hidden), KProj: mk(L.kBQ, L.kS, kvDim, hidden),
				VProj: mk(L.vBQ, L.vS, kvDim, hidden), OProj: mk(L.oBQ, L.oS, hidden, qDim),
				InvFreq: invD, KCache: kc, VCache: vc,
			},
			MLPNorm: up32(L.mn), Gate: mk(L.gBQ, L.gS, inter, hidden), Up: mk(L.uBQ, L.uS, inter, hidden), Down: mk(L.dBQ, L.dS, hidden, inter),
		})
	}
	got, err := ctx.DecodeToken(x0, mw, hidden, nH, nKV, hd, inter, pos, 0, eps, scale, false)
	if err != nil {
		t.Fatalf("DecodeToken: %v", err)
	}
	cos, maxAbs := cosine(got, refLogits)
	t.Logf("DecodeToken parity (%d layers): cosine=%.6f maxAbs=%.3e", L, cos, maxAbs)
	if cos < 0.999 {
		t.Errorf("DecodeToken diverges: cosine=%.6f maxAbs=%.3e", cos, maxAbs)
	}

	// The fast single-encoder variant must match the reference too (same math,
	// one command buffer instead of ~500 submits).
	gotF, err := ctx.DecodeTokenFused(x0, mw, hidden, nH, nKV, hd, inter, pos, 0, eps, scale, false)
	if err != nil {
		t.Fatalf("DecodeTokenFused: %v", err)
	}
	cosF, maxAbsF := cosine(gotF, refLogits)
	t.Logf("DecodeTokenFused parity: cosine=%.6f maxAbs=%.3e", cosF, maxAbsF)
	if cosF < 0.999 {
		t.Errorf("DecodeTokenFused diverges: cosine=%.6f maxAbs=%.3e", cosF, maxAbsF)
	}

	// The persistent runner (pre-built buffers/bindgroups) must also match.
	runner, err := ctx.NewDecodeRunner(mw, hidden, nH, nKV, hd, inter, 0, eps, scale, false)
	if err != nil {
		t.Fatalf("NewDecodeRunner: %v", err)
	}
	defer runner.Release()
	// P1 (own-forward residency bridge): both layers share one attention geometry, so the
	// value-keyed geomFor must dedup them to a single attnGeom. A regression that allocated
	// one uniform per layer would leave these logits byte-identical while inflating this to
	// the layer count — the assertion catches what the parity check cannot see.
	if gv := runner.GeomVariantCount(); gv != 1 {
		t.Errorf("GeomVariantCount = %d, want 1 (uniform %d-layer model must dedup to one geometry)", gv, L)
	}
	gotR, err := runner.Run(x0, pos)
	if err != nil {
		t.Fatalf("DecodeRunner.Run: %v", err)
	}
	cosR, maxAbsR := cosine(gotR, refLogits)
	t.Logf("DecodeRunner parity: cosine=%.6f maxAbs=%.3e", cosR, maxAbsR)
	if cosR < 0.999 {
		t.Errorf("DecodeRunner diverges: cosine=%.6f maxAbs=%.3e", cosR, maxAbsR)
	}
}
