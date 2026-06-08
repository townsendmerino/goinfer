//go:build gpu

package gpu

import (
	"math"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

func refRMSNorm(x, w []float32, H int, eps float32, addOne bool) []float32 {
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	inv := float32(1.0 / math.Sqrt(ss/float64(H)+float64(eps)))
	out := make([]float32, H)
	for i := range out {
		wi := w[i]
		if addOne {
			wi += 1
		}
		out[i] = x[i] * inv * wi
	}
	return out
}

func refRoPE(vec []float32, heads, hd, pos int, invFreq []float32) {
	half := hd / 2
	for d := 0; d < half; d++ {
		theta := float64(pos) * float64(invFreq[d])
		c, s := math.Cos(theta), math.Sin(theta)
		for h := 0; h < heads; h++ {
			off := h * hd
			x1, x2 := float64(vec[off+d]), float64(vec[off+half+d])
			vec[off+d] = float32(x1*c - x2*s)
			vec[off+half+d] = float32(x2*c + x1*s)
		}
	}
}

func refAttn(q, keys, vals []float32, nH, nKV, hd, nKeys int, scale float32) []float32 {
	kvDim := nKV * hd
	group := nH / nKV
	out := make([]float32, nH*hd)
	for qh := 0; qh < nH; qh++ {
		kvh := qh / group
		maxS := math.Inf(-1)
		sc := make([]float64, nKeys)
		for s := 0; s < nKeys; s++ {
			var dot float64
			for d := 0; d < hd; d++ {
				dot += float64(q[qh*hd+d]) * float64(keys[s*kvDim+kvh*hd+d])
			}
			sc[s] = dot * float64(scale)
			if sc[s] > maxS {
				maxS = sc[s]
			}
		}
		var sum float64
		for s := 0; s < nKeys; s++ {
			sc[s] = math.Exp(sc[s] - maxS)
			sum += sc[s]
		}
		for s := 0; s < nKeys; s++ {
			w := sc[s] / sum
			for d := 0; d < hd; d++ {
				out[qh*hd+d] += float32(w * float64(vals[s*kvDim+kvh*hd+d]))
			}
		}
	}
	return out
}

// TestAttnBlock_parity validates the whole on-device attention sub-block
// (rmsnorm → qkv → RoPE → KV-append → attention → o_proj → residual, one fence)
// against a Go reference of the same math.
func TestAttnBlock_parity(t *testing.T) {
	ctx, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	defer ctx.Close()

	const hidden, nH, nKV, hd, pos = 1536, 12, 2, 128, 20
	qDim, kvDim := nH*hd, nKV*hd
	half := hd / 2
	eps := float32(1e-6)
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	x := randMat(hidden, 5)
	normW := randMat(hidden, 6)
	invFreq := make([]float32, half)
	for d := range invFreq {
		invFreq[d] = float32(1.0 / math.Pow(1e6, float64(2*d)/float64(hd)))
	}
	qBQ, qS := quantW(qDim, hidden, 1)
	kBQ, kS := quantW(kvDim, hidden, 2)
	vBQ, vS := quantW(kvDim, hidden, 3)
	oBQ, oS := quantW(hidden, qDim, 4)
	priorK := randMat(pos*kvDim, 7)
	priorV := randMat(pos*kvDim, 8)

	// --- Go reference ---
	xn := refRMSNorm(x, normW, hidden, eps, false)
	q := make([]float32, qDim)
	linalg.MatmulBTW8A8(xn, qBQ, qS, q, 1, hidden, qDim)
	k := make([]float32, kvDim)
	linalg.MatmulBTW8A8(xn, kBQ, kS, k, 1, hidden, kvDim)
	v := make([]float32, kvDim)
	linalg.MatmulBTW8A8(xn, vBQ, vS, v, 1, hidden, kvDim)
	refRoPE(q, nH, hd, pos, invFreq)
	refRoPE(k, nKV, hd, pos, invFreq)
	kFull := append(append([]float32(nil), priorK...), k...)
	vFull := append(append([]float32(nil), priorV...), v...)
	ctxv := refAttn(q, kFull, vFull, nH, nKV, hd, pos+1, scale)
	attnOut := make([]float32, hidden)
	linalg.MatmulBTW8A8(ctxv, oBQ, oS, attnOut, 1, qDim, hidden)
	ref := make([]float32, hidden)
	for i := range ref {
		ref[i] = x[i] + attnOut[i]
	}

	// --- GPU on-device sub-block ---
	mk := func(bq []int8, s []float32, N, K int) *ResidentW8A8 {
		rm, e := ctx.UploadW8A8(bq, s, N, K)
		if e != nil {
			t.Fatal(e)
		}
		return rm
	}
	normD, _ := ctx.UploadF32(normW)
	invD, _ := ctx.UploadF32(invFreq)
	kc, _ := ctx.NewKVCache(priorK, (pos+1)*kvDim)
	vc, _ := ctx.NewKVCache(priorV, (pos+1)*kvDim)
	w := AttnWeights{
		Norm: normD, QProj: mk(qBQ, qS, qDim, hidden), KProj: mk(kBQ, kS, kvDim, hidden),
		VProj: mk(vBQ, vS, kvDim, hidden), OProj: mk(oBQ, oS, hidden, qDim),
		InvFreq: invD, KCache: kc, VCache: vc,
	}
	got, err := ctx.AttnBlock(x, w, hidden, nH, nKV, hd, pos, 0, eps, scale, false)
	if err != nil {
		t.Fatalf("AttnBlock: %v", err)
	}
	cos, maxAbs := cosine(got, ref)
	t.Logf("AttnBlock parity: cosine=%.6f maxAbs=%.3e", cos, maxAbs)
	if cos < 0.999 {
		t.Errorf("AttnBlock diverges: cosine=%.6f maxAbs=%.3e", cos, maxAbs)
	}
}
