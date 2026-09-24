//go:build gpu

package gpu

import (
	"fmt"
	"math"
	"testing"
)

// TestDecodeRunnerW4A8_geometries is TestDecodeRunnerW4A8_parity re-run over attention
// geometries the resident decode path had never been exercised on by any synthetic test
// (every one of them used qwen2.5-1.5B's (nH,nKV,hd) = (12,2,128)), against the same CPU
// oracle. It exists for docs/completed/task-webgpu-nogqa-decode-bug.md: phi3-mini's resident
// decode diverges from CPU, and the doc's open question is whether the kernels are wrong for
// its shape — no GQA (nH == nKV, group 1) AND head_dim 96 (not a power of two) AND a
// [3072,3072] Q/K/V/O with a 32064-row LM head — or whether the divergence is numerical.
// A synthetic model with EXACTLY phi3-mini's dims (2 layers) answers the shape half of that
// question on any adapter, including the CI software one: the oracle runs the same
// int4/int8 math, so a pass here is a kernel/wiring pass for the geometry (cosine ~1.0),
// independent of what a real checkpoint's activations do to int8 quantization.
//
// Position 0 is covered as well as a mid-context position, because the doc reports the
// real-model divergence already at position 0 — where attention has one key and is the
// identity on V, so only the projection/norm/MLP/LM-head chain can differ.
func TestDecodeRunnerW4A8_geometries(t *testing.T) {
	ctx := newOrSkip(t)
	defer ctx.Close()
	info := ctx.adapter.GetInfo()
	t.Logf("adapter: %s %q (%v)", info.Device, info.Description, info.AdapterType)

	type geom struct {
		name                              string
		hidden, nH, nKV, hd, inter, vocab int
		eps                               float32
	}
	geoms := []geom{
		{"qwen2.5-1.5b-control", 1536, 12, 2, 128, 8960, 4096, 1e-6},
		{"phi3-mini (MHA, hd=96)", 3072, 32, 32, 96, 8192, 32064, 1e-5},
		{"mha-hd128 (group=1 only)", 1536, 12, 12, 128, 8960, 4096, 1e-6},
		{"gqa-hd96 (hd=96 only)", 1536, 12, 2, 96, 8960, 4096, 1e-6},
	}
	for _, g := range geoms {
		for _, pos := range []int{0, 20} {
			t.Run(fmt.Sprintf("%s/pos%d", g.name, pos), func(t *testing.T) {
				runGeomParity(t, ctx, g.hidden, g.nH, g.nKV, g.hd, g.inter, g.vocab, pos, g.eps)
			})
		}
	}
}

func runGeomParity(t *testing.T, ctx *Context, hidden, nH, nKV, hd, inter, vocab, pos int, eps float32) {
	const L = 2
	qDim, kvDim := nH*hd, nKV*hd
	half := hd / 2
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	x0 := randMat(hidden, 100)
	invFreq := make([]float32, half)
	for d := range invFreq {
		invFreq[d] = float32(1.0 / math.Pow(1e4, float64(2*d)/float64(hd)))
	}

	type lw struct {
		an, mn                     []float32
		qN, kN, vN, oN, gN, uN, dN []uint8
		qS, kS, vS, oS, gS, uS, dS []float32
		priorK, priorV             []float32
	}
	layers := make([]lw, L)
	seed := uint64(1)
	for l := range layers {
		mk := func(N, K int) ([]uint8, []float32) { seed++; return quantInt4Group(randMat(N*K, seed), N, K, w4Group) }
		layers[l].an = randMat(hidden, uint64(200+l))
		layers[l].mn = randMat(hidden, uint64(300+l))
		layers[l].qN, layers[l].qS = mk(qDim, hidden)
		layers[l].kN, layers[l].kS = mk(kvDim, hidden)
		layers[l].vN, layers[l].vS = mk(kvDim, hidden)
		layers[l].oN, layers[l].oS = mk(hidden, qDim)
		layers[l].gN, layers[l].gS = mk(inter, hidden)
		layers[l].uN, layers[l].uS = mk(inter, hidden)
		layers[l].dN, layers[l].dS = mk(hidden, inter)
		layers[l].priorK = randMat(pos*kvDim, uint64(400+l))
		layers[l].priorV = randMat(pos*kvDim, uint64(500+l))
	}
	fnorm := randMat(hidden, 600)
	lmN, lmS := quantInt4Group(randMat(vocab*hidden, 999), vocab, hidden, w4Group)

	// --- CPU oracle (same math as the GPU W4A8 forward) ---
	silu := func(g float32) float32 { return g / (1 + float32(math.Exp(float64(-g)))) }
	x := append([]float32(nil), x0...)
	for l := range layers {
		L := &layers[l]
		xn := refRMSNorm(x, L.an, hidden, eps, false)
		aq, as := refQuantInt8(xn)
		q := refMatmulW4A8(aq, as, L.qN, L.qS, qDim, hidden, w4Group)
		k := refMatmulW4A8(aq, as, L.kN, L.kS, kvDim, hidden, w4Group)
		v := refMatmulW4A8(aq, as, L.vN, L.vS, kvDim, hidden, w4Group)
		refRoPE(q, nH, hd, pos, invFreq)
		refRoPE(k, nKV, hd, pos, invFreq)
		kFull := append(append([]float32(nil), L.priorK...), k...)
		vFull := append(append([]float32(nil), L.priorV...), v...)
		cv := refAttn(q, kFull, vFull, nH, nKV, hd, pos+1, scale)
		cq, cs := refQuantInt8(cv)
		ao := refMatmulW4A8(cq, cs, L.oN, L.oS, hidden, qDim, w4Group)
		for i := range x {
			x[i] += ao[i]
		}
		xn2 := refRMSNorm(x, L.mn, hidden, eps, false)
		mq, ms := refQuantInt8(xn2)
		gate := refMatmulW4A8(mq, ms, L.gN, L.gS, inter, hidden, w4Group)
		up := refMatmulW4A8(mq, ms, L.uN, L.uS, inter, hidden, w4Group)
		mid := make([]float32, inter)
		for i := range mid {
			mid[i] = silu(gate[i]) * up[i]
		}
		dq, ds := refQuantInt8(mid)
		down := refMatmulW4A8(dq, ds, L.dN, L.dS, hidden, inter, w4Group)
		for i := range x {
			x[i] += down[i]
		}
	}
	xnf := refRMSNorm(x, fnorm, hidden, eps, false)
	fq, fs := refQuantInt8(xnf)
	refLogits := refMatmulW4A8(fq, fs, lmN, lmS, vocab, hidden, w4Group)

	// --- GPU W4A8 DecodeRunner ---
	var keep []func()
	defer func() {
		for i := len(keep) - 1; i >= 0; i-- {
			keep[i]()
		}
	}()
	up32 := func(v []float32) *DeviceBuffer {
		d, e := ctx.UploadF32(v)
		if e != nil {
			t.Fatal(e)
		}
		keep = append(keep, d.Release)
		return d
	}
	w4 := func(nib []uint8, sc []float32, N, K int) *ResidentW4A8 {
		rm, e := ctx.UploadW4A8(nib, sc, N, K)
		if e != nil {
			t.Fatal(e)
		}
		keep = append(keep, rm.Release)
		return rm
	}
	invD := up32(invFreq)
	rm := runModel{finalNorm: up32(fnorm).buf, lmHead: w4(lmN, lmS, vocab, hidden)}
	for l := range layers {
		L := &layers[l]
		kc, e1 := ctx.NewKVCache(L.priorK, (pos+1)*kvDim)
		vc, e2 := ctx.NewKVCache(L.priorV, (pos+1)*kvDim)
		if e1 != nil || e2 != nil {
			t.Fatalf("kv alloc: %v %v", e1, e2)
		}
		keep = append(keep, kc.Release, vc.Release)
		rm.layers = append(rm.layers, runLayer{
			attnNorm: up32(L.an).buf, invFreq: invD.buf, kCache: kc.buf, vCache: vc.buf, mlpNorm: up32(L.mn).buf,
			q: w4(L.qN, L.qS, qDim, hidden), k: w4(L.kN, L.kS, kvDim, hidden), v: w4(L.vN, L.vS, kvDim, hidden),
			o:    w4(L.oN, L.oS, hidden, qDim),
			gate: w4(L.gN, L.gS, inter, hidden), up: w4(L.uN, L.uS, inter, hidden), down: w4(L.dN, L.dS, hidden, inter),
		})
	}
	runner, err := ctx.newDecodeRunner(rm, hidden, nH, nKV, hd, inter, 0, eps, scale, false)
	if err != nil {
		t.Fatalf("newDecodeRunner(W4A8): %v", err)
	}
	defer runner.Release()
	got, err := runner.Run(x0, pos, pos)
	if err != nil {
		t.Fatalf("W4A8 Run: %v", err)
	}
	cos, maxAbs := cosine(got, refLogits)
	am := func(v []float32) int {
		best := 0
		for i, x := range v {
			if x > v[best] {
				best = i
			}
		}
		return best
	}
	t.Logf("(hidden=%d nH=%d nKV=%d hd=%d inter=%d vocab=%d pos=%d): cosine=%.6f maxAbs=%.3e argmax gpu=%d ref=%d",
		hidden, nH, nKV, hd, inter, vocab, pos, cos, maxAbs, am(got), am(refLogits))
	if cos < 0.9999 {
		t.Errorf("W4A8 DecodeRunner diverges for this geometry: cosine=%.6f maxAbs=%.3e", cos, maxAbs)
	}
}
