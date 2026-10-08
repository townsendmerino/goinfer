//go:build gpu

package gpu

import (
	"math"
	"testing"
)

// synthPrefill builds an L-layer synthetic W8A8 model (Qwen-shaped, with Qwen2's q/k/v biases when withBias) with
// prior K/V at positions [0, start) and returns a function that runs one PrefillLastW8A8 of M rows on fresh caches and
// returns the last row's logits and every layer's K and V cache, read back whole. Everything it uploads, on every run, is added to own for the caller to close.
func synthPrefill(t *testing.T, ctx *Context, own *closers, L int, withBias bool) (run func() (logits []float32, caches [][]float32), start, M, kvDim int) {
	const hidden, nH, nKV, hd, inter, vocab = 256, 8, 2, 32, 688, 512
	start, M = 5, 24
	qDim := nH * hd
	kvDim = nKV * hd
	half := hd / 2
	eps := float32(1e-6)
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	capElems := (start + M) * kvDim
	invFreq := make([]float32, half)
	for d := range invFreq {
		invFreq[d] = float32(1.0 / math.Pow(1e6, float64(2*d)/float64(hd)))
	}
	up32 := func(v []float32) *DeviceBuffer { d, _ := ctx.UploadF32(v); return add(own, d) }
	seed := uint64(7)
	W := func(N, K int) *ResidentW8A8 {
		seed++
		bq, s := quantW(N, K, seed)
		rm, e := ctx.UploadW8A8(bq, s, N, K)
		if e != nil {
			t.Fatal(e)
		}
		return add(own, rm)
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
	add(own, lmHead)
	invD := up32(invFreq)
	xs := make([][]float32, M)
	for r := range xs {
		xs[r] = randMat(hidden, uint64(1000+r))
	}
	positions := make([]int, M)
	for r := range positions {
		positions[r] = start + r
	}
	run = func() ([]float32, [][]float32) {
		mw := ModelW{FinalNorm: fnorm, LMHead: lmHead}
		for l := range layers {
			kc, _ := ctx.NewKVCache(layers[l].priorK, capElems)
			add(own, kc)
			vc, _ := ctx.NewKVCache(layers[l].priorV, capElems)
			add(own, vc)
			a := AttnWeights{Norm: up32(layers[l].an), QProj: layers[l].q, KProj: layers[l].k, VProj: layers[l].v,
				OProj: layers[l].o, InvFreq: invD, KCache: kc, VCache: vc}
			if withBias {
				a.QBias, a.KBias, a.VBias = up32(layers[l].qb), up32(layers[l].kb), up32(layers[l].vb)
			}
			mw.Layers = append(mw.Layers, LayerW{Attn: a, MLPNorm: up32(layers[l].mn), Gate: layers[l].g, Up: layers[l].u, Down: layers[l].d})
		}
		out, err := ctx.PrefillLastW8A8(xs, mw, hidden, nH, nKV, hd, inter, positions, 0, eps, scale, false)
		if err != nil {
			t.Fatalf("bias %v: %v", withBias, err)
		}
		var caches [][]float32
		for l := range mw.Layers {
			for _, c := range []*DeviceBuffer{mw.Layers[l].Attn.KCache, mw.Layers[l].Attn.VCache} {
				v, err := ctx.Readback(c)
				if err != nil {
					t.Fatal(err)
				}
				caches = append(caches, v)
			}
		}
		return append([]float32(nil), out...), caches
	}
	return run, start, M, kvDim
}

// samePrefill fails unless two synthPrefill results agree bit for bit: the logits and the written K/V range.
func samePrefill(t *testing.T, what string, refL, gotL []float32, refC, gotC [][]float32, n int) {
	t.Helper()
	for j := range refL {
		if math.Float32bits(refL[j]) != math.Float32bits(gotL[j]) {
			t.Fatalf("%s: logit %d is %v, want %v", what, j, gotL[j], refL[j])
		}
	}
	for i := range refC {
		for j := range refC[i][:n] {
			if math.Float32bits(refC[i][j]) != math.Float32bits(gotC[i][j]) {
				t.Fatalf("%s: cache %d element %d differs", what, i, j)
			}
		}
	}
}

// TestPrefillLastW8A8_sharedQuantBitIdentical is R-25's gate (docs/tasks/task-recompute-audit.md): the batched prefill
// quantizes xn once for q, k and v and xn2 once for gate and up (sharedQ) instead of once per projection. Against the
// same prefill with a quantization per projection (prefillQuantPerProj), the last row's logits and every element of
// every layer's K and V cache must be equal bit for bit, with and without Qwen2's q/k/v biases (the bias-epilogue GEMM),
// and the shared path must have run: 2 shared quantizations per layer.
func TestPrefillLastW8A8_sharedQuantBitIdentical(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()
	var own closers
	defer own.closeAll()
	defer func() { prefillQuantPerProj = false }()
	const L = 2
	for _, withBias := range []bool{false, true} {
		run, start, M, kvDim := synthPrefill(t, ctx, &own, L, withBias)
		arm := func(perProj bool) ([]float32, [][]float32, int64) {
			prefillQuantPerProj = perProj
			s0 := prefillSharedQuants.Load()
			l, c := run()
			return l, c, prefillSharedQuants.Load() - s0
		}
		refL, refC, refS := arm(true)
		gotL, gotC, gotS := arm(false)
		if refS != 0 || gotS != 2*L {
			t.Fatalf("bias %v: shared quantizations %d per-projection, %d shared; want 0 and %d", withBias, refS, gotS, 2*L)
		}
		samePrefill(t, "shared vs per projection", refL, gotL, refC, gotC, (start+M)*kvDim)
		t.Logf("bias %v: %d rows, logits and K/V bit-identical; %d quantizations shared", withBias, M, gotS)
	}
}

// TestPrefillLastW8A8_bufferRecycleBitIdentical is the gate for R-25's second half: each layer of the batched prefill
// reuses the buffers of the layer before it (storF's free list) instead of creating fresh, zero-filled ones. Over 3
// layers, against fresh buffers for every call (prefillBufFresh), the last row's logits and all K/V must be equal bit
// for bit, and again with every recycled buffer filled with 0xFF bytes (NaN as f32, -1 as packed int8) before its
// reuse (prefillBufPoison), which proves no kernel reads a byte of a reused buffer it did not write; with and without
// q/k/v biases. The recycling must have run (prefillBufReused), and not in the fresh arm.
func TestPrefillLastW8A8_bufferRecycleBitIdentical(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()
	var own closers
	defer own.closeAll()
	defer func() { prefillBufFresh, prefillBufPoison = false, false }()
	for _, withBias := range []bool{false, true} {
		run, start, M, kvDim := synthPrefill(t, ctx, &own, 3, withBias)
		arm := func(fresh, poison bool) ([]float32, [][]float32, int64) {
			prefillBufFresh, prefillBufPoison = fresh, poison
			r0 := prefillBufReused.Load()
			l, c := run()
			return l, c, prefillBufReused.Load() - r0
		}
		refL, refC, refR := arm(true, false)
		gotL, gotC, gotR := arm(false, false)
		poiL, poiC, poiR := arm(false, true)
		if refR != 0 || gotR == 0 || poiR != gotR {
			t.Fatalf("bias %v: recycled buffers %d fresh, %d recycled, %d poisoned; want 0, >0 and equal", withBias, refR, gotR, poiR)
		}
		n := (start + M) * kvDim
		samePrefill(t, "recycled vs fresh", refL, gotL, refC, gotC, n)
		samePrefill(t, "poisoned recycled vs fresh", refL, poiL, refC, poiC, n)
		t.Logf("bias %v: %d buffers recycled over 3 layers; logits and K/V bit-identical fresh, recycled and poisoned", withBias, gotR)
	}
}
