//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"runtime"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
	gpu "github.com/townsendmerino/goinfer/metal/internal/gpudev"
	"github.com/townsendmerino/goinfer/metal/kernels"
)

// Test hooks: entry points into the resident that only goinfer_testhooks tests call (bisection, sub-layer capture, the
// confirmers, the q4k lane's segments on the unlined plan line). Production code does not reach them.

// q4kLaneProj is q4kLaneProjOn on gpu.UnlinedLine: tests call it; production code names its plan line.
func q4kLaneProj(d *Device, kq *decoder.KQuantReader, wms ...*linalg.WeightMat) []q4kSeg {
	return q4kLaneProjOn(gpu.UnlinedLine, d, kq, wms...)
}

// forwardTrunkForTest runs the trunk over the first nLayers layers ONLY and returns the residual
// stream (r.x) at that depth — the GPU half of a per-layer bisect against decoder.ForwardCapture.
//
// It is the seam for locating WHERE a parity gap enters rather than only that it exists: a
// whole-model cosine says a backend is wrong, never which op. Truncation is exact because
// encodeTrunkInto reads r.nL at ENCODE time and re-encodes every call, so lowering it drops the
// tail layers with no other effect; the final norm still runs but writes r.aq, never r.x.
//
// Side effect worth knowing: the layers that DO run write their K/V for pos as usual, so calling
// this repeatedly at one position is idempotent but calling it out of order is not.
func (r *resident) forwardTrunkForTest(emb []float32, pos, nLayers int) []float32 {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	saved := r.nL
	r.nL = nLayers
	defer func() { r.nL = saved }()
	r.loadEmb(emb)
	r.setPos(pos)
	e := r.q.Begin()
	r.encodeTrunkInto(e)
	e.End()
	return append([]float32(nil), r.x.Floats()...)
}

// forwardSubCaptureForTest is the Metal twin of decoder.ForwardSubCapture: it returns, per layer,
// the attention contribution (o-proj output AFTER the post-attn sandwich norm, BEFORE the residual
// add) and the MLP contribution (down output after the post-MLP sandwich norm, before the add) —
// exactly the two sublayer contributions the CUDA box traced against f32 truth. It mirrors
// encodeTrunkInto's sandwich path dispatch-for-dispatch, but flushes after each sublayer norm to
// read r.oO / r.dO before the add consumes them. Sandwich (Gemma) only; nil otherwise.
func (r *resident) forwardSubCaptureForTest(emb []float32, pos int) (attn, mlp, mlpPre, ctx, cqDeq [][]float32) {
	if !r.sandwich {
		return nil, nil, nil, nil, nil
	}
	for l := range r.layers {
		if r.layers[l].kvShared || r.layers[l].ffnI != r.I {
			return nil, nil, nil, nil, nil // its own attention/FFN dispatches assume owned KV and one FFN width (S1.3/S1.4)
		}
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	r.loadEmb(emb)
	r.setPos(pos)
	grab := func() []float32 { return append([]float32(nil), r.oO.Floats()...) }
	grabD := func() []float32 { return append([]float32(nil), r.dO.Floats()...) }
	for l := 0; l < r.nL; l++ {
		L := &r.layers[l]
		g := L.geom
		nHhd := r.nH * g.hd
		qkvRows := nHhd + 2*g.kvDim
		kOff, vOff := nHhd*4, (nHhd+g.kvDim)*4
		e := r.q.Begin()
		e.Dispatch(r.pRms, tgReduceNorm, tgReduceNorm, r.x, L.preNorm, r.aq, r.aSc, r.uH, r.uEps, r.uAddOne)
		e.DispatchTG(r.pSABias, qkvRows*32, 256, r.H*2, L.qkvW, L.qkvS, r.aq, r.aSc, r.qkv, L.qkvBias, r.uH)
		if r.qkNorm {
			e.Dispatch(r.pQKNorm, (r.nH+g.nKV)*tgReduceAttn, tgReduceAttn, r.qkv, L.qNorm, L.kNorm, r.uNH, g.uNKV, g.uHd, g.uNHhd, r.uEps, r.uAddOne)
		}
		e.Dispatch(r.pRope2, r.nH*g.half+g.nKV*g.half, 64, r.qkv, L.invf, g.uHd, r.uPos, g.uQtotal, g.uKtotal, g.uHalf, L.mscale, g.uNHhd, r.uQTempScale)
		if r.kvI8 {
			e.Dispatch(r.pKvI8, g.nKV, 1, r.qkv.At(kOff), r.qkv.At(vOff), r.kc[l], r.vc[l], r.ks[l], r.vs[l], g.uNKV, g.uHd, r.uPos)
			e.Dispatch(r.pAttnI8, r.nH*tgReduceAttn, tgReduceAttn, r.qkv, r.kc[l], r.vc[l], r.ks[l], r.vs[l], r.ctx, r.uNH, g.uNKV, g.uHd, r.uNKeys, r.uScale, L.uWindow, L.attnSinks, L.uHasSink)
		} else {
			e.Dispatch(r.pKv, g.kvDim, 64, r.qkv.At(kOff), r.qkv.At(vOff), r.kc[l], r.vc[l], g.uKvDim, r.uPos)
			e.Dispatch(r.pAttn, r.nH*tgReduceAttn, tgReduceAttn, r.qkv, r.kc[l], r.vc[l], r.ctx, r.uNH, g.uNKV, g.uHd, r.uNKeys, r.uScale, L.uWindow, L.attnSinks, L.uHasSink)
		}
		e.Dispatch(r.pQv, 256, 256, r.ctx, r.cq, r.cSc, g.uNHhd)
		e.DispatchTG(r.pSA, r.H*32, 256, r.nH*g.hd*2, L.oW, L.oS, r.cq, r.cSc, r.oO, g.uNHhd)
		e.Dispatch(r.pRmsF32, tgReduceNorm, tgReduceNorm, r.oO, L.postAttnNorm, r.uH, r.uEps, r.uAddOne)
		e.End()
		attn = append(attn, grab()) // oO now = attention contribution (post-norm, pre-add)
		// ctx (f32 attention output) and cq (its int8 quant) are still valid here — o-proj READ
		// them but never wrote them. Capture both to isolate quant_vec (context int8-quant) vs the
		// context itself as the o-proj INPUT under test.
		ctx = append(ctx, append([]float32(nil), r.ctx.Floats()[:r.nH*g.hd]...))
		csc := r.cSc.Floats()[0]
		cq8 := r.cq.Int8s()
		deq := make([]float32, r.nH*g.hd)
		for i := range deq {
			deq[i] = float32(cq8[i]) * csc
		}
		cqDeq = append(cqDeq, deq)

		e = r.q.Begin()
		e.Dispatch(r.pRes, r.H, 256, r.x, r.oO)
		e.Dispatch(r.pRms, tgReduceNorm, tgReduceNorm, r.x, L.postNorm, r.mq, r.mSc, r.uH, r.uEps, r.uAddOne)
		e.DispatchTG(r.pSA, (2*L.ffnI)*32, 256, r.H*2, L.guW, L.guS, r.mq, r.mSc, r.gu, r.uH)
		e.Dispatch(r.pSw, 256, 256, r.gu, r.gu.At(L.ffnI*4), r.dq, r.dSc, L.uFFNI, r.uAct)
		e.Dispatch(r.pGemv, r.H*32, 32, L.dW, L.dS, r.dq, r.dSc, r.dO, L.uFFNI)
		e.End()
		mlpPre = append(mlpPre, grabD()) // dO = down output BEFORE post-MLP sandwich norm (compute)

		e = r.q.Begin()
		e.Dispatch(r.pRmsF32, tgReduceNorm, tgReduceNorm, r.dO, L.postMLPNorm, r.uH, r.uEps, r.uAddOne)
		e.End()
		mlp = append(mlp, grabD()) // dO now = MLP contribution (post-norm, pre-add)

		e = r.q.Begin()
		e.Dispatch(r.pRes, r.H, 256, r.x, r.dO)
		e.End()
	}
	return attn, mlp, mlpPre, ctx, cqDeq
}

// l0GegluForTest captures L0's MLP intermediates at the BOS for the geglu-vs-down cut: the f32
// gate|up activation (r.gu) and the int8 round-trip geglu (r.dq * r.dSc) that the down-proj
// consumes. Runs L0's attention + MLP-to-swiglu in one buffer, then reads. Sandwich only.
func (r *resident) l0GegluForTest(emb []float32, pos int) (gateUp, geglu8 []float32, gSc float32) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	r.loadEmb(emb)
	r.setPos(pos)
	L := &r.layers[0]
	g := L.geom
	nHhd := r.nH * g.hd
	qkvRows := nHhd + 2*g.kvDim
	kOff, vOff := nHhd*4, (nHhd+g.kvDim)*4
	e := r.q.Begin()
	e.Dispatch(r.pRms, tgReduceNorm, tgReduceNorm, r.x, L.preNorm, r.aq, r.aSc, r.uH, r.uEps, r.uAddOne)
	e.DispatchTG(r.pSABias, qkvRows*32, 256, r.H*2, L.qkvW, L.qkvS, r.aq, r.aSc, r.qkv, L.qkvBias, r.uH)
	if r.qkNorm {
		e.Dispatch(r.pQKNorm, (r.nH+g.nKV)*tgReduceAttn, tgReduceAttn, r.qkv, L.qNorm, L.kNorm, r.uNH, g.uNKV, g.uHd, g.uNHhd, r.uEps, r.uAddOne)
	}
	e.Dispatch(r.pRope2, r.nH*g.half+g.nKV*g.half, 64, r.qkv, L.invf, g.uHd, r.uPos, g.uQtotal, g.uKtotal, g.uHalf, L.mscale, g.uNHhd, r.uQTempScale)
	if r.kvI8 {
		e.Dispatch(r.pKvI8, g.nKV, 1, r.qkv.At(kOff), r.qkv.At(vOff), r.kc[0], r.vc[0], r.ks[0], r.vs[0], g.uNKV, g.uHd, r.uPos)
		e.Dispatch(r.pAttnI8, r.nH*tgReduceAttn, tgReduceAttn, r.qkv, r.kc[0], r.vc[0], r.ks[0], r.vs[0], r.ctx, r.uNH, g.uNKV, g.uHd, r.uNKeys, r.uScale, L.uWindow, L.attnSinks, L.uHasSink)
	} else {
		e.Dispatch(r.pKv, g.kvDim, 64, r.qkv.At(kOff), r.qkv.At(vOff), r.kc[0], r.vc[0], g.uKvDim, r.uPos)
		e.Dispatch(r.pAttn, r.nH*tgReduceAttn, tgReduceAttn, r.qkv, r.kc[0], r.vc[0], r.ctx, r.uNH, g.uNKV, g.uHd, r.uNKeys, r.uScale, L.uWindow, L.attnSinks, L.uHasSink)
	}
	e.Dispatch(r.pQv, 256, 256, r.ctx, r.cq, r.cSc, g.uNHhd)
	e.DispatchTG(r.pSA, r.H*32, 256, r.nH*g.hd*2, L.oW, L.oS, r.cq, r.cSc, r.oO, g.uNHhd)
	e.Dispatch(r.pRmsF32, tgReduceNorm, tgReduceNorm, r.oO, L.postAttnNorm, r.uH, r.uEps, r.uAddOne)
	e.Dispatch(r.pRes, r.H, 256, r.x, r.oO)
	e.Dispatch(r.pRms, tgReduceNorm, tgReduceNorm, r.x, L.postNorm, r.mq, r.mSc, r.uH, r.uEps, r.uAddOne)
	e.DispatchTG(r.pSA, (2*L.ffnI)*32, 256, r.H*2, L.guW, L.guS, r.mq, r.mSc, r.gu, r.uH)
	e.Dispatch(r.pSw, 256, 256, r.gu, r.gu.At(L.ffnI*4), r.dq, r.dSc, L.uFFNI, r.uAct)
	e.End()
	gateUp = append([]float32(nil), r.gu.Floats()[:2*L.ffnI]...)
	gSc = r.dSc.Floats()[0]
	dq8 := r.dq.Int8s()
	geglu8 = make([]float32, L.ffnI)
	for i := range geglu8 {
		geglu8[i] = float32(dq8[i]) * gSc
	}
	return gateUp, geglu8, gSc
}

// attnConfirmForTest is the Metal half of the matched-input confirmer (CUDA box's
// TestGemmaConfirmerReference). It runs ONE layer's attention block over an INJECTED residual and
// INJECTED post-RoPE K / raw V (goinfer's exact CPU-int4 state), skipping kv_store, and returns the
// resulting context. Comparing to the reference target context isolates the L1 crater:
//
//	match on matched input  -> the crater is accumulated f16/precision drift in the residual+KV
//	still inflates          -> Metal's per-layer attention block (norm/QKV/RoPE/softmax) has a bug
//
// K is injected post-RoPE (kv_store stores post-RoPE K), so only Q gets RoPE here.
func (r *resident) attnConfirmForTest(resid, kHist, vHist []float32, layer, pos int, injectKV bool) []float32 {
	if r.layers[layer].kvShared {
		panic(fmt.Sprintf("metal: attnConfirmForTest on KV-shared layer %d: it owns no K/V to project or inject", layer))
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	g := r.layers[layer].geom
	nHhd := r.nH * g.hd
	qkvRows := nHhd + 2*g.kvDim
	kOff, vOff := nHhd*4, (nHhd+g.kvDim)*4
	copy(r.x.Floats(), resid)
	r.setPos(pos)
	if injectKV {
		// Inject goinfer's post-RoPE K and raw V — matched KV history. f32 cache (Gemma) takes the
		// values verbatim; the f16 cache narrows them (still cos 1.0 for the CORRECT values — the
		// crater is walked-value drift, not storage of the right ones).
		if r.kvF32 {
			o := r.kvHostOff(layer, 4)
			kc, vc := r.kc[layer].Floats()[o:], r.vc[layer].Floats()[o:]
			copy(kc[:len(kHist)], kHist)
			copy(vc[:len(vHist)], vHist)
		} else {
			o := r.kvHostOff(layer, 2)
			kc, vc := r.kc[layer].U16s()[o:], r.vc[layer].U16s()[o:]
			for i := range kHist {
				kc[i] = f32ToF16(kHist[i])
			}
			for i := range vHist {
				vc[i] = f32ToF16(vHist[i])
			}
		}
	}
	L := &r.layers[layer]
	e := r.q.Begin()
	e.Dispatch(r.pRms, tgReduceNorm, tgReduceNorm, r.x, L.preNorm, r.aq, r.aSc, r.uH, r.uEps, r.uAddOne)
	e.DispatchTG(r.pSABias, qkvRows*32, 256, r.H*2, L.qkvW, L.qkvS, r.aq, r.aSc, r.qkv, L.qkvBias, r.uH)
	if r.qkNorm { // Gemma3/Qwen3: per-head Q/K RMSNorm before RoPE — the injected K is post-QK-norm
		e.Dispatch(r.pQKNorm, (r.nH+g.nKV)*tgReduceAttn, tgReduceAttn, r.qkv, L.qNorm, L.kNorm, r.uNH, g.uNKV, g.uHd, g.uNHhd, r.uEps, r.uAddOne)
	}
	e.Dispatch(r.pRope, r.nH*g.half, 64, r.qkv, L.invf, g.uHd, r.uPos, g.uQtotal, g.uHalf, L.mscale) // Q
	if !injectKV {
		// Use Metal's OWN walked KV history: RoPE K and store pos into the cache (isolates the
		// f16-KV drift of positions 0..pos-1 from Metal's walk, with a matched residual).
		e.Dispatch(r.pRope, g.nKV*g.half, 64, r.qkv.At(kOff), L.invf, g.uHd, r.uPos, g.uKtotal, g.uHalf, L.mscale)
		if r.kvI8 {
			e.Dispatch(r.pKvI8, g.nKV, 1, r.qkv.At(kOff), r.qkv.At(vOff), r.kc[layer], r.vc[layer], r.ks[layer], r.vs[layer], g.uNKV, g.uHd, r.uPos)
		} else {
			e.Dispatch(r.pKv, g.kvDim, 64, r.qkv.At(kOff), r.qkv.At(vOff), r.kc[layer], r.vc[layer], g.uKvDim, r.uPos)
		}
	}
	if r.kvI8 {
		e.Dispatch(r.pAttnI8, r.nH*tgReduceAttn, tgReduceAttn, r.qkv, r.kc[layer], r.vc[layer], r.ks[layer], r.vs[layer], r.ctx, r.uNH, g.uNKV, g.uHd, r.uNKeys, r.uScale, L.uWindow, L.attnSinks, L.uHasSink)
	} else {
		e.Dispatch(r.pAttn, r.nH*tgReduceAttn, tgReduceAttn, r.qkv, r.kc[layer], r.vc[layer], r.ctx, r.uNH, g.uNKV, g.uHd, r.uNKeys, r.uScale, L.uWindow, L.attnSinks, L.uHasSink)
	}
	e.End()
	return append([]float32(nil), r.ctx.Floats()[:nHhd]...)
}

// forwardHeadForTest runs the FULL trunk (final norm included) then the LM head, and returns
// both the head's INPUT activation as it actually sees it — r.aq dequantized by r.aSc, i.e. the
// int8-quantized final-norm output — and the logits. It splits the one step the per-layer bisect
// cannot: final-norm-quant vs the head matmul. Comparing the returned activation to the CPU's
// f32 final-norm output isolates the quantization of the head's input; comparing the logits
// isolates the matmul on top of it.
func (r *resident) forwardHeadForTest(emb []float32, pos int) (act, logits []float32) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	r.loadEmb(emb)
	r.setPos(pos)
	e := r.q.Begin()
	r.encodeTrunkInto(e)
	e.Dispatch(r.pGemvW8, (r.V)*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.logits, r.uH)
	e.End()
	q, sc := r.aq.Int8s(), r.aSc.Floats()[0]
	act = make([]float32, r.H)
	for i := range act {
		act[i] = float32(q[i]) * sc
	}
	return act, append([]float32(nil), r.logits.Floats()...)
}

func dequantG32(q []int8, sc []float32, rows, cols int) []float32 {
	return kernels.DequantG32(q, sc, rows, cols)
}
