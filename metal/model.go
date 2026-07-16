//go:build darwin

// Real-model resident Metal decoder — the final GO/NO-GO piece of the spike. Loads a
// dense Qwen2/Llama model's int8 weights out of the goinfer decoder, uploads them to
// Metal once, and runs the full layer stack per token in ONE command buffer (the tax
// requirement). cgo-free throughout (purego-objc + dlopen Metal).
package metal

import (
	"fmt"
	"runtime"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

const metalCtxCap = 4096 // resident KV positions (spike)

type residLayer struct {
	qW, qS, kW, kS, vW, vS, oW, oS, gW, gS, uW, uS, dW, dS Buffer
	qb, kb, vb, preNorm, postNorm                         Buffer
	hasBias                                               bool
}

// Resident is a Metal-resident dense decoder. Weights uploaded once in BuildResident;
// per token only the embedding + pos uniforms change.
type Resident struct {
	d                                             *Device
	q                                             Queue
	pRms, pQv, pGemv, pRope, pKv, pAttn, pSw, pRes Pipeline

	H, nL, nH, nKV, hd, I, V, kvDim, half int
	embed                                 *linalg.WeightMat

	layers    []residLayer
	finalNorm Buffer
	lmW, lmS  Buffer
	kc, vc    []Buffer

	x, aq, aSc, qB, kB, vB, ctx, cq, cSc, oO, mq, mSc, gO, uO, dq, dSc, dO, logits Buffer
	invf, uHd, uKvDim, uH, uI, uHH, uNH, uNKV, uScale, uEps, uQtotal, uKtotal      Buffer
	uPos, uNKeys                                                                   Buffer
	logitsHost                                                                    []float32
}

func byteBuf(d *Device, n int) Buffer {
	return Buffer{id: d.id.Send(selNewBufferLen, uintptr(n), uintptr(0)), n: n}
}

func int8Buf(d *Device, w *linalg.WeightMat) (Buffer, Buffer, error) {
	q8, sc, _, ok := w.Int8()
	if !ok {
		return Buffer{}, Buffer{}, fmt.Errorf("metal: weight kind %q is not int8 (load Options{Quant:\"int8int8\"})", w.Kind())
	}
	return d.NewBufferInt8(q8), d.NewBufferFloats(sc), nil
}

// BuildResident builds a Metal resident decoder from an int8-loaded dense Qwen2/Llama
// Model. Handles Qwen2 q/k/v bias; assumes no QK-norm / sliding-window / embed-scale
// (the DecodeRunnerEligible dense shape), full RoPE via the model's own inv-freq table.
func BuildResident(m *decoder.Model) (*Resident, error) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		return nil, err
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		return nil, err
	}
	pipe := func(name string) Pipeline {
		p, e := d.NewComputePipeline(lib, name)
		if e != nil {
			panic(fmt.Sprintf("metal pipeline %s: %v", name, e))
		}
		return p
	}
	H, nL, nH, nKV, hd, I, V := m.Dims()
	r := &Resident{d: d, H: H, nL: nL, nH: nH, nKV: nKV, hd: hd, I: I, V: V, kvDim: nKV * hd, half: hd / 2}
	r.pRms, r.pQv, r.pGemv = pipe("rmsnorm_quant"), pipe("quant_vec"), pipe("gemv_w8a8_coal")
	r.pRope, r.pKv, r.pAttn = pipe("rope"), pipe("kv_store"), pipe("attention")
	r.pSw, r.pRes = pipe("swiglu_quant"), pipe("residual")
	r.q = d.NewCommandQueue()

	w := m.Weights()
	r.embed = &w.Embed
	mk := func(wm *linalg.WeightMat) (Buffer, Buffer) {
		q, s, e := int8Buf(d, wm)
		if e != nil {
			panic(e)
		}
		return q, s
	}
	r.layers = make([]residLayer, nL)
	r.kc, r.vc = make([]Buffer, nL), make([]Buffer, nL)
	for l := 0; l < nL; l++ {
		lw := &w.Layers[l]
		var L residLayer
		L.qW, L.qS = mk(&lw.QProj)
		L.kW, L.kS = mk(&lw.KProj)
		L.vW, L.vS = mk(&lw.VProj)
		L.oW, L.oS = mk(&lw.OProj)
		L.gW, L.gS = mk(&lw.GateProj)
		L.uW, L.uS = mk(&lw.UpProj)
		L.dW, L.dS = mk(&lw.DownProj)
		L.preNorm = d.NewBufferFloats(lw.PreAttnNorm)
		L.postNorm = d.NewBufferFloats(lw.PreMLPNorm)
		if lw.QBias != nil {
			L.qb, L.kb, L.vb, L.hasBias = d.NewBufferFloats(lw.QBias), d.NewBufferFloats(lw.KBias), d.NewBufferFloats(lw.VBias), true
		}
		r.layers[l] = L
		r.kc[l] = d.NewBufferLen(metalCtxCap * r.kvDim)
		r.vc[l] = d.NewBufferLen(metalCtxCap * r.kvDim)
	}
	r.finalNorm = d.NewBufferFloats(w.FinalNorm)
	lm := &w.LMHead
	if lm.Rows() == 0 {
		lm = &w.Embed // tied
	}
	if r.lmW, r.lmS, err = int8Buf(d, lm); err != nil {
		return nil, err
	}

	r.x = d.NewBufferLen(H)
	r.aq, r.aSc = byteBuf(d, H), d.NewBufferLen(1)
	r.qB, r.kB, r.vB = d.NewBufferLen(nH*hd), d.NewBufferLen(r.kvDim), d.NewBufferLen(r.kvDim)
	r.ctx, r.cq, r.cSc = d.NewBufferLen(nH*hd), byteBuf(d, nH*hd), d.NewBufferLen(1)
	r.oO, r.mq, r.mSc = d.NewBufferLen(H), byteBuf(d, H), d.NewBufferLen(1)
	r.gO, r.uO, r.dq, r.dSc, r.dO = d.NewBufferLen(I), d.NewBufferLen(I), byteBuf(d, I), d.NewBufferLen(1), d.NewBufferLen(H)
	r.logits = d.NewBufferLen(V)
	r.invf = d.NewBufferFloats(m.RopeInvFreq())
	r.uHd, r.uKvDim = d.NewBufferU32(uint32(hd)), d.NewBufferU32(uint32(r.kvDim))
	r.uH, r.uI, r.uHH = d.NewBufferU32(uint32(H)), d.NewBufferU32(uint32(I)), d.NewBufferU32(uint32(nH*hd))
	r.uNH, r.uNKV = d.NewBufferU32(uint32(nH)), d.NewBufferU32(uint32(nKV))
	r.uScale, r.uEps = d.NewBufferFloats([]float32{m.AttnScale()}), d.NewBufferFloats([]float32{m.NormEps()})
	r.uQtotal, r.uKtotal = d.NewBufferU32(uint32(nH*r.half)), d.NewBufferU32(uint32(nKV*r.half))
	r.uPos, r.uNKeys = d.NewBufferU32(0), d.NewBufferU32(1)
	r.logitsHost = make([]float32, V)
	return r, nil
}

// Forward runs token `id` at absolute position `pos` and returns logits[V]. The whole
// layer stack + LM head is encoded into ONE command buffer, one commit/wait.
func (r *Resident) Forward(id, pos int) []float32 {
	// Pin to one OS thread for the whole call: the NSAutoreleasePool (begin/end) is
	// per-OS-thread, and Go can migrate goroutines mid-call — draining a pool on a
	// different thread than it was pushed is UB (intermittent SIGSEGV). Same discipline
	// the CUDA backend's LockOSThread executor uses.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	r.embed.Row(id, r.x.Floats()) // CPU dequant embedding into the shared buffer
	r.uPos.SetU32(uint32(pos))
	r.uNKeys.SetU32(uint32(pos + 1))

	nHhd := r.nH * r.hd
	e := r.q.begin()
	for l := 0; l < r.nL; l++ {
		L := &r.layers[l]
		e.dispatch(r.pRms, 256, 256, r.x, L.preNorm, r.aq, r.aSc, r.uH, r.uEps)
		e.dispatch(r.pGemv, (nHhd)*32, 32, r.aq, r.aSc, L.qW, L.qS, r.qB, r.uH)
		e.dispatch(r.pGemv, (r.kvDim)*32, 32, r.aq, r.aSc, L.kW, L.kS, r.kB, r.uH)
		e.dispatch(r.pGemv, (r.kvDim)*32, 32, r.aq, r.aSc, L.vW, L.vS, r.vB, r.uH)
		if L.hasBias {
			e.dispatch(r.pRes, nHhd, 64, r.qB, L.qb)
			e.dispatch(r.pRes, r.kvDim, 64, r.kB, L.kb)
			e.dispatch(r.pRes, r.kvDim, 64, r.vB, L.vb)
		}
		e.dispatch(r.pRope, nHhd/2, 64, r.qB, r.invf, r.uHd, r.uPos, r.uQtotal)
		e.dispatch(r.pRope, r.kvDim/2, 64, r.kB, r.invf, r.uHd, r.uPos, r.uKtotal)
		e.dispatch(r.pKv, r.kvDim, 64, r.kB, r.vB, r.kc[l], r.vc[l], r.uKvDim, r.uPos)
		e.dispatch(r.pAttn, r.nH, r.nH, r.qB, r.kc[l], r.vc[l], r.ctx, r.uNH, r.uNKV, r.uHd, r.uNKeys, r.uScale)
		e.dispatch(r.pQv, 256, 256, r.ctx, r.cq, r.cSc, r.uHH)
		e.dispatch(r.pGemv, (r.H)*32, 32, r.cq, r.cSc, L.oW, L.oS, r.oO, r.uHH)
		e.dispatch(r.pRes, r.H, 64, r.x, r.oO)
		e.dispatch(r.pRms, 256, 256, r.x, L.postNorm, r.mq, r.mSc, r.uH, r.uEps)
		e.dispatch(r.pGemv, (r.I)*32, 32, r.mq, r.mSc, L.gW, L.gS, r.gO, r.uH)
		e.dispatch(r.pGemv, (r.I)*32, 32, r.mq, r.mSc, L.uW, L.uS, r.uO, r.uH)
		e.dispatch(r.pSw, 256, 256, r.gO, r.uO, r.dq, r.dSc, r.uI)
		e.dispatch(r.pGemv, (r.H)*32, 32, r.dq, r.dSc, L.dW, L.dS, r.dO, r.uI)
		e.dispatch(r.pRes, r.H, 64, r.x, r.dO)
	}
	e.dispatch(r.pRms, 256, 256, r.x, r.finalNorm, r.aq, r.aSc, r.uH, r.uEps)
	e.dispatch(r.pGemv, (r.V)*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.logits, r.uH)
	e.end()

	copy(r.logitsHost, r.logits.Floats())
	return r.logitsHost
}
