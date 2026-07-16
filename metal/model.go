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
	qkvW, qkvS, guW, guS, oW, oS, dW, dS Buffer // fused QKV + fused gate/up + o + down
	qkvBias, preNorm, postNorm          Buffer
}

// Resident is a Metal-resident dense decoder. Weights uploaded once in BuildResident;
// per token only the embedding + pos uniforms change.
type Resident struct {
	d                                             *Device
	q                                             Queue
	pRms, pQv, pGemv, pGemvBias, pGemvResid, pRope, pKv, pAttn, pSw, pRes Pipeline
	qkv, gu                                                              Buffer // fused QKV out, fused gate/up out

	H, nL, nH, nKV, hd, I, V, kvDim, half int
	embed                                 *linalg.WeightMat

	layers    []residLayer
	finalNorm Buffer
	lmW, lmS  Buffer
	kc, vc    []Buffer

	x, aq, aSc, ctx, cq, cSc, oO, mq, mSc, dq, dSc, dO, logits Buffer
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

// int4Buf re-quantizes an int8 WeightMat to W4A8 (int4, group=32) via the validated packer
// and uploads packed nibbles + f32 group scales — half the weight bytes of int8 (the
// target-quant bandwidth win). One-time at build.
func int4Buf(d *Device, w *linalg.WeightMat) (Buffer, Buffer, error) {
	q8, sc, _, ok := w.Int8()
	if !ok {
		return Buffer{}, Buffer{}, fmt.Errorf("metal: weight kind %q is not int8", w.Kind())
	}
	N, K := w.Rows(), w.Cols()
	words := make([]uint32, N*(K/8))
	scales := make([]float32, N*(K/32))
	for n := 0; n < N; n++ {
		wd, sd := packW4A8Row(dequantInt8ToF32Row(q8[n*K:(n+1)*K], sc[n], K, K))
		copy(words[n*(K/8):(n+1)*(K/8)], wd)
		copy(scales[n*(K/32):(n+1)*(K/32)], sd)
	}
	return d.NewBufferUint32s(words), d.NewBufferFloats(scales), nil
}

// int4Concat re-quantizes and row-concatenates several same-K WeightMats into ONE W4A8
// buffer — the fusion enabler (combined QKV → one GEMV, combined gate/up → one GEMV).
func int4Concat(d *Device, wms ...*linalg.WeightMat) (Buffer, Buffer) {
	var words []uint32
	var scales []float32
	for _, w := range wms {
		q8, sc, _, ok := w.Int8()
		if !ok {
			panic(fmt.Sprintf("metal: int4Concat weight kind %q not int8", w.Kind()))
		}
		N, K := w.Rows(), w.Cols()
		for n := 0; n < N; n++ {
			wd, sd := packW4A8Row(dequantInt8ToF32Row(q8[n*K:(n+1)*K], sc[n], K, K))
			words = append(words, wd...)
			scales = append(scales, sd...)
		}
	}
	return d.NewBufferUint32s(words), d.NewBufferFloats(scales)
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
	r.pRms, r.pQv, r.pGemv = pipe("rmsnorm_quant"), pipe("quant_vec"), pipe("gemv_w4a8_coal")
	r.pGemvBias, r.pGemvResid = pipe("gemv_w4a8_bias"), pipe("gemv_w4a8_resid")
	r.pRope, r.pKv, r.pAttn = pipe("rope"), pipe("kv_store"), pipe("attention")
	r.pSw, r.pRes = pipe("swiglu_quant"), pipe("residual")
	r.q = d.NewCommandQueue()

	w := m.Weights()
	r.embed = &w.Embed
	mk := func(wm *linalg.WeightMat) (Buffer, Buffer) {
		q, s, e := int4Buf(d, wm)
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
		L.qkvW, L.qkvS = int4Concat(d, &lw.QProj, &lw.KProj, &lw.VProj) // fused QKV
		L.guW, L.guS = int4Concat(d, &lw.GateProj, &lw.UpProj)          // fused gate/up
		L.oW, L.oS = mk(&lw.OProj)
		L.dW, L.dS = mk(&lw.DownProj)
		L.preNorm = d.NewBufferFloats(lw.PreAttnNorm)
		L.postNorm = d.NewBufferFloats(lw.PreMLPNorm)
		// combined qkv bias (zeros where absent) so the fused GEMV epilogue is uniform.
		qb, kb, vb := lw.QBias, lw.KBias, lw.VBias
		if qb == nil {
			qb, kb, vb = make([]float32, nH*hd), make([]float32, r.kvDim), make([]float32, r.kvDim)
		}
		L.qkvBias = d.NewBufferFloats(append(append(append([]float32{}, qb...), kb...), vb...))
		r.layers[l] = L
		r.kc[l] = d.NewBufferLen(metalCtxCap * r.kvDim)
		r.vc[l] = d.NewBufferLen(metalCtxCap * r.kvDim)
	}
	r.finalNorm = d.NewBufferFloats(w.FinalNorm)
	lm := &w.LMHead
	if lm.Rows() == 0 {
		lm = &w.Embed // tied
	}
	if r.lmW, r.lmS, err = int4Buf(d, lm); err != nil {
		return nil, err
	}

	r.x = d.NewBufferLen(H)
	r.aq, r.aSc = byteBuf(d, H), d.NewBufferLen(1)
	r.qkv = d.NewBufferLen(nH*hd + 2*r.kvDim) // fused [q | k | v]
	r.gu = d.NewBufferLen(2 * I)              // fused [gate | up]
	r.ctx, r.cq, r.cSc = d.NewBufferLen(nH*hd), byteBuf(d, nH*hd), d.NewBufferLen(1)
	r.oO, r.mq, r.mSc = d.NewBufferLen(H), byteBuf(d, H), d.NewBufferLen(1)
	r.dq, r.dSc, r.dO = byteBuf(d, I), d.NewBufferLen(1), d.NewBufferLen(H)
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
	qkvRows := nHhd + 2*r.kvDim
	kOff, vOff := nHhd*4, (nHhd+r.kvDim)*4 // byte offsets of k, v within the fused qkv buffer
	e := r.q.begin()
	for l := 0; l < r.nL; l++ {
		L := &r.layers[l]
		// --- attention block (12 dispatches vs 19: QKV fused +bias, residual fused) ---
		e.dispatch(r.pRms, 256, 256, r.x, L.preNorm, r.aq, r.aSc, r.uH, r.uEps)
		e.dispatch(r.pGemvBias, qkvRows*32, 32, L.qkvW, L.qkvS, r.aq, r.aSc, r.qkv, L.qkvBias, r.uH)
		e.dispatch(r.pRope, nHhd/2, 64, r.qkv, r.invf, r.uHd, r.uPos, r.uQtotal)                 // q @ off 0
		e.dispatch(r.pRope, r.kvDim/2, 64, r.qkv.At(kOff), r.invf, r.uHd, r.uPos, r.uKtotal)      // k
		e.dispatch(r.pKv, r.kvDim, 64, r.qkv.At(kOff), r.qkv.At(vOff), r.kc[l], r.vc[l], r.uKvDim, r.uPos)
		e.dispatch(r.pAttn, r.nH*128, 128, r.qkv, r.kc[l], r.vc[l], r.ctx, r.uNH, r.uNKV, r.uHd, r.uNKeys, r.uScale)
		e.dispatch(r.pQv, 256, 256, r.ctx, r.cq, r.cSc, r.uHH)
		e.dispatch(r.pGemvResid, r.H*32, 32, L.oW, L.oS, r.cq, r.cSc, r.x, r.uHH) // o-proj + residual
		// --- ffn block ---
		e.dispatch(r.pRms, 256, 256, r.x, L.postNorm, r.mq, r.mSc, r.uH, r.uEps)
		e.dispatch(r.pGemv, (2*r.I)*32, 32, L.guW, L.guS, r.mq, r.mSc, r.gu, r.uH) // fused gate|up
		e.dispatch(r.pSw, 256, 256, r.gu, r.gu.At(r.I*4), r.dq, r.dSc, r.uI)        // gate @0, up @I
		e.dispatch(r.pGemvResid, r.H*32, 32, L.dW, L.dS, r.dq, r.dSc, r.x, r.uI) // down + residual
	}
	e.dispatch(r.pRms, 256, 256, r.x, r.finalNorm, r.aq, r.aSc, r.uH, r.uEps)
	e.dispatch(r.pGemv, (r.V)*32, 32, r.lmW, r.lmS, r.aq, r.aSc, r.logits, r.uH)
	e.end()

	copy(r.logitsHost, r.logits.Floats())
	return r.logitsHost
}
