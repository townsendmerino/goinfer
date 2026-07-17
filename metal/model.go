//go:build darwin

// Real-model resident Metal decoder — the final GO/NO-GO piece of the spike. Loads a
// dense Qwen2/Llama model's int8 weights out of the goinfer decoder, uploads them to
// Metal once, and runs the full layer stack per token in ONE command buffer (the tax
// requirement). cgo-free throughout (purego-objc + dlopen Metal).
package metal

import (
	"fmt"
	"runtime"
	"sync"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

const metalCtxCap = 4096 // resident KV positions (spike)

// prefillFeatures is what the f16 MMA prefill kernels (prefill.go) actually implement: a dense
// SiLU FFN, per-head QK-norm, a MODEL-LEVEL rope table and a MODEL-LEVEL window. Anything else
// — MoE (which never packs the dense FFN buffers at all), or Gemma's sandwich norms / (1+w) RMS
// / GeGLU / per-layer rope — must decline prefill (see Resident.prefillOK). Deriving the
// predicate from the shared taxonomy keeps it honest as features land.
var prefillFeatures = map[decoder.ResidentFeature]bool{
	decoder.FeatQKNorm:        true,
	decoder.FeatSlidingWindow: true,
	decoder.FeatPartialRotary: true,
}

type residLayer struct {
	qkvW, qkvS, guW, guS, oW, oS, dW, dS Buffer // fused QKV + fused gate/up + o + down
	qkvBias, preNorm, postNorm           Buffer
	qNorm, kNorm                         Buffer    // per-head QK-RMSNorm weights (Qwen3; zero if !qkNorm)
	moe                                  *moeLayer // non-nil ⇒ this layer's FFN is MoE (dense guW/dW unused)
	postAttnNorm, postMLPNorm            Buffer    // Gemma sandwich norms on each sublayer OUTPUT (zero if !sandwich)
	invf                                 Buffer    // per-layer RoPE inv-freq (Gemma local 10k vs global 1M base)
	uWindow                              Buffer    // per-layer attention window (0 = full causal; Gemma mixes local/global)
}

// Resident is a Metal-resident dense decoder. Weights uploaded once in BuildResident;
// per token only the embedding + pos uniforms change.
type Resident struct {
	d                                                                     *Device
	q                                                                     Queue
	pRms, pQv, pGemv, pGemvBias, pGemvResid, pRope, pKv, pAttn, pSw, pRes Pipeline
	pSA, pSABias, pSAResid                                                Pipeline // Stage A gemv (K<=1536)
	pSAAmax, pArgFinish                                                   Pipeline // fused block-argmax lm head
	pQKNorm                                                               Pipeline // per-head QK-RMSNorm (Qwen3)
	pRmsF32                                                               Pipeline // Gemma sandwich: in-place RMSNorm of a sublayer output
	pGemvW8, pGemvW8Amax                                                  Pipeline // int8 GEMV + fused block-argmax — the logit-critical LM head (see lmW)
	qkNorm                                                                bool     // arch has QK-norm
	sandwich                                                              bool     // Gemma NormSandwich4: norm each sublayer output before the residual add
	uAct                                                                  Buffer   // gated-MLP activation ordinal (decoder.ActKind: 0=GELU-tanh, 1=SiLU)
	uNHhd, uAddOne, uHalf, uWindow                                        Buffer   // qk_norm + rope + sliding-window uniforms
	qkv, gu                                                               Buffer   // fused QKV out, fused gate/up out

	H, nL, nH, nKV, hd, I, V, kvDim, half int
	embed                                 *linalg.WeightMat

	layers    []residLayer
	finalNorm Buffer
	lmW, lmS  Buffer
	kc, vc    []Buffer
	moe       *moeResident // non-nil ⇒ MoE model (router + stacked experts); see moe.go

	// prefillOK reports whether the f16 MMA prefill kernels (prefill.go) actually implement
	// this model's shape. They run a DENSE FFN out of L.guW/L.dW with a model-level rope +
	// window and a SiLU-only swiglu — so a model whose decode path diverges (MoE leaves the
	// dense FFN buffers unset entirely) MUST decline prefill and let the caller fall back to
	// the sequential Forward loop: correct, just a slower TTFT.
	prefillOK bool

	x, aq, aSc, ctx, cq, cSc, oO, mq, mSc, dq, dSc, dO, logits                Buffer
	invf, uHd, uKvDim, uH, uI, uHH, uNH, uNKV, uScale, uEps, uQtotal, uKtotal Buffer
	uPos, uNKeys                                                              Buffer
	part, tok, uP                                                             Buffer // fused-argmax: tile partials, token out, tile count
	logitsHost                                                                []float32
	gpuStart, gpuEnd, kernStart, kernEnd                                      float64 // last-Forward GPU timing (Step 0)

	// pipelined logits executor (encode-ahead): a persistent OS-thread-pinned goroutine that
	// commits token t, pre-encodes t+1 while the GPU runs t, then waits — hiding the ~0.9ms
	// host encode bubble. Lazily started on first ForwardEmbPipe; stopped in Close.
	execOnce sync.Once
	execReq  chan execJob
	execAck  chan []float32
	execDone chan struct{} // closed when execLoop returns — Close must WAIT on this before freeing

	pf *prefillState // lazily-compiled f16 MMA prefill pipelines (opt-in)
}

type execJob struct {
	emb []float32
	pos int
}

func byteBuf(d *Device, n int) Buffer {
	return d.mustBuf(d.id.Send(selNewBufferLen, uintptr(n), uintptr(0)), n, "bytes")
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
	scales := make([]uint16, N*(K/32)) // f16 group scales (L1: −10% token traffic, parity-neutral)
	for n := range N {
		wd, sd := packW4A8Row(dequantInt8ToF32Row(q8[n*K:(n+1)*K], sc[n], K, K))
		copy(words[n*(K/8):(n+1)*(K/8)], wd)
		for g, s := range sd {
			scales[n*(K/32)+g] = f32ToF16(s)
		}
	}
	return d.NewBufferUint32s(words), d.NewBufferU16s(scales), nil
}

// int4Concat re-quantizes and row-concatenates several same-K WeightMats into ONE W4A8
// buffer — the fusion enabler (combined QKV → one GEMV, combined gate/up → one GEMV).
func int4Concat(d *Device, wms ...*linalg.WeightMat) (Buffer, Buffer) {
	var words []uint32
	var scales []uint16 // f16 group scales (L1)
	for _, w := range wms {
		q8, sc, _, ok := w.Int8()
		if !ok {
			panic(fmt.Sprintf("metal: int4Concat weight kind %q not int8", w.Kind()))
		}
		N, K := w.Rows(), w.Cols()
		for n := range N {
			wd, sd := packW4A8Row(dequantInt8ToF32Row(q8[n*K:(n+1)*K], sc[n], K, K))
			words = append(words, wd...)
			for _, s := range sd {
				scales = append(scales, f32ToF16(s))
			}
		}
	}
	return d.NewBufferUint32s(words), d.NewBufferU16s(scales)
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
	r.pSA, r.pSABias, r.pSAResid = pipe("gemv_w4a8_sa"), pipe("gemv_w4a8_sa_bias"), pipe("gemv_w4a8_sa_resid")
	r.pSAAmax, r.pArgFinish = pipe("gemv_w4a8_sa_amax"), pipe("argmax_finish")
	r.pRope, r.pKv, r.pAttn = pipe("rope"), pipe("kv_store"), pipe("attention")
	r.pSw, r.pRes = pipe("swiglu_quant"), pipe("residual")
	r.pGemvW8, r.pGemvW8Amax = pipe("gemv_w8a8_coal"), pipe("gemv_w8a8_amax")
	r.pQKNorm, r.pRmsF32 = pipe("qk_norm"), pipe("rmsnorm_f32")
	r.qkNorm = m.HasQKNorm()
	r.sandwich = m.SandwichNormResident()
	// Gated-MLP activation: ordinals ARE decoder.ActKind's iota (0=GELU-tanh, 1=SiLU), so this
	// passes straight through to glu_act. Gemma is GeGLU; everything else admitted is SwiGLU.
	r.uAct = d.NewBufferU32(uint32(m.GatedActResident()))
	addOne := uint32(0)
	if m.RMSAddOne() {
		addOne = 1
	}
	r.uNHhd, r.uAddOne = d.NewBufferU32(uint32(nH*hd)), d.NewBufferU32(addOne)
	win := m.SlidingWindowResident() // 0 = full causal; Mistral is all-local with this window
	if win < 0 {
		win = 0
	}
	r.uWindow = d.NewBufferU32(uint32(win))
	if r.moe, err = buildMoE(d, m, pipe, H); err != nil { // nil for a dense model; error ⇒ decline
		return nil, err
	}
	// prefillOK, derived rather than hand-listed: the f16 prefill kernels implement exactly the
	// features below, so ANY model needing more (MoE, Gemma's sandwich/(1+w)/GeGLU/per-layer
	// rope) declines prefill and falls back to the sequential Forward loop.
	r.prefillOK = len(m.MissingResidentFeatures(prefillFeatures)) == 0
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
	for l := range nL {
		lw := &w.Layers[l]
		var L residLayer
		L.qkvW, L.qkvS = int4Concat(d, &lw.QProj, &lw.KProj, &lw.VProj) // fused QKV
		L.oW, L.oS = mk(&lw.OProj)
		if r.moe != nil && len(lw.Experts) > 0 { // MoE layer: stacked experts instead of a dense FFN
			L.moe = buildMoELayer(d, lw, r.moe)
		} else { // dense FFN (also GLM/DeepSeek's FirstKDense prefix layers)
			L.guW, L.guS = int4Concat(d, &lw.GateProj, &lw.UpProj) // fused gate/up
			L.dW, L.dS = mk(&lw.DownProj)
		}
		L.preNorm = d.NewBufferFloats(lw.PreAttnNorm)
		L.postNorm = d.NewBufferFloats(lw.PreMLPNorm)
		if r.qkNorm { // Qwen3 per-head Q/K norm weights [hd]
			L.qNorm, L.kNorm = d.NewBufferFloats(lw.QNorm), d.NewBufferFloats(lw.KNorm)
		}
		// Gemma sandwich norms on each sublayer OUTPUT. Required to be present when the arch
		// declares them — a silently-missing one would DROP the norm rather than error.
		if r.sandwich {
			if len(lw.PostAttnNorm) != H || len(lw.PostMLPNorm) != H {
				return nil, fmt.Errorf("metal: layer %d declares sandwich norms but PostAttnNorm/PostMLPNorm are not len==hidden(%d) (got %d/%d)",
					l, H, len(lw.PostAttnNorm), len(lw.PostMLPNorm))
			}
			L.postAttnNorm = d.NewBufferFloats(lw.PostAttnNorm)
			L.postMLPNorm = d.NewBufferFloats(lw.PostMLPNorm)
		}
		// Per-layer RoPE table (Gemma local 10k vs global 1M base) and per-layer window. The rope
		// KERNEL is unchanged — all per-layer variation rides in the table contents, and the width
		// (r.half) stays model-level. Uniform-rope families hand back identical tables per layer.
		L.invf = d.NewBufferFloats(m.RopeInvFreqLayer(l))
		lw2 := uint32(0) // 0 = full causal; only local layers carry the window
		if m.LayerIsLocalResident(l) {
			lw2 = uint32(win)
		}
		L.uWindow = d.NewBufferU32(lw2)
		// combined qkv bias (zeros where absent) so the fused GEMV epilogue is uniform.
		qb, kb, vb := lw.QBias, lw.KBias, lw.VBias
		if qb == nil {
			qb, kb, vb = make([]float32, nH*hd), make([]float32, r.kvDim), make([]float32, r.kvDim)
		}
		L.qkvBias = d.NewBufferFloats(append(append(append([]float32{}, qb...), kb...), vb...))
		r.layers[l] = L
		r.kc[l] = byteBuf(d, metalCtxCap*r.kvDim*2) // f16 KV: 2 bytes/elem (halves the cache)
		r.vc[l] = byteBuf(d, metalCtxCap*r.kvDim*2)
	}
	r.finalNorm = d.NewBufferFloats(w.FinalNorm)
	lm := &w.LMHead
	if lm.Rows() == 0 {
		lm = &w.Embed // tied
	}
	// The LM head is LOGIT-CRITICAL and must stay int8. decoder/weightmat.go: "at int4 they flip
	// the argmax and tank the cosine (the tied head dots every logit against them)" — which is
	// why the decoder PINS the embedding/LM-head at int8 even in int4 mode. Metal was
	// int4-quantizing it anyway, violating that pin. Worst for Gemma: a TIED head, 262k x 2560,
	// so every one of 262k logits is dotted against int4-mangled embedding rows.
	if r.lmW, r.lmS, err = int8Buf(d, lm); err != nil {
		return nil, err
	}

	// The gate|up and down scratch must fit the widest FFN width: the dense I, or (for MoE) the
	// larger of the expert and shared-expert intermediate dims.
	guDim := I
	if r.moe != nil {
		guDim = max(I, max(r.moe.inter, r.moe.sharedInter))
	}
	r.x = d.NewBufferLen(H)
	r.aq, r.aSc = byteBuf(d, H), d.NewBufferLen(1)
	r.qkv = d.NewBufferLen(nH*hd + 2*r.kvDim) // fused [q | k | v]
	r.gu = d.NewBufferLen(2 * guDim)          // fused [gate | up]
	r.ctx, r.cq, r.cSc = d.NewBufferLen(nH*hd), byteBuf(d, nH*hd), d.NewBufferLen(1)
	r.oO, r.mq, r.mSc = d.NewBufferLen(H), byteBuf(d, H), d.NewBufferLen(1)
	r.dq, r.dSc, r.dO = byteBuf(d, guDim), d.NewBufferLen(1), d.NewBufferLen(H)
	r.logits = d.NewBufferLen(V)
	nTiles := V / 8 // one (maxLogit,rowIdx) partial per threadgroup (8 rows) — V divisible by 8
	r.part, r.tok, r.uP = d.NewBufferLen(nTiles*2), d.NewBufferLen(1), d.NewBufferU32(uint32(nTiles))
	invf := m.RopeInvFreq()
	r.half = len(invf) // rotaryDim/2 (= hd/2 for full rotary; < for Phi partial rotary)
	r.invf = d.NewBufferFloats(invf)
	r.uHalf = d.NewBufferU32(uint32(r.half))
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
	return r.forwardLogits(pos)
}

// ForwardEmb is Forward given a precomputed embedding[H] (the decoder.ResidentForward shape)
// instead of a token id — it copies the embedding into the shared input buffer and skips the
// internal embedding lookup. Numerically identical to Forward(id,pos) when emb = Embed.Row(id)
// (the eligible dense archs have no embed scale).
func (r *Resident) ForwardEmb(emb []float32, pos int) []float32 {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	copy(r.x.Floats(), emb)
	return r.forwardLogits(pos)
}

// forwardLogits encodes the trunk + full lm head and reads back logits[V]. Caller must hold
// the OS thread and have filled r.x with the input embedding.
func (r *Resident) forwardLogits(pos int) []float32 {
	r.uPos.SetU32(uint32(pos))
	r.uNKeys.SetU32(uint32(pos + 1))
	e := r.q.begin()
	r.encodeTrunkInto(e)                                                           // 28 layers → final norm → r.aq/r.aSc
	e.dispatch(r.pGemvW8, (r.V)*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.logits, r.uH) // full lm head (int8 — logit-critical)
	e.end()
	r.gpuStart, r.gpuEnd, r.kernStart, r.kernEnd = e.gpuStart, e.gpuEnd, e.kernStart, e.kernEnd
	copy(r.logitsHost, r.logits.Floats())
	return r.logitsHost
}

// ForwardEmbPipe is ForwardEmb through the pipelined executor (encode-ahead) — the production
// decode path. Returns logits[V] (reused buffer; consume before the next call). Synchronous to
// the caller (one job in, one logits out), but the executor overlaps the next token's encode
// with this token's GPU execution.
func (r *Resident) ForwardEmbPipe(emb []float32, pos int) []float32 {
	r.execOnce.Do(func() {
		r.execReq = make(chan execJob)
		r.execAck = make(chan []float32)
		r.execDone = make(chan struct{})
		go r.execLoop()
	})
	r.execReq <- execJob{emb: emb, pos: pos}
	return <-r.execAck
}

// encodeLogitsCB builds a complete, un-committed command buffer (trunk + full lm head) with no
// per-call pool — it autoreleases into the executor's long-lived pool.
func (r *Resident) encodeLogitsCB() *Encoder {
	e := r.q.beginNP()
	r.encodeTrunkInto(e)
	e.dispatch(r.pGemvW8, (r.V)*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.logits, r.uH)
	e.finishEncoding()
	return e
}

// execLoop is the pinned executor: pipeline commit(t) → pre-encode(t+1) → wait(t). One shared
// autorelease pool, drained every drainEvery tokens (with a one-token non-overlapped hiccup so
// no un-committed command buffer is live across the drain — keeps the pool LIFO-safe).
func (r *Resident) execLoop() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	defer close(r.execDone) // Close waits here: no command buffer is live once this returns
	const drainEvery = 64
	pool := newARPool()
	var cur *Encoder
	count := 0
	for job := range r.execReq {
		if cur == nil {
			cur = r.encodeLogitsCB()
		}
		copy(r.x.Floats(), job.emb) // this token's embedding + pos (set at commit time, not encode)
		r.uPos.SetU32(uint32(job.pos))
		r.uNKeys.SetU32(uint32(job.pos + 1))
		cur.commit()

		count++
		drain := count%drainEvery == 0
		var next *Encoder
		if !drain {
			next = r.encodeLogitsCB() // overlaps cur's GPU execution — the encode-ahead win
		}
		cur.waitDone()
		r.gpuStart, r.gpuEnd, r.kernStart, r.kernEnd = cur.gpuStart, cur.gpuEnd, cur.kernStart, cur.kernEnd
		copy(r.logitsHost, r.logits.Floats())

		if drain { // no un-committed cb live now → safe to drain the shared pool
			pool.drain()
			pool = newARPool()
			cur = nil
		} else {
			cur = next
		}
		r.execAck <- r.logitsHost
	}
	pool.drain()
}

// stopExec shuts down the executor goroutine (if started) and BLOCKS until it has returned —
// which is what makes freeing safe. Closing the channel only signals; the loop may still be
// waiting on an in-flight command buffer, and releasing a buffer it references is a
// use-after-free. (CUDA hit the mirror-image ordering constraint in d8e81cb.)
func (r *Resident) stopExec() {
	if r.execReq != nil {
		close(r.execReq)
		r.execReq = nil
		<-r.execDone
	}
}

// Close stops the executor, waits for it, then releases every MTLBuffer this resident allocated.
//
// It used to do only the first of those, with the comment "Metal buffers are freed at process
// exit (single-model lifetime)". That assumption was false and had teeth: cmd/serve is
// multi-model (--model name=path is repeatable) with /admin/models/{load,unload}, so every
// load+unload leaked the whole model — weights + per-layer KV + the MoE stacked experts, i.e.
// GIGABYTES of unified (system) memory, until the process exited. purego has no ARC and Metal
// has no context-destroy to reclaim in bulk, so each buffer must be released explicitly.
// Idempotent: ReleaseAll empties the ledger, so a second Close is a no-op.
func (r *Resident) Close() {
	r.stopExec()
	if r.d != nil {
		r.d.ReleaseAll()
	}
}

// LastGPUTimes returns, for the last Forward, the GPU-busy window and the kernel window
// (incl scheduling) in seconds — GPUEnd-GPUStart and kernelEnd-kernelStart from the command
// buffer. wall - gpuBusy is the per-token host bubble (Step 0 of the headroom decision tree).
func (r *Resident) LastGPUTimes() (gpuBusy, kernTotal float64) {
	return r.gpuEnd - r.gpuStart, r.kernEnd - r.kernStart
}

// ForwardArgmax runs the identical trunk but replaces the full lm head + 608KB readback with
// Fable's fused block-argmax (per-tile (maxLogit,rowIdx) → argmax_finish → 4-byte token). It
// returns argmax(Forward's logits) — same values, tie-broken first-max-wins — without ever
// materializing the logit vector. This is the fastest greedy decode path.
func (r *Resident) ForwardArgmax(id, pos int) uint32 {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	r.embed.Row(id, r.x.Floats())
	r.uPos.SetU32(uint32(pos))
	r.uNKeys.SetU32(uint32(pos + 1))
	e := r.q.begin()
	r.encodeTrunkInto(e)
	e.dispatch(r.pGemvW8Amax, (r.V)*32, 256, r.aq, r.aSc, r.lmW, r.lmS, r.part, r.uH) // tile partials (int8 head)
	e.dispatch(r.pArgFinish, 256, 256, r.part, r.tok, r.uP)                           // reduce tiles → token
	e.end()
	return r.tok.U32()
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
func (r *Resident) forwardTrunkForTest(emb []float32, pos, nLayers int) []float32 {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	saved := r.nL
	r.nL = nLayers
	defer func() { r.nL = saved }()
	copy(r.x.Floats(), emb)
	r.uPos.SetU32(uint32(pos))
	r.uNKeys.SetU32(uint32(pos + 1))
	e := r.q.begin()
	r.encodeTrunkInto(e)
	e.end()
	return append([]float32(nil), r.x.Floats()...)
}

// forwardHeadForTest runs the FULL trunk (final norm included) then the LM head, and returns
// both the head's INPUT activation as it actually sees it — r.aq dequantized by r.aSc, i.e. the
// int8-quantized final-norm output — and the logits. It splits the one step the per-layer bisect
// cannot: final-norm-quant vs the head matmul. Comparing the returned activation to the CPU's
// f32 final-norm output isolates the quantization of the head's input; comparing the logits
// isolates the matmul on top of it.
func (r *Resident) forwardHeadForTest(emb []float32, pos int) (act, logits []float32) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	copy(r.x.Floats(), emb)
	r.uPos.SetU32(uint32(pos))
	r.uNKeys.SetU32(uint32(pos + 1))
	e := r.q.begin()
	r.encodeTrunkInto(e)
	e.dispatch(r.pGemvW8, (r.V)*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.logits, r.uH)
	e.end()
	q, sc := r.aq.Int8s(), r.aSc.Floats()[0]
	act = make([]float32, r.H)
	for i := range act {
		act[i] = float32(q[i]) * sc
	}
	return act, append([]float32(nil), r.logits.Floats()...)
}

// encodeTrunkInto encodes all decoder layers + the final norm into e, leaving the quantized
// final hidden state in r.aq/r.aSc ready for an lm head. It ONLY records dispatches (referencing
// the shared buffers) — it does NOT set uPos/uNKeys or fill r.x; the caller sets those before
// commit. This value-independence is what lets the executor pre-encode token t+1 while token t
// runs (encode-ahead).
func (r *Resident) encodeTrunkInto(e *Encoder) {
	nHhd := r.nH * r.hd
	qkvRows := nHhd + 2*r.kvDim
	kOff, vOff := nHhd*4, (nHhd+r.kvDim)*4 // byte offsets of k, v within the fused qkv buffer
	for l := 0; l < r.nL; l++ {
		L := &r.layers[l]
		// --- attention block (12 dispatches vs 19: QKV fused +bias, residual fused) ---
		e.dispatch(r.pRms, 256, 256, r.x, L.preNorm, r.aq, r.aSc, r.uH, r.uEps, r.uAddOne)
		e.dispatchTG(r.pSABias, qkvRows*32, 256, r.H*2, L.qkvW, L.qkvS, r.aq, r.aSc, r.qkv, L.qkvBias, r.uH)
		if r.qkNorm { // Qwen3: per-head Q/K RMSNorm before RoPE
			e.dispatch(r.pQKNorm, (r.nH+r.nKV)*128, 128, r.qkv, L.qNorm, L.kNorm, r.uNH, r.uNKV, r.uHd, r.uNHhd, r.uEps, r.uAddOne)
		}
		e.dispatch(r.pRope, r.nH*r.half, 64, r.qkv, L.invf, r.uHd, r.uPos, r.uQtotal, r.uHalf)           // q @ off 0
		e.dispatch(r.pRope, r.nKV*r.half, 64, r.qkv.At(kOff), L.invf, r.uHd, r.uPos, r.uKtotal, r.uHalf) // k
		e.dispatch(r.pKv, r.kvDim, 64, r.qkv.At(kOff), r.qkv.At(vOff), r.kc[l], r.vc[l], r.uKvDim, r.uPos)
		e.dispatch(r.pAttn, r.nH*128, 128, r.qkv, r.kc[l], r.vc[l], r.ctx, r.uNH, r.uNKV, r.uHd, r.uNKeys, r.uScale, L.uWindow)
		e.dispatch(r.pQv, 256, 256, r.ctx, r.cq, r.cSc, r.uHH)
		if r.sandwich {
			// Gemma: the sublayer OUTPUT is normed BEFORE the residual add, which the fused
			// _resid epilogue can't express — project into the (otherwise dead) oO scratch,
			// norm it, then add. Three dispatches instead of one.
			e.dispatchTG(r.pSA, r.H*32, 256, r.nH*r.hd*2, L.oW, L.oS, r.cq, r.cSc, r.oO, r.uHH)
			e.dispatch(r.pRmsF32, 256, 256, r.oO, L.postAttnNorm, r.uH, r.uEps, r.uAddOne)
			e.dispatch(r.pRes, r.H, 256, r.x, r.oO)
		} else {
			e.dispatchTG(r.pSAResid, r.H*32, 256, r.nH*r.hd*2, L.oW, L.oS, r.cq, r.cSc, r.x, r.uHH) // o-proj + residual
		}
		// --- ffn block (dense SwiGLU/GeGLU, or MoE router + experts + shared) ---
		if L.moe != nil {
			r.encodeMoEFFN(e, L)
		} else {
			e.dispatch(r.pRms, 256, 256, r.x, L.postNorm, r.mq, r.mSc, r.uH, r.uEps, r.uAddOne)
			e.dispatchTG(r.pSA, (2*r.I)*32, 256, r.H*2, L.guW, L.guS, r.mq, r.mSc, r.gu, r.uH) // fused gate|up
			e.dispatch(r.pSw, 256, 256, r.gu, r.gu.At(r.I*4), r.dq, r.dSc, r.uI, r.uAct)       // gate @0, up @I
			if r.sandwich {
				e.dispatch(r.pGemv, r.H*32, 32, L.dW, L.dS, r.dq, r.dSc, r.dO, r.uI) // down → scratch
				e.dispatch(r.pRmsF32, 256, 256, r.dO, L.postMLPNorm, r.uH, r.uEps, r.uAddOne)
				e.dispatch(r.pRes, r.H, 256, r.x, r.dO)
			} else {
				e.dispatch(r.pGemvResid, r.H*32, 32, L.dW, L.dS, r.dq, r.dSc, r.x, r.uI) // down + residual
			}
		}
	}
	e.dispatch(r.pRms, 256, 256, r.x, r.finalNorm, r.aq, r.aSc, r.uH, r.uEps, r.uAddOne)
}
