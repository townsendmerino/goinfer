//go:build darwin

// Real-model resident Metal decoder — the final GO/NO-GO piece of the spike. Loads a
// dense Qwen2/Llama model's int8 weights out of the goinfer decoder, uploads them to
// Metal once, and runs the full layer stack per token in ONE command buffer (the tax
// requirement). cgo-free throughout (purego-objc + dlopen Metal).
package metal

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

const metalCtxCap = 4096 // resident KV positions (spike)

// attnScoreKeyBound is the attention kernel's static score-buffer capacity: the
// `threadgroup float sc[4096]` in kernels.go holds one score per key, so the kernel
// is only correct for nKeys ≤ this. metalCtxCap MUST NOT exceed it (checkCap refuses
// nKeys > metalCtxCap, so the cap is what keeps the kernel in-bounds) — TestMetalCtxCapWithinKernelBound
// asserts the invariant, and TestAttention_ShippedKernelShapes measures correctness at the exact
// boundary (nKeys=4096). Bumping metalCtxCap past this without resizing sc[] is a silent OOB
// threadgroup write; on Metal's unified memory that corrupts adjacent buffers.
const attnScoreKeyBound = 4096

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
	qNorm, kNorm                         Buffer          // per-head QK-RMSNorm weights (Qwen3; zero if !qkNorm)
	moe                                  *moeLayer       // non-nil ⇒ this layer's FFN is MoE (dense guW/dW unused)
	g4moe                                *gemma4MoeLayer // non-nil ⇒ Gemma-4 parallel dense‖MoE FFN (gemma4_moe.go)
	postAttnNorm, postMLPNorm            Buffer          // Gemma sandwich norms on each sublayer OUTPUT (zero if !sandwich)
	invf                                 Buffer          // per-layer RoPE inv-freq (Gemma local 10k vs global 1M base)
	uWindow                              Buffer          // per-layer attention window (0 = full causal; Gemma mixes local/global)
	geom                                 *attnGeom       // per-layer attention geometry (hd/nKV/half/kEqV + uniforms); see geom.go
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
	kvF32                                                                 bool     // f32 KV cache (Gemma sandwich path) — f16 rounding craters Gemma's sensitive contexts
	uAct                                                                  Buffer   // gated-MLP activation ordinal (decoder.ActKind: 0=GELU-tanh, 1=SiLU)
	uAddOne, uWindow                                                      Buffer   // qk_norm + sliding-window uniforms
	uZero                                                                 Buffer   // constant 0 (nH=0 / nHhd=0 / addOne=0 for the scale-less v_norm dispatch)
	vNormUnit                                                             Buffer   // [maxHd] of 1.0 — unit weight so qk_norm (x·rms·w, addOne=0) = scale-less v_norm for K=V layers

	qkv, gu Buffer // fused QKV out, fused gate/up out

	// Model-level (constant across a family's layers). Per-layer attention geometry —
	// hd/nKV/kvDim/half and their uniform buffers — lives on residLayer.geom (see geom.go);
	// it was REMOVED from here so a launch site cannot bind the uniform shape by mistake.
	H, nL, nH, I, V int
	finalSoftcap    float32 // Gemma final-logit softcap (30); 0 ⇒ none. Applied host-side in finalizeLogits (FeatFinalLogitSoftcap).
	embed           *linalg.WeightMat

	layers    []residLayer
	finalNorm Buffer
	lmW, lmS  Buffer
	kc, vc    []Buffer
	moe       *moeResident       // non-nil ⇒ MoE model (router + stacked experts); see moe.go
	g4moe     *gemma4MoeResident // non-nil ⇒ Gemma-4 enable_moe_block (parallel dense‖MoE); see gemma4_moe.go

	// prefillOK reports whether the f16 MMA prefill kernels (prefill.go) actually implement
	// this model's shape. They run a DENSE FFN out of L.guW/L.dW with a model-level rope +
	// window and a SiLU-only swiglu — so a model whose decode path diverges (MoE leaves the
	// dense FFN buffers unset entirely) MUST decline prefill and let the caller fall back to
	// the sequential Forward loop: correct, just a slower TTFT.
	prefillOK bool

	x, aq, aSc, ctx, cq, cSc, oO, mq, mSc, dq, dSc, dO, logits Buffer
	invf, uH, uI, uNH, uScale, uEps                            Buffer // invf = model-level rope (prefill only); geometry uniforms live on residLayer.geom
	uPos, uNKeys                                               Buffer
	part, tok, uP                                              Buffer // fused-argmax: tile partials, token out, tile count
	logitsHost                                                 []float32
	gpuStart, gpuEnd, kernStart, kernEnd                       float64      // last-Forward GPU timing (Step 0)
	prof                                                       pagedProfile // per-phase paging decomposition (accumulates; snapshot+diff over a timed window)
	residency                                                  ResidencySet // pinned working set (GOINFER_MOE_RESIDENCY, paged path); zero value if unused
	residencyBufs                                              []Buffer     // exactly the buffers added to `residency` (for the teardown-consistency gate)

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
	return d.NewBufferBytes(n)
}

func int8Buf(d *Device, w *linalg.WeightMat) (Buffer, Buffer, error) {
	q8, sc, _, ok := w.Int8()
	if !ok {
		return Buffer{}, Buffer{}, fmt.Errorf("metal: weight kind %q is not int8 (load Options{Quant:\"int8int8\"})", w.Kind())
	}
	return d.NewBufferInt8(q8), d.NewBufferFloats(sc), nil
}

// int4DirectWords converts a decoder int4 WeightMat's packed nibbles + f32 group scales straight
// into Metal's W4A8 buffers (uint32 words + f16 scales) — NO int8 intermediate. aikit's group=32
// packing (nib = q+8, byte k/2 low/high) and Metal's packW4A8Row (element k → word k/8, bit
// 4·(k%8)) are the SAME bytes on little-endian, so the nibbles copy verbatim; only the group
// scales narrow f32→f16. Returns ok=false if the weight is not group-32 int4.
//
// This is the fix for Gemma's dormant residual: BuildResident's default path double-quantizes
// (f32→int8→int4), and Gemma's low-magnitude attention contexts amplify that int8-intermediate
// drift into a catastrophic context error (metal/gemma_sublayer_test.go: L1 cosine craters to
// 0.649 vs the direct-int4 reference's ~0.93). Consuming the decoder's int4 directly — exactly
// what CUDA/WebGPU do — removes the int8 step. Qwen is insensitive to it (ships clean either way).
func int4DirectWords(w *linalg.WeightMat) (words []uint32, scales []uint16, ok bool) {
	q4, q4s, group, ok := w.Int4()
	if !ok || group != 32 {
		return nil, nil, false
	}
	words = bytesToU32(q4)
	scales = make([]uint16, len(q4s))
	for i, s := range q4s {
		scales[i] = f32ToF16(s)
	}
	return words, scales, true
}

// bytesToU32 reinterprets a little-endian byte slice as uint32 words (len must be a multiple of 4).
func bytesToU32(b []byte) []uint32 {
	w := make([]uint32, len(b)/4)
	for i := range w {
		w[i] = uint32(b[4*i]) | uint32(b[4*i+1])<<8 | uint32(b[4*i+2])<<16 | uint32(b[4*i+3])<<24
	}
	return w
}

// int4DirectBytes is int4DirectWords' zero-copy sibling for the paging hot path: it returns the
// packed nibble bytes ALIASED straight from the mmap (no bytesToU32 reconstruction, no per-stage
// []uint32 allocation) plus the f16 group scales. The nibble bytes are byte-for-byte the words
// int4DirectWords would build (little-endian), so a byte-copy into a uint32 slot buffer reproduces
// them exactly on LE — the paged forward's copy does exactly that (expertpool.copyBytesToU32Buf).
// Measured: bytesToU32 over the 26B's per-token staged nibbles is ~215 ms/1.9 GB of pure
// reconstruction + a 1.9 GB/run allocation, both removed here; the words path (int4DirectWords)
// stays for the one-time non-paged build where the []uint32 shape is wanted.
func int4DirectBytes(w *linalg.WeightMat) (q4 []byte, scales []uint16, ok bool) {
	b, q4s, group, ok := w.Int4()
	if !ok || group != 32 {
		return nil, nil, false
	}
	scales = make([]uint16, len(q4s))
	for i, s := range q4s {
		scales[i] = f32ToF16(s)
	}
	return b, scales, true
}

// int4Buf uploads a WeightMat as W4A8 (int4, group=32) + f16 group scales. If the weight is
// ALREADY int4 (a Quant:"int4" load), it consumes the nibbles directly (int4-direct, no int8
// step); otherwise it re-quantizes the int8 weight through the validated packer. One-time at build.
func int4Buf(d *Device, w *linalg.WeightMat) (Buffer, Buffer, error) {
	if words, scales, ok := int4DirectWords(w); ok {
		return d.NewBufferUint32s(words), d.NewBufferU16s(scales), nil
	}
	q8, sc, _, ok := w.Int8()
	if !ok {
		return Buffer{}, Buffer{}, fmt.Errorf("metal: weight kind %q is not int8 or int4", w.Kind())
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
		if dw, ds, ok := int4DirectWords(w); ok { // int4-direct: consume nibbles verbatim
			words = append(words, dw...)
			scales = append(scales, ds...)
			continue
		}
		q8, sc, _, ok := w.Int8()
		if !ok {
			panic(fmt.Sprintf("metal: int4Concat weight kind %q not int8 or int4", w.Kind()))
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
	// M24(c): pin the thread and hold ONE autorelease pool for the whole build — CompileLibrary,
	// NewComputePipeline, and every NewBuffer*/nsString create autoreleased temporaries that would
	// otherwise leak on this unpinned, pool-less thread.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	pool := NewARPool()
	defer pool.Drain()
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		return nil, err
	}
	// M24(a): every buffer + objc object below lands on d's ledgers. On ANY early return or panic
	// (a sandwich shape-check error, a buildMoE/int4/int8-kind error, or a mustBuf/pipe OOM panic
	// that backend.go recovers into a clean CPU decline), release it all — otherwise a declined
	// build leaks gigabytes while serve continues on CPU. Cleared once construction completes.
	ok := false
	defer func() {
		if !ok {
			d.ReleaseAll()
			d.ReleaseObjects()
		}
	}()
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
	H, nL, nH, _, _, I, V := m.Dims() // model-level hd/nKV dropped — geometry is per-layer (geom.go)
	r := &Resident{d: d, H: H, nL: nL, nH: nH, I: I, V: V}
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
	// f32 KV path (kv_store_f32/attention_f32) exists but is DISABLED: the matched-input confirmer
	// proved f16-vs-f32 KV storage is NOT the Gemma crater. The whole crater is Metal's BOS
	// (position-0) K/V being computed wrong (cos 0.40 vs goinfer); overwriting just that recovers
	// the context to 0.999. So precision was a red herring (mine and the CUDA box's) — the real
	// bug is the position-0/attention-sink K/V compute. Kept off until that's fixed.
	r.kvF32 = false
	if r.kvF32 {
		r.pKv, r.pAttn = pipe("kv_store_f32"), pipe("attention_f32")
	}
	// Gated-MLP activation: ordinals ARE decoder.ActKind's iota (0=GELU-tanh, 1=SiLU), so this
	// passes straight through to glu_act. Gemma is GeGLU; everything else admitted is SwiGLU.
	r.uAct = d.NewBufferU32(uint32(m.GatedActResident()))
	addOne := uint32(0)
	if m.RMSAddOne() {
		addOne = 1
	}
	r.uAddOne = d.NewBufferU32(addOne)
	win := m.SlidingWindowResident() // 0 = full causal; Mistral is all-local with this window
	if win < 0 {
		win = 0
	}
	r.uWindow = d.NewBufferU32(uint32(win))
	if r.moe, err = buildMoE(d, m, pipe, H); err != nil { // nil for a dense (or gemma4-MoE) model; error ⇒ decline
		return nil, err
	}
	if r.g4moe, err = buildGemma4MoE(d, m, pipe, H, nL); err != nil { // Gemma-4 enable_moe_block; nil otherwise; error ⇒ decline
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
	// Per-layer attention geometry, deduped by value (geom.go). Uniform families resolve every
	// layer to one geom; Gemma 4's local/global interleave resolves to two. maxNHhd/maxKvDim size
	// the shared per-token scratch to the widest layer (== the model shape for a uniform family).
	geomCache := map[[4]int]*attnGeom{}
	var maxNHhd, maxKvDim, maxHd int
	for l := range nL {
		lw := &w.Layers[l]
		var L residLayer
		// K=V (attention_k_eq_v, Gemma 4 global layers): NO v_proj — V = v_norm(the RAW k_proj
		// output). Fuse the V slot with k_proj (so the fused proj yields qkv=[Q|K_raw|K_raw]); the
		// V slot is then scale-less-v_norm'd and left un-roped at encode time (encodeTrunkInto),
		// while the K slot gets k_norm+RoPE. This is a BUILD-TIME weight-layout difference, so the
		// value-independent ForwardEmbPipe pre-encode stays correct — see geom.kEqV.
		kEqV := m.VFromKResident(l)
		if kEqV {
			L.qkvW, L.qkvS = int4Concat(d, &lw.QProj, &lw.KProj, &lw.KProj) // V slot = raw k_proj
		} else {
			L.qkvW, L.qkvS = int4Concat(d, &lw.QProj, &lw.KProj, &lw.VProj) // fused QKV
		}
		L.oW, L.oS = mk(&lw.OProj)
		g4b, isG4MoE := decoder.Gemma4MoEResidentBundle{}, false
		if r.g4moe != nil {
			g4b, isG4MoE = m.Gemma4MoEResidentLayer(l)
		}
		switch {
		case isG4MoE: // Gemma-4 enable_moe_block: parallel dense‖MoE FFN (its own norms/router/experts)
			L.g4moe = buildGemma4MoELayer(d, m, &g4b, r.g4moe)
		case r.moe != nil && len(lw.Experts) > 0: // generic MoE layer: stacked experts instead of a dense FFN
			L.moe = buildMoELayer(d, lw, r.moe)
		default: // dense FFN (also GLM/DeepSeek's FirstKDense prefix layers, and gemma4 dense layers)
			L.guW, L.guS = int4Concat(d, &lw.GateProj, &lw.UpProj) // fused gate/up
			L.dW, L.dS = mk(&lw.DownProj)
		}
		L.preNorm = d.NewBufferFloats(lw.PreAttnNorm)
		if L.g4moe == nil { // dense/generic FFN entry norm (PreMLPNorm); g4moe carries its five norms in the bundle
			L.postNorm = d.NewBufferFloats(lw.PreMLPNorm)
		}
		if r.qkNorm { // Qwen3 per-head Q/K norm weights [hd]
			L.qNorm, L.kNorm = d.NewBufferFloats(lw.QNorm), d.NewBufferFloats(lw.KNorm)
		}
		// Gemma sandwich norms on each sublayer OUTPUT. Required to be present when the arch
		// declares them — a silently-missing one would DROP the norm rather than error. A g4moe
		// layer still runs the ATTENTION sandwich (postAttnNorm) but its FFN block is the parallel
		// dense‖MoE, which carries its own post-norms (postFFN1/postFFN2/postFFN) in the bundle and
		// never touches postMLPNorm — exactly as decoder/forward_gemma4.go skips PostMLPNorm for
		// enable_moe_block layers. So require/build postMLPNorm only for the non-g4moe FFN path.
		if r.sandwich {
			if len(lw.PostAttnNorm) != H {
				return nil, fmt.Errorf("metal: layer %d declares sandwich norms but PostAttnNorm is not len==hidden(%d) (got %d)",
					l, H, len(lw.PostAttnNorm))
			}
			L.postAttnNorm = d.NewBufferFloats(lw.PostAttnNorm)
			if L.g4moe == nil {
				if len(lw.PostMLPNorm) != H {
					return nil, fmt.Errorf("metal: layer %d declares sandwich norms but PostMLPNorm is not len==hidden(%d) (got %d)",
						l, H, len(lw.PostMLPNorm))
				}
				L.postMLPNorm = d.NewBufferFloats(lw.PostMLPNorm)
			}
		}
		// Per-layer RoPE table (Gemma local 10k vs global 1M base) and per-layer window. The rope
		// KERNEL is unchanged — all per-layer variation rides in the table contents AND the width
		// (geom.half). RopeInvFreqLayerResident is the resident per-layer table (== the generic
		// RopeInvFreqLayer for every uniform family; Gemma 4's global proportional-rotary tail is
		// what makes it differ). Its length is this layer's rhalf (rotated pairs/head).
		invf := m.RopeInvFreqLayerResident(l)
		L.invf = d.NewBufferFloats(invf)
		// Per-layer attention geometry — the whole point of the 9c seam. Uniform families resolve
		// every layer to the same {hd, nKV, half, kEqV}; Gemma 4 varies hd/nKV/kEqV between its
		// local and global layers. geomFor dedups by value so a uniform model shares one geom.
		L.geom = r.geomFor(geomCache, m.HeadDimAtResident(l), m.KVHeadsAtResident(l), len(invf), kEqV)
		if L.geom.hd > maxHd {
			maxHd = L.geom.hd
		}
		if L.geom.kvDim > maxKvDim {
			maxKvDim = L.geom.kvDim
		}
		if r.nH*L.geom.hd > maxNHhd {
			maxNHhd = r.nH * L.geom.hd
		}
		lw2 := uint32(0) // 0 = full causal; only local layers carry the window
		if m.LayerIsLocalResident(l) {
			lw2 = uint32(win)
		}
		L.uWindow = d.NewBufferU32(lw2)
		// combined qkv bias (zeros where absent) so the fused GEMV epilogue is uniform.
		qb, kb, vb := lw.QBias, lw.KBias, lw.VBias
		if qb == nil {
			qb, kb, vb = make([]float32, r.nH*L.geom.hd), make([]float32, L.geom.kvDim), make([]float32, L.geom.kvDim)
		}
		L.qkvBias = d.NewBufferFloats(append(append(append([]float32{}, qb...), kb...), vb...))
		r.layers[l] = L
		kvBytes := metalCtxCap * L.geom.kvDim * 2 // f16 KV: 2 bytes/elem (halves the cache)
		if r.kvF32 {
			kvBytes = metalCtxCap * L.geom.kvDim * 4 // Gemma: f32 KV — see r.kvF32
		}
		r.kc[l] = byteBuf(d, kvBytes)
		r.vc[l] = byteBuf(d, kvBytes)
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
	if r.g4moe != nil { // Gemma-4 dense‖MoE: the gate|up/down scratch must fit BOTH the dense branch and the experts
		guDim = max(guDim, max(r.g4moe.denseInter, r.g4moe.moeInter))
	}
	r.x = d.NewBufferLen(H)
	r.aq, r.aSc = byteBuf(d, H), d.NewBufferLen(1)
	r.qkv = d.NewBufferLen(maxNHhd + 2*maxKvDim) // fused [q | k | v], sized to the widest layer
	r.gu = d.NewBufferLen(2 * guDim)             // fused [gate | up]
	r.ctx, r.cq, r.cSc = d.NewBufferLen(maxNHhd), byteBuf(d, maxNHhd), d.NewBufferLen(1)
	r.oO, r.mq, r.mSc = d.NewBufferLen(H), byteBuf(d, H), d.NewBufferLen(1)
	r.dq, r.dSc, r.dO = byteBuf(d, guDim), d.NewBufferLen(1), d.NewBufferLen(H)
	r.logits = d.NewBufferLen(V)
	nTiles := V / 8 // one (maxLogit,rowIdx) partial per threadgroup (8 rows) — V divisible by 8
	r.part, r.tok, r.uP = d.NewBufferLen(nTiles*2), d.NewBufferLen(1), d.NewBufferU32(uint32(nTiles))
	// Model-level rope table for the (uniform-only) prefill path; decode uses each layer's L.invf.
	r.invf = d.NewBufferFloats(m.RopeInvFreq())
	r.uH, r.uI = d.NewBufferU32(uint32(H)), d.NewBufferU32(uint32(I))
	r.uNH = d.NewBufferU32(uint32(nH)) // query heads: constant across a family, so model-level
	r.uScale, r.uEps = d.NewBufferFloats([]float32{m.AttnScale()}), d.NewBufferFloats([]float32{m.NormEps()})
	r.uPos, r.uNKeys = d.NewBufferU32(0), d.NewBufferU32(1)
	// Scale-less v_norm plumbing for K=V layers (Gemma 4 globals): a unit-1.0 weight and a shared
	// zero (nH=0 / nHhd=0 / addOne=0) so qk_norm reduces to x·rms·1 on the V slot. Sized to the
	// widest head; harmless (a few KB) for models with no K=V layer.
	r.uZero = d.NewBufferU32(0)
	ones := make([]float32, maxHd)
	for i := range ones {
		ones[i] = 1
	}
	r.vNormUnit = d.NewBufferFloats(ones)
	r.finalSoftcap = m.FinalLogitSoftcapResident() // Gemma 4: 30 (host-side softcap); 0 for every other family
	r.logitsHost = make([]float32, V)

	// Residency set (default ON when supported + paged; GOINFER_MOE_RESIDENCY=0 opts out). The paged
	// path submits per-layer, and the pread stage CPU-writes the slot buffers each token — dirtying
	// their residency so the driver re-validates them every phase-2 commit (~9 ms/CB of GPU-idle-in-
	// wait). Pinning the SLOT POOL resident holds it across those writes → ~0.44 ms/CB (measured
	// −11%: 0.61→0.68 tok/s at N=32 on an idle 16 GB box). SLOTS ONLY: a five-arm bisect showed
	// pinning anything more (weights/KV/scratch) regresses phase 1 in proportion to pin-set size,
	// read/write-agnostic — pinning helps only the pread-INVALIDATED buffers. FOOTPRINT: the slot pool
	// is N × MoE-layers × per-expert-bytes (≈3 GB at N=32), PERMANENTLY requested resident; re-measure
	// the win if N grows or the box is under other load. Capability-gated (macOS 15+); older OSes keep
	// the correct, slower per-submit path.
	if r.g4moe != nil && r.g4moe.paged && os.Getenv("GOINFER_MOE_RESIDENCY") != "0" && ResidencySetsSupported() {
		rs, rerr := d.NewResidencySet()
		if rerr != nil {
			fmt.Fprintf(os.Stderr, "metal: residency set unavailable (%v) — per-submit validation stands\n", rerr)
		} else {
			slots := r.slotBuffers()      // GPU-read-only (CPU-written by pread)
			written := r.writtenBuffers() // GPU-WRITTEN per token: KV cache + intermediates/scratch
			kv := r.kvBuffers()
			scratch := r.scratchBuffers()
			addAll := func(bs []Buffer) {
				for _, b := range bs {
					rs.Add(b)
				}
			}
			switch os.Getenv("GOINFER_MOE_RESIDENCY_SCOPE") {
			case "slots":
				addAll(slots)
			case "slots+kv": // bisect: does the KV cache cause the phase-1 regression?
				addAll(slots)
				addAll(kv)
			case "slots+scratch": // bisect: do the intermediates cause it?
				addAll(slots)
				addAll(scratch)
			case "readonly", "slots+weights": // diagnostic: pin all buffers EXCEPT the written set
				rs.AddAllDeviceBuffersExcept(d, written)
			case "all": // diagnostic: pin every device buffer (regresses phase 1 — set-size overhead)
				rs.AddAllDeviceBuffers(d)
			default: // SHIP DEFAULT: slots only. The five-arm bisect showed pinning anything BEYOND the
				// pread-invalidated slot buffers regresses phase 1 in proportion to the pinned set size
				// (read/write-agnostic), so slots-only is the sole net win (phase 2 idle 9→0.44 ms/CB).
				addAll(slots)
				r.residencyBufs = slots
				_ = kv
				_ = scratch
			}
			rs.Commit()
			rs.RequestResidency()
			// KNOWN COST (cold A/B): attaching the set at the QUEUE rides it on EVERY command buffer,
			// so phase 1 now carries the ~3 GB of pinned slots in its referenced set even though it
			// never touches them — +2.07 ms/CB → +62 ms/tok (p1 idle scales with pin-set size; see the
			// bisect). Net is still −189 ms (p2 −248), so it ships. FIX (fold into the next aikit
			// release, not worth a release alone): a PHASE-SCOPED residency set attached only to
			// phase-2's command buffers (per-encoder useResidencySet) recovers the 62 ms.
			r.q.AddResidencySet(rs)
			r.residency = rs
		}
	}

	ok = true // construction complete — the Resident owns everything; Close (not the defer) frees it
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
	if r.g4moe != nil && r.g4moe.paged { // synchronous expert paging: per-layer submit+wait, staged experts
		return r.forwardLogitsPaged(pos)
	}
	return r.forwardLogits(pos)
}

// forwardLogits encodes the trunk + full lm head and reads back logits[V]. Caller must hold
// the OS thread and have filled r.x with the input embedding.
func (r *Resident) forwardLogits(pos int) []float32 {
	r.uPos.SetU32(uint32(pos))
	r.uNKeys.SetU32(uint32(pos + 1))
	e := r.q.Begin()
	r.encodeTrunkInto(e)                                                           // 28 layers → final norm → r.aq/r.aSc
	e.Dispatch(r.pGemvW8, (r.V)*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.logits, r.uH) // full lm head (int8 — logit-critical)
	e.End()
	r.gpuStart, r.gpuEnd, r.kernStart, r.kernEnd = e.GPUStart(), e.GPUEnd(), e.KernStart(), e.KernEnd()
	r.finalizeLogits()
	return r.logitsHost
}

// finalizeLogits copies the device logits into the host buffer and applies Gemma's final-logit
// softcap host-side (softcap·tanh(logits/softcap)) — the single site FeatFinalLogitSoftcap
// declares on this backend, exactly as the CPU path (logitsFromHidden) and CUDA (resident.go)
// do it. finalSoftcap is 0 for every non-softcapped family (no-op). The softcap is MONOTONIC, so
// ForwardArgmax's on-device argmax needs it not — only the full-logits paths (sampling,
// temperature, parity) do, and both readback sites (forwardLogits, execLoop) route through here.
func (r *Resident) finalizeLogits() {
	copy(r.logitsHost, r.logits.Floats())
	if r.finalSoftcap > 0 {
		sc := r.finalSoftcap
		for j, v := range r.logitsHost {
			r.logitsHost[j] = sc * float32(math.Tanh(float64(v/sc)))
		}
	}
}

// ForwardEmbPipe is ForwardEmb through the pipelined executor (encode-ahead) — the production
// decode path. Returns logits[V] (reused buffer; consume before the next call). Synchronous to
// the caller (one job in, one logits out), but the executor overlaps the next token's encode
// with this token's GPU execution.
func (r *Resident) ForwardEmbPipe(emb []float32, pos int) []float32 {
	if r.g4moe != nil && r.g4moe.paged {
		// Paging tears each MoE layer into two submits with a host readback between — the encode-ahead
		// executor (one static command buffer/token) cannot express it. Fall back to the synchronous
		// paged path; there is no pipelining to lose (Step-0: paging is submit-bound, not encode-bound).
		return r.ForwardEmb(emb, pos)
	}
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
	e := r.q.BeginNP()
	r.encodeTrunkInto(e)
	e.Dispatch(r.pGemvW8, (r.V)*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.logits, r.uH)
	e.FinishEncoding()
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
	pool := NewARPool()
	var cur *Encoder
	count := 0
	for job := range r.execReq {
		if cur == nil {
			cur = r.encodeLogitsCB()
		}
		copy(r.x.Floats(), job.emb) // this token's embedding + pos (set at commit time, not encode)
		r.uPos.SetU32(uint32(job.pos))
		r.uNKeys.SetU32(uint32(job.pos + 1))
		cur.Commit()

		count++
		drain := count%drainEvery == 0
		var next *Encoder
		if !drain {
			next = r.encodeLogitsCB() // overlaps cur's GPU execution — the encode-ahead win
		}
		cur.WaitDone()
		r.gpuStart, r.gpuEnd, r.kernStart, r.kernEnd = cur.GPUStart(), cur.GPUEnd(), cur.KernStart(), cur.KernEnd()
		r.finalizeLogits()

		if drain { // no un-committed cb live now → safe to drain the shared pool
			pool.Drain()
			pool = NewARPool()
			cur = nil
		} else {
			cur = next
		}
		r.execAck <- r.logitsHost
	}
	pool.Drain()
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
// slotBuffers returns the paged MoE slot-pool buffers — GPU-READ-ONLY (phase 2 reads them; the pread
// stage CPU-writes their contents). Safe to pin resident.
func (r *Resident) slotBuffers() []Buffer {
	var out []Buffer
	for l := range r.layers {
		if p := r.layers[l].g4moe; p != nil && p.pool != nil {
			for s := range p.pool.slots {
				out = append(out, p.pool.slots[s].guW, p.pool.slots[s].guS, p.pool.slots[s].dW, p.pool.slots[s].dS)
			}
		}
	}
	return out
}

// kvBuffers returns the per-layer KV cache buffers — GPU-WRITTEN by attention every token.
func (r *Resident) kvBuffers() []Buffer {
	out := make([]Buffer, 0, len(r.kc)+len(r.vc))
	out = append(out, r.kc...)
	out = append(out, r.vc...)
	return out
}

// scratchBuffers returns the per-token GPU-WRITTEN intermediates (attention/MLP staging, MoE scratch,
// logits, and the per-token uniform buffers). NOT the KV cache (see kvBuffers).
func (r *Resident) scratchBuffers() []Buffer {
	out := []Buffer{
		r.x, r.aq, r.aSc, r.ctx, r.cq, r.cSc, r.oO, r.mq, r.mSc, r.dq, r.dSc, r.dO,
		r.logits, r.qkv, r.gu, r.part, r.tok, r.uPos, r.uNKeys,
	}
	if g := r.g4moe; g != nil {
		out = append(out, g.rLogits, g.rIdx, g.rWgt, g.g4x1, g.g4x2, g.g4rn)
	}
	return out
}

// writtenBuffers is every GPU-written buffer (KV cache + scratch) — the set that MUST be excluded when
// pinning "all read-only" buffers (pinning a written member regresses the writing phase).
func (r *Resident) writtenBuffers() []Buffer {
	return append(r.kvBuffers(), r.scratchBuffers()...)
}

func (r *Resident) Close() {
	r.stopExec()
	if r.g4moe != nil && r.g4moe.giwFile != nil {
		_ = r.g4moe.giwFile.Close() // the pread-staging fd (GOINFER_MOE_PREAD)
		r.g4moe.giwFile = nil
	}
	if r.d != nil {
		r.d.ReleaseAll()     // every MTLBuffer
		r.d.ReleaseObjects() // command queue, ~40 pipelines, 1-2 libraries, and the MTLDevice (M24b)
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
	e := r.q.Begin()
	r.encodeTrunkInto(e)
	e.Dispatch(r.pGemvW8Amax, (r.V)*32, 256, r.aq, r.aSc, r.lmW, r.lmS, r.part, r.uH) // tile partials (int8 head)
	e.Dispatch(r.pArgFinish, 256, 256, r.part, r.tok, r.uP)                           // reduce tiles → token
	e.End()
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
func (r *Resident) forwardSubCaptureForTest(emb []float32, pos int) (attn, mlp, mlpPre, ctx, cqDeq [][]float32) {
	if !r.sandwich {
		return nil, nil, nil, nil, nil
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	copy(r.x.Floats(), emb)
	r.uPos.SetU32(uint32(pos))
	r.uNKeys.SetU32(uint32(pos + 1))
	grab := func() []float32 { return append([]float32(nil), r.oO.Floats()...) }
	grabD := func() []float32 { return append([]float32(nil), r.dO.Floats()...) }
	for l := 0; l < r.nL; l++ {
		L := &r.layers[l]
		g := L.geom
		nHhd := r.nH * g.hd
		qkvRows := nHhd + 2*g.kvDim
		kOff, vOff := nHhd*4, (nHhd+g.kvDim)*4
		e := r.q.Begin()
		e.Dispatch(r.pRms, 256, 256, r.x, L.preNorm, r.aq, r.aSc, r.uH, r.uEps, r.uAddOne)
		e.DispatchTG(r.pSABias, qkvRows*32, 256, r.H*2, L.qkvW, L.qkvS, r.aq, r.aSc, r.qkv, L.qkvBias, r.uH)
		if r.qkNorm {
			e.Dispatch(r.pQKNorm, (r.nH+g.nKV)*128, 128, r.qkv, L.qNorm, L.kNorm, r.uNH, g.uNKV, g.uHd, g.uNHhd, r.uEps, r.uAddOne)
		}
		e.Dispatch(r.pRope, r.nH*g.half, 64, r.qkv, L.invf, g.uHd, r.uPos, g.uQtotal, g.uHalf)
		e.Dispatch(r.pRope, g.nKV*g.half, 64, r.qkv.At(kOff), L.invf, g.uHd, r.uPos, g.uKtotal, g.uHalf)
		e.Dispatch(r.pKv, g.kvDim, 64, r.qkv.At(kOff), r.qkv.At(vOff), r.kc[l], r.vc[l], g.uKvDim, r.uPos)
		e.Dispatch(r.pAttn, r.nH*128, 128, r.qkv, r.kc[l], r.vc[l], r.ctx, r.uNH, g.uNKV, g.uHd, r.uNKeys, r.uScale, L.uWindow)
		e.Dispatch(r.pQv, 256, 256, r.ctx, r.cq, r.cSc, g.uNHhd)
		e.DispatchTG(r.pSA, r.H*32, 256, r.nH*g.hd*2, L.oW, L.oS, r.cq, r.cSc, r.oO, g.uNHhd)
		e.Dispatch(r.pRmsF32, 256, 256, r.oO, L.postAttnNorm, r.uH, r.uEps, r.uAddOne)
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
		e.Dispatch(r.pRms, 256, 256, r.x, L.postNorm, r.mq, r.mSc, r.uH, r.uEps, r.uAddOne)
		e.DispatchTG(r.pSA, (2*r.I)*32, 256, r.H*2, L.guW, L.guS, r.mq, r.mSc, r.gu, r.uH)
		e.Dispatch(r.pSw, 256, 256, r.gu, r.gu.At(r.I*4), r.dq, r.dSc, r.uI, r.uAct)
		e.Dispatch(r.pGemv, r.H*32, 32, L.dW, L.dS, r.dq, r.dSc, r.dO, r.uI)
		e.End()
		mlpPre = append(mlpPre, grabD()) // dO = down output BEFORE post-MLP sandwich norm (compute)

		e = r.q.Begin()
		e.Dispatch(r.pRmsF32, 256, 256, r.dO, L.postMLPNorm, r.uH, r.uEps, r.uAddOne)
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
func (r *Resident) l0GegluForTest(emb []float32, pos int) (gateUp, geglu8 []float32, gSc float32) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	copy(r.x.Floats(), emb)
	r.uPos.SetU32(uint32(pos))
	r.uNKeys.SetU32(uint32(pos + 1))
	L := &r.layers[0]
	g := L.geom
	nHhd := r.nH * g.hd
	qkvRows := nHhd + 2*g.kvDim
	kOff, vOff := nHhd*4, (nHhd+g.kvDim)*4
	e := r.q.Begin()
	e.Dispatch(r.pRms, 256, 256, r.x, L.preNorm, r.aq, r.aSc, r.uH, r.uEps, r.uAddOne)
	e.DispatchTG(r.pSABias, qkvRows*32, 256, r.H*2, L.qkvW, L.qkvS, r.aq, r.aSc, r.qkv, L.qkvBias, r.uH)
	if r.qkNorm {
		e.Dispatch(r.pQKNorm, (r.nH+g.nKV)*128, 128, r.qkv, L.qNorm, L.kNorm, r.uNH, g.uNKV, g.uHd, g.uNHhd, r.uEps, r.uAddOne)
	}
	e.Dispatch(r.pRope, r.nH*g.half, 64, r.qkv, L.invf, g.uHd, r.uPos, g.uQtotal, g.uHalf)
	e.Dispatch(r.pRope, g.nKV*g.half, 64, r.qkv.At(kOff), L.invf, g.uHd, r.uPos, g.uKtotal, g.uHalf)
	e.Dispatch(r.pKv, g.kvDim, 64, r.qkv.At(kOff), r.qkv.At(vOff), r.kc[0], r.vc[0], g.uKvDim, r.uPos)
	e.Dispatch(r.pAttn, r.nH*128, 128, r.qkv, r.kc[0], r.vc[0], r.ctx, r.uNH, g.uNKV, g.uHd, r.uNKeys, r.uScale, L.uWindow)
	e.Dispatch(r.pQv, 256, 256, r.ctx, r.cq, r.cSc, g.uNHhd)
	e.DispatchTG(r.pSA, r.H*32, 256, r.nH*g.hd*2, L.oW, L.oS, r.cq, r.cSc, r.oO, g.uNHhd)
	e.Dispatch(r.pRmsF32, 256, 256, r.oO, L.postAttnNorm, r.uH, r.uEps, r.uAddOne)
	e.Dispatch(r.pRes, r.H, 256, r.x, r.oO)
	e.Dispatch(r.pRms, 256, 256, r.x, L.postNorm, r.mq, r.mSc, r.uH, r.uEps, r.uAddOne)
	e.DispatchTG(r.pSA, (2*r.I)*32, 256, r.H*2, L.guW, L.guS, r.mq, r.mSc, r.gu, r.uH)
	e.Dispatch(r.pSw, 256, 256, r.gu, r.gu.At(r.I*4), r.dq, r.dSc, r.uI, r.uAct)
	e.End()
	gateUp = append([]float32(nil), r.gu.Floats()[:2*r.I]...)
	gSc = r.dSc.Floats()[0]
	dq8 := r.dq.Int8s()
	geglu8 = make([]float32, r.I)
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
func (r *Resident) attnConfirmForTest(resid, kHist, vHist []float32, layer, pos int, injectKV bool) []float32 {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	g := r.layers[layer].geom
	nHhd := r.nH * g.hd
	qkvRows := nHhd + 2*g.kvDim
	kOff, vOff := nHhd*4, (nHhd+g.kvDim)*4
	copy(r.x.Floats(), resid)
	r.uPos.SetU32(uint32(pos))
	r.uNKeys.SetU32(uint32(pos + 1))
	if injectKV {
		// Inject goinfer's post-RoPE K and raw V — matched KV history. f32 cache (Gemma) takes the
		// values verbatim; the f16 cache narrows them (still cos 1.0 for the CORRECT values — the
		// crater is walked-value drift, not storage of the right ones).
		if r.kvF32 {
			kc, vc := r.kc[layer].Floats(), r.vc[layer].Floats()
			copy(kc[:len(kHist)], kHist)
			copy(vc[:len(vHist)], vHist)
		} else {
			kc, vc := r.kc[layer].U16s(), r.vc[layer].U16s()
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
	e.Dispatch(r.pRms, 256, 256, r.x, L.preNorm, r.aq, r.aSc, r.uH, r.uEps, r.uAddOne)
	e.DispatchTG(r.pSABias, qkvRows*32, 256, r.H*2, L.qkvW, L.qkvS, r.aq, r.aSc, r.qkv, L.qkvBias, r.uH)
	if r.qkNorm { // Gemma3/Qwen3: per-head Q/K RMSNorm before RoPE — the injected K is post-QK-norm
		e.Dispatch(r.pQKNorm, (r.nH+g.nKV)*128, 128, r.qkv, L.qNorm, L.kNorm, r.uNH, g.uNKV, g.uHd, g.uNHhd, r.uEps, r.uAddOne)
	}
	e.Dispatch(r.pRope, r.nH*g.half, 64, r.qkv, L.invf, g.uHd, r.uPos, g.uQtotal, g.uHalf) // Q
	if !injectKV {
		// Use Metal's OWN walked KV history: RoPE K and store pos into the cache (isolates the
		// f16-KV drift of positions 0..pos-1 from Metal's walk, with a matched residual).
		e.Dispatch(r.pRope, g.nKV*g.half, 64, r.qkv.At(kOff), L.invf, g.uHd, r.uPos, g.uKtotal, g.uHalf)
		e.Dispatch(r.pKv, g.kvDim, 64, r.qkv.At(kOff), r.qkv.At(vOff), r.kc[layer], r.vc[layer], g.uKvDim, r.uPos)
	}
	e.Dispatch(r.pAttn, r.nH*128, 128, r.qkv, r.kc[layer], r.vc[layer], r.ctx, r.uNH, g.uNKV, g.uHd, r.uNKeys, r.uScale, L.uWindow)
	e.End()
	return append([]float32(nil), r.ctx.Floats()[:nHhd]...)
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

// encodeTrunkInto encodes all decoder layers + the final norm into e, leaving the quantized
// final hidden state in r.aq/r.aSc ready for an lm head. It ONLY records dispatches (referencing
// the shared buffers) — it does NOT set uPos/uNKeys or fill r.x; the caller sets those before
// commit. This value-independence is what lets the executor pre-encode token t+1 while token t
// runs (encode-ahead).
func (r *Resident) encodeTrunkInto(e *Encoder) {
	for l := 0; l < r.nL; l++ {
		r.encodeLayer(e, l)
	}
	e.Dispatch(r.pRms, 256, 256, r.x, r.finalNorm, r.aq, r.aSc, r.uH, r.uEps, r.uAddOne)
}

// encodeLayer encodes one decoder layer (attention block + FFN block) into e. Factored out of
// encodeTrunkInto so Step 6's expert-paging work has a per-layer SEAM: the pre-encode-cost
// measurement wraps each call in its own command buffer (the per-layer submit+wait regime), and the
// eventual paging path hangs its router-readback / expert-stage handshake here. Byte-identical to the
// old inline loop body (same dispatches, same order), so encodeTrunkInto stays the single-command-
// buffer, value-independent trunk it was.
func (r *Resident) encodeLayer(e *Encoder, l int) {
	L := &r.layers[l]
	r.encodeAttention(e, l)
	// --- ffn block (dense SwiGLU/GeGLU, generic MoE, or Gemma-4 parallel dense‖MoE) ---
	if L.g4moe != nil {
		r.encodeGemma4MoEFFN(e, L)
	} else if L.moe != nil {
		r.encodeMoEFFN(e, L)
	} else {
		e.Dispatch(r.pRms, 256, 256, r.x, L.postNorm, r.mq, r.mSc, r.uH, r.uEps, r.uAddOne)
		e.DispatchTG(r.pSA, (2*r.I)*32, 256, r.H*2, L.guW, L.guS, r.mq, r.mSc, r.gu, r.uH) // fused gate|up
		e.Dispatch(r.pSw, 256, 256, r.gu, r.gu.At(r.I*4), r.dq, r.dSc, r.uI, r.uAct)       // gate @0, up @I
		if r.sandwich {
			e.Dispatch(r.pGemv, r.H*32, 32, L.dW, L.dS, r.dq, r.dSc, r.dO, r.uI) // down → scratch
			e.Dispatch(r.pRmsF32, 256, 256, r.dO, L.postMLPNorm, r.uH, r.uEps, r.uAddOne)
			e.Dispatch(r.pRes, r.H, 256, r.x, r.dO)
		} else {
			e.Dispatch(r.pGemvResid, r.H*32, 32, L.dW, L.dS, r.dq, r.dSc, r.x, r.uI) // down + residual
		}
	}
}

// encodeAttention records one layer's attention block (through the o-proj + residual/sandwich norm).
// Split from encodeLayer so the paged Gemma-4 MoE forward can put [attention + dense + router] in one
// command buffer, submit+wait, read the router idx, stage experts, then encode [experts + join] in a
// second — the value-dependent seam paging forces. Byte-identical to the old inline attention block.
func (r *Resident) encodeAttention(e *Encoder, l int) {
	L := &r.layers[l]
	g := L.geom
	nHhd := r.nH * g.hd
	qkvRows := nHhd + 2*g.kvDim
	kOff, vOff := nHhd*4, (nHhd+g.kvDim)*4 // byte offsets of k, v within the fused qkv buffer
	// --- attention block (12 dispatches vs 19: QKV fused +bias, residual fused) ---
	e.Dispatch(r.pRms, 256, 256, r.x, L.preNorm, r.aq, r.aSc, r.uH, r.uEps, r.uAddOne)
	e.DispatchTG(r.pSABias, qkvRows*32, 256, r.H*2, L.qkvW, L.qkvS, r.aq, r.aSc, r.qkv, L.qkvBias, r.uH)
	if r.qkNorm { // Qwen3: per-head Q/K RMSNorm before RoPE
		e.Dispatch(r.pQKNorm, (r.nH+g.nKV)*128, 128, r.qkv, L.qNorm, L.kNorm, r.uNH, g.uNKV, g.uHd, g.uNHhd, r.uEps, r.uAddOne)
	}
	if g.kEqV {
		// K=V (Gemma 4 globals): the V slot holds the RAW k_proj output ([Q|K|K] fusion). Apply
		// scale-less v_norm to it — qk_norm over the V slot (qkv.At(vOff)) with nH=0 so every
		// head takes the K branch at base 0+head*hd, a UNIT weight, and addOne=0 → x·rms·1. Runs
		// AFTER qk_norm (which touched the K slot, not V) and BEFORE RoPE (which never touches V),
		// so V = v_norm(raw k), un-rotated — exactly the CPU path (copy(v,k) pre-k_norm; then
		// rmsNormNoWeight(v)). See TestVNorm_scaleless.
		e.Dispatch(r.pQKNorm, g.nKV*128, 128, r.qkv.At(vOff), r.vNormUnit, r.vNormUnit, r.uZero, g.uNKV, g.uHd, r.uZero, r.uEps, r.uZero)
	}
	e.Dispatch(r.pRope, r.nH*g.half, 64, r.qkv, L.invf, g.uHd, r.uPos, g.uQtotal, g.uHalf)           // q @ off 0
	e.Dispatch(r.pRope, g.nKV*g.half, 64, r.qkv.At(kOff), L.invf, g.uHd, r.uPos, g.uKtotal, g.uHalf) // k (V slot at vOff is NOT roped)
	e.Dispatch(r.pKv, g.kvDim, 64, r.qkv.At(kOff), r.qkv.At(vOff), r.kc[l], r.vc[l], g.uKvDim, r.uPos)
	e.Dispatch(r.pAttn, r.nH*128, 128, r.qkv, r.kc[l], r.vc[l], r.ctx, r.uNH, g.uNKV, g.uHd, r.uNKeys, r.uScale, L.uWindow)
	e.Dispatch(r.pQv, 256, 256, r.ctx, r.cq, r.cSc, g.uNHhd)
	if r.sandwich {
		// Gemma: the sublayer OUTPUT is normed BEFORE the residual add, which the fused
		// _resid epilogue can't express — project into the (otherwise dead) oO scratch,
		// norm it, then add. Three dispatches instead of one.
		e.DispatchTG(r.pSA, r.H*32, 256, r.nH*g.hd*2, L.oW, L.oS, r.cq, r.cSc, r.oO, g.uNHhd)
		e.Dispatch(r.pRmsF32, 256, 256, r.oO, L.postAttnNorm, r.uH, r.uEps, r.uAddOne)
		e.Dispatch(r.pRes, r.H, 256, r.x, r.oO)
	} else {
		e.DispatchTG(r.pSAResid, r.H*32, 256, r.nH*g.hd*2, L.oW, L.oS, r.cq, r.cSc, r.x, g.uNHhd) // o-proj + residual
	}
}
