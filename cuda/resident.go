//go:build cuda

package cuda

import (
	"fmt"
	"math"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

// cudaResident is the production resident decode runner: the parity-green cgo-free forward
// (TestRealForwardParity), promoted from the test harness. All CUDA state is owned by a
// single LockOSThread-pinned executor goroutine (guardrail #3); Forward routes one channel
// round-trip per token. Dense residency only (Qwen2/Llama, DecodeRunnerEligible), mixed
// int4/int8/f32 weights as the real q4_k_m checkpoint stores them.
var (
	_ decoder.ResidentForward = (*cudaResident)(nil)
	_ decoder.ResidentGreedy  = (*cudaResident)(nil)
)

const cudaCtxCap = 4096 // resident KV capacity (positions); staged path handles longer.

// cudaWQ is a device projection weight in whatever precision the checkpoint stored it.
type cudaWQ struct {
	kind string
	W    Buffer // packed weights (int4 fast-layout nibbles, or int8x4)
	ws   Buffer // int8 row scales (N)
	ws16 Buffer // int4 group scales as f16 (N*K/32) — f32 would be 20% of the
	//              int4 byte stream; f16 halves that (decode is byte-bound).
	N, K int
}

type cudaLayer struct {
	q, k, v, o, g, u, d cudaWQ
	qb, kb, vb          Buffer // QKV bias (absent ⇒ none)
	qNorm, kNorm        Buffer // per-head QK-norm weights (absent ⇒ arch has none)
	window              int32  // sliding-window span for THIS layer; 0 = full causal
	// Per-layer attention geometry (9a-P2, own-forward residency bridge). Uniform families
	// set these to the model values on every layer; Gemma 4 varies them (local head_dim=16 /
	// global head_dim=512, K=V). launchToken reads ONLY these — the model-level hd/nKV/qDim/
	// kvDim/rhalf were removed from cudaResident so a launch site physically cannot bind the
	// wrong (uniform) source; nH stays model-level (constant across a family's layers).
	// qDim = nH*hd, kvDim = nKV*hd, rhalf = rotaryDim/2 (rotated pairs per head).
	hd, nKV, qDim, kvDim, rhalf int
	// kEqV (attention_k_eq_v, Gemma 4 global layers): this layer has NO v_proj — V is
	// v_norm(the raw pre-RoPE k_proj output), stored un-rotated in its OWN vCache (NOT aliased
	// to kCache; kvDim is NOT halved). launchToken derives V from k before rope_kv mutates it.
	kEqV              bool
	preNorm, postNorm Buffer
	// Gemma sandwich norms (absent unless NormSandwich4): applied to the SUBLAYER OUTPUT before
	// the residual add, not to a GEMV input.
	postAttnNorm, postMLPNorm Buffer
	invF                      Buffer // per-layer RoPE inv-freq (local vs global base)
	hasBias                   bool

	// Sparse MoE FFN. Per LAYER, not per model: GLM/DeepSeek's first_k_dense_replace makes the
	// first FirstKDense layers plain dense MLPs while the rest route, so the two blocks coexist
	// in one model and the dispatch picks per layer (the decoder keys off the same thing —
	// mlp.go: `arch.MoE != nil && lw.Experts != nil`).
	isMoE   bool
	routerW Buffer // [nE, hidden] f32 — see cudaResident.moe on why it is not quantized
	routerB Buffer // [nE] selection bias; ALWAYS allocated (zeros when the arch has none)
	expGU   cudaWQ // stacked [nE * 2*moeInter, hidden]: expert e's gate at e*2*moeInter, up at +moeInter
	expDown cudaWQ // stacked [nE * hidden, moeInter]

	// Always-on shared expert (GLM/DeepSeek): an ungated SwiGLU MLP at sharedInter, added to the
	// routed output. hasShared is false for a plain MoE (Mixtral). gate‖up is concatenated the
	// same way the routed experts are, so one dense GEMV + the glu_quant offset split covers it.
	hasShared bool
	shGU      cudaWQ // [2*sharedInter, hidden]
	shDown    cudaWQ // [hidden, sharedInter]
}

type cudaResident struct {
	reqCh chan func() error
	ackCh chan error

	hidden, nLayers, inter, vocab int
	// nH (query-head count) is the ONE model-level attention dimension — constant across a
	// family's layers (Gemma 4 is 16 query heads in both variants), so GQA still tracks
	// per-layer nKV via nH/Ly.nKV. hd/nKV/qDim/kvDim/rhalf are DELIBERATELY per-layer only
	// (cudaLayer): removing them here makes a uniform-source threading bug a compile error,
	// not a silent byte-identical pass on uniform models. Allocation-time maxima live as
	// backend.go locals; the per-layer KV cache and UploadKV read r.layers[l].kvDim.
	nH             int
	eps, attnScale float32
	finalSoftcap   float32 // Gemma final-logit softcap (30); 0 ⇒ none. Applied host-side in step().
	vNormUnit      Buffer  // [maxHd] of 1.0 — unit weight so qk_norm (x*inv*w, addOne=0) computes scale-less v_norm for K=V layers. nil unless any layer is kEqV.
	qkNorm         bool    // arch needs per-head Q/K RMSNorm before RoPE
	rmsAddOne      bool    // (1+w) offset — false for Qwen3/Llama
	act            int32   // gated MLP activation, decoder.ActKind (0=gelu-tanh, 1=silu)
	sandwich       bool    // Gemma 4-norm sandwich: extra post-attn / post-MLP norms

	// Sparse MoE. The router projection stays f32 (gemv_f32_a8) while the experts are int4:
	// the router's output steers a DISCRETE choice, so a quantization error near a tie does not
	// perturb the result slightly — it runs a DIFFERENT expert and the output is unrelated.
	// goinfer has already paid for that class once (the Granite SSM work traced a 66%-agreement
	// wall to discrete expert flips and proved no precision knob recovered it), so the router is
	// the one place in this backend where the cheap thing is not worth it.
	moe                     bool
	nE, topK, moeInter      int
	moeSigmoid, moeNormTopK int32
	moeScale                float32
	nGroup, topkGroup       int

	// Per-sublayer contribution capture (diagnostic; off in production, zero cost). When subCap
	// is set, launchToken copies the sandwich-normed o-proj output (attention contribution) and
	// down output (MLP contribution) per layer — the exact dp4a-path analogue of the decoder's
	// ForwardSubCapture, so a cross-backend per-sublayer diff is possible.
	subCap                                 bool
	subAttnC, subMLPC, subCtxC, subMLPpreC [][]float32

	sharedInter int // width of the always-on shared expert (0 ⇒ none)

	// device state — touched ONLY on the executor thread.
	dev                                                                                                *Device
	stream                                                                                             Queue
	gemvW4, gemvW8, kvStore, ropeKV, fRms, fRmsF32, fQ, fRope, fAttn, fSw, fRes, fArg, fQKV, fGU, fQKN Pipeline
	fRoute, fRouterGemv, fMoEGemv, fMoEWacc, fSharedCombine                                            Pipeline

	fuseQKV   bool  // all of Q/K/V/gate/up int4 ⇒ the fused K1 (fQKV) + fGU super-kernels are usable
	launchErr error // sticky first launch error within a launchToken call (reset per token) — M23
	layers    []cudaLayer
	lmW       cudaWQ
	finalNorm Buffer

	// per-token scratch + KV caches (device).
	x, aSc, qB, kB, vB, cctx, cSc, oO, mSc, gO, uO, dSc, dScr, dO, logits Buffer
	aq, cq, mq, dq, argIdx                                                Buffer
	argVal                                                                Buffer
	kc, vc                                                                []Buffer

	// MoE per-token scratch (allocated only when moe). Sized to the MoE expert width, which is
	// NOT the dense one — Mellum's moe_intermediate_size differs from intermediate_size, so
	// reusing gO/uO/dq here would overrun on some archs and silently under-read on others.
	rLogits, rWgt, moeGU, moeSc, moeScr Buffer
	rIdx                                Buffer
	moeQ                                Buffer

	// Shared-expert scratch (allocated only when any layer hasShared). Sized to sharedInter,
	// which is its own width — distinct from both the dense inter and the routed moeInter.
	shGUout, shSc, shScr, shDownOut Buffer
	shQ                             Buffer

	// logitsPinned is PAGE-LOCKED host memory for the per-token logits readback. A pageable
	// D2H of 594 KB measured only ~1.26 GB/s (it stages through a driver bounce buffer);
	// pinned memory DMAs straight out. Slice() is a zero-copy view, so Forward still returns
	// without an extra copy. Reused across calls (decode consumes each before the next).
	logitsPinned *HostBuffer[float32]
	logitsHost   []float32 // zero-copy view of logitsPinned
	setupErr     error     // first alloc/upload error during BuildResident's setup job

}

// alloc/upload helpers — called ONLY inside the setup job (r.dev's context current on the
// executor thread). gpu.NewBufferLenOf PANICS on OOM (recorded into the Device ledger);
// BuildResident's defer recovers that panic and declines gracefully (→ staged fallback)
// instead of proceeding with unusable buffers.
func (r *cudaResident) af(n int) Buffer {
	return gpu.NewBufferLenOf[float32](r.dev, n)
}
func (r *cudaResident) ai(n int) Buffer {
	return gpu.NewBufferLenOf[int32](r.dev, n)
}
func (r *cudaResident) au32(n int) Buffer {
	return gpu.NewBufferLenOf[uint32](r.dev, n)
}
func (r *cudaResident) up32(v []float32) Buffer {
	b := r.af(len(v))
	_ = gpu.Upload(b, v)
	return b
}
func (r *cudaResident) upu32(v []uint32) Buffer {
	b := r.au32(len(v))
	_ = gpu.Upload(b, v)
	return b
}
func (r *cudaResident) upu16(v []uint16) Buffer {
	b := gpu.NewBufferLenOf[uint16](r.dev, len(v))
	_ = gpu.Upload(b, v)
	return b
}
func (r *cudaResident) upW(h hostW) cudaWQ {
	w := cudaWQ{kind: h.kind, W: r.upu32(h.wpk), N: h.N, K: h.K}
	if h.kind == "int4" {
		w.ws16 = r.upu16(h.ws16)
	} else {
		w.ws = r.up32(h.ws)
	}
	return w
}

func (r *cudaResident) do(j func() error) error { r.reqCh <- j; return <-r.ackCh }

// checkCap guards the resident KV allocation. Every layer's cache is sized cudaCtxCap*kvDim, so
// a write for absolute position p lands at kc[p*kvDim ...]; valid positions are [0, cudaCtxCap).
// Writing past it (rope_kv/kv_store) is an out-of-bounds DEVICE write — silent memory corruption,
// UB, and the attention launch's shared-mem request eventually exceeds the block limit. Nothing
// upstream clamps prompt+max_tokens to the cap, so return an error here; the decode loop stops on
// it (model.go) and the caller can fall back to the staged path, which handles longer contexts.
func (r *cudaResident) checkCap(pos, n int) error {
	if pos < 0 || pos+n > cudaCtxCap {
		return fmt.Errorf("cuda: KV position %d(+%d) exceeds resident context cap %d — use the staged path for longer contexts", pos, n, cudaCtxCap)
	}
	return nil
}

// ContextCap is the resident KV capacity in positions (queryable so callers can clamp max_tokens
// up front rather than discover the limit mid-generation).
func (r *cudaResident) ContextCap() int { return cudaCtxCap }

// Forward runs one token at absolute position pos and returns logits[vocab].
func (r *cudaResident) Forward(embedding []float32, pos int) ([]float32, error) {
	if e := r.checkCap(pos, 1); e != nil {
		return nil, e
	}
	var out []float32
	err := r.do(func() error {
		o, e := r.step(embedding, pos)
		out = o
		return e
	})
	return out, err
}

// ForwardN runs K tokens at consecutive positions. Correctness-first: sequential steps in a
// single executor round-trip (bit-identical to K Forward calls; amortizes the channel hop).
func (r *cudaResident) ForwardN(embeddings [][]float32, startPos int) ([][]float32, error) {
	if e := r.checkCap(startPos, len(embeddings)); e != nil {
		return nil, e
	}
	out := make([][]float32, len(embeddings))
	err := r.do(func() error {
		for i, emb := range embeddings {
			l, e := r.step(emb, startPos+i)
			if e != nil {
				return e
			}
			out[i] = append([]float32(nil), l...) // each row kept, so copy off the reused host buf
		}
		return nil
	})
	return out, err
}

// UploadKV writes a layer's post-RoPE K and raw V into the resident caches from position 0
// (prefill bridge, same packed layout the kernels read: [pos*kvDim + head*hd + d]).
func (r *cudaResident) UploadKV(layer int, keys, vals []float32) error {
	if kvDim := r.layers[layer].kvDim; kvDim > 0 {
		if e := r.checkCap(0, len(keys)/kvDim); e != nil {
			return e
		}
	}
	return r.do(func() error {
		if e := gpu.Upload(r.kc[layer], keys); e != nil {
			return e
		}
		return gpu.Upload(r.vc[layer], vals)
	})
}

// TruncateTo is a no-op: KV is positional and Forward sets nKeys=pos+1, so entries past pos
// are never read and get overwritten (matches the WebGPU path).
func (r *cudaResident) TruncateTo(pos int) {}

// Reset clears resident KV for a fresh generation (positions are overwritten on write).
func (r *cudaResident) Reset() {}

// Close shuts down the executor goroutine (and unpins its OS thread). Device buffers are
// freed by primary-context teardown at process exit; a per-buffer free is unnecessary for
// the single-model serve lifetime.
// Close releases the model's GPU memory and tears down the pinned executor.
//
// Freeing the DEVICE memory is the whole job: a resident model owns the weight buffers and the
// per-layer KV cache — gigabytes for a real checkpoint. This once freed only the page-locked
// HOST buffer and closed the channel, so every Load(cuda)+Close leaked the entire model until
// the process exited: invisible in a one-model run, fatal for a model zoo, an
// /admin/models/unload, or a test binary loading models in sequence (it saturated an 8 GB card
// mid-suite, after which every Alloc silently returned nil and the zero-filled buffers looked
// like a parity bug rather than an OOM).
//
// Every buffer is freed EXPLICITLY rather than by leaning on context destruction. Releasing our
// primary-context reference only reclaims memory if the refcount reaches ZERO — and
// dev.Primary() hands out a refcounted per-device singleton, so any other holder (a second
// model in a zoo, another subsystem, a test's own probe context) keeps the context alive and
// the "freed" model's VRAM never comes back. That is precisely the multi-model case unloading
// exists for, so the release-the-context shortcut was wrong exactly where it mattered most;
// TestResidentCloseFreesVRAM pins it.
//
// All of it runs ON the executor thread — that thread made the context current — and therefore
// before reqCh closes. Page-locked host memory goes first: it must be freed before the context.
func (r *cudaResident) Close() error {
	if r.reqCh == nil {
		return nil
	}
	r.reqCh <- func() error {
		// Page-locked host memory must be freed before the context (its free reaches the
		// context's executor), so it goes first and out of the device ledger.
		if r.logitsPinned != nil {
			_ = r.logitsPinned.Close()
			r.logitsPinned, r.logitsHost = nil, nil
		}
		// The Device OWNS every device allocation (weights, KV caches, scratch) in its ledger;
		// ReleaseObjects frees all of them in the correct order, then the modules + stream, then
		// releases our primary-context ref. One call replaces the old per-field free loops.
		if r.dev != nil {
			r.dev.ReleaseObjects()
			r.dev = nil
		}
		r.layers = nil
		r.kc, r.vc = nil, nil
		return nil
	}
	<-r.ackCh
	close(r.reqCh)
	r.reqCh = nil
	return nil
}

// --- launch helpers (executor-thread only) ---

func g1cfg(n, b int) LaunchConfig {
	return LaunchConfig{GridX: uint32((n + b - 1) / b), GridY: 1, GridZ: 1, BlockX: uint32(b), BlockY: 1, BlockZ: 1}
}
func onecfg(b, sh int) LaunchConfig {
	return LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: uint32(b), BlockY: 1, BlockZ: 1, SharedMemBytes: uint32(sh)}
}

func (r *cudaResident) launch(f Pipeline, cfg LaunchConfig, args ...KernelArg) error {
	e := r.stream.Launch(f, cfg, args...)
	if e != nil && r.launchErr == nil {
		// Sticky: launchToken's dense hot chain discards many launch errors (`_ = r.launch(...)`),
		// so a config error (bad shared-mem size, bad args) would let the token "succeed" with
		// stale buffers. Record the first here; launchToken returns it (M23). doG/rms funnel through
		// launch too, so this covers the whole chain without touching every call site.
		r.launchErr = e
	}
	return e
}

// capVec copies the first n elements of a device vector to host into dst[l] (diagnostic
// sublayer capture — n is hidden for the o-proj/down contributions, qDim for the pre-o-proj
// context). Runs on the executor thread inside launchToken, so it syncs before the readback.
func (r *cudaResident) capVec(src Buffer, dst [][]float32, l, n int) {
	_ = r.stream.Sync()
	h := make([]float32, n)
	_ = gpu.Download(src, h)
	dst[l] = h
}

// addOneArg is the (1+w) RMS selector as the kernels take it (Architecture.RMSAddOne).
// Gemma stores norm weights as deviations from 1.0; Llama/Qwen scale by w directly.
func (r *cudaResident) addOneArg() int32 {
	if r.rmsAddOne {
		return 1
	}
	return 0
}

func (r *cudaResident) rms(src, nrm Buffer, qOut Buffer, sOut Buffer) error {
	return r.launch(r.fRms, onecfg(256, (r.hidden+256)*4),
		Arg(src), Arg(nrm), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(r.eps),
		gpu.ArgValue(r.addOneArg()), Arg(qOut), Arg(sOut))
}

// normF32 is Gemma's sandwich post-norm: a plain in-place RMSNorm of a SUBLAYER OUTPUT
// (no quant — it lands straight in the f32 residual stream). No-op when the arch has no
// sandwich norms, so non-Gemma families pay nothing.
func (r *cudaResident) normF32(x, w Buffer) error {
	if w.Len() == 0 {
		return nil
	}
	return r.launch(r.fRmsF32, onecfg(256, 256*4),
		Arg(x), Arg(w), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(r.eps), gpu.ArgValue(r.addOneArg()))
}

// doG launches the projection GEMV. accum=1 makes the epilogue do dst[n] += result, which
// absorbs the separate `residual` launch (bit-identical: same operands, same rounding, just
// no round-trip through a temp buffer). Only lane 0 of the row's warp touches dst[n], and the
// GEMV's input activation is never x, so accumulating straight into the residual stream is
// race-free.
func (r *cudaResident) doG(wt cudaWQ, a Buffer, as Buffer, bias KernelArg, dst Buffer, accum int32) error {
	cfg := LaunchConfig{GridX: uint32((wt.N + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
	if wt.kind == "int4" {
		return r.launch(r.gemvW4, cfg, Arg(wt.W), Arg(a), Arg(wt.ws16), Arg(as), bias,
			gpu.ArgValue(int32(wt.N)), gpu.ArgValue(int32(wt.K/8)), gpu.ArgValue(int32(wt.K/32)), Arg(dst), gpu.ArgValue(accum))
	}
	return r.launch(r.gemvW8, cfg, Arg(wt.W), Arg(a), Arg(wt.ws), Arg(as), bias,
		gpu.ArgValue(int32(wt.N)), gpu.ArgValue(int32(wt.K/4)), Arg(dst), gpu.ArgValue(accum))
}

// moeMLP issues one MoE FFN block for layer Ly, accumulating straight into the residual
// stream r.x. Mirrors decoder/mlp.go's moeMLP exactly:
//
//	h        = rmsNorm(x, PreMLPNorm)          — the SAME normed activation feeds router AND experts
//	logits   = Router · h                       (f32; see cudaResident.moe)
//	idx, wgt = route(logits, bias, ...)
//	x       += Σ_j wgt[j] · Down_e(silu(Gate_e·h) * Up_e·h)     where e = idx[j]
//
// The k experts are dispatched SEQUENTIALLY, one slot at a time, but every launch has the same
// geometry regardless of which experts the router picked — the expert is chosen by ARITHMETIC on
// the weight-row index inside the kernel, not by binding a different buffer. That is what lets a
// resident runner keep a static dispatch chain: the routing changes per token, the launches do
// not.
//
// The final GEMV weight-accumulates into r.x, so the per-expert combine and the residual add are
// the same instruction — no scratch, no separate combine pass. This is why the block `continue`s
// the layer loop rather than falling through to the dense epilogue.
func (r *cudaResident) moeMLP(Ly *cudaLayer) error {
	// Explicit rmsnorm, NOT the fused fGU path: the fused kernel folds the norm into the dense
	// gate/up GEMV and never writes r.mq, but the router needs that quantized activation too.
	if e := r.rms(r.x, Ly.postNorm, r.mq, r.mSc); e != nil {
		return e
	}
	// Router logits (one block per expert row) → top-k idx/wgt, both left on the device. The
	// selection never round-trips to the host: a D2H here would serialize the whole token.
	if e := r.launch(r.fRouterGemv, LaunchConfig{GridX: uint32(r.nE), GridY: 1, GridZ: 1,
		BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
		Arg(Ly.routerW), Arg(r.mq), Arg(r.mSc), gpu.ArgValue(int32(r.nE)),
		gpu.ArgValue(int32(r.hidden)), Arg(r.rLogits)); e != nil {
		return e
	}
	if e := r.launch(r.fRoute, onecfg(1, 0),
		Arg(r.rLogits), Arg(Ly.routerB), Arg(r.rIdx), Arg(r.rWgt),
		gpu.ArgValue(int32(r.nE)), gpu.ArgValue(int32(r.topK)), gpu.ArgValue(r.moeSigmoid),
		gpu.ArgValue(r.moeNormTopK), gpu.ArgValue(r.moeScale),
		gpu.ArgValue(int32(r.nGroup)), gpu.ArgValue(int32(r.topkGroup))); e != nil {
		return e
	}
	gu := 2 * r.moeInter
	for j := 0; j < r.topK; j++ {
		// gate‖up for the routed expert, in ONE indexed GEMV: the stack interleaves each
		// expert's gate and up rows (packWeightStack(g0,u0,g1,u1,...)), so one row range of
		// width 2*moeInter is exactly this expert's pair.
		if e := r.launch(r.fMoEGemv, LaunchConfig{GridX: uint32((gu + 7) / 8), GridY: 1, GridZ: 1,
			BlockX: 256, BlockY: 1, BlockZ: 1},
			Arg(Ly.expGU.W), Arg(r.mq), Arg(Ly.expGU.ws16), Arg(r.mSc),
			Arg(r.rIdx), gpu.ArgValue(int32(j)), gpu.ArgValue(int32(gu)),
			gpu.ArgValue(int32(gu)), gpu.ArgValue(int32(r.hidden/8)), gpu.ArgValue(int32(r.hidden/32)),
			Arg(r.moeGU)); e != nil {
			return e
		}
		// SwiGLU over the halves of that one buffer. gocudrv exposes no buffer view/offset, so
		// the split is the kernel's gOff/uOff rather than Go-side pointer arithmetic.
		if e := r.launch(r.fSw, onecfg(256, 256*4),
			Arg(r.moeGU), Arg(r.moeGU), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(r.moeInter)),
			gpu.ArgValue(int32(r.moeInter)), gpu.ArgValue(r.act),
			Arg(r.moeQ), Arg(r.moeSc), Arg(r.moeScr)); e != nil {
			return e
		}
		// down-proj, weight-accumulating into the residual: x += wgt[j] * (Down_e · act).
		if e := r.launch(r.fMoEWacc, LaunchConfig{GridX: uint32((r.hidden + 7) / 8), GridY: 1, GridZ: 1,
			BlockX: 256, BlockY: 1, BlockZ: 1},
			Arg(Ly.expDown.W), Arg(r.moeQ), Arg(Ly.expDown.ws16), Arg(r.moeSc),
			Arg(r.rIdx), Arg(r.rWgt), gpu.ArgValue(int32(j)), gpu.ArgValue(int32(r.hidden)),
			gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(int32(r.moeInter/8)), gpu.ArgValue(int32(r.moeInter/32)),
			Arg(r.x)); e != nil {
			return e
		}
	}
	// Always-on shared expert (GLM/DeepSeek): an ungated SwiGLU MLP over the SAME normed
	// activation, added to the residual. Structurally a routed expert with no routing — a dense
	// gate‖up GEMV, the same glu_quant offset split, a dense down-proj, then the combine adds
	// it in ungated (dst += shDown). decoder/mlp.go does exactly this after the routed sum.
	if Ly.hasShared {
		nullBias := ArgNull()
		if e := r.doG(Ly.shGU, r.mq, r.mSc, nullBias, r.shGUout, 0); e != nil {
			return e
		}
		if e := r.launch(r.fSw, onecfg(256, 256*4),
			Arg(r.shGUout), Arg(r.shGUout), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(r.sharedInter)),
			gpu.ArgValue(int32(r.sharedInter)), gpu.ArgValue(r.act),
			Arg(r.shQ), Arg(r.shSc), Arg(r.shScr)); e != nil {
			return e
		}
		if e := r.doG(Ly.shDown, r.shQ, r.shSc, nullBias, r.shDownOut, 0); e != nil {
			return e
		}
		// ungated=1: dst[i] += shDown[i]. The gl pointer is unread when ungated, but the kernel
		// still takes it, so pass a valid buffer (shSc, spare) rather than a null.
		if e := r.launch(r.fSharedCombine, g1cfg(r.hidden, 256),
			Arg(r.x), Arg(r.shDownOut), Arg(r.shSc), gpu.ArgValue(int32(r.hidden)),
			gpu.ArgValue(int32(1))); e != nil {
			return e
		}
	}
	return nil
}

// launchToken issues one token's whole kernel chain, leaving logits[vocab] on the device.
func (r *cudaResident) launchToken(emb []float32, pos int) error {
	r.launchErr = nil // reset the sticky launch-error accumulator for this token (M23)
	nullBias := ArgNull()
	if e := gpu.Upload(r.x, emb); e != nil {
		return e
	}
	for l := 0; l < r.nLayers; l++ {
		Ly := &r.layers[l]
		qb, kb, vb := nullBias, nullBias, nullBias
		if Ly.hasBias {
			qb, kb, vb = Arg(Ly.qb), Arg(Ly.kb), Arg(Ly.vb)
		}
		if r.fuseQKV {
			// K1: rmsnorm+quant redundantly per block + this block's Q/K/V rows — one
			// launch instead of four, and the GridX:1 rmsnorm disappears.
			nrows := Ly.qDim + 2*Ly.kvDim
			cfg := LaunchConfig{GridX: uint32((nrows + 7) / 8), GridY: 1, GridZ: 1,
				BlockX: 256, BlockY: 1, BlockZ: 1,
				SharedMemBytes: uint32((r.hidden + 256 + r.hidden/4) * 4)}
			if e := r.launch(r.fQKV, cfg,
				Arg(r.x), Arg(Ly.preNorm), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(r.eps),
				gpu.ArgValue(r.addOneArg()),
				Arg(Ly.q.W), Arg(Ly.q.ws16), qb,
				Arg(Ly.k.W), Arg(Ly.k.ws16), kb,
				Arg(Ly.v.W), Arg(Ly.v.ws16), vb,
				gpu.ArgValue(int32(Ly.qDim)), gpu.ArgValue(int32(Ly.kvDim)),
				gpu.ArgValue(int32(r.hidden/8)), gpu.ArgValue(int32(r.hidden/32)),
				Arg(r.qB), Arg(r.kB), Arg(r.vB)); e != nil {
				return e
			}
		} else {
			if e := r.rms(r.x, Ly.preNorm, r.aq, r.aSc); e != nil {
				return e
			}
			if e := r.doG(Ly.q, r.aq, r.aSc, qb, r.qB, 0); e != nil {
				return e
			}
			_ = r.doG(Ly.k, r.aq, r.aSc, kb, r.kB, 0)
			if Ly.kEqV {
				// K=V (attention_k_eq_v): no v_proj. V = v_norm(RAW k_proj output), so recompute
				// the k projection into vB (bit-identical to kB before k-norm/RoPE touch it) — a
				// second GEMV rather than a device copy (no D2D helper), the raw-k source v_norm
				// needs. v-norm is applied below, after qk-norm, before rope_kv rotates k.
				_ = r.doG(Ly.k, r.aq, r.aSc, kb, r.vB, 0)
			} else {
				_ = r.doG(Ly.v, r.aq, r.aSc, vb, r.vB, 0)
			}
		}
		// QK-norm (Qwen3/GLM/Mellum): per-head RMSNorm of Q and K over head_dim, BEFORE RoPE
		// (decoder/attention.go:94-96). One block per head; only dispatched for archs that
		// need it, so plain-dense models pay nothing.
		if r.qkNorm {
			addOne := int32(0)
			if r.rmsAddOne {
				addOne = 1
			}
			if e := r.launch(r.fQKN, LaunchConfig{GridX: uint32(r.nH + Ly.nKV), GridY: 1, GridZ: 1,
				BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 8}, // f64 reduction
				Arg(r.qB), Arg(r.kB), Arg(Ly.qNorm), Arg(Ly.kNorm),
				gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)),
				gpu.ArgValue(r.eps), gpu.ArgValue(addOne)); e != nil {
				return e
			}
		}
		if Ly.kEqV {
			// K=V: scale-less v_norm(vB) = qk_norm reused with nH=0 (skip the Q pass), k=vB, UNIT
			// weight, addOne=0 → x*inv*1. One block per V head (GridX=nKV). Runs AFTER qk-norm
			// (which touched kB, not vB) and BEFORE rope_kv (which stores vB un-rotated) — vB still
			// holds the RAW k here, so this is v_norm(raw k), not v_norm(k_norm(k)). Proven
			// bit-identical to the CPU oracle by TestVNorm_scaleless.
			_ = r.launch(r.fQKN, LaunchConfig{GridX: uint32(Ly.nKV), GridY: 1, GridZ: 1,
				BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 8},
				Arg(r.vB), Arg(r.vB), Arg(r.vNormUnit), Arg(r.vNormUnit),
				gpu.ArgValue(int32(0)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)),
				gpu.ArgValue(r.eps), gpu.ArgValue(int32(0)))
		}
		// fused rope(q)+rope(k)+kv_store(k)+kv_store(v): 4 launches → 1 (same math/order).
		// rhalf == hd/2 for full rotary (tail group empty, bit-identical to the pre-partial
		// kernel); rotaryDim/2 for partial rotary, where the tail threads carry the un-rotated
		// remainder into the KV cache.
		_ = r.launch(r.ropeKV, g1cfg(r.nH*Ly.rhalf+Ly.nKV*Ly.rhalf+Ly.nKV*(Ly.hd-2*Ly.rhalf), 256),
			Arg(r.qB), Arg(r.kB), Arg(r.vB), Arg(Ly.invF), Arg(r.kc[l]), Arg(r.vc[l]),
			gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)),
			gpu.ArgValue(int32(pos)), gpu.ArgValue(int32(Ly.rhalf)))
		nKeys := pos + 1
		// Sliding window (per layer: Mistral is all-local, Mellum interleaves). Shared is sized
		// to the ATTENDED span, so a windowed layer's request stays bounded as context grows.
		nWin := nKeys
		if Ly.window > 0 && nKeys > int(Ly.window) {
			nWin = int(Ly.window)
		}
		_ = r.launch(r.fAttn, LaunchConfig{GridX: uint32(r.nH), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((nWin + 128) * 4)},
			Arg(r.qB), Arg(r.kc[l]), Arg(r.vc[l]), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)), gpu.ArgValue(int32(nKeys)), gpu.ArgValue(r.attnScale), gpu.ArgValue(Ly.window), Arg(r.cctx))
		if r.subCap { // pre-o-proj attention context (qDim), before quant — the cross-box discriminator
			r.capVec(r.cctx, r.subCtxC, l, Ly.qDim)
		}
		_ = r.launch(r.fQ, onecfg(256, 256*4), Arg(r.cctx), gpu.ArgValue(int32(Ly.qDim)), Arg(r.cq), Arg(r.cSc))
		// Normally the out-proj accumulates straight into the residual stream (accum=1),
		// absorbing the `residual` launch. Gemma's sandwich norm CANNOT use that epilogue: it
		// must normalize the sublayer output BETWEEN the projection and the residual add
		// (decoder/model.go: `a = attn(...); if sandwich { a = rmsNorm(a, PostAttnNorm) }; h += a`).
		// So the sandwich path projects into a scratch buffer, norms it, then adds.
		if r.sandwich {
			// oO/dO are the pre-existing [hidden] sublayer-output buffers, dead since the
			// accum=1 epilogue absorbed the residual launch — the sandwich path needs exactly
			// them back.
			_ = r.doG(Ly.o, r.cq, r.cSc, nullBias, r.oO, 0)
			_ = r.normF32(r.oO, Ly.postAttnNorm)
			if r.subCap { // attention contribution about to hit the residual
				r.capVec(r.oO, r.subAttnC, l, r.hidden)
			}
			_ = r.launch(r.fRes, g1cfg(r.hidden, 256), Arg(r.x), Arg(r.oO), gpu.ArgValue(int32(r.hidden)))
		} else {
			_ = r.doG(Ly.o, r.cq, r.cSc, nullBias, r.x, 1)
		}
		if Ly.isMoE {
			if e := r.moeMLP(Ly); e != nil {
				return e
			}
			continue // the MoE block ends the layer: its combine already hit the residual stream
		}
		if r.fuseQKV { // same guard: all layer projections int4
			// K3a: pre-MLP rmsnorm+quant redundantly per block + this block's gate/up rows —
			// one launch instead of three; the layer's second GridX:1 rmsnorm disappears.
			cfg := LaunchConfig{GridX: uint32((2*r.inter + 63) / 64), GridY: 1, GridZ: 1, // 64 rows/block
				BlockX: 256, BlockY: 1, BlockZ: 1,
				SharedMemBytes: uint32((r.hidden + 256 + r.hidden/4) * 4)}
			if e := r.launch(r.fGU, cfg,
				Arg(r.x), Arg(Ly.postNorm), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(r.eps),
				gpu.ArgValue(r.addOneArg()),
				Arg(Ly.g.W), Arg(Ly.g.ws16),
				Arg(Ly.u.W), Arg(Ly.u.ws16),
				gpu.ArgValue(int32(r.inter)), gpu.ArgValue(int32(r.hidden/8)), gpu.ArgValue(int32(r.hidden/32)),
				Arg(r.gO), Arg(r.uO)); e != nil {
				return e
			}
		} else {
			_ = r.rms(r.x, Ly.postNorm, r.mq, r.mSc)
			_ = r.doG(Ly.g, r.mq, r.mSc, nullBias, r.gO, 0)
			_ = r.doG(Ly.u, r.mq, r.mSc, nullBias, r.uO, 0)
		}
		_ = r.launch(r.fSw, onecfg(256, 256*4), Arg(r.gO), Arg(r.uO), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(r.inter)),
			gpu.ArgValue(r.act), Arg(r.dq), Arg(r.dSc), Arg(r.dScr))
		if r.sandwich {
			if e := r.doG(Ly.d, r.dq, r.dSc, nullBias, r.dO, 0); e != nil {
				return e
			}
			if r.subCap { // down output BEFORE the post-MLP sandwich norm
				r.capVec(r.dO, r.subMLPpreC, l, r.hidden)
			}
			_ = r.normF32(r.dO, Ly.postMLPNorm)
			if r.subCap { // MLP contribution about to hit the residual
				r.capVec(r.dO, r.subMLPC, l, r.hidden)
			}
			_ = r.launch(r.fRes, g1cfg(r.hidden, 256), Arg(r.x), Arg(r.dO), gpu.ArgValue(int32(r.hidden)))
		} else if e := r.doG(Ly.d, r.dq, r.dSc, nullBias, r.x, 1); e != nil {
			return e
		}
	}
	if e := r.rms(r.x, r.finalNorm, r.aq, r.aSc); e != nil {
		return e
	}
	if e := r.doG(r.lmW, r.aq, r.aSc, nullBias, r.logits, 0); e != nil {
		return e
	}
	return r.launchErr // surface any launch error discarded in the dense chain above (M23)
}

// step returns full logits — the general contract (sampler / constrained decode / logprobs).
// Costs a vocab*4 B D2H every token (594 KB at a 151936 vocab).
func (r *cudaResident) step(emb []float32, pos int) ([]float32, error) {
	if e := r.launchToken(emb, pos); e != nil {
		return nil, e
	}
	if e := r.stream.Sync(); e != nil {
		return nil, e
	}
	if e := gpu.ReadToHost(r.logits, r.logitsPinned); e != nil {
		return nil, e
	}
	// Final-logit softcap (Gemma 2/4): softcap·tanh(logits/softcap), applied ONCE to the logit
	// vector after the LM head — host-side, exactly as the CPU path (forwardn.go / logitsFromHidden)
	// and as FeatEmbedScale's √hidden is host-side. This is what FeatFinalLogitSoftcap declares on
	// this backend; 0 for every non-softcapped family (no-op). Covers Forward and ForwardN (both
	// route through step).
	if r.finalSoftcap > 0 {
		sc := r.finalSoftcap
		for j, v := range r.logitsHost {
			r.logitsHost[j] = sc * float32(math.Tanh(float64(v/sc)))
		}
	}
	return r.logitsHost, nil
}

// ForwardArgmax is the greedy fast path (decoder.ResidentGreedy): reduce the argmax on-device
// and read back 4 B instead of the whole logits vector. Same kernel chain, same numerics —
// only the readback differs, so the id equals argmax(Forward(...)) exactly.
func (r *cudaResident) ForwardArgmax(embedding []float32, pos int) (int, error) {
	if e := r.checkCap(pos, 1); e != nil {
		return 0, e
	}
	var id int
	err := r.do(func() error {
		if e := r.launchToken(embedding, pos); e != nil {
			return e
		}
		if e := r.launch(r.fArg, onecfg(256, 256*4+256*4), Arg(r.logits),
			gpu.ArgValue(int32(r.vocab)), Arg(r.argIdx), Arg(r.argVal)); e != nil {
			return e
		}
		if e := r.stream.Sync(); e != nil {
			return e
		}
		out := make([]int32, 1)
		if e := gpu.Download(r.argIdx, out); e != nil {
			return e
		}
		id = int(out[0])
		return nil
	})
	return id, err
}

// --- host-side weight packing (CPU; runs before any CUDA) ---

type hostW struct {
	kind string
	wpk  []uint32
	ws   []float32 // int8 row scales
	ws16 []uint16  // int4 group scales (f16)
	N, K int
}

func packI8(q8 []int8, N, K int) []uint32 {
	p := make([]uint32, N*(K/4))
	for i := range p {
		p[i] = uint32(uint8(q8[i*4])) | uint32(uint8(q8[i*4+1]))<<8 | uint32(uint8(q8[i*4+2]))<<16 | uint32(uint8(q8[i*4+3]))<<24
	}
	return p
}

// packWeight quantizes/repacks a projection into the resident device layout, or errors
// (so BuildResident can decline gracefully → staged fallback) for shapes it can't handle.
func packWeight(w *linalg.WeightMat) (hostW, error) {
	N, K := w.Rows(), w.Cols()
	if K%4 != 0 {
		return hostW{}, fmt.Errorf("cuda: K=%d not a multiple of 4", K)
	}
	switch w.Kind() {
	case "int4":
		if K%32 != 0 {
			return hostW{}, fmt.Errorf("cuda: int4 K=%d not a multiple of 32", K)
		}
		q4, sc, _, _ := w.Int4()
		wpk := make([]uint32, N*(K/8))
		for i := range wpk {
			b := q4[i*4 : i*4+4]
			wpk[i] = permuteFast(uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24)
		}
		gs := make([]uint16, len(sc))
		for i, v := range sc {
			gs[i] = f32tof16(v)
		}
		return hostW{kind: "int4", wpk: wpk, ws16: gs, N: N, K: K}, nil
	case "int8":
		q8, sc, _, _ := w.Int8()
		return hostW{kind: "int8", wpk: packI8(q8, N, K), ws: sc, N: N, K: K}, nil
	case "f32":
		f32, _ := w.F32()
		q8, sc := linalg.QuantizeRowsInt8(f32, N, K)
		return hostW{kind: "int8", wpk: packI8(q8, N, K), ws: sc, N: N, K: K}, nil
	default:
		return hostW{}, fmt.Errorf("cuda: unsupported projection kind %q", w.Kind())
	}
}

// packWeightStack row-stacks several same-K weights into ONE packed buffer, so a routed expert
// is selected by INDEXING a row range rather than by binding a different buffer per token
// (gemv_w4a8_moe: wrow = idx[slot]*rowsPerExpert + row). Fixed launch geometry is what lets the
// resident runner keep a static dispatch chain regardless of which experts a token picks.
//
// It composes packWeight rather than re-implementing the layout, so a stacked expert's bytes
// are IDENTICAL to the same weight packed alone — the nibble permutation, group-scale f16
// rounding and row order all come from the one packer. A second copy of that layout is how the
// indexed reads would silently land on garbage: the GEMV cannot tell a mis-packed row from a
// real one, it just returns a plausible wrong number.
//
// Every input must share kind and K: the kernel derives its stride from one Kwords/Kgroups, so
// a ragged stack would read across row boundaries.
func packWeightStack(ws ...*linalg.WeightMat) (hostW, error) {
	if len(ws) == 0 {
		return hostW{}, fmt.Errorf("cuda: packWeightStack needs at least one weight")
	}
	var out hostW
	for i, w := range ws {
		h, err := packWeight(w)
		if err != nil {
			return hostW{}, fmt.Errorf("cuda: packWeightStack[%d]: %w", i, err)
		}
		if i == 0 {
			out.kind, out.K = h.kind, h.K
		}
		if h.kind != out.kind {
			return hostW{}, fmt.Errorf("cuda: packWeightStack[%d]: kind %q != %q — a mixed-precision "+
				"stack cannot share one kernel's unpack path", i, h.kind, out.kind)
		}
		if h.K != out.K {
			return hostW{}, fmt.Errorf("cuda: packWeightStack[%d]: K=%d != %d — the kernel strides by a "+
				"single Kwords, so a ragged stack reads across row boundaries", i, h.K, out.K)
		}
		out.wpk = append(out.wpk, h.wpk...)
		out.ws = append(out.ws, h.ws...)
		out.ws16 = append(out.ws16, h.ws16...)
		out.N += h.N
	}
	return out, nil
}
