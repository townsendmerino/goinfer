//go:build cuda

package cuda

import (
	"fmt"
	"math"
	"os"
	"strings"
	"unsafe"

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
	// C′ VRAM expert cache: srcW/srcS are the FULL expert stack in pinned host memory (the DMA
	// source); W/ws16 are then small nSlots-deep DEVICE slot buffers the GEMV reads. perExpertW /
	// perExpertS are the per-expert strides (uint32 words / uint16 scales) for the H2D fill.
	srcW, srcS             *gpu.MappedHostBuffer
	perExpertW, perExpertS int
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
	isMoE    bool
	routerW  Buffer       // [nE, hidden] f32 — see cudaResident.moe on why it is not quantized
	routerB  Buffer       // [nE] selection bias; ALWAYS allocated (zeros when the arch has none)
	expGU    cudaWQ       // stacked [nE * 2*moeInter, hidden]: expert e's gate at e*2*moeInter, up at +moeInter
	expDown  cudaWQ       // stacked [nE * hidden, moeInter]
	expCache *expertCache // C′ step 2: per-layer LRU slot residency (nil unless cacheExperts)

	// Always-on shared expert (GLM/DeepSeek): an ungated SwiGLU MLP at sharedInter, added to the
	// routed output. hasShared is false for a plain MoE (Mixtral). gate‖up is concatenated the
	// same way the routed experts are, so one dense GEMV + the glu_quant offset split covers it.
	hasShared bool
	shGU      cudaWQ // [2*sharedInter, hidden]
	shDown    cudaWQ // [hidden, sharedInter]

	// Gemma-4 enable_moe_block layer (parallel dense‖MoE FFN — its own gemma4MoeMLP, NOT the generic
	// moeMLP). The dense branch reuses g/u/d; the router reuses routerW (the f32 proj with
	// routerScale·hidden^-0.5 folded into its columns at build) + routerB; the experts reuse
	// expGU/expDown. g4moe selects the path; these are its extra params.
	g4moe                                                  bool
	g4preFFN, g4postFFN1, g4preFFN2, g4postFFN2, g4postFFN Buffer  // the 5 gemma4 RMSNorm weights
	perExpertScaleB                                        Buffer  // [nE] learned scale on the renormalized top-k weights
	layerScalar                                            float32 // per-layer output scalar (out = (h+combined)*layerScalar)

	// CUDA-graph capture of this layer's three STATIC launch segments (r.graphs). segA = QKV proj +
	// qk/v-norm (pre-RoPE); segB = ctx-quant + o-proj + the MLP up to the router readback; segC =
	// the expert loop + join (nil for a dense layer — no readback gap). rope_kv, attention, the g4x2
	// zero-upload and the loadRoutedExperts D2H stay LIVE in the gaps (per-token pos/nKeys, or a host
	// round-trip — neither is graph-capturable). Captured once at build, replayed per token; each
	// replay reads the CURRENT buffer contents (TestCUDA_graphReplay), so the routing that changes
	// per token flows through unchanged. nil unless r.graphs.
	gSegA, gSegB, gSegC *gpu.Graph
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
	cacheExperts   bool    // C′: routed experts DMA'd host→VRAM slots per token (device read; correct)
	graphs         bool    // CUDA graphs: replay each layer's static segments instead of re-issuing launches (off ⇒ byte-identical)
	graphsSync     bool    // DEBUG probe: r.stream.Sync() after each segment replay (bisects inter- vs intra-segment ordering hazards)
	graphMask      string  // DEBUG probe: if non-empty, replay ONLY the named segments (e.g. "A","B","C","AB") and issue the rest live — localizes a replay hazard to a segment
	layerCap       bool    // DEBUG probe: snapshot the residual r.x after every layer (localizes where a full-forward divergence first appears)
	layerCapBuf    [][]float32
	launchN        int      // diagnostic: per-forward dispatch count (graph-capturable-fraction bound)
	cacheSlots     int      // C′ step 2: device slots per layer (≥ topK; = topK ⇒ step-1 fresh-load, no reuse)
	slotIdx        Buffer   // C′: per-token slot ids for the routed experts, bound as the GEMV's idx
	hostIdx        []uint32 // C′: scratch for the per-layer rIdx device→host readback
	hostSlot       []uint32 // C′: scratch for the per-token slot ids uploaded to slotIdx

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

	// Gemma-4 parallel dense‖MoE (any layer g4moe ⇒ JIT router_f32, alloc g4 scratch, take
	// gemma4MoeMLP). g4cap (GOINFER_G4_CAPTURE) is a DEBUG readback of the four MoE-layer buffers
	// (rn / wgt / x1 / x2) so a whole-forward miss localizes to router vs dense vs expert vs join in
	// ONE run — the observation wired BEFORE the gate, not bolted on after it reds.
	gemma4Moe                           bool
	g4cap                               bool
	g4capRn, g4capWgt, g4capX1, g4capX2 [][]float32
	g4capIdx                            [][]uint32 // APPEND order (token-outer, layer-inner), matching the CPU routerCaptureBuf — for the per-POSITION routing-agreement check (a top-k flip at pos N reads like accumulation in a cosine)

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
	// Batched (M=len) prefill pipelines (prefill_batched.ptx) — the weight-stationary path that fixes
	// the ~128-token Ollama crossover. bGemv is the batched W4A8 GEMV; the rest are the M=1 glue
	// kernels with an M dimension, each bit-identical per row. Loaded once at build (small module).
	bGemv, bRN, bRms, bRopeKV, bAttn, bQuant, bSw, bRes     Pipeline
	prefillReady                                            bool         // batched kernels loaded; PrefillLast usable
	prof                                                    *prefillProf // non-nil ⇒ PrefillLast times each kernel category (test-only; adds stream syncs)
	fRoute, fRouterGemv, fMoEGemv, fMoEWacc, fSharedCombine Pipeline
	fRouterF32, fScaleWgt, fRmsNW, fScaleVec                Pipeline // gemma4 MoE (router_f32 module)

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

	// Gemma-4 MoE branch scratch [hidden] (allocated only when gemma4Moe). x1 = dense branch, x2 =
	// expert-sum branch, rn = the router's weightless-normed raw-h input. Kept SEPARATE from r.x
	// because the join norms (x1+x2) BEFORE adding the residual h — the experts can't wacc into the
	// residual stream the way the generic moeMLP does.
	g4x1, g4x2, g4rn Buffer
	g4zero           []float32 // host [hidden] zeros to clear x2 before the wacc loop (no D2D helper)

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

// upExperts uploads an expert stack, VRAM-slot-cached (C′) when cacheExperts is set and the weight
// is int4 (the only kind the resident MoE GEMVs accept), else fully VRAM-resident. (The A′ zero-copy
// direct-read path was removed: correct in isolation but mis-read by gemv_w4a8_moe at width — see
// docs/task-moe-streaming.md and NewMappedHostBuffer's doc; C′ reads device memory and is bit-exact.)
func (r *cudaResident) upExperts(h hostW) cudaWQ {
	if r.cacheExperts && h.kind == "int4" {
		return r.cacheWQ(h)
	}
	return r.upW(h)
}

// cacheWQ (C′) keeps the FULL expert stack in pinned host memory (srcW/srcS, the DMA source) and
// allocates a small nSlots-deep DEVICE slot buffer (W/ws16) that the GEMV actually reads. Per token
// the routed experts are DMA'd from the host source into the slots (loadWQ); the kernel indexes by
// SLOT id, so it runs unmodified against the smaller stack (no moe.ptx change). The device stack is
// r.cacheSlots deep (≥ topK); loadRoutedExperts fills it via the per-layer LRU expertCache — at
// nSlots=topK it degenerates to step-1 fresh-load, above that it reuses across tokens.
// rowsPerExpert = N/nE (the stack has nE experts, N total rows).
func (r *cudaResident) cacheWQ(h hostW) cudaWQ {
	w := cudaWQ{kind: h.kind, N: h.N, K: h.K}
	rowsPerExpert := h.N / r.nE
	w.perExpertW = rowsPerExpert * (h.K / 8)  // uint32 words per expert weight (Kwords = K/8)
	w.perExpertS = rowsPerExpert * (h.K / 32) // uint16 group scales per expert (Kgroups = K/32)
	w.srcW = r.mapBytes(u32bytes(h.wpk))
	w.srcS = r.mapBytes(u16bytes(h.ws16))
	// Device slot buffers (W/ws16) are NOT allocated here — allocSlots does it after the core +
	// KV are up, so it can size r.cacheSlots to the MEASURED free VRAM and never OOM.
	return w
}

// slotBytesPerLayer is the device VRAM one slot's worth of BOTH expert projections costs (int4
// weight + f16 scales), used to size the cache to free VRAM.
func (r *cudaResident) slotBytesPerLayer() int {
	gu, dn := &r.layers[0].expGU, &r.layers[0].expDown
	return (gu.perExpertW+dn.perExpertW)*4 + (gu.perExpertS+dn.perExpertS)*2
}

// allocSlots caps r.cacheSlots to the MEASURED free device VRAM (with a safety margin) and then
// allocates each MoE layer's slot buffers + its LRU cache. Called after the resident core + KV +
// scratch are up, so the cap reflects what is actually left — an over-large GOINFER_MOE_CACHE_SLOTS
// is capped-and-logged (the repo's "decline honestly at load" discipline), never OOM'd mid-build.
func (r *cudaResident) allocSlots() error {
	var moeLayers []int
	for i := range r.layers {
		if r.layers[i].expGU.srcW != nil {
			moeLayers = append(moeLayers, i)
		}
	}
	if len(moeLayers) == 0 {
		return nil
	}
	perLayer := r.slotBytesPerLayer()
	const marginBytes = 384 << 20 // headroom for the greedy-argmax readback + driver overhead
	if free, _, err := r.dev.Context().MemInfo(); err == nil {
		budget := int64(free) - marginBytes
		if fit := int(budget / int64(len(moeLayers)) / int64(perLayer)); fit < r.cacheSlots {
			capped := fit
			if capped < r.topK {
				capped = r.topK // topK always fits — one token's routed set must be resident
			}
			if capped < r.cacheSlots {
				fmt.Fprintf(os.Stderr, "[cuda] C′ cache: %d slots/layer would need %.1f GB VRAM but only %.1f GB free — "+
					"capping to %d (%.1f GB)\n", r.cacheSlots,
					float64(len(moeLayers))*float64(r.cacheSlots)*float64(perLayer)/1e9, float64(free)/1e9,
					capped, float64(len(moeLayers))*float64(capped)*float64(perLayer)/1e9)
				r.cacheSlots = capped
			}
		}
	}
	for _, i := range moeLayers {
		L := &r.layers[i]
		for _, w := range []*cudaWQ{&L.expGU, &L.expDown} {
			w.W = r.au32(r.cacheSlots * w.perExpertW)
			w.ws16 = gpu.NewBufferLenOf[uint16](r.dev, r.cacheSlots*w.perExpertS)
		}
		L.expCache = newExpertCache(r.nE, r.cacheSlots)
	}
	return nil
}

// expertCache is C′ step 2's per-layer LRU slot residency: which cached expert occupies which of the
// nSlots device slots, so a routed expert already resident skips its H2D DMA. nSlots = topK is the
// step-1 staging degenerate case (every token evicts, no reuse); nSlots > topK gives cross-token
// reuse — LRU because the router signal is a stationary skew where recency is a sufficient statistic
// for frequency (the Lever-2 verdict; see docs/task-moe-streaming.md). expGU and expDown share the
// slot index (loaded together), so the cache is per-LAYER, not per-projection.
type expertCache struct {
	nSlots       int
	slotOf       []int32  // [nE]     expert → slot, -1 = not resident
	inSlot       []int32  // [nSlots] slot → expert, -1 = empty
	used         []uint64 // [nSlots] last-touch clock (LRU)
	clock        uint64
	hits, misses uint64 // reuse accounting (a miss is one expert's H2D DMA)
}

func newExpertCache(nE, nSlots int) *expertCache {
	c := &expertCache{nSlots: nSlots, slotOf: make([]int32, nE), inSlot: make([]int32, nSlots), used: make([]uint64, nSlots)}
	for i := range c.slotOf {
		c.slotOf[i] = -1
	}
	for i := range c.inSlot {
		c.inSlot[i] = -1
	}
	return c
}

// admit returns the slot holding expert e, evicting the LRU slot on a miss. hit=false means the
// caller must DMA e's weights into `slot` before the GEMV reads it. An expert admitted earlier in
// the SAME token is the newest, so it is never the victim (nSlots ≥ topK guarantees room).
func (c *expertCache) admit(e uint32) (slot int, hit bool) {
	c.clock++
	if s := c.slotOf[e]; s >= 0 {
		c.used[s] = c.clock
		c.hits++
		return int(s), true
	}
	c.misses++
	victim, oldest := 0, ^uint64(0)
	for s := 0; s < c.nSlots; s++ {
		if c.inSlot[s] < 0 { // an empty slot always wins
			victim = s
			break
		}
		if c.used[s] < oldest {
			oldest = c.used[s]
			victim = s
		}
	}
	if old := c.inSlot[victim]; old >= 0 {
		c.slotOf[old] = -1
	}
	c.inSlot[victim] = int32(e)
	c.slotOf[e] = int32(victim)
	c.used[victim] = c.clock
	return victim, false
}

// loadExpertSlot DMAs one expert's weight+scales from the pinned host source into device slot `slot`.
// Synchronous H2D (gocudrv v0.2.0); the async-overlap bump rides C′ step 2's perf work, not this.
func (r *cudaResident) loadExpertSlot(w *cudaWQ, e, slot int) error {
	srcW, srcS := w.srcW.Bytes(), w.srcS.Bytes()
	wOff, wLen := e*w.perExpertW*4, w.perExpertW*4
	sOff, sLen := e*w.perExpertS*2, w.perExpertS*2
	if err := gpu.Upload(w.W.At(slot*w.perExpertW*4), srcW[wOff:wOff+wLen]); err != nil {
		return err
	}
	return gpu.Upload(w.ws16.At(slot*w.perExpertS*2), srcS[sOff:sOff+sLen])
}

// loadRoutedExperts reads back the router's idx (device→host — C′'s acknowledged per-layer sync),
// admits each routed expert into the layer's slot cache (DMAing only cache misses), and uploads the
// per-token slot ids the GEMV binds as `idx` (slot j ← the slot now holding routed expert j).
func (r *cudaResident) loadRoutedExperts(L *cudaLayer) error {
	if e := r.stream.Sync(); e != nil {
		return e
	}
	if e := gpu.Download(r.rIdx, r.hostIdx[:r.topK]); e != nil {
		return e
	}
	c := L.expCache
	for j := 0; j < r.topK; j++ {
		e := r.hostIdx[j]
		slot, hit := c.admit(e)
		r.hostSlot[j] = uint32(slot)
		if !hit {
			if err := r.loadExpertSlot(&L.expGU, int(e), slot); err != nil {
				return err
			}
			if err := r.loadExpertSlot(&L.expDown, int(e), slot); err != nil {
				return err
			}
		}
	}
	return gpu.Upload(r.slotIdx, r.hostSlot[:r.topK])
}

// CacheStatsForTest sums the LRU cache hits/misses across all layers (C′ step 2 measurement). A
// miss is one expert's H2D DMA; hit rate = hits/(hits+misses) is the fraction of per-token expert
// bytes reuse saves.
func (r *cudaResident) CacheStatsForTest() (hits, misses uint64) {
	for i := range r.layers {
		if c := r.layers[i].expCache; c != nil {
			hits += c.hits
			misses += c.misses
		}
	}
	return hits, misses
}

// expIdx is the idx argument the expert GEMVs bind: the constant slot ids [0..topK-1] when caching
// (slot j holds routed expert j), else the router's real rIdx (fully-resident path).
func (r *cudaResident) expIdx() Buffer {
	if r.cacheExperts {
		return r.slotIdx
	}
	return r.rIdx
}

func u32bytes(v []uint32) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*4)
}

func u16bytes(v []uint16) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*2)
}

// mapBytes allocates pinned (device-mapped) host memory and copies src into it — the C′ DMA source.
// Panics on the UVA guard failing (NewMappedHostBuffer): eligibility already asserted a UVA device,
// so a failure here is a broken invariant, not a runtime condition.
func (r *cudaResident) mapBytes(src []byte) *gpu.MappedHostBuffer {
	mb, err := r.dev.NewMappedHostBuffer(len(src))
	if err != nil {
		panic(fmt.Sprintf("cacheWQ: NewMappedHostBuffer(%d): %v", len(src), err))
	}
	copy(mb.Bytes(), src)
	return mb
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

// ForwardNoLogits (ResidentPrefillKV) runs the token's forward to build ONLY its resident K/V —
// skipping the final norm + LM head matmul + logits readback + softcap. Used for prompt[:-1] during
// prefill; the layer chain (hence the K/V written at pos) is identical to Forward, so decode from the
// last prompt token is byte-identical. No readback, so nothing is returned but the error.
func (r *cudaResident) ForwardNoLogits(embedding []float32, pos int) error {
	if e := r.checkCap(pos, 1); e != nil {
		return e
	}
	return r.do(func() error {
		if e := r.launchToken(embedding, pos, false); e != nil {
			return e
		}
		return r.stream.Sync() // step()'s trailing sync is skipped here; drain so the KV write completes
	})
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
		// A′: host-mapped expert stacks are page-locked host memory too (not in the device
		// ledger), so free them here alongside logitsPinned, before ReleaseObjects.
		for i := range r.layers {
			for _, m := range []*gpu.MappedHostBuffer{
				r.layers[i].expGU.srcW, r.layers[i].expGU.srcS, r.layers[i].expDown.srcW, r.layers[i].expDown.srcS} {
				if m != nil {
					_ = m.Close()
				}
			}
			// CUDA graphs own a driver graphExec each; destroy them before the context teardown.
			for _, g := range []*gpu.Graph{r.layers[i].gSegA, r.layers[i].gSegB, r.layers[i].gSegC} {
				if g != nil {
					_ = g.Close()
				}
			}
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
	r.launchN++ // per-token dispatch count (diagnostic: graph-capturable-fraction bound)
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
// moeMLPPre issues the pre-readback half of the MoE FFN: the shared normed activation (mq/mSc) and
// the router (logits → top-k idx/wgt, left on the device). It ends exactly at the point where the
// cacheExperts path must read rIdx back to the host — so this half is graph-static (segB), and the
// loadRoutedExperts D2H stays live in the gap between segB and segC.
func (r *cudaResident) moeMLPPre(Ly *cudaLayer) error {
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
	return r.launch(r.fRoute, onecfg(1, 0),
		Arg(r.rLogits), Arg(Ly.routerB), Arg(r.rIdx), Arg(r.rWgt),
		gpu.ArgValue(int32(r.nE)), gpu.ArgValue(int32(r.topK)), gpu.ArgValue(r.moeSigmoid),
		gpu.ArgValue(r.moeNormTopK), gpu.ArgValue(r.moeScale),
		gpu.ArgValue(int32(r.nGroup)), gpu.ArgValue(int32(r.topkGroup)))
}

// moeMLPPost issues the post-readback half: the sequential expert loop (each expert selected by
// ARITHMETIC on r.expIdx(), so the launch geometry is identical regardless of routing — the property
// that makes this graph-static) weight-accumulating into the residual r.x, then the always-on shared
// expert. The cacheExperts readback (if any) has already filled the slots when this runs.
func (r *cudaResident) moeMLPPost(Ly *cudaLayer) error {
	gu := 2 * r.moeInter
	for j := 0; j < r.topK; j++ {
		// gate‖up for the routed expert, in ONE indexed GEMV: the stack interleaves each
		// expert's gate and up rows (packWeightStack(g0,u0,g1,u1,...)), so one row range of
		// width 2*moeInter is exactly this expert's pair.
		if e := r.launch(r.fMoEGemv, LaunchConfig{GridX: uint32((gu + 7) / 8), GridY: 1, GridZ: 1,
			BlockX: 256, BlockY: 1, BlockZ: 1},
			Arg(Ly.expGU.W), Arg(r.mq), Arg(Ly.expGU.ws16), Arg(r.mSc),
			Arg(r.expIdx()), gpu.ArgValue(int32(j)), gpu.ArgValue(int32(gu)),
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
			Arg(r.expIdx()), Arg(r.rWgt), gpu.ArgValue(int32(j)), gpu.ArgValue(int32(r.hidden)),
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

// gemma4MoeMLP issues Gemma-4's parallel dense‖MoE FFN for layer Ly (enable_moe_block). Unlike the
// generic moeMLP (one branch, wacc straight into the residual), Gemma-4 runs TWO branches off the
// SAME residual h and joins them under a shared post-norm, so h is read three times and written only
// at the end. Mirrors decoder/forward_gemma4_moe.go exactly:
//
//	x1 = postFFNNorm1( mlpDown( geluTanh(mlpGate·xd)·(mlpUp·xd) ) )      xd = preFFNNorm(h)   [dense]
//	rn = rmsnorm_nw(h);  logits = RouterProjScaled·rn;  idx,wgt = route  wgt *= perExpertScale[idx]
//	x2 = postFFNNorm2( Σ_j wgt[j]·expertDown_j(geluTanh(gu_j)·up_j) )    xe = preFFNNorm2(h)  [MoE]
//	h  = (h + postFFNNorm(x1 + x2)) · layerScalar                                             [join]
//
// The router GEMV is PURE f32 (gemv_f32_f32) with routerScale·hidden^-0.5 folded into RouterProjScaled
// at build; rn is the weightless OUT-OF-PLACE norm (rmsnorm_nw) so h stays intact for the other two
// branches and the residual add.
// gemma4MoeMLPPre issues the pre-readback half of Gemma-4's parallel dense‖MoE FFN: the dense branch
// (→ g4x1), the router (on RAW h → idx/wgt with the per-expert scale folded in), and the expert-branch
// input norm (xe → mq/mSc). It ends before the g4x2 accumulator clear + the cacheExperts readback,
// both of which stay live in the segB→segC gap (an H2D and, optionally, a D2H — neither capturable).
func (r *cudaResident) gemma4MoeMLPPre(Ly *cudaLayer, l int) error {
	nullBias := ArgNull()

	// --- dense branch → g4x1 ---
	if e := r.rms(r.x, Ly.g4preFFN, r.mq, r.mSc); e != nil { // xd = preFFNNorm(h), int8
		return e
	}
	if e := r.doG(Ly.g, r.mq, r.mSc, nullBias, r.gO, 0); e != nil {
		return e
	}
	if e := r.doG(Ly.u, r.mq, r.mSc, nullBias, r.uO, 0); e != nil {
		return e
	}
	if e := r.launch(r.fSw, onecfg(256, 256*4), Arg(r.gO), Arg(r.uO), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(0)),
		gpu.ArgValue(int32(r.inter)), gpu.ArgValue(r.act), Arg(r.dq), Arg(r.dSc), Arg(r.dScr)); e != nil {
		return e
	}
	if e := r.doG(Ly.d, r.dq, r.dSc, nullBias, r.g4x1, 0); e != nil {
		return e
	}
	if e := r.normF32(r.g4x1, Ly.g4postFFN1); e != nil {
		return e
	}

	// --- router (on RAW h): rmsnorm_nw → gemv_f32_f32(folded proj) → moe_route → per-expert-scale fold ---
	if e := r.launch(r.fRmsNW, onecfg(256, 256*4), Arg(r.x), Arg(r.g4rn), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(r.eps)); e != nil {
		return e
	}
	if e := r.launch(r.fRouterF32, LaunchConfig{GridX: uint32(r.nE), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
		Arg(Ly.routerW), Arg(r.g4rn), gpu.ArgValue(int32(r.nE)), gpu.ArgValue(int32(r.hidden)), Arg(r.rLogits)); e != nil {
		return e
	}
	// softmax (sigmoid=0), UNCONDITIONAL renorm (norm=1), scale=1, no group routing.
	if e := r.launch(r.fRoute, onecfg(1, 0), Arg(r.rLogits), Arg(Ly.routerB), Arg(r.rIdx), Arg(r.rWgt),
		gpu.ArgValue(int32(r.nE)), gpu.ArgValue(int32(r.topK)), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(1)),
		gpu.ArgValue(float32(1)), gpu.ArgValue(int32(1)), gpu.ArgValue(int32(1))); e != nil {
		return e
	}
	if e := r.launch(r.fScaleWgt, LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: uint32(r.topK), BlockY: 1, BlockZ: 1},
		Arg(r.rWgt), Arg(r.rIdx), Arg(Ly.perExpertScaleB), gpu.ArgValue(int32(r.topK))); e != nil {
		return e
	}

	// xe = preFFNNorm2(h); reuse mq/mSc (dense branch done with them). This is the last static op
	// before the gap: launchToken clears g4x2 (H2D) and, if caching, reads back the routing (D2H).
	return r.rms(r.x, Ly.g4preFFN2, r.mq, r.mSc)
}

// gemma4MoeMLPPost issues the post-readback half: the expert loop accumulating into the (already
// cleared) g4x2, its post-norm, and the join — sum x1+x2, joint post-norm, add the residual h, then
// the per-layer scalar. g4x2 was zeroed and the expert slots filled in the gap before this runs.
func (r *cudaResident) gemma4MoeMLPPost(Ly *cudaLayer, l int) error {
	gu := 2 * r.moeInter
	for j := 0; j < r.topK; j++ {
		if e := r.launch(r.fMoEGemv, LaunchConfig{GridX: uint32((gu + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
			Arg(Ly.expGU.W), Arg(r.mq), Arg(Ly.expGU.ws16), Arg(r.mSc), Arg(r.expIdx()),
			gpu.ArgValue(int32(j)), gpu.ArgValue(int32(gu)), gpu.ArgValue(int32(gu)),
			gpu.ArgValue(int32(r.hidden/8)), gpu.ArgValue(int32(r.hidden/32)), Arg(r.moeGU)); e != nil {
			return e
		}
		if e := r.launch(r.fSw, onecfg(256, 256*4), Arg(r.moeGU), Arg(r.moeGU), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(r.moeInter)),
			gpu.ArgValue(int32(r.moeInter)), gpu.ArgValue(r.act), Arg(r.moeQ), Arg(r.moeSc), Arg(r.moeScr)); e != nil {
			return e
		}
		if e := r.launch(r.fMoEWacc, LaunchConfig{GridX: uint32((r.hidden + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
			Arg(Ly.expDown.W), Arg(r.moeQ), Arg(Ly.expDown.ws16), Arg(r.moeSc), Arg(r.expIdx()), Arg(r.rWgt),
			gpu.ArgValue(int32(j)), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(int32(r.hidden)),
			gpu.ArgValue(int32(r.moeInter/8)), gpu.ArgValue(int32(r.moeInter/32)), Arg(r.g4x2)); e != nil {
			return e
		}
	}
	if e := r.normF32(r.g4x2, Ly.g4postFFN2); e != nil {
		return e
	}

	// DEBUG capture BEFORE the join — the four buffers that localize a whole-forward miss to router
	// (rn/wgt) vs dense (x1) vs expert (x2) vs join (logits). Off unless GOINFER_G4_CAPTURE.
	if r.g4cap {
		r.capVec(r.g4rn, r.g4capRn, l, r.hidden)
		r.capVec(r.rWgt, r.g4capWgt, l, r.topK)
		r.capVec(r.g4x1, r.g4capX1, l, r.hidden)
		r.capVec(r.g4x2, r.g4capX2, l, r.hidden)
		_ = r.stream.Sync()
		idx := make([]uint32, r.topK)
		_ = gpu.Download(r.rIdx, idx)
		r.g4capIdx = append(r.g4capIdx, idx) // append order = CPU routerCaptureBuf order (per-position routing check)
	}

	// --- JOIN (Phase-1a ordering, get it EXACT): sum x1+x2 BEFORE the joint norm; add the residual h
	// AFTER it; then the per-layer scalar. A mis-order is plausible-but-wrong that cosine shows but
	// won't localize. ---
	if e := r.launch(r.fRes, g1cfg(r.hidden, 256), Arg(r.g4x1), Arg(r.g4x2), gpu.ArgValue(int32(r.hidden))); e != nil { // x1 += x2
		return e
	}
	if e := r.normF32(r.g4x1, Ly.g4postFFN); e != nil { // x1 = postFFNNorm(x1 + x2)
		return e
	}
	if e := r.launch(r.fRes, g1cfg(r.hidden, 256), Arg(r.x), Arg(r.g4x1), gpu.ArgValue(int32(r.hidden))); e != nil { // r.x = h + comb
		return e
	}
	return r.launch(r.fScaleVec, g1cfg(r.hidden, 256), Arg(r.x), gpu.ArgValue(Ly.layerScalar), gpu.ArgValue(int32(r.hidden))) // r.x *= layerScalar
}

// segA / segB / segC are the three graph-STATIC launch runs of one layer, factored out of
// launchToken so a SINGLE source of truth serves both the live path (call them in sequence — byte-
// identical to the pre-graph launchToken) and the graph path (replay Ly.gSegA/B/C). Every launch here
// binds only fixed buffers and per-layer constants; the per-token dynamics (rope_kv/attention bind
// pos/nKeys; the g4x2 clear and the routing readback are host copies) stay live in the gaps between
// them. See cudaLayer.gSegA and captureGraphs.

// segA: QKV projection (fused K1 super-kernel when every projection is int4, else rmsnorm+quant +
// three GEMVs) + per-head QK-norm + scale-less K=V v-norm. Ends before rope_kv.
func (r *cudaResident) segA(Ly *cudaLayer, l int) error {
	nullBias := ArgNull()
	qb, kb, vb := nullBias, nullBias, nullBias
	if Ly.hasBias {
		qb, kb, vb = Arg(Ly.qb), Arg(Ly.kb), Arg(Ly.vb)
	}
	if r.fuseQKV {
		// K1: rmsnorm+quant redundantly per block + this block's Q/K/V rows — one launch instead of four.
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
			// K=V: recompute the k projection into vB (raw pre-norm k), which v_norm consumes below.
			_ = r.doG(Ly.k, r.aq, r.aSc, kb, r.vB, 0)
		} else {
			_ = r.doG(Ly.v, r.aq, r.aSc, vb, r.vB, 0)
		}
	}
	if r.qkNorm { // per-head Q/K RMSNorm before RoPE (Qwen3/GLM/Mellum)
		addOne := int32(0)
		if r.rmsAddOne {
			addOne = 1
		}
		if e := r.launch(r.fQKN, LaunchConfig{GridX: uint32(r.nH + Ly.nKV), GridY: 1, GridZ: 1,
			BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 8},
			Arg(r.qB), Arg(r.kB), Arg(Ly.qNorm), Arg(Ly.kNorm),
			gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)),
			gpu.ArgValue(r.eps), gpu.ArgValue(addOne)); e != nil {
			return e
		}
	}
	if Ly.kEqV { // scale-less v_norm(raw k in vB), BEFORE rope_kv rotates k
		_ = r.launch(r.fQKN, LaunchConfig{GridX: uint32(Ly.nKV), GridY: 1, GridZ: 1,
			BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 8},
			Arg(r.vB), Arg(r.vB), Arg(r.vNormUnit), Arg(r.vNormUnit),
			gpu.ArgValue(int32(0)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)),
			gpu.ArgValue(r.eps), gpu.ArgValue(int32(0)))
	}
	return nil
}

// segB: context-quant + o-proj (accum into the residual, or sandwich-norm then add) + the MLP up to
// the router readback — the whole dense MLP for a dense layer (no readback gap, segC is nil), or the
// MoE pre-readback half (moeMLPPre / gemma4MoeMLPPre) for a routed layer.
func (r *cudaResident) segB(Ly *cudaLayer, l int) error {
	nullBias := ArgNull()
	_ = r.launch(r.fQ, onecfg(256, 256*4), Arg(r.cctx), gpu.ArgValue(int32(Ly.qDim)), Arg(r.cq), Arg(r.cSc))
	if r.sandwich {
		_ = r.doG(Ly.o, r.cq, r.cSc, nullBias, r.oO, 0)
		_ = r.normF32(r.oO, Ly.postAttnNorm)
		if r.subCap {
			r.capVec(r.oO, r.subAttnC, l, r.hidden)
		}
		_ = r.launch(r.fRes, g1cfg(r.hidden, 256), Arg(r.x), Arg(r.oO), gpu.ArgValue(int32(r.hidden)))
	} else {
		_ = r.doG(Ly.o, r.cq, r.cSc, nullBias, r.x, 1)
	}
	if Ly.g4moe {
		return r.gemma4MoeMLPPre(Ly, l)
	}
	if Ly.isMoE {
		return r.moeMLPPre(Ly)
	}
	// Dense MLP (whole): no readback gap, so segC is nil for this layer.
	if r.fuseQKV {
		cfg := LaunchConfig{GridX: uint32((2*r.inter + 63) / 64), GridY: 1, GridZ: 1,
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
		if r.subCap {
			r.capVec(r.dO, r.subMLPpreC, l, r.hidden)
		}
		_ = r.normF32(r.dO, Ly.postMLPNorm)
		if r.subCap {
			r.capVec(r.dO, r.subMLPC, l, r.hidden)
		}
		_ = r.launch(r.fRes, g1cfg(r.hidden, 256), Arg(r.x), Arg(r.dO), gpu.ArgValue(int32(r.hidden)))
	} else if e := r.doG(Ly.d, r.dq, r.dSc, nullBias, r.x, 1); e != nil {
		return e
	}
	return nil
}

// segC: the post-readback MoE half (expert loop + combine/join). nil for a dense layer.
func (r *cudaResident) segC(Ly *cudaLayer, l int) error {
	if Ly.g4moe {
		return r.gemma4MoeMLPPost(Ly, l)
	}
	if Ly.isMoE {
		return r.moeMLPPost(Ly)
	}
	return nil
}

// captureGraphs records each layer's three static segments once (on the executor thread, so the
// thread-local capture matches the replay thread). Off unless r.graphs; incompatible with the subCap/
// g4cap diagnostics (they sync inside a segment, which stream capture forbids) — the caller gates on
// !g4cap and launchToken's useGraphs gates on !subCap. A capture failure is a build error → the
// resident declines to the staged path, never runs a half-captured chain.
func (r *cudaResident) captureGraphs() error {
	for l := range r.layers {
		Ly, ll := &r.layers[l], l
		gA, e := r.stream.Capture(func() error { return r.segA(Ly, ll) })
		if e != nil {
			return fmt.Errorf("layer %d segA: %w", l, e)
		}
		gB, e := r.stream.Capture(func() error { return r.segB(Ly, ll) })
		if e != nil {
			return fmt.Errorf("layer %d segB: %w", l, e)
		}
		r.layers[l].gSegA, r.layers[l].gSegB = gA, gB
		if Ly.g4moe || Ly.isMoE {
			gC, e := r.stream.Capture(func() error { return r.segC(Ly, ll) })
			if e != nil {
				return fmt.Errorf("layer %d segC: %w", l, e)
			}
			r.layers[l].gSegC = gC
		}
	}
	r.launchErr = nil // capture invoked launch() to RECORD (not execute); clear the sticky accumulator
	return nil
}

// launchToken issues one token's whole kernel chain, leaving logits[vocab] on the device.
func (r *cudaResident) launchToken(emb []float32, pos int, head bool) error {
	r.launchErr = nil // reset the sticky launch-error accumulator for this token (M23)
	nullBias := ArgNull()
	if e := gpu.Upload(r.x, emb); e != nil {
		return e
	}
	// useGraphs replays the captured static segments instead of re-issuing their launches. Gated on
	// !subCap: the sublayer-capture diagnostic syncs mid-segment, which a captured graph cannot do
	// (and which a test may enable on a graphs-built runner) — so it falls back to the live seg calls.
	useGraphs := r.graphs && !r.subCap
	gA := useGraphs && (r.graphMask == "" || strings.Contains(r.graphMask, "A"))
	gB := useGraphs && (r.graphMask == "" || strings.Contains(r.graphMask, "B"))
	gC := useGraphs && (r.graphMask == "" || strings.Contains(r.graphMask, "C"))
	for l := 0; l < r.nLayers; l++ {
		Ly := &r.layers[l]
		// segA: QKV proj + qk/v-norm (pre-RoPE, static).
		if gA {
			if e := Ly.gSegA.Replay(); e != nil {
				return e
			}
			if r.graphsSync {
				_ = r.stream.Sync()
			}
		} else if e := r.segA(Ly, l); e != nil {
			return e
		}
		// --- dynamic gap: rope_kv + attention (bind pos/nKeys; attention's shared-mem grows with the
		// attended span — never graph-static). Same stream as the segments, so ordering is preserved.
		// fused rope(q)+rope(k)+kv_store(k)+kv_store(v): rhalf == hd/2 for full rotary, rotaryDim/2 for partial.
		_ = r.launch(r.ropeKV, g1cfg(r.nH*Ly.rhalf+Ly.nKV*Ly.rhalf+Ly.nKV*(Ly.hd-2*Ly.rhalf), 256),
			Arg(r.qB), Arg(r.kB), Arg(r.vB), Arg(Ly.invF), Arg(r.kc[l]), Arg(r.vc[l]),
			gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)),
			gpu.ArgValue(int32(pos)), gpu.ArgValue(int32(Ly.rhalf)))
		nKeys := pos + 1
		// Sliding window (per layer: Mistral all-local, Mellum interleaves); shared sized to the attended span.
		nWin := nKeys
		if Ly.window > 0 && nKeys > int(Ly.window) {
			nWin = int(Ly.window)
		}
		if r.prefillReady {
			// Coalesced M=1 decode attention: attn_batched with M=1 is BIT-IDENTICAL to the glue
			// `attention` (TestAttnBatched_bitIdentical) but reads K via float4 — 21.96%→98% bytes/sector.
			// ncu found the glue decode attention L1TEX-latency-bound at 2048 (~63% of the decode budget,
			// the 221→97 tok/s long-context deficit vs current Ollama); the coalesced read recovers it.
			// startPos=pos, M=1 → nKeys = pos+1; same GridX/block/shared/ctx-layout as the glue launch, so
			// decode stays byte-identical. glue `attention` (audited) is UNTOUCHED and is the fallback below.
			_ = r.launch(r.bAttn, LaunchConfig{GridX: uint32(r.nH), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((nWin + 128) * 4)},
				Arg(r.qB), Arg(r.kc[l]), Arg(r.vc[l]), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)), gpu.ArgValue(int32(pos)), gpu.ArgValue(r.attnScale), gpu.ArgValue(Ly.window), gpu.ArgValue(int32(1)), Arg(r.cctx))
		} else {
			_ = r.launch(r.fAttn, LaunchConfig{GridX: uint32(r.nH), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((nWin + 128) * 4)},
				Arg(r.qB), Arg(r.kc[l]), Arg(r.vc[l]), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)), gpu.ArgValue(int32(nKeys)), gpu.ArgValue(r.attnScale), gpu.ArgValue(Ly.window), Arg(r.cctx))
		}
		if r.subCap { // pre-o-proj attention context (qDim), before quant — the cross-box discriminator (live path only)
			r.capVec(r.cctx, r.subCtxC, l, Ly.qDim)
		}
		// segB: ctx-quant + o-proj + MLP up to the router readback (whole dense MLP for a dense layer).
		if gB {
			if e := Ly.gSegB.Replay(); e != nil {
				return e
			}
			if r.graphsSync {
				_ = r.stream.Sync()
			}
		} else if e := r.segB(Ly, l); e != nil {
			return e
		}
		// --- dynamic gap: clear the g4moe accumulator (H2D), then, if caching, read the routing back
		// to the host and DMA the routed experts into their VRAM slots (D2H+H2D). Host copies, not
		// graph-capturable; synchronous, so the slots are filled before segC replays.
		if Ly.g4moe {
			if e := gpu.Upload(r.g4x2, r.g4zero); e != nil {
				return e
			}
		}
		if (Ly.g4moe || Ly.isMoE) && r.cacheExperts {
			if e := r.loadRoutedExperts(Ly); e != nil {
				return e
			}
		}
		// segC: the post-readback MoE half (expert loop + join). Dense layers have none.
		if Ly.g4moe || Ly.isMoE {
			if gC {
				if e := Ly.gSegC.Replay(); e != nil {
					return e
				}
				if r.graphsSync {
					_ = r.stream.Sync()
				}
			} else if e := r.segC(Ly, l); e != nil {
				return e
			}
		}
		if r.layerCap { // DEBUG: snapshot the residual after this layer (divergence-localization probe)
			_ = r.stream.Sync()
			h := make([]float32, r.hidden)
			_ = gpu.Download(r.x, h)
			r.layerCapBuf = append(r.layerCapBuf, h)
		}
	}
	// Final norm + LM head — skipped for KV-only prefill (head=false): prompt[:-1] tokens need only
	// their K/V in the cache, and the head is a big-vocab matmul + ~1 MB readback + softcap. The layer
	// loop above already wrote this position's K/V identically, so decode stays byte-identical.
	if head {
		if e := r.rms(r.x, r.finalNorm, r.aq, r.aSc); e != nil {
			return e
		}
		if e := r.doG(r.lmW, r.aq, r.aSc, nullBias, r.logits, 0); e != nil {
			return e
		}
	}
	return r.launchErr // surface any launch error discarded in the dense chain above (M23)
}

// step returns full logits — the general contract (sampler / constrained decode / logprobs).
// Costs a vocab*4 B D2H every token (594 KB at a 151936 vocab).
func (r *cudaResident) step(emb []float32, pos int) ([]float32, error) {
	if e := r.launchToken(emb, pos, true); e != nil {
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
		if e := r.launchToken(embedding, pos, true); e != nil {
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
