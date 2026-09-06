//go:build cuda

package cuda

import (
	"context"
	"errors"
	"fmt"
	"os"
	"reflect"
	"runtime/debug"
	"sort"
	"strings"
	"sync/atomic"
	"time"
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

// cudaCtxCapDefault is the resident KV capacity in positions when nothing asks for more; the staged
// path handles longer. It is a DEFAULT, not a ceiling: decoder.Options.ResidentContext raises it (see
// resolveCtxCap). It stays 4096 so that a caller who did not ask never allocates deep-KV VRAM —
// raising the default would silently multiply every resident model's KV footprint.
//
// 4096 is a round, conservative choice, NOT a value tuned against real VRAM headroom — nothing has
// ever measured how much of a real card's free VRAM it leaves unused. Measured 2026-09-06
// (docs/task-kv-cache-streaming.md): on an RTX 2070 SUPER (8 GB) with a dense 7B at int4,
// checkKVFits accepts -ctx 20000 (7257/8192 MiB used) and refuses -ctx 24576 (needs 2.82 GB of KV,
// 2.90 GB free) — a true per-card ceiling roughly 5-6x this default. Whether that ratio holds for
// other model sizes/quants/cards is unmeasured; raise -ctx and read checkKVFits' own error to find
// the real number for a given deployment rather than assuming this default reflects it.
const cudaCtxCapDefault = 4096

// ctxCapMarginBytes is the VRAM left free beside weights+KV at the load-time fit check: driver
// overhead plus the transient allocations decode makes (logits readback, split-KV scratch). Same
// margin the C′ expert cache uses, for the same reason.
const ctxCapMarginBytes = 384 << 20

// resolveCtxCap turns a request into the effective resident KV capacity:
//
//	cap = min(model context window, request)   — request 0 ⇒ cudaCtxCapDefault
//
// Clamping to the model's own context window matters because the KV beyond it can never be attended:
// allocating it would burn VRAM to hold positions the RoPE tables and the model's training never
// cover. modelCtx 0 means "unknown" (some architectures do not report one), in which case the request
// stands on its own — the VRAM fit check is then the only guard, which is why that check is not
// optional.
func resolveCtxCap(request, modelCtx int) int {
	if request <= 0 {
		return cudaCtxCapDefault
	}
	if modelCtx > 0 && request > modelCtx {
		return modelCtx
	}
	return request
}

// kvBytesForCap is the device bytes the resident K+V caches occupy at a given capacity: every layer
// holds K and V as f32[cap*kvDim]. Measured against this formula: 24.0 KB/position for
// qwen2.5-coder-0.5b (24 layers × 128 kvDim × 2 × 4 B) and 56.0 KB/position for the 1.5B
// (28 × 256 × 2 × 4 B), which is what the deep-context sizing in docs/benchmarks.md is derived from.
func kvBytesForCap(cap int, layers []cudaLayer) int64 {
	var perPos int64
	for i := range layers {
		perPos += int64(layers[i].kvDim)
	}
	return perPos * 2 /* K+V */ * 4 /* f32 */ * int64(cap)
}

// splitkvNever disables the split-KV decode attention for a geometry (no depth within the resident's
// capacity pays for it). Larger than any reachable nWin, so the gate comparison stays a plain >=.
const splitkvNever = 1 << 30

// splitkvThreshold returns the EFFECTIVE attended-key count (nWin — window-clamped, not the raw
// position) at/above which the split-KV decode attention beats the single-block attn_batched(M=1)
// for this geometry, or splitkvNever to disable it.
//
// WHY A TABLE AND NOT A FORMULA. Split-KV buys occupancy and pays for it in DRAM. attn_batched
// launches nH blocks and keeps the whole score row in SHARED memory (SharedMemBytes=(nWin+128)*4);
// split-KV materializes an nH×nWin f32 score array in GLOBAL memory and touches it three times
// (splitkv_scores writes, splitkv_softmax reads+writes, splitkv_vsum reads) in exchange for filling
// the SMs that nH blocks leave idle. So
//
//	net(nWin) ≈ (A−B)·nWin − 2·nLayers·T_launch
//
// where A grows with the occupancy deficit (it needs nH ≪ SM count) and B grows with nH (score
// materialization). A > B gives a crossover; A < B means split-KV NEVER wins and the deficit WIDENS
// with depth — which is exactly what phi3-mini measures (nH=32 on a 40-SM part: almost no deficit to
// recover, and the largest score array of the four). No one-parameter law reproduces all four
// geometries — nLayers/(nH·hd) and nLayers/(nKV·hd) both underpredict the 0.5B crossover by ~2×, and
// neither can express phi3's "never" at any threshold. An honest lookup beats a false formula.
//
// MEASURED (e2e decode-only tok/s through serve, int4, RTX 2070 SUPER / 40 SMs, ON÷OFF; >1 = split-KV
// wins). Full table and method: docs/benchmarks.md §B6.
//
//	geometry              nH  nKV  hd   L   256    512    1024   2048   3900    crossover
//	qwen2.5-0.5b          14   2    64  24  0.839  0.819  0.869  0.955  1.197   ~2560 (see below)
//	qwen2.5-1.5b          12   2   128  28  0.941  0.939  1.078  1.191  1.280   ( 512, 1024]
//	gemma3-1b (win 512)    4   1   256  26  0.890  0.909  0.919  0.941  1.084   windowed — see below
//	phi3-mini (MHA)       32  32    96  32  0.993  0.969  0.919  0.815  0.754   NONE (monotone)
//
// qwen2.5-0.5b was localized further inside the (2048, 3900] band: 2560 → 1.019 (break-even), 3072 →
// 1.061 (first clear win). So splitkvConservative = 3072 is the measured first-clear-win depth for
// that geometry, not a guess; it forfeits ~2% at 2560, which is the asymmetric-loss trade taken
// deliberately.
//
// The 256/512 columns are where the old constant fired: it cost the 0.5B up to 18%. The old comment
// here claimed "break-even 256, clear win from 384+" from TestSplitKVCrossover on the 1.5B — that is
// refuted even on its own geometry (the 1.5B loses at 256 AND 512). That test measures a tight
// in-process ForwardArgmax loop and takes best-of-3 MINIMUM; both choices flatter split-KV relative
// to serving (the loop hides per-token CPU dispatch that e2e exposes, and best-of-min favours the
// higher-variance arm — ON's spread is 3.6–6.4 tok/s vs OFF's 0.1–0.6).
//
// ASYMMETRIC LOSS: firing early costs up to 18–25%, firing late costs a few percent (OFF's slope is
// mild near the crossover). Every threshold is therefore rounded UP, and unmeasured geometries get
// the conservative default rather than an extrapolation.
//
// NOT DEVICE-PORTABLE: the occupancy term scales with SM count and every cell above is one 40-SM
// Turing part. On a much wider GPU nH=32 would be starved and phi3's "never" would not hold.
// Re-measure per device class before trusting these on other hardware; do not scale them by SM count
// on paper.
// singleBlockAttnShmemLimit is the dynamic shared memory a single-block attention launch may
// request. The glue/batched attention kernels size their scratch (nWin+128)*4, with no ceiling.
//
// M-16, MEASURED on the RTX 2070 SUPER (Turing) rather than inferred:
//
//	CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK        49152 (48 KB)  -> nWin <= 12160 keys
//	CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN  65536 (64 KB)  -> nWin <= 16256 keys
//
// 48 KB is the operative one: nothing here calls cuFuncSetAttribute to raise a kernel into the
// opt-in range, so the driver enforces the default. Past 12,160 attended keys the launch is
// refused — decode fails outright at that position, and batched prefill errors at layer 0 and
// silently falls back to the ~9x slower sequential path.
//
// Conservative by choice: the constant is the DEFAULT limit, not the opt-in one, because raising
// it needs a per-kernel SetAttribute call that does not exist yet. If that lands, this becomes a
// device query.
const singleBlockAttnShmemLimit = 48 * 1024

// attnShmemBytes is the dynamic shared memory a single-block attention launch needs for an
// attended span of nWin keys — the sizing every such launch site uses.
func attnShmemBytes(nWin int) int { return (nWin + 128) * 4 }

// splitKVRequired reports whether the single-block attention kernel CANNOT run at this span, so
// split-KV is the only option rather than the faster one (M-16). The perf table below decides
// when split-KV is preferred; this decides when it is mandatory.
func splitKVRequired(nWin int) bool { return attnShmemBytes(nWin) > singleBlockAttnShmemLimit }

func splitkvThreshold(nH, hd int) int {
	switch {
	case nH >= splitkvMaxHeads:
		// Enough blocks to keep a 40-SM part busy: no deficit to buy, and the score array is at its
		// largest. phi3-mini (nH=32) measures a monotone loss to 0.754 at 3900 — it never crosses.
		return splitkvNever
	case nH == 12 && hd == 128:
		return 1024 // qwen2.5-1.5b class: measured crossover in (512, 1024]
	default:
		return splitkvConservative
	}
}

// splitkvMaxHeads: at/above this many query heads the single-block kernel already fills the device,
// so split-KV is pure cost. Anchored at phi3-mini's measured nH=32 ("never") and lowered to 24
// because the asymmetric loss says be conservative between the measured points (the next geometry
// down is nH=14, which does cross over).
const splitkvMaxHeads = 24

// splitkvConservative: qwen2.5-0.5b's measured first-clear-win depth (2560 break-even, 3072 → 1.061),
// which also serves as the default for any geometry not in the table. Three of the four measured
// geometries do not cross over until past 2048, so an unmeasured one is assumed not to either; that
// forfeits a few percent at depth rather than risk the 18–25% shallow regression.
const splitkvConservative = 3072

// splitkvMin is splitkvThreshold with the runtime override applied. The override exists because the
// previous constant could only be re-characterized by rebuilding, which is part of why a refuted
// number survived a release: GOINFER_SPLITKV_MIN_KEYS=<n> re-gates a stock binary (0 ⇒ always take
// the split path, the force-on A/B arm), GOINFER_SPLITKV_ATTN=0 still force-disables entirely.
func (r *cudaResident) splitkvMin(hd int) int {
	if r.skMinKeys >= 0 {
		return r.skMinKeys
	}
	return splitkvThreshold(r.nH, hd)
}

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
	idx                 int // this layer's index, for per-layer side tables (gpt-oss sinks / expert biases)
	q, k, v, o, g, u, d cudaWQ
	qb, kb, vb          Buffer // QKV bias (absent ⇒ none)
	ob                  Buffer // attention output-projection bias (GPT-2 c_proj, gpt-oss o_proj); absent ⇒ none
	hasOBias            bool
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
	// mscale: YaRN's attention_factor for THIS layer (decoder.Model.RopeMscaleLayer). 1.0 for
	// every family without YaRN, and 1.0 on a YaRN family's non-scaled layers — Mellum carries
	// 1.2772588722239782 on its full-attention layers and 1.0 on the sliding ones, so this is
	// per-layer and not per-model. Passed to rope_kv / rope_kv_batched, which fold it into
	// cos/sin. Metal and WebGPU both carry the same per-layer value for the same reason.
	mscale  float32
	hasBias bool

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
	shGateW   cudaWQ // [1, hidden] sigmoid gate (Qwen-MoE); zero-value ⇒ ungated combine

	// Gemma-4 enable_moe_block layer (parallel dense‖MoE FFN — its own gemma4MoeMLP, NOT the generic
	// moeMLP). The dense branch reuses g/u/d; the router reuses routerW (the f32 proj with
	// routerScale·hidden^-0.5 folded into its columns at build) + routerB; the experts reuse
	// expGU/expDown. g4moe selects the path; these are its extra params.
	g4moe                                                  bool
	g4preFFN, g4postFFN1, g4preFFN2, g4postFFN2, g4postFFN Buffer  // the 5 gemma4 RMSNorm weights
	perExpertScaleB                                        Buffer  // [nE] learned scale on the renormalized top-k weights
	layerScalar                                            float32 // per-layer output scalar (out = (h+combined)*layerScalar)

	// Gated-DeltaNet mixer layer (Qwen3.5/3.6-MoE, Qwen3-Next, Qwen3.8). When isDeltaNet this
	// layer's sequence mixer is the recurrent delta rule (deltanet.cu) instead of attention: no
	// KV cache, no q/k/v/o, and TWO persistent state buffers updated in place per token.
	//
	// dnState is [nv*hv*hk] and stored TRANSPOSED relative to the CPU's [hk,hv] so each thread
	// owns a contiguous row. dnWin is the causal-conv ring, [(K-1)*convDim]. Both COMPOUND, so
	// both are re-zeroed per generation (Reset) — unlike a KV cache, which the next sequence
	// simply overwrites.
	isDeltaNet                   bool
	dnQKV, dnZ, dnOut, dnB, dnA  cudaWQ // the five DeltaNet projections
	dnConvW, dnDtBias, dnNegExpA Buffer // conv taps, dt bias, precomputed -exp(A_log)
	dnNormW                      Buffer // [hv] gated-RMSNorm weight, shared across heads
	dnWin, dnState               Buffer // persistent: conv ring, recurrent matrix state
	// qGate marks a SOFTMAX layer of the same family: q_proj is double width ([query ‖ gate] per
	// head) and the context is scaled by sigmoid(gate) before o_proj. The weight stays fused
	// because it is quantized; the split happens on the activation.
	qGate bool

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
	cacheProf      bool    // GOINFER_MOE_CACHE_PROF: time the per-layer routing round trip
	profStall      time.Duration
	profHost       time.Duration
	profDMA        time.Duration
	// Phase 0 (G31 follow-up): split the expert DMA into the BIG weight copy and the TINY scale
	// copy, with bytes and call counts for each. If a ~4 KB scale upload costs nearly as much as a
	// ~600 KB weight upload, the path is PER-CALL-OVERHEAD bound rather than bandwidth bound, and
	// the fix is batching rather than anything cleverer. Zero cost when cacheProf is off.
	profWBytes, profSBytes uint64
	profWCalls, profSCalls uint64
	// The batch that replaced them. profWCalls/profSCalls still count logical COPIES (so the
	// per-token copy count is unchanged and comparable across the change); profSyncCalls counts
	// the SYNCHRONIZES, which is the quantity batching actually moves — ~240/token to ~40.
	// Per-kind timing is gone rather than kept at ~0: with the copies merely appended here and
	// issued together later, a "weight upload took Xµs" figure would name something that no
	// longer happens.
	expBatch []gpu.HostCopy
	// pendingAdmits are the slot claims loadRoutedExperts made before its DMA, rolled back if
	// the upload fails so the cache cannot assert residency for bytes never copied (N-09).
	pendingAdmits []pendingAdmit
	// resetErr holds the last Reset() failure. Reset cannot return one (the cross-backend
	// interface returns nothing), so Forward/ForwardN surface it — N-08.
	resetErr      error
	profBatchTime time.Duration
	profSyncCalls uint64
	profCalls     uint64
	graphs        bool   // CUDA graphs: replay each layer's static segments instead of re-issuing launches (off ⇒ byte-identical)
	graphsSync    bool   // DEBUG probe: r.stream.Sync() after each segment replay (bisects inter- vs intra-segment ordering hazards)
	graphMask     string // DEBUG probe: if non-empty, replay ONLY the named segments (e.g. "A","B","C","AB") and issue the rest live — localizes a replay hazard to a segment
	layerCap      bool   // DEBUG probe: snapshot the residual r.x after every layer (localizes where a full-forward divergence first appears)
	layerCapBuf   [][]float32

	// hidCap is the PRODUCTION hidden-state seam (P10 / docs/spec/08): the resident
	// analogue of decoder.Model.ForwardCapture, which exists only on the CPU forward. A
	// hidden-state drafter (DFlash, DSpark) reads a handful of the target's layer outputs
	// per token; without this a resident target cannot feed one at all.
	//
	// Distinct from layerCap above, deliberately. layerCap is a divergence-localization
	// probe: EVERY layer, a stream.Sync() and a download each, appended to an unbounded
	// buffer. At 36 layers that is 36 syncs per token — fine to bisect a bug with, far too
	// expensive to decode against. hidCap copies only the TAPPED layers into fixed slots,
	// so a 5-tap drafter costs 5, not 36.
	hidCapTaps []int       // layer indices to capture, ascending; nil ⇒ seam off
	hidCapOut  [][]float32 // [len(hidCapTaps)][hidden], overwritten per token
	// The BATCHED counterpart, for the block-drafting verify: one download per tap covering
	// all M rows, instead of the per-token seam's one per tap PER TOKEN.
	capBTaps   []int       // layer indices, ascending; nil ⇒ off
	capBOut    [][]float32 // [len(capBTaps)][M*hidden], overwritten per batched call
	launchN    int         // diagnostic: per-forward dispatch count (graph-capturable-fraction bound)
	cacheSlots int         // C′ step 2: device slots per layer (≥ topK; = topK ⇒ step-1 fresh-load, no reuse)
	slotIdx    Buffer      // C′: per-token slot ids for the routed experts, bound as the GEMV's idx
	hostIdx    []uint32    // C′: scratch for the per-layer rIdx device→host readback
	hostSlot   []uint32    // C′: scratch for the per-token slot ids uploaded to slotIdx

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
	// g4capLayer is g4capIdx's layer index, recorded in parallel so a consumer does not have to
	// INFER the (token, layer) shape from the append order and a divisor. G33 needs to replay the
	// trace through a per-layer cache, and a decisions-count that does not divide evenly by the
	// layer count (2730 over 64 tokens) is exactly the kind of thing an inference gets wrong.
	g4capLayer []int

	// Per-sublayer contribution capture (diagnostic; off in production, zero cost). When subCap
	// is set, launchToken copies the sandwich-normed o-proj output (attention contribution) and
	// down output (MLP contribution) per layer — the exact dp4a-path analogue of the decoder's
	// ForwardSubCapture, so a cross-backend per-sublayer diff is possible.
	subCap                                 bool
	subAttnC, subMLPC, subCtxC, subMLPpreC [][]float32

	sharedInter int // width of the always-on shared expert (0 ⇒ none)

	// device state — touched ONLY on the executor thread.
	dev                                                                                *Device
	stream                                                                             Queue
	gemvW4, gemvW8, ropeKV, fRms, fRmsF32, fQ, fAttn, fSw, fRes, fArg, fQKV, fGU, fQKN Pipeline
	// Batched (M=len) prefill pipelines (prefill_batched.ptx) — the weight-stationary path that fixes
	// the ~128-token Ollama crossover. bGemv is the batched W4A8 GEMV; the rest are the M=1 glue
	// kernels with an M dimension, each bit-identical per row. Loaded once at build (small module).
	bRN, bRms, bRopeKV, bAttn, bQuant, bSw, bRes Pipeline
	bW8                                          Pipeline // batched W8A8 GEMV (int8 bundles); §C6. nil ⇒ int8 prefill declines
	bQKN                                         Pipeline // batched per-head Q/K RMSNorm (qwen3 etc.); loaded with the batched set
	bNormF32                                     Pipeline // batched plain f32 RMSNorm for Gemma sandwich post-norms; loaded with the batched set
	skScores, skSoftmax, skVsum                  Pipeline // Campaign-A split-KV decode attention (high-occupancy, bit-identical)
	skScoreBuf, skInvBuf                         Buffer   // split-KV scratch: [nH·ctxCap] raw/exp scores, [nH] inverse denominators

	// L2 (docs/task-prefill-gap.md §4 L2): the fused prefill attention, one instantiation per
	// supported head dim. Zero-valued unless GOINFER_CUDA_FAST_PREFILL selected it AND the module
	// loaded; every selection site treats the zero Pipeline as "use attn_batched". Kept in its own
	// alignment group so adding them does not re-align (and so re-diff) the fields above.
	bAttnFused64, bAttnFused128 Pipeline
	// L3 (§4 L3): the tensor-core int4 GEMM. Zero-valued unless selected AND loaded; bGemvB
	// treats the zero Pipeline as "use gemv_w4a8_rn", which is the exact path.
	bGemmMMA Pipeline
	// Per-lever selection (§5 attribution): fastAttn drives L2, fastGemm drives L3.
	fastAttn, fastGemm bool
	// gpt-oss only (nil elsewhere, and every other family's launches pass ArgNull so the
	// kernels stay bit-identical): the clamped interleaved-SwiGLU expert epilogue, plus the
	// per-layer attention sinks and the per-expert gate‖up bias table it needs.
	// Gated-DeltaNet mixer (deltanet.ptx — own module; nothing else here is recurrent).
	// Loaded only for that family; every other model leaves these zero and never dispatches them.
	dnConv, dnGates, dnNorm, dnRule, dnGNorm Pipeline
	dnQSplit, dnAttnGate                     Pipeline // the family's fused double-width q_proj + output gate
	dnet                                     *dnetParams

	gptOssSw                 Pipeline // glu_quant_gptoss (own module — audited glue.ptx/moe.ptx untouched)
	gptOssRoute              Pipeline // route_gptoss — top-k + softmax over the BIASED logits (moe_route's contract is wrong for this family)
	gptOssAlpha, gptOssLimit float32
	gptOssSinks              []Buffer // [layer] → [nH] learned per-head attention sink logits
	gptOssDownBias           []Buffer // [layer] → [nExpert*hidden] per-expert down-projection bias
	gptOssExpBias            []Buffer // [layer] → [nExpert·2·moeInter] gate‖up biases, indexed on-device by the router
	splitkvAttn              bool     // GOINFER_SPLITKV_ATTN: use the split-KV decode attention (else the A1 attn_batched(M=1))
	skMinKeys                int      // GOINFER_SPLITKV_MIN_KEYS: -1 ⇒ per-geometry table; ≥0 overrides it (0 ⇒ always split)
	prefillReady             bool     // batched kernels loaded; PrefillLast usable
	// prefillChunkCap is prefillChunked's LEARNED row budget: 0 until a pass OOMs, then the width
	// that worked. It exists so a card that cannot hold the default chunk is discovered ONCE rather
	// than on every prompt. Repeatedly driving the context to CUDA_ERROR_OUT_OF_MEMORY is not merely
	// wasteful: per backend.go's A13 note a context taken to refusal and kept in use can afterwards
	// launch kernels that "return SUCCESS and execute NOTHING". Atomic because prefillChunked runs on
	// the CALLER's goroutine (only the per-pass job is serialized through the executor).
	prefillChunkCap atomic.Int64
	prof            *prefillProf // non-nil ⇒ PrefillLast times each kernel category (test-only; adds stream syncs)
	// passPromptLen is the TOTAL prompt length this batched pass belongs to (startPos+M), set once
	// at the top of prefillCore. The fast-prefill floor is a property of the PROMPT, not of the
	// chunk: prefillChunked splits a long prompt into passes of <=512 rows, so gating on M alone
	// would judge a 3900-token prompt by its 512-row chunk. Per-pass mutable state on the resident,
	// the same shape as prof above.
	passPromptLen                                           int
	fRoute, fRouterGemv, fMoEGemv, fMoEWacc, fSharedCombine Pipeline
	fMoEWaccBias                                            Pipeline // gpt-oss: wacc + per-expert down bias
	fRouterF32, fScaleWgt, fRmsNW, fScaleVec                Pipeline // gemma4 MoE (router_f32 module)

	fuseQKV     bool  // all of Q/K/V/gate/up int4 ⇒ the fused K1 (fQKV) + fGU super-kernels are usable
	launchErr   error // sticky first launch error within a launchToken call (reset per token) — M23
	layers      []cudaLayer
	ctxExplicit bool // the cap came from configuration (Options.ResidentContext), not the default — decides whether a VRAM miss is a hard error or a decline
	ctxCap      int  // effective resident KV capacity in positions = resolveCtxCap(request, model ctx). Every kc/vc is sized cap*kvDim; checkCap guards against it.
	lmW         cudaWQ
	finalNorm   Buffer

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

	// Gated-DeltaNet per-token scratch (allocated only when any layer isDeltaNet). Sized from
	// dnetParams, which is NOT the attention geometry: convDim is 2*nk*hk + nv*hv and has no
	// relation to qDim/kvDim, so none of the buffers above can be reused for it.
	dnMixed, dnConvOut, dnQn, dnKn, dnHeadP, dnBt, dnAt, dnZOut, dnCore, dnGated Buffer
	dnGq, dnGSc                                                                  Buffer // int8 activation + scale for the gated output's out_proj GEMV
	dnQg                                                                         Buffer // [2*qDim] the fused [query ‖ gate] q_proj output, before the split
	dnAGate                                                                      Buffer // [qDim] attention output gate (qGate layers)
	moeQ                                                                         Buffer

	// Gemma-4 MoE branch scratch [hidden] (allocated only when gemma4Moe). x1 = dense branch, x2 =
	// expert-sum branch, rn = the router's weightless-normed raw-h input. Kept SEPARATE from r.x
	// because the join norms (x1+x2) BEFORE adding the residual h — the experts can't wacc into the
	// residual stream the way the generic moeMLP does.
	g4x1, g4x2, g4rn Buffer
	g4zero           []float32 // host [hidden] zeros to clear x2 before the wacc loop (no D2D helper)

	// Shared-expert scratch (allocated only when any layer hasShared). Sized to sharedInter,
	// which is its own width — distinct from both the dense inter and the routed moeInter.
	shGUout, shSc, shScr, shDownOut Buffer
	shGl                            Buffer // [1] the Qwen-MoE shared-gate logit; unused when ungated
	shQ                             Buffer

	// logitsPinned is PAGE-LOCKED host memory for the per-token logits readback. A pageable
	// D2H of 594 KB measured only ~1.26 GB/s (it stages through a driver bounce buffer);
	// pinned memory DMAs straight out. Slice() is a zero-copy view, so Forward still returns
	// without an extra copy. Reused across calls (decode consumes each before the next).
	logitsPinned *HostBuffer[float32]
	logitsHost   []float32 // zero-copy view of logitsPinned
	// Batched-head verify scratch (PrefillLastNArgmax). Allocated lazily INSIDE the executor
	// job — af/ai require r.dev's context current — and reused across rounds. logitsBCap is the
	// row count the current allocation covers; a wider block reallocates once and then holds.
	logitsB    Buffer
	logitsBCap int
	setupErr   error // first alloc/upload error during BuildResident's setup job
	// A1 instrument, RECORDING ONLY — never read by production logic. Free device VRAM immediately
	// before and after allocSlots in the SAME process, so consumption is measured without the
	// cross-run assumption the earlier sweep depended on.
	dbgFreeBefore, dbgFreeAfter uint64
	dbgPredInline               int64  // consumption predicted by the arithmetic production executes
	dbgPredCapSlots             int64  // consumption predicted from capSlots' chosen count
	dbgSlotsInline              int    // slot count the INLINE copy at allocSlots chose
	dbgSlotsCapSlots            int    // slot count capSlots would choose from the same free
	dbgFreeBeforeLaunch         uint64 // free VRAM immediately before the first forward's launches
	dbgFreePreLaunch            uint64 // free VRAM immediately before the MOST RECENT launch
	dbgLaunchTrace              []string
	// dbgProbe enables the per-launch pre-launch probe. A FIELD, not a package var read from the
	// environment: a package var is initialised at package-init time, which is before t.Setenv can
	// run, so the first attempt at this probe silently recorded nothing. The test sets the field on
	// the resident it already holds, which has no ordering question in it. Off by default — the
	// probe costs a MemInfo round-trip and a reflection scan on EVERY launch.
	dbgProbe      bool
	cacheSlotsReq int   // slots REQUESTED (pre-cap), so errors can name both
	dbgAllocSizes []int // every requested slot-buffer size, recording only
}

// alloc/upload helpers — called ONLY inside the setup job (r.dev's context current on the
// executor thread). gpu.NewBufferLenOf PANICS on OOM (recorded into the Device ledger); the
// executor's runJob recover (NOT BuildResident's own defer, which runs on a different goroutine and
// cannot reach an executor-thread panic — that was the C-24 finding) turns it into setupErr, so
// BuildResident declines gracefully (→ staged fallback) instead of proceeding with unusable buffers.
func (r *cudaResident) af(n int) Buffer {
	return gpu.NewBufferLenOf[float32](r.dev, n)
}
func (r *cudaResident) ai(n int) Buffer {
	return gpu.NewBufferLenOf[int32](r.dev, n)
}
func (r *cudaResident) au32(n int) Buffer {
	return gpu.NewBufferLenOf[uint32](r.dev, n)
}

// recordUpload captures the FIRST alloc/upload error hit during BuildResident's setup job into
// r.setupErr (audit C-08). The up* helpers used to discard gpu.Upload's error with `_ =`, so a failed
// weight upload left a ZEROED device buffer and the build still returned ok=true — a resident that
// decodes garbage. The setup job's last statement returns r.setupErr, which BuildResident turns into a
// graceful decline (→ staged/CPU fallback); recording here is what makes that check ever fire. Only the
// first error is kept (later uploads in the same doomed job are noise). Called only at load time.
func (r *cudaResident) recordUpload(e error) {
	if e != nil && r.setupErr == nil {
		r.setupErr = e
	}
}
func (r *cudaResident) up32(v []float32) Buffer {
	b := r.af(len(v))
	r.recordUpload(gpu.Upload(b, v))
	return b
}
func (r *cudaResident) upu32(v []uint32) Buffer {
	b := r.au32(len(v))
	r.recordUpload(gpu.Upload(b, v))
	return b
}
func (r *cudaResident) upu16(v []uint16) Buffer {
	b := gpu.NewBufferLenOf[uint16](r.dev, len(v))
	r.recordUpload(gpu.Upload(b, v))
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
//
// It must be measured from a ROUTED layer, not from layer 0 (audit C-25). Layer 0 is dense on every
// family with `first_k_dense_replace` — GLM-4.5/4.6, DeepSeek-V2/V3, Kimi — so its expGU/expDown
// strides are zero, and the caller's `budget / len(moeLayers) / perLayer` is then an integer divide
// by zero. That panic raised on the EXECUTOR goroutine, which (before C-24) killed the process
// rather than declining. Trigger: GOINFER_MOE_CACHE_EXPERTS=1 on any dense-prefix MoE.
func (r *cudaResident) slotBytesPerLayer(layer int) int {
	if layer < 0 || layer >= len(r.layers) {
		return 0
	}
	gu, dn := &r.layers[layer].expGU, &r.layers[layer].expDown
	return (gu.perExpertW+dn.perExpertW)*4 + (gu.perExpertS+dn.perExpertS)*2
}

// slotStrides is the per-slot byte size of each buffer allocated per MoE layer, IN THE ORDER AND
// GROUPING allocSlots allocates them. The sum is slotBytesPerLayer; the split matters because the
// driver charges each allocation its own whole quanta, so the total is not a function of the sum.
func (r *cudaResident) slotStrides(layer int) []int64 {
	if layer < 0 || layer >= len(r.layers) {
		return nil
	}
	gu, dn := &r.layers[layer].expGU, &r.layers[layer].expDown
	return []int64{
		int64(gu.perExpertW) * 4, int64(gu.perExpertS) * 2,
		int64(dn.perExpertW) * 4, int64(dn.perExpertS) * 2,
	}
}

// allocQuantumBytes is the driver's allocation granularity, MEASURED (cuda/allocgran_test.go:
// 5 MiB -> 6, 6 -> 6, 9 -> 10, so 2 MiB granular and NOT next-power-of-two, which would over-charge
// by up to 2x on any buffer not sitting just above a power of two).
const allocQuantumBytes = 2 << 20

// slotRequirement is the device VRAM n slots/layer actually costs: each buffer rounded up to its own
// quantum, times the layer count. Monotone non-decreasing in n, which is what makes capSlots'
// binary search valid.
func slotRequirement(n int, nLayers int64, strides []int64) int64 {
	var perLayer int64
	for _, p := range strides {
		b := int64(n) * p
		perLayer += (b + allocQuantumBytes - 1) / allocQuantumBytes * allocQuantumBytes
	}
	return nLayers * perLayer
}

// slotMarginBytes is the launch-time headroom the cap must leave free. NOTE it is currently the
// only unmeasured constant in this path: whether 384 MiB is the right figure is open, and the
// per-layer allocation overhead it must absorb has not yet been measured within a single process.
const slotMarginBytes = 384 << 20

// capSlots is the sizing arithmetic, and it is a SEARCH rather than a division.
//
// Per-buffer 2 MiB rounding makes the requirement a step function of the slot count, and
//
//	fit := (free - slotMarginBytes) / nLayers / perLayer
//
// cannot invert a step function — it is wrong precisely at the boundaries the failure lives on. On
// the real 26B that division returned 34, where the true requirement at 34 exceeds free by
// 203,816,960 B: at n=34 the ratio n*123904/2MiB crosses 2 and all four buffers tip a quantum AT
// ONCE, a 4-quanta step. The forward then generated zero tokens. A division plus a correction term
// would reproduce the same class one boundary over, so there is no fudge factor here.
//
// The requirement is monotone non-decreasing in n, so bisect it. Pure function of its inputs, so
// synthetic free-VRAM figures exercise a branch that in production binds only on models far larger
// than any fixture — the exercised-but-never-triggered shape.
//
// Returns the slot count to use, or decline=true when not even topK fits.
func capSlots(free, nLayers int64, strides []int64, topK, request int) (slots int, decline bool) {
	if nLayers <= 0 || len(strides) == 0 {
		return request, false
	}
	var any bool
	for _, p := range strides {
		if p > 0 {
			any = true
		}
	}
	if !any {
		return request, false
	}
	fits := func(n int) bool { return slotRequirement(n, nLayers, strides)+slotMarginBytes <= free }
	if fits(request) {
		return request, false
	}
	// Largest n in [0, request) that fits. lo always fits (n=0 costs nothing), hi never does.
	lo, hi := 0, request
	for hi-lo > 1 {
		mid := (lo + hi) / 2
		if fits(mid) {
			lo = mid
		} else {
			hi = mid
		}
	}
	if lo < topK {
		return lo, true
	}
	return lo, false
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
	// Size from the FIRST ROUTED layer (moeLayers[0]), not layer 0 — see slotBytesPerLayer (C-25).
	perLayer := r.slotBytesPerLayer(moeLayers[0])
	if perLayer <= 0 {
		// No routed layer reports a per-expert stride (a degenerate blob): nothing to cache and,
		// more importantly, nothing to divide by. Clear cacheExperts so the decode path can't later
		// read slots/expCache that were never allocated → nil-deref (audit R-25). NOTE (F-05): this
		// does NOT yield a working "hold every expert" resident path — with caching off and no stride,
		// upExperts left the expert stacks host-mapped-only, so the expert GEMVs would bind zero-value
		// device buffers. This branch is unreachable with any real MoE checkpoint (they all report a
		// per-expert stride); it exists only to fail safe, not to serve. A blob that trips it should be
		// declined to the staged/CPU path upstream.
		r.cacheExperts = false
		fmt.Fprintf(os.Stderr, "[cuda] C′ cache: routed layer %d reports zero per-expert bytes — expert cache disabled\n", moeLayers[0])
		return nil
	}
	strides := r.slotStrides(moeLayers[0])
	if f0, _, e0 := r.dev.Context().MemInfo(); e0 == nil {
		r.dbgFreeBefore = f0 // A1 instrument, recording only
	}
	if free, _, err := r.dev.Context().MemInfo(); err == nil {
		// ONE implementation. This used to be an inline copy of the same arithmetic, with capSlots
		// existing only for the gate — so the gate corroborated a parallel copy and a change to
		// either was uncontradicted by the other (the sibling-drift instance in
		// docs/parity-coverage-policy.md). The gate now points at the shipping path.
		fit, decline := capSlots(int64(free), int64(len(moeLayers)), strides, r.topK, r.cacheSlots)
		if decline {
			// FLOOR. topK slots is the minimum that can work — one token's routed set must be
			// simultaneously resident — so if even that does not fit, DECLINE naming the shortfall
			// instead of allocating and discovering it at the first kernel launch.
			//
			// This used to read `if capped < r.topK { capped = r.topK }`, commented "topK always
			// fits". That is an assumption written as a check: when false it clamps UP to a figure
			// it has just computed does not fit, allocates it, and the failure surfaces later as
			// CUDA_ERROR_OUT_OF_MEMORY from cuLaunchKernel or a generation loop returning nothing —
			// neither of which points back here.
			need := slotRequirement(r.topK, int64(len(moeLayers)), strides)
			return fmt.Errorf("expert cache (C′) cannot fit its MINIMUM: top-%d routed experts across %d MoE "+
				"layers need %.2f GB of slots, but only %.2f GB is free — %d slots/layer fit. Free VRAM, "+
				"lower --ctx, or drop GOINFER_MOE_CACHE_EXPERTS and use a card that holds the experts outright",
				r.topK, len(moeLayers), float64(need)/1e9, float64(free)/1e9, fit)
		}
		if fit < r.cacheSlots {
			fmt.Fprintf(os.Stderr, "[cuda] C′ cache: %d slots/layer would need %.1f GB VRAM but only %.1f GB free — "+
				"capping to %d (%.1f GB)\n", r.cacheSlots,
				float64(slotRequirement(r.cacheSlots, int64(len(moeLayers)), strides))/1e9, float64(free)/1e9,
				fit, float64(slotRequirement(fit, int64(len(moeLayers)), strides))/1e9)
			r.cacheSlots = fit
		}
	}
	// A10 instrument, recording only. Read at CALL time, not package-init time, so a test's
	// t.Setenv reaches it — the opposite mistake cost a full 26B run once already.
	//
	// The question it answers: allocSlots can fail mid-sequence on an individual buffer while the
	// closed form says the TOTAL fits with room to spare (observed at 34 slots: a 67,403,776 B
	// request refused with 61,014,016 B of predicted total slack, and 182,648,832 B predicted free
	// at that point). Recording free before every allocation separates "free was genuinely below the
	// request", which would mean the form is wrong near the boundary, from "free was ample and the
	// allocation failed anyway", which is servability rather than capacity.
	a10 := os.Getenv("GOINFER_A10_PROBE") != ""
	allocN := 0
	probe := func(nBytes int, alloc func()) {
		if !a10 {
			alloc()
			allocN++
			return
		}
		var freeNow uint64
		if f, _, e := r.dev.Context().MemInfo(); e == nil {
			freeNow = f
		}
		// The failure arrives as a panic from gpu's MustBuf, and the resident is discarded on
		// decline — so the diagnostic has to be emitted HERE, before the panic propagates, or the
		// run reports only that something declined.
		defer func() {
			if p := recover(); p != nil {
				fmt.Fprintf(os.Stderr, "[a10] alloc #%d FAILED: requested %d B, free immediately "+
					"before = %d B (%.1f MiB), ratio free/request = %.2f — %v\n",
					allocN, nBytes, freeNow, float64(freeNow)/(1<<20),
					float64(freeNow)/float64(nBytes), p)
				panic(p)
			}
		}()
		fmt.Fprintf(os.Stderr, "[a10] alloc #%3d: %10d B, free before %13d B\n", allocN, nBytes, freeNow)
		alloc()
		allocN++
	}
	// Issue LARGEST FIRST across all layers, rather than group-by-group. Total is identical either
	// way — capSlots and the granularity form are untouched, only the order moves.
	//
	// HONEST ACCOUNT OF WHY: this was A10's hypothesised FIX and it was REFUTED as one. The theory
	// was that group-by-group (30 repetitions of {64.3, 8.0, 32.1, 4.0} MiB) leaves the last layer's
	// biggest request facing the most carved-up heap. Largest-first was predicted to complete at 34
	// slots; it did not. It failed on a 4,212,736 B request with 155,385,856 B free — a ratio of
	// 36.88, which no contiguity story survives.
	//
	// The real constraint is a driver ALLOCATION FLOOR: 151,191,552 B (144.2 MiB) that cuMemGetInfo
	// reports as free and cuMemAlloc will not hand out at ANY request size down to 1 MiB, measured
	// directly in TestAllocFloor. Leftover after allocSlots must exceed it, which is what the margin
	// now provides.
	//
	// This ordering is KEPT anyway, on its measured merit and not on the refuted theory: it drains
	// 27 MiB further before hitting the floor (155,385,856 vs 182,648,832) and packs more bytes
	// before failing, at zero cost. It is a packing improvement, not a fix.
	//
	// Sorting by MEASURED request size rather than by an assumed stride order, because per-layer
	// geometry is not guaranteed uniform (Gemma 4's KV widths already differ per layer, and
	// slotBytesPerLayer exists because layer 0 can be dense). A stable sort keeps layer order within
	// each size class, so the sequence stays deterministic.
	type slotReq struct {
		w    *cudaWQ
		n    int // element count for the allocator
		size int // bytes, for ordering and for the probe
		isW  bool
	}
	reqs := make([]slotReq, 0, 4*len(moeLayers))
	for _, i := range moeLayers {
		L := &r.layers[i]
		for _, w := range []*cudaWQ{&L.expGU, &L.expDown} {
			reqs = append(reqs,
				slotReq{w, r.cacheSlots * w.perExpertW, r.cacheSlots * w.perExpertW * 4, true},
				slotReq{w, r.cacheSlots * w.perExpertS, r.cacheSlots * w.perExpertS * 2, false})
		}
	}
	sort.SliceStable(reqs, func(a, b int) bool { return reqs[a].size > reqs[b].size })
	for _, q := range reqs {
		q := q
		if q.isW {
			probe(q.size, func() { q.w.W = r.au32(q.n) })
		} else {
			probe(q.size, func() { q.w.ws16 = gpu.NewBufferLenOf[uint16](r.dev, q.n) })
		}
		// A1, recording only: the REQUESTED byte size of each allocation, so predicted consumption
		// can be recomputed with per-buffer 2 MiB rounding instead of a raw sum. Recorded in ISSUE
		// order, which is now sorted; every gate on this reads it order-independently (counts,
		// distinct sizes, occurrences, sum).
		r.dbgAllocSizes = append(r.dbgAllocSizes, q.size)
	}
	for _, i := range moeLayers {
		r.layers[i].expCache = newExpertCache(r.nE, r.cacheSlots)
	}
	if f1, _, e1 := r.dev.Context().MemInfo(); e1 == nil {
		r.dbgFreeAfter = f1 // A1 instrument, recording only
	}
	// One prediction now, because there is one implementation. These used to record BOTH the inline
	// copy's choice and capSlots' choice, precisely because they could disagree.
	r.dbgSlotsInline = r.cacheSlots
	r.dbgPredInline = slotRequirement(r.cacheSlots, int64(len(moeLayers)), strides)
	r.dbgSlotsCapSlots = r.cacheSlots
	r.dbgPredCapSlots = r.dbgPredInline
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

	// DEMAND accounting, and it answers a different question from hits/misses. hits/misses say how
	// much DMA reuse SAVED; these say how much the caller ASKED FOR per staging event, which is the
	// quantity that separates "speculation blows the slot budget" from "speculation is just slow
	// here". stages counts loadRoutedExperts calls on this layer; distinct sums the UNIQUE experts
	// each stage requested. The ratio distinct/stages is the causal number: a verify that presented
	// its whole width to the pager at once would push it ABOVE topK and grow with the width, while
	// one that stages position-by-position pins it AT topK no matter how wide the verify is.
	stages   uint64
	distinct uint64
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

// unadmit marks slot empty and forgets expert e, so the next admit(e) misses and re-uploads.
//
// N-09: admit commits the slot mapping BEFORE the DMA, which is what lets the caller batch a
// whole layer's copies into one UploadBatch. If that upload fails, the cache is left claiming
// expert e is resident in a slot that holds the EVICTED expert's bytes — and the next token to
// route e gets hit=true, skips the DMA, and runs the wrong expert with no error anywhere.
//
// Rolled back to EMPTY rather than to the evicted expert: a failed batch may have completed some
// of its copies, so the slot's contents are unknown. Claiming nothing costs one re-upload;
// claiming the old expert would be the same bug with a different victim.
func (c *expertCache) unadmit(slot int, e uint32) {
	if c.inSlot[slot] == int32(e) {
		c.inSlot[slot] = -1
	}
	if c.slotOf[e] == int32(slot) {
		c.slotOf[e] = -1
	}
}

// appendExpertSlot QUEUES one expert's weight+scales copy from the pinned host source into device
// slot `slot`. It does not upload: loadRoutedExperts submits the layer's whole batch with a single
// gpu.UploadBatch, so one synchronize covers every miss in the layer instead of two per miss.
//
// Why the change is the sync count and not the copy. Each gpu.Upload ends in a full device
// Synchronize — right for per-request uploads, wrong here: a MoE decode token loads ~120 slots at
// two uploads each, so ~240 synchronizes land on one token. Measured on an RTX 2070 SUPER that is
// ~3.6 ms of a 64 ms token (5.6%), paid for nothing, since the bytes are already in flight and one
// sync at the end covers them all.
//
// The correctness property is PRESERVED, not weakened. UploadBatch still synchronizes before it
// returns, so the guarantee is identical to Upload's and the race that sync was added for
// (non-blocking streams unordered against the null stream) stays covered. Nothing moves into a
// caller's hands — that is the whole difference from the declined async variant.
func (r *cudaResident) appendExpertSlot(w *cudaWQ, e, slot int) {
	srcW, srcS := w.srcW.Bytes(), w.srcS.Bytes()
	wOff, wLen := e*w.perExpertW*4, w.perExpertW*4
	sOff, sLen := e*w.perExpertS*2, w.perExpertS*2
	r.expBatch = append(r.expBatch,
		gpu.HostCopy{Dst: w.W.At(slot * w.perExpertW * 4), Src: srcW[wOff : wOff+wLen]},
		gpu.HostCopy{Dst: w.ws16.At(slot * w.perExpertS * 2), Src: srcS[sOff : sOff+sLen]},
	)
	if r.cacheProf {
		r.profWBytes += uint64(wLen)
		r.profWCalls++
		r.profSBytes += uint64(sLen)
		r.profSCalls++
	}
}

// loadRoutedExperts reads back the router's idx (device→host — C′'s acknowledged per-layer sync),
// admits each routed expert into the layer's slot cache (DMAing only cache misses), and uploads the
// per-token slot ids the GEMV binds as `idx` (slot j ← the slot now holding routed expert j).
func (r *cudaResident) loadRoutedExperts(L *cudaLayer) error {
	// C′ TIMING SEAM (GOINFER_MOE_CACHE_PROF). This function is the only host round trip on the
	// decode path, and it happens once per MoE layer per token — 40 times for the 35B. It splits
	// into three costs that call for completely different fixes, and tok/s alone cannot separate
	// them:
	//
	//   stall  the stream drain, waiting for the router kernel. Fixing this means removing the
	//          round trip (device-side slot mapping), not making it faster.
	//   host   the LRU bookkeeping. Pure CPU; fixing it is ordinary optimization.
	//   dma    the H2D of missed experts. Fixing this means more slots or fewer bytes — and the
	//          slot sweep already showed that lever saturating, so if dma is small the knee is
	//          explained and more slots really are pointless.
	//
	// Off by default and zero cost when off (one branch); it adds no syncs of its own, because the
	// stall it measures is a sync that already exists.
	var t0 time.Time
	if r.cacheProf {
		t0 = time.Now()
	}
	if e := r.stream.Sync(); e != nil {
		return e
	}
	if r.cacheProf {
		r.profStall += time.Since(t0)
		t0 = time.Now()
	}
	if e := gpu.Download(r.rIdx, r.hostIdx[:r.topK]); e != nil {
		return e
	}
	c := L.expCache
	// Demand accounting (see expertCache). Counted BEFORE admission on purpose: the number must
	// describe what this stage REQUESTED, not what the cache happened to already hold, or it would
	// fall as the cache warmed and stop being a measure of demand at all. topK is ~8, so the
	// quadratic scan is orders of magnitude cheaper than the DMA it describes.
	c.stages++
	for j := 0; j < r.topK; j++ {
		dup := false
		for k := 0; k < j; k++ {
			if r.hostIdx[k] == r.hostIdx[j] {
				dup = true
				break
			}
		}
		if !dup {
			c.distinct++
		}
	}
	r.expBatch = r.expBatch[:0] // reused across layers; the slice keeps its capacity
	// N-09: every admit made in THIS call is provisional until the batch lands. On any error
	// below they are rolled back, so a failed upload cannot leave the cache asserting residency
	// for bytes that were never copied.
	r.pendingAdmits = r.pendingAdmits[:0]
	for j := 0; j < r.topK; j++ {
		e := r.hostIdx[j]
		slot, hit := c.admit(e)
		r.hostSlot[j] = uint32(slot)
		if !hit {
			r.pendingAdmits = append(r.pendingAdmits, pendingAdmit{slot: slot, expert: e})
			var td time.Time
			if r.cacheProf {
				r.profHost += time.Since(t0)
				td = time.Now()
			}
			r.appendExpertSlot(&L.expGU, int(e), slot)
			r.appendExpertSlot(&L.expDown, int(e), slot)
			if r.cacheProf {
				r.profDMA += time.Since(td)
				t0 = time.Now()
			}
		}
	}
	// One synchronize for the WHOLE layer: every expert-slot miss plus the per-token slot-index
	// upload the GEMV reads this round's routing from, folded into the SAME batch (P-21). This
	// used to be two separate calls — UploadBatch for the misses, then a lone gpu.Upload for
	// slotIdx — each paying its own synchronize; slotIdx is always present (hit or miss) and
	// tiny, so there was nothing to gain from keeping it apart. All destinations are slot buffers
	// on the one resident context, so UploadBatch's mixed-context refusal should never fire here
	// — if it does, something about the layer's buffers is not what this code believes.
	r.expBatch = append(r.expBatch, gpu.HostCopy{Dst: r.slotIdx, Src: u32bytes(r.hostSlot[:r.topK])})
	var tb time.Time
	if r.cacheProf {
		r.profHost += time.Since(t0)
		tb = time.Now()
	}
	e := gpu.UploadBatch(r.expBatch)
	if e != nil {
		// The expert bytes and/or the slot-index table may be partially applied — the cheap
		// conservative move is to make the next call re-establish both (N-09).
		r.rollbackAdmits(c)
	}
	if r.cacheProf {
		r.profBatchTime += time.Since(tb)
		r.profSyncCalls++
		r.profDMA += time.Since(tb)
		r.profCalls++
	}
	return e
}

// pendingAdmit is one provisional slot claim made by loadRoutedExperts before its DMA.
type pendingAdmit struct {
	slot   int
	expert uint32
}

// rollbackAdmits undoes every admit made in the current loadRoutedExperts call (N-09).
func (r *cudaResident) rollbackAdmits(c *expertCache) {
	for _, p := range r.pendingAdmits {
		c.unadmit(p.slot, p.expert)
	}
	r.pendingAdmits = r.pendingAdmits[:0]
}

// CacheProfForTest reports the C′ round-trip decomposition (stall / host / dma) and the call
// count. Zero unless GOINFER_MOE_CACHE_PROF is set.
func (r *cudaResident) CacheProfForTest() (stall, host, dma time.Duration, calls uint64) {
	return r.profStall, r.profHost, r.profDMA, r.profCalls
}

// UploadProfForTest reports the expert-DMA split: the big weight copies vs the tiny scale copies,
// by bytes moved and COPY count. Zero unless GOINFER_MOE_CACHE_PROF is set.
//
// The per-kind ELAPSED TIMES this used to return are gone, deliberately. The copies are now queued
// by appendExpertSlot and issued together by one UploadBatch, so there is no longer a per-kind
// upload to time; returning the append cost under the old names would have been an accessor whose
// name promised a measurement it no longer makes. The transfer+sync time is BatchProfForTest's.
func (r *cudaResident) UploadProfForTest() (wB, sB, wC, sC uint64) {
	return r.profWBytes, r.profSBytes, r.profWCalls, r.profSCalls
}

// BatchProfForTest reports the batched-upload cost: total time inside gpu.UploadBatch and the
// number of SYNCHRONIZES it performed (one per layer with at least one miss). This is the quantity
// the batching change moves — copy count is unchanged, sync count is not. Zero unless
// GOINFER_MOE_CACHE_PROF is set.
func (r *cudaResident) BatchProfForTest() (batchTime time.Duration, syncCalls uint64) {
	return r.profBatchTime, r.profSyncCalls
}

// THERE ARE TWO INDEX SPACES HERE AND THEY DIVERGE ONLY WHEN EXPERT CACHING IS ON. Keeping them
// as two named accessors rather than one is the whole guard: with caching off they return the
// same buffer, so a site that binds the wrong one is correct in every configuration anyone has
// run, and wrong — silently, with plausible logits — in the one configuration that needs it.
//
//	expIdx        WHERE the weights live  → slot ids when caching, expert ids otherwise
//	expertBiasIdx WHICH expert is running → ALWAYS expert ids
//
// expIdx is the idx argument the expert GEMVs bind: the slot ids when caching (slot j holds
// routed expert j), else the router's real rIdx (fully-resident path).
func (r *cudaResident) expIdx() Buffer {
	if r.cacheExperts {
		return r.slotIdx
	}
	return r.rIdx
}

// expertBiasIdx is the index for PER-EXPERT TABLES that are uploaded ONCE for all experts and
// indexed on the device — today only gpt-oss's [nExpert][2*I] gate‖up bias table. It is always
// the router's real expert ids, NEVER the slot ids: the table is expert-indexed and does not
// move when an expert is streamed into a slot.
//
// Binding expIdx here instead was a live defect (fixed 2026-08-31, never shipped in a run):
// glu_quant_gptoss does `biasGU + idx[slot]*2*I`, so with caching on it would have selected the
// bias row by SLOT id — the wrong expert's gate/up biases, no error, plausible output. It could
// not be caught by any test that exists because gpt-oss has never been admitted on CUDA, and
// expert caching is exactly the path gpt-oss needs to fit an 8 GB card at all.
func (r *cudaResident) expertBiasIdx() Buffer { return r.rIdx }

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

// runJob is the executor goroutine's PANIC BOUNDARY: it runs one job and converts a panic into
// an ordinary error, so the pinned thread survives and `do` returns to its caller (audit C-24).
//
// Why it has to live here and not at a call site. Every job runs on the executor goroutine, but
// `do` blocks on a *different* goroutine — so a `defer recover()` in BuildResident, or in any
// caller, cannot catch a panic raised inside `j()`. Two comments (resident.go's setup path and
// prefill.go's scratch note) asserted this was already handled; it was not, and the gap is
// reachable by design rather than by accident: `gpu.NewBufferLenOf` PANICS on allocation failure
// per its own contract, and prefillCore allocates M*(2*inter+2*hidden+…) floats — hundreds of MB
// on a long prompt. So a long prompt against a nearly-full card killed the serve process, at the
// one seam whose entire job is to decline to the sequential path instead.
//
// The recovered error is deliberately NOT wrapped in errPrefillDeclined here: `do` is shared by
// every job (setup, decode, prefill), and only the prefill caller knows a decline is the right
// response. prefillCore wraps it at its own boundary.
func runJob(j func() error) (err error) {
	defer func() {
		if p := recover(); p != nil {
			// The STACK, not just the value. A recovered executor panic becomes a resident
			// DECLINE, printed once to stderr — and "device allocation failed (0 bytes)" with no
			// frame is unactionable: it took four ~9-minute 35B load cycles to localize one to a
			// dense-FFN scratch buffer that a pure-MoE model has no width for. The stack is a few
			// KB on a path that runs at most once per model load.
			err = fmt.Errorf("cuda: executor job panicked: %v\n%s", p, debug.Stack())
		}
	}()
	return j()
}

// checkKVFits fails the LOAD if the KV the configured cap implies does not fit beside what is
// already on the device. It runs after the weights are uploaded and before the K/V caches are
// allocated, so `free` is genuinely "what is left for KV".
//
// The point is the failure MODE, not the arithmetic: without this, an over-large cap surfaces as an
// allocation failure part-way through sizing the per-layer caches, or worse as an OOM mid-decode
// under load — after the server reported ready. Naming the number at startup turns a production
// incident into a config error, so the message states what was asked for, what it costs, and what is
// actually free.
// errKVWontFit marks the one setup failure that must NOT degrade quietly to the staged path: an
// explicitly configured resident context whose KV does not fit. BuildResident turns it into a hard
// startup error instead of a decline. An UNCONFIGURED (default-cap) miss stays a decline, because
// that is the historical behaviour for "this device cannot host this model".
var errKVWontFit = errors.New("resident KV does not fit in device memory")

// kvWontFitError carries the full operator-facing message while still matching errKVWontFit under
// errors.Is. A plain %w wrap would append the sentinel's text to a message that already says all of
// this, so the reader sees it twice — the classification must not leak into the prose.
type kvWontFitError struct{ msg string }

func (e *kvWontFitError) Error() string        { return e.msg }
func (e *kvWontFitError) Is(target error) bool { return target == errKVWontFit }

func (r *cudaResident) checkKVFits() error {
	need := kvBytesForCap(r.ctxCap, r.layers)
	free, _, err := r.dev.Context().MemInfo()
	if err != nil {
		// No MemInfo ⇒ no fit check possible. Do not fail the load on that: the default cap has
		// always been allocated without one, and a hard failure here would regress every driver
		// that does not report memory. The allocation itself still errors if it truly cannot fit.
		return nil
	}
	if need+ctxCapMarginBytes <= int64(free) {
		return nil
	}
	perPos := float64(need) / float64(max(r.ctxCap, 1)) / 1024
	e := &kvWontFitError{msg: fmt.Sprintf("cuda: resident context %d positions needs %.2f GB of KV "+
		"(%.1f KB/position across %d layers) but only %.2f GB is free on the device beside the weights "+
		"(plus %.0f MB reserved for driver and decode scratch) — lower the serve context setting (-ctx), "+
		"or use a smaller/more-quantized model",
		r.ctxCap, float64(need)/1e9, perPos, len(r.layers), float64(free)/1e9, float64(ctxCapMarginBytes)/(1<<20))}
	if !r.ctxExplicit {
		// Default cap: keep the historical decline. Strip the sentinel so BuildResident treats it as
		// an ordinary "cannot host this here" and the staged path takes over, as it always has.
		return fmt.Errorf("cuda: default resident context %d positions does not fit (%.2f GB of KV, %.2f GB free) — staged path",
			r.ctxCap, float64(need)/1e9, float64(free)/1e9)
	}
	return e
}

// checkCap guards the resident KV allocation. Every layer's cache is sized r.ctxCap*kvDim, so
// a write for absolute position p lands at kc[p*kvDim ...]; valid positions are [0, r.ctxCap).
// Writing past it (rope_kv/kv_store) is an out-of-bounds DEVICE write — silent memory corruption,
// UB, and the attention launch's shared-mem request eventually exceeds the block limit. Nothing
// upstream clamps prompt+max_tokens to the cap, so return an error here; the decode loop stops on
// it (model.go) and the caller can fall back to the staged path, which handles longer contexts.
//
// The cap is now configuration-derived (resolveCtxCap) rather than a constant, but the invariant
// this guards is unchanged: it is always the capacity the caches were actually SIZED with, read from
// the same field the allocation used. A request beyond a configured cap still fails here, cleanly.
func (r *cudaResident) checkCap(pos, n int) error {
	if pos < 0 || pos+n > r.ctxCap {
		return fmt.Errorf("cuda: KV position %d(+%d) exceeds resident context cap %d — raise it with the serve context setting (bounded by the model's own context window), or use the staged path for longer contexts", pos, n, r.ctxCap)
	}
	return nil
}

// ContextCap is the resident KV capacity in positions (queryable so callers can clamp max_tokens
// up front rather than discover the limit mid-generation).
func (r *cudaResident) ContextCap() int { return r.ctxCap }

// Forward runs one token at absolute position pos and returns logits[vocab].
func (r *cudaResident) Forward(embedding []float32, pos int) ([]float32, error) {
	if e := r.checkCap(pos, 1); e != nil {
		return nil, e
	}
	if pos == 0 {
		// Fresh sequence: re-zero the compounding Gated-DeltaNet state so it does not carry over
		// from a prior Generate on this *Model (audit C-01). No-op for every other family.
		r.Reset()
		if r.resetErr != nil {
			return nil, r.resetErr // N-08: a failed re-zero decodes from the previous sequence
		}
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
	if pos == 0 {
		r.Reset() // prefill from 0 is also a fresh sequence — same reason as Forward
		if r.resetErr != nil {
			return r.resetErr // N-08
		}
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
	if len(embeddings) == 0 {
		return nil, nil // no-op on empty, matching the cpu/webgpu ForwardN contract (audit R-21) —
		// prefillReady would otherwise route into prefillCore and return a spurious "empty prompt" error.
	}
	if e := r.checkCap(startPos, len(embeddings)); e != nil {
		return nil, e
	}
	// Batched verify: the whole [cur, draft…] run in ONE weight-stationary pass (prefillCore,
	// allLogits=true) instead of len(embeddings) sequential decode steps — the amortization that
	// makes speculative decode a win on the resident CUDA path (each weight read once for all M
	// positions, not once per token). Bit-identical to the sequential step loop below (every batched
	// kernel is the M=1 kernel with an M dimension + explicit-FMA, gated by TestSpecDecodeCurve's
	// lossless-vs-sequential check). Falls back to the per-token loop for archs the batched path
	// doesn't cover (MoE / K=V / non-int4 / non-uniform geometry) — errPrefillDeclined only; a real
	// compute error propagates. spec verify runs M≤9, so batched allocation is tiny (no OOM concern).
	// The BATCHED (weight-stationary) path has no notion of recurrent state: it runs M rows in one
	// pass over the weights, while a Gated-DeltaNet layer's conv ring and matrix state must advance
	// strictly one token at a time and in order. Take the sequential loop below, which is the same
	// r.step this family's decode uses, so prefill and decode see the same state evolution.
	if r.prefillReady && r.dnet == nil {
		// context.Background(): ForwardN is the spec-decode verify, M<=9 rows, and its own
		// interface carries no context. Nothing here is long enough to want cancelling.
		if outs, _, err := r.prefillCore(context.Background(), embeddings, startPos, tailAllLogits); err == nil {
			return outs, nil
		} else if !errors.Is(err, errPrefillDeclined) {
			return nil, err
		}
	}
	if startPos == 0 {
		r.Reset() // fresh sequence — the recurrent state compounds and is not positional
		if r.resetErr != nil {
			return nil, r.resetErr // N-08
		}
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
// Reset re-zeroes the compounding recurrent state for a fresh generation. A KV cache needs no
// reset — the next sequence overwrites it positionally, which is why this was empty — but a
// Gated-DeltaNet layer's {conv ring, matrix state} is NOT positional: it accumulates, and without
// this the next Generate on the same *Model decodes from the previous one's state (audit C-01's
// shape). The state buffers are allocated zeroed, so the FIRST generation is correct either way,
// which is exactly what makes the omission invisible to a single-sequence test.
func (r *cudaResident) Reset() {
	if r.dnet == nil {
		return
	}
	// N-08: the error was dropped. Reset() satisfies the cross-backend ResidentForward interface
	// and returns nothing, so it cannot be returned — but a failed re-zero is precisely C-01's
	// shape with a silent failure mode: the next sequence decodes from the PREVIOUS one's
	// recurrent state, and the state buffers being allocated zeroed means the first generation
	// is correct either way, so nothing single-sequence can see it. Recorded and surfaced by the
	// next Forward/ForwardN, which is the same shape gpu/residency.go uses (N-15).
	r.resetErr = r.do(func() error { return r.resetState() })
}

// resetState is Reset's body WITHOUT the executor hop, for callers already running on the executor
// thread. graphsSelfTest is one: it runs inside r.do, so calling Reset there posts to reqCh from
// the goroutine that services reqCh and deadlocks — a hang, not an error, and one that only appears
// once graphs are actually admitted for a recurrent model.
func (r *cudaResident) resetState() error {
	if r.dnet == nil {
		return nil
	}
	dp := r.dnet
	winZ := make([]float32, (dp.convK-1)*dp.convDim)
	stZ := make([]float32, dp.stateElems)
	for i := range r.layers {
		L := &r.layers[i]
		if !L.isDeltaNet {
			continue
		}
		if e := gpu.Upload(L.dnWin, winZ); e != nil {
			return e
		}
		if e := gpu.Upload(L.dnState, stZ); e != nil {
			return e
		}
	}
	return nil
}

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
//
// Returns error to satisfy io.Closer (audit B-12; see the assertion in backend.go); the native
// releases are best-effort and can't meaningfully fail, so it always returns nil — like metal's.
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
	if r.dbgFreeBeforeLaunch == 0 { // A1: free VRAM at the FIRST launch, recording only
		if f0, _, e0 := r.dev.Context().MemInfo(); e0 == nil {
			r.dbgFreeBeforeLaunch = f0
		}
	}
	// A1 item 2, recording only: free VRAM immediately BEFORE the launch, i.e. on the other side of
	// the event from describeLaunchErr's reading. That reading is reached only after Launch returns
	// non-nil, so it cannot distinguish "memory was released before this launch was attempted" from
	// "the failed attempt released it while unwinding" — the two hypotheses differ by where the
	// probe sits, not by what happened. Recording the pre-launch value at every launch also yields
	// the decrement A9 asks for (free at first launch → free at the failing launch) from one run.
	if r.dbgProbe {
		if f0, _, e0 := r.dev.Context().MemInfo(); e0 == nil {
			r.dbgFreePreLaunch = f0
			if len(r.dbgLaunchTrace) < 512 {
				r.dbgLaunchTrace = append(r.dbgLaunchTrace,
					fmt.Sprintf("%3d %-14s free=%d", len(r.dbgLaunchTrace), r.pipeName(f), f0))
			}
		}
	}
	r.launchN++ // per-token dispatch count (diagnostic: graph-capturable-fraction bound)
	e := r.stream.Launch(f, cfg, args...)
	if e != nil {
		e = r.describeLaunchErr(f, e)
	}
	if e != nil && r.launchErr == nil {
		// Sticky: launchToken's dense hot chain discards many launch errors (`_ = r.launch(...)`),
		// so a config error (bad shared-mem size, bad args) would let the token "succeed" with
		// stale buffers. Record the first here; launchToken returns it (M23). doG/rms funnel through
		// launch too, so this covers the whole chain without touching every call site.
		r.launchErr = e
	}
	return e
}

// pipeName recovers the struct-field name a Pipeline was bound to, by scanning cudaResident's
// fields for one equal to it. Pipeline is a single-pointer struct, so it compares by identity.
//
// aikit's Pipeline carries no name (it is `struct{ f *gc.Function }`) and lives in another module,
// so the name cannot come from the value itself — but goinfer binds every pipeline to a named
// field, which makes the mapping recoverable here. Error path only; the reflection cost never
// touches a working launch.
func (r *cudaResident) pipeName(f Pipeline) (name string) {
	// TOTAL BY CONSTRUCTION. This runs only when something has already failed, so it must not be
	// able to turn a diagnosable error into a panic — not on a nil receiver, not on an unexported
	// field, not on a pipeline held somewhere other than a named field. The recover is the
	// guarantee; "all 48 sites resolve today" is an observation about today.
	name = "?"
	defer func() {
		if recover() != nil {
			name = "?"
		}
	}()
	if r == nil {
		return "?"
	}
	v := reflect.ValueOf(r).Elem()
	t := v.Type()
	pt := reflect.TypeOf(Pipeline{})
	for i := 0; i < t.NumField(); i++ {
		if t.Field(i).Type != pt {
			continue
		}
		fv := reflect.NewAt(pt, unsafe.Pointer(v.Field(i).UnsafeAddr())).Elem().Interface().(Pipeline)
		if fv == f {
			return t.Field(i).Name
		}
	}
	return "?"
}

// describeLaunchErr turns a bare driver status into something that names what ran out and what the
// operator can do about it.
//
// The motivating case: a 26B at 34 expert-cache slots died with exactly
// `cuLaunchKernel: CUDA_ERROR_OUT_OF_MEMORY` — the API call, not the kernel, and no connection to
// the setting that caused it. A day of investigation started from that string. The decline floor
// added alongside does NOT cover this: it fires below topK, and this failed at 34 slots with topK
// of 8, so the path most in need of an honest failure had none.
//
// Message content only — the error is wrapped with %w, so type and classification are unchanged.
func (r *cudaResident) describeLaunchErr(f Pipeline, e error) error {
	name := r.pipeName(f)
	// Reframe an out-of-memory launch against the expert cache when it is on: the slot count is by
	// far the largest operator-controlled VRAM consumer on this path. Deliberately does NOT suggest
	// a specific safe value — the computation that would produce one (capSlots) is the thing under
	// suspicion, and printing a number from it would launder a suspect figure into advice.
	if r.cacheExperts && strings.Contains(e.Error(), "OUT_OF_MEMORY") {
		// Name the REQUESTED and EFFECTIVE counts both. They differ when the cap fires, and naming
		// only the effective one sends a user who set 48 to lower it to 40 — which caps to the same
		// value, fails identically, and makes the advice look wrong.
		slots := fmt.Sprintf("%d slots/layer", r.cacheSlots)
		if r.cacheSlotsReq != r.cacheSlots {
			slots = fmt.Sprintf("%d slots/layer requested, capped to %d", r.cacheSlotsReq, r.cacheSlots)
		}
		// Deliberately NO free-VRAM reading here. This site is reached only after Launch has
		// returned non-nil, so any figure taken here is a POST-failure state and cannot be
		// distinguished by the reader from a pre-launch one — the number's meaning depends on where
		// the probe sits, which is not something an error string can carry. It was carrying one, and
		// the 64 MiB apparently "released" between two launches was an artifact of exactly that.
		// Free VRAM belongs to instrumentation, which can state its own probe position.
		//
		// Also deliberately NOT suggesting a specific safe slot count: the computation that would
		// produce one is the thing under suspicion whenever this fires, and printing a number from
		// it would launder a suspect figure into advice.
		return fmt.Errorf("launch %s: out of device memory with the expert cache at %s — the slot "+
			"count is the likely cause; lower GOINFER_MOE_CACHE_SLOTS below the effective value "+
			"above and retry: %w", name, slots, e)
	}
	return fmt.Errorf("launch %s: %w", name, e)
}

// capVec copies the first n elements of a device vector to host into dst[l] (diagnostic
// sublayer capture — n is hidden for the o-proj/down contributions, qDim for the pre-o-proj
// context). Runs on the executor thread inside launchToken, so it syncs before the readback.
// N-08: both errors were dropped, and this feeds HiddenCapture() — so a failed sync or download
// handed the caller a slice of ZEROS that is indistinguishable from a real capture. That is the
// input a speculative drafter fuses from, and zeros are a plausible-looking vector.
//
// capVec cannot return an error (its callers are the capture hooks inside launchToken, which run
// on the executor thread and have no error channel), so it records into setupErr — the same
// field the build path uses — and leaves dst[l] nil rather than zero-filled. A nil row is
// something the caller can notice; a zero row is not.
func (r *cudaResident) capVec(src Buffer, dst [][]float32, l, n int) {
	if e := r.stream.Sync(); e != nil {
		r.recordUpload(fmt.Errorf("cuda capVec: sync: %w", e))
		return
	}
	h := make([]float32, n)
	if e := gpu.Download(src, h); e != nil {
		r.recordUpload(fmt.Errorf("cuda capVec: download: %w", e))
		return
	}
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

// splitKVAttnDecode runs the high-occupancy, bit-identical decode attention (Campaign A) for layer
// l at position pos (M=1): three launches replacing the single attn_batched(M=1). scores tile over
// keys (nH·⌈nWin/128⌉ blocks), softmax keeps the exact 128-wide partition+tree (byte-identical max +
// denominator), vsum tiles over output dims (nH·⌈hd/32⌉ blocks, each thread the whole per-d fold).
// Writes r.cctx exactly as attn_batched would. See docs/task-decode-splitkv-attention.md.
func (r *cudaResident) splitKVAttnDecode(l, pos int) error {
	Ly := &r.layers[l]
	nKeys := pos + 1
	winStart := 0
	if Ly.window > 0 && nKeys > int(Ly.window) {
		winStart = nKeys - int(Ly.window)
	}
	nWin := nKeys - winStart
	const dTile = 32 // 1 warp/block; keeps the coalesced V-read, maximizes blocks without sub-warp waste
	// 1. scores → r.skScoreBuf[h*nWin + i] (raw, ·scale). One thread per key; no reduction.
	if e := r.launch(r.skScores, LaunchConfig{GridX: uint32(r.nH), GridY: uint32((nWin + 127) / 128), GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1},
		Arg(r.qB), Arg(r.kc[l]), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)),
		gpu.ArgValue(int32(winStart)), gpu.ArgValue(int32(nKeys)), gpu.ArgValue(r.attnScale), Arg(r.skScoreBuf), gpu.ArgValue(int32(nWin))); e != nil {
		return e
	}
	// 2. softmax in place (block 128 — MUST match attn_batched for byte-identical max/denominator).
	if e := r.launch(r.skSoftmax, LaunchConfig{GridX: uint32(r.nH), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 4},
		// ArgNull: no attention sink. gpt-oss is the only family with one, and it is not
		// resident-eligible yet (its clamped-SwiGLU expert kernel does not exist), so every
		// caller today passes null and the kernel stays bit-identical to before.
		Arg(r.skScoreBuf), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(nWin)), Arg(r.skInvBuf), r.sinkArg(l)); e != nil {
		return e
	}
	// 3. V-sum → r.cctx (each thread the whole ascending-s fold for one output dim).
	return r.launch(r.skVsum, LaunchConfig{GridX: uint32(r.nH), GridY: uint32((Ly.hd + dTile - 1) / dTile), GridZ: 1, BlockX: dTile, BlockY: 1, BlockZ: 1},
		Arg(r.skScoreBuf), Arg(r.vc[l]), Arg(r.skInvBuf), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)),
		gpu.ArgValue(int32(winStart)), gpu.ArgValue(int32(nKeys)), gpu.ArgValue(int32(nWin)), Arg(r.cctx))
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
func (r *cudaResident) moeMLPPre(Ly *cudaLayer, x Buffer) error {
	// Explicit rmsnorm, NOT the fused fGU path: the fused kernel folds the norm into the dense
	// gate/up GEMV and never writes r.mq, but the router needs that quantized activation too.
	if e := r.rms(x, Ly.postNorm, r.mq, r.mSc); e != nil {
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
	// TRAP: the nGroup/topkGroup mapping below has NEVER been exercised through this path.
	//
	// moe_route's group-limited routing is well gated as a KERNEL — TestMoERoute constructs
	// nGroup != topkGroup (8/4) across four cases and asserts against cpuRoute. But that test
	// builds its own argument list and calls the kernel directly, so it validates the kernel's
	// math, not this call's argument ORDER. Every CUDA-admissible MoE model has
	// nGroup == topkGroup (glm-tiny sets 1/1; mixtral and gemma4-moe set neither, so both
	// default to 0; the gemma4 path below hardcodes 1, 1), because the group-routed families
	// (DeepSeek, Kimi) decline earlier on FeatMLA, which CUDA does not declare.
	//
	// Consequence, measured not assumed: transposing these two arguments and re-running every
	// gate — kernel-level suite, resident parity gates, TestMoERoute itself — passes all of
	// them. The transposition is currently INERT rather than silent-wrong, since nGroup ==
	// topkGroup makes the swap a no-op, but nothing here would notice if it were live.
	//
	// Typed launch wrappers (one generated type per (parameter name, C type)) make a
	// transposition at this site a COMPILE error. They do not verify that the values are the
	// right way round — a wrong value of the right kind still compiles. See the reciprocal trap
	// at decoder/features.go's cuda entry, and docs/parity-coverage-policy.md § "Relevant".
	// gpt-oss routes through its OWN kernel: moe_route takes the mixing weight from the
	// UNBIASED score (bias steers selection only), while gpt-oss softmaxes over the SELECTED
	// BIASED logits. Same selection, different weights — which is why the wrong one produces
	// plausible output rather than an error.
	if r.gptOssRoute != (Pipeline{}) {
		return r.launch(r.gptOssRoute, onecfg(1, 0),
			Arg(r.rLogits), Arg(Ly.routerB), Arg(r.rIdx), Arg(r.rWgt),
			gpu.ArgValue(int32(r.nE)), gpu.ArgValue(int32(r.topK)))
	}
	return r.launch(r.fRoute, onecfg(1, 0),
		Arg(r.rLogits), Arg(Ly.routerB), Arg(r.rIdx), Arg(r.rWgt),
		gpu.ArgValue(int32(r.nE)), gpu.ArgValue(int32(r.topK)), gpu.ArgValue(r.moeSigmoid),
		gpu.ArgValue(r.moeNormTopK), gpu.ArgValue(r.moeScale),
		gpu.ArgValue(int32(r.nGroup)), gpu.ArgValue(int32(r.topkGroup)))
}

// launchGluSplit runs the fused-gate‖up SwiGLU (glu_quant): dscratch[k]=act(gu[k])*gu[inter+k], then
// symmetric-int8-quantizes it into (outQ,outSc). The stacked-expert GEMV lays gate and up CONTIGUOUS
// in one buffer, so this passes the same pointer twice with gOff=0, uOff=inter — the ONE place that
// gate/up split convention lives, so every MoE call site (routed, shared, gemma-4) shares it. This
// centralization is deliberate: a gOff/uOff swap here silently computes silu(up)*gate, which the e2e
// MoE parity gate CANNOT catch on random-weight experts (silu(up)*gate ≈ silu(gate)*up in magnitude,
// the final-logit near-tie washout) — so TestMoeSwigluWiring exercises this exact helper with crafted
// gate≠up and asserts gate-first (audit C-15, bug B). Keep this the sole gate/up-split dispatch.
func (r *cudaResident) launchGluSplit(gu Buffer, inter int, outQ, outSc, outScr Buffer) error {
	return r.launchGluSplitExpert(gu, inter, outQ, outSc, outScr, Buffer{}, -1)
}

// expBiasArg returns this layer's per-expert gate‖up bias table, or the zero Buffer when the
// family has none. Keyed off cudaLayer.idx because moeMLPPost works from *cudaLayer; threading
// a separate index alongside it would be one more thing that can fall out of step with the
// weights the bias must match.
func (r *cudaResident) expBiasArg(Ly *cudaLayer) Buffer {
	if l := Ly.idx; l >= 0 && l < len(r.gptOssExpBias) {
		return r.gptOssExpBias[l]
	}
	return Buffer{}
}

// expDownBiasArg returns layer Ly's per-expert down-projection bias table, or the zero Buffer
// when the family has none. A per-layer SIDE TABLE like the sinks and the gate‖up biases, not a
// cudaLayer field: these are uploaded in the gpt-oss block that runs BEFORE r.layers is
// allocated, so a cudaLayer field there writes into a nil slice.
func (r *cudaResident) expDownBiasArg(Ly *cudaLayer) Buffer {
	if l := Ly.idx; l >= 0 && l < len(r.gptOssDownBias) {
		return r.gptOssDownBias[l]
	}
	return Buffer{}
}

// sinkArg returns layer l's attention-sink argument, or null when the family has none.
// Centralized so the DECODE and PREFILL attention launches cannot disagree — a sink applied
// on one path and not the other presents as drift partway through a sequence, which is much
// harder to attribute than a missing term everywhere.
func (r *cudaResident) sinkArg(l int) gpu.KernelArg {
	if l >= 0 && l < len(r.gptOssSinks) && r.gptOssSinks[l] != (Buffer{}) {
		return Arg(r.gptOssSinks[l])
	}
	return ArgNull()
}

// oBiasArg returns layer Ly's attention output-projection bias, or null when the family has
// none. Centralized for exactly the reason sinkArg above is: the DECODE and the PREFILL o_proj
// launches must not disagree. A bias applied on one path and not the other does not read as a
// wrong answer — it reads as drift partway through a sequence, once decode takes over from
// prefill, which is far harder to attribute than a term missing everywhere.
//
// Note there is NO new kernel here. Both GEMV pairs already fold the bias into the value BEFORE
// the accumulate select (aikit gemv_quant.cu: `val = fma(facc, aScale, bias?bias[n]:0);
// dst[n] = accum ? dst[n]+val : val`, and goinfer's batched gemv_w4a8_rn.cu:117 identically), so
// bias-plus-residual is one instruction on this backend. That is where CUDA differs from Metal,
// which needed a genuinely new gemv_w4a8_sa_bias_resid because no SA GEMV there combined the two.
func (r *cudaResident) oBiasArg(Ly *cudaLayer) gpu.KernelArg {
	if Ly != nil && Ly.hasOBias {
		return Arg(Ly.ob)
	}
	return ArgNull()
}

// launchGluSplitExpert is launchGluSplit with the routed-expert context the gpt-oss epilogue
// needs. bias is that layer's [nExpert·2·inter] gate‖up table and slot is the top-k position
// whose expert is running; the KERNEL does biasRow = idx[slot]*2*inter, because which expert
// runs is a device-side routing decision and the launch geometry must not depend on it.
//
// Families other than gpt-oss route through the nil/-1 path and land on glu_quant exactly as
// before — same kernel, same arguments, bit-identical. gpt-oss is the only family whose
// activation is not act(g)*u, so branching here rather than adding a mode to glu_quant keeps
// the clamped variant out of the audited glue.ptx.
func (r *cudaResident) launchGluSplitExpert(gu Buffer, inter int, outQ, outSc, outScr Buffer, bias Buffer, slot int) error {
	// Buffer and Pipeline are STRUCTS here, not interfaces, so "absent" is the zero value
	// rather than nil — the same test resident.go already uses for the optional split-KV
	// pipelines (r.skScores != (Pipeline{})).
	if r.gptOssSw != (Pipeline{}) {
		// expertBiasIdx, NOT expIdx: this kernel uses idx ONLY to pick a row of the per-expert
		// bias table (it consumes already-computed gate/up activations and runs no GEMV), so it
		// needs the expert id even when the weights it followed were read from a cache slot.
		idx := Arg(r.expertBiasIdx())
		bArg := ArgNull()
		if bias != (Buffer{}) {
			bArg = Arg(bias)
		}
		if slot < 0 {
			slot = 0 // the shared/dense caller has no routing slot; gpt-oss has no shared expert
		}
		return r.launch(r.gptOssSw, onecfg(256, 256*4),
			Arg(gu), Arg(gu), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(inter)),
			gpu.ArgValue(int32(inter)), bArg, idx, gpu.ArgValue(int32(slot)),
			gpu.ArgValue(r.gptOssAlpha), gpu.ArgValue(r.gptOssLimit),
			Arg(outQ), Arg(outSc), Arg(outScr))
	}
	return r.launch(r.fSw, onecfg(256, 256*4),
		Arg(gu), Arg(gu), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(inter)),
		gpu.ArgValue(int32(inter)), gpu.ArgValue(r.act),
		Arg(outQ), Arg(outSc), Arg(outScr))
}

// moeMLPPost issues the post-readback half: the sequential expert loop (each expert selected by
// ARITHMETIC on r.expIdx(), so the launch geometry is identical regardless of routing — the property
// that makes this graph-static) weight-accumulating into the residual r.x, then the always-on shared
// expert. The cacheExperts readback (if any) has already filled the slots when this runs.
func (r *cudaResident) moeMLPPost(Ly *cudaLayer, x Buffer) error {
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
		if e := r.launchGluSplitExpert(r.moeGU, r.moeInter, r.moeQ, r.moeSc, r.moeScr, r.expBiasArg(Ly), j); e != nil {
			return e
		}
		// down-proj, weight-accumulating into the residual: x += wgt[j] * (Down_e · act).
		// gpt-oss adds its per-expert down bias INSIDE that product (x += wgt[j] * (Down_e·act +
		// downBias_e)) via its own kernel; every other family keeps the untouched wacc, so their
		// numerics are byte-identical to before this branch existed.
		if db := r.expDownBiasArg(Ly); db != (Buffer{}) && r.fMoEWaccBias != (Pipeline{}) {
			if e := r.launch(r.fMoEWaccBias, LaunchConfig{GridX: uint32((r.hidden + 7) / 8), GridY: 1, GridZ: 1,
				BlockX: 256, BlockY: 1, BlockZ: 1},
				Arg(Ly.expDown.W), Arg(r.moeQ), Arg(Ly.expDown.ws16), Arg(r.moeSc),
				Arg(r.expIdx()), Arg(r.expertBiasIdx()), Arg(db), Arg(r.rWgt),
				gpu.ArgValue(int32(j)), gpu.ArgValue(int32(r.hidden)),
				gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(int32(r.moeInter/8)), gpu.ArgValue(int32(r.moeInter/32)),
				Arg(x)); e != nil {
				return e
			}
		} else if e := r.launch(r.fMoEWacc, LaunchConfig{GridX: uint32((r.hidden + 7) / 8), GridY: 1, GridZ: 1,
			BlockX: 256, BlockY: 1, BlockZ: 1},
			Arg(Ly.expDown.W), Arg(r.moeQ), Arg(Ly.expDown.ws16), Arg(r.moeSc),
			Arg(r.expIdx()), Arg(r.rWgt), gpu.ArgValue(int32(j)), gpu.ArgValue(int32(r.hidden)),
			gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(int32(r.moeInter/8)), gpu.ArgValue(int32(r.moeInter/32)),
			Arg(x)); e != nil {
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
		if e := r.launchGluSplit(r.shGUout, r.sharedInter, r.shQ, r.shSc, r.shScr); e != nil {
			return e
		}
		if e := r.doG(Ly.shDown, r.shQ, r.shSc, nullBias, r.shDownOut, 0); e != nil {
			return e
		}
		// GLM/DeepSeek add the shared output ungated; Qwen-MoE scales it by sigmoid(SharedGate·h)
		// first. Same kernel either way. When ungated the gl pointer is unread, but the kernel
		// still takes it, so pass a valid buffer (shSc, spare) rather than a null.
		gl, ungated := r.shSc, int32(1)
		if Ly.shGateW.W != (Buffer{}) {
			if e := r.doG(Ly.shGateW, r.mq, r.mSc, nullBias, r.shGl, 0); e != nil {
				return e
			}
			gl, ungated = r.shGl, 0
		}
		if e := r.launch(r.fSharedCombine, g1cfg(r.hidden, 256),
			Arg(x), Arg(r.shDownOut), Arg(gl), gpu.ArgValue(int32(r.hidden)),
			gpu.ArgValue(ungated)); e != nil {
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
func (r *cudaResident) gemma4MoeMLPPre(Ly *cudaLayer, l int, x Buffer) error {
	nullBias := ArgNull()

	// --- dense branch → g4x1 ---
	if e := r.rms(x, Ly.g4preFFN, r.mq, r.mSc); e != nil { // xd = preFFNNorm(h), int8
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
	if e := r.launch(r.fRmsNW, onecfg(256, 256*4), Arg(x), Arg(r.g4rn), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(r.eps)); e != nil {
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
	return r.rms(x, Ly.g4preFFN2, r.mq, r.mSc)
}

// gemma4MoeMLPPost issues the post-readback half: the expert loop accumulating into the (already
// cleared) g4x2, its post-norm, and the join — sum x1+x2, joint post-norm, add the residual h, then
// the per-layer scalar. g4x2 was zeroed and the expert slots filled in the gap before this runs.
func (r *cudaResident) gemma4MoeMLPPost(Ly *cudaLayer, l int, x Buffer) error {
	gu := 2 * r.moeInter
	for j := 0; j < r.topK; j++ {
		if e := r.launch(r.fMoEGemv, LaunchConfig{GridX: uint32((gu + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
			Arg(Ly.expGU.W), Arg(r.mq), Arg(Ly.expGU.ws16), Arg(r.mSc), Arg(r.expIdx()),
			gpu.ArgValue(int32(j)), gpu.ArgValue(int32(gu)), gpu.ArgValue(int32(gu)),
			gpu.ArgValue(int32(r.hidden/8)), gpu.ArgValue(int32(r.hidden/32)), Arg(r.moeGU)); e != nil {
			return e
		}
		if e := r.launchGluSplitExpert(r.moeGU, r.moeInter, r.moeQ, r.moeSc, r.moeScr, r.expBiasArg(Ly), j); e != nil {
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
		if err := r.stream.Sync(); err != nil {
			return err
		}
		idx := make([]uint32, r.topK)
		if err := gpu.Download(r.rIdx, idx); err != nil {
			return err
		}
		r.g4capIdx = append(r.g4capIdx, idx) // append order = CPU routerCaptureBuf order (per-position routing check)
		r.g4capLayer = append(r.g4capLayer, l)
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
	if e := r.launch(r.fRes, g1cfg(r.hidden, 256), Arg(x), Arg(r.g4x1), gpu.ArgValue(int32(r.hidden))); e != nil { // x = h + comb
		return e
	}
	return r.launch(r.fScaleVec, g1cfg(r.hidden, 256), Arg(x), gpu.ArgValue(Ly.layerScalar), gpu.ArgValue(int32(r.hidden))) // x *= layerScalar
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
		if Ly.qGate {
			// attn_output_gate: q_proj emits [query ‖ gate] PER HEAD at double width. Project into
			// a 2*qDim scratch and split on the ACTIVATION — the weight stays fused because it is
			// quantized, and slicing rows out of an int4 bundle with its per-group scales is real
			// surgery. Interleaved per head, NOT two concatenated blocks.
			if e := r.doG(Ly.q, r.aq, r.aSc, qb, r.dnQg, 0); e != nil {
				return e
			}
			if e := r.launch(r.dnQSplit, g1cfg(Ly.qDim, 256),
				Arg(r.dnQg), Arg(r.qB), Arg(r.dnAGate),
				gpu.ArgValue(int32(Ly.qDim)), gpu.ArgValue(int32(Ly.hd))); e != nil {
				return e
			}
		} else if e := r.doG(Ly.q, r.aq, r.aSc, qb, r.qB, 0); e != nil {
			return e
		}
		if err := r.doG(Ly.k, r.aq, r.aSc, kb, r.kB, 0); err != nil {
			return err
		}
		if Ly.kEqV {
			// K=V: recompute the k projection into vB (raw pre-norm k), which v_norm consumes below.
			if err := r.doG(Ly.k, r.aq, r.aSc, kb, r.vB, 0); err != nil {
				return err
			}
		} else {
			if err := r.doG(Ly.v, r.aq, r.aSc, vb, r.vB, 0); err != nil {
				return err
			}
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
		if err := r.launch(r.fQKN, LaunchConfig{GridX: uint32(Ly.nKV), GridY: 1, GridZ: 1,
			BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 8},
			Arg(r.vB), Arg(r.vB), Arg(r.vNormUnit), Arg(r.vNormUnit),
			gpu.ArgValue(int32(0)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)),
			gpu.ArgValue(r.eps), gpu.ArgValue(int32(0))); err != nil {
			return err
		}
	}
	return nil
}

// segB: context-quant + o-proj (accum into the residual, or sandwich-norm then add) + the MLP up to
// the router readback — the whole dense MLP for a dense layer (no readback gap, segC is nil), or the
// MoE pre-readback half (moeMLPPre / gemma4MoeMLPPre) for a routed layer.
func (r *cudaResident) segB(Ly *cudaLayer, l int, x Buffer) error {
	if Ly.qGate { // ctx *= sigmoid(gate), before o_proj (matches the CPU qwen35Attention)
		if err := r.launch(r.dnAttnGate, g1cfg(Ly.qDim, 256),
			Arg(r.cctx), Arg(r.dnAGate), gpu.ArgValue(int32(Ly.qDim))); err != nil {
			return err
		}
	}
	if err := r.launch(r.fQ, onecfg(256, 256*4), Arg(r.cctx), gpu.ArgValue(int32(Ly.qDim)), Arg(r.cq), Arg(r.cSc)); err != nil {
		return err
	}
	if r.sandwich {
		if err := r.doG(Ly.o, r.cq, r.cSc, r.oBiasArg(Ly), r.oO, 0); err != nil {
			return err
		}
		if err := r.normF32(r.oO, Ly.postAttnNorm); err != nil {
			return err
		}
		if r.subCap {
			r.capVec(r.oO, r.subAttnC, l, r.hidden)
		}
		if err := r.launch(r.fRes, g1cfg(r.hidden, 256), Arg(r.x), Arg(r.oO), gpu.ArgValue(int32(r.hidden))); err != nil {
			return err
		}
	} else {
		if err := r.doG(Ly.o, r.cq, r.cSc, r.oBiasArg(Ly), r.x, 1); err != nil {
			return err
		}
	}
	return r.segBFFN(Ly, l, x)
}

// segBFFN is segB's FFN half, split out so the Gated-DeltaNet mixer can reuse it. The mixer
// replaces everything segB does BEFORE this point (ctx-quant, o-proj, residual) and nothing after
// it: a DeltaNet layer's FFN sub-block is the ordinary one, dense or MoE, and the router readback
// gap + segC that follow in launchToken are likewise unchanged.
func (r *cudaResident) segBFFN(Ly *cudaLayer, l int, x Buffer) error {
	nullBias := ArgNull()
	if Ly.g4moe {
		return r.gemma4MoeMLPPre(Ly, l, x)
	}
	if Ly.isMoE {
		return r.moeMLPPre(Ly, x)
	}
	// Dense MLP (whole): no readback gap, so segC is nil for this layer.
	if r.fuseQKV {
		cfg := LaunchConfig{GridX: uint32((2*r.inter + 63) / 64), GridY: 1, GridZ: 1,
			BlockX: 256, BlockY: 1, BlockZ: 1,
			SharedMemBytes: uint32((r.hidden + 256 + r.hidden/4) * 4)}
		if e := r.launch(r.fGU, cfg,
			Arg(x), Arg(Ly.postNorm), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(r.eps),
			gpu.ArgValue(r.addOneArg()),
			Arg(Ly.g.W), Arg(Ly.g.ws16),
			Arg(Ly.u.W), Arg(Ly.u.ws16),
			gpu.ArgValue(int32(r.inter)), gpu.ArgValue(int32(r.hidden/8)), gpu.ArgValue(int32(r.hidden/32)),
			Arg(r.gO), Arg(r.uO)); e != nil {
			return e
		}
	} else {
		if err := r.rms(x, Ly.postNorm, r.mq, r.mSc); err != nil {
			return err
		}
		if err := r.doG(Ly.g, r.mq, r.mSc, nullBias, r.gO, 0); err != nil {
			return err
		}
		if err := r.doG(Ly.u, r.mq, r.mSc, nullBias, r.uO, 0); err != nil {
			return err
		}
	}
	if err := r.launch(r.fSw, onecfg(256, 256*4), Arg(r.gO), Arg(r.uO), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(r.inter)),
		gpu.ArgValue(r.act), Arg(r.dq), Arg(r.dSc), Arg(r.dScr)); err != nil {
		return err
	}
	if r.sandwich {
		if e := r.doG(Ly.d, r.dq, r.dSc, nullBias, r.dO, 0); e != nil {
			return e
		}
		if r.subCap {
			r.capVec(r.dO, r.subMLPpreC, l, r.hidden)
		}
		if err := r.normF32(r.dO, Ly.postMLPNorm); err != nil {
			return err
		}
		if r.subCap {
			r.capVec(r.dO, r.subMLPC, l, r.hidden)
		}
		if err := r.launch(r.fRes, g1cfg(r.hidden, 256), Arg(x), Arg(r.dO), gpu.ArgValue(int32(r.hidden))); err != nil {
			return err
		}
	} else if e := r.doG(Ly.d, r.dq, r.dSc, nullBias, x, 1); e != nil {
		return e
	}
	return nil
}

// segC: the post-readback MoE half (expert loop + combine/join). nil for a dense layer.
func (r *cudaResident) segC(Ly *cudaLayer, l int, x Buffer) error {
	if Ly.g4moe {
		return r.gemma4MoeMLPPost(Ly, l, x)
	}
	if Ly.isMoE {
		return r.moeMLPPost(Ly, x)
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
		if Ly.isDeltaNet {
			// A Gated-DeltaNet layer is MORE graph-static than an attention one, not less. It has
			// no rope and no attention, so it has no per-token dynamic uniform at all — the
			// recurrence's only per-token input is the residual, which is buffer CONTENTS. Replay
			// reads current contents (TestCUDA_graphReplay), which is exactly why the MoE routing
			// already flows through a captured segC unchanged.
			//
			// So the whole pre-routing half of the layer captures as ONE segment: mixer + FFN-pre.
			// segB stays nil — there is no attention gap to split around.
			gM, e := r.stream.Capture(func() error {
				if err := r.deltaNetMixer(Ly, ll); err != nil {
					return err
				}
				return r.segBFFN(Ly, ll, r.x)
			})
			if e != nil {
				return fmt.Errorf("layer %d deltanet mixer+ffn: %w", l, e)
			}
			r.layers[l].gSegA = gM
			if Ly.g4moe || Ly.isMoE {
				gC, e := r.stream.Capture(func() error { return r.segC(Ly, ll, r.x) })
				if e != nil {
					return fmt.Errorf("layer %d segC: %w", l, e)
				}
				r.layers[l].gSegC = gC
			}
			continue
		}
		gA, e := r.stream.Capture(func() error { return r.segA(Ly, ll) })
		if e != nil {
			return fmt.Errorf("layer %d segA: %w", l, e)
		}
		gB, e := r.stream.Capture(func() error { return r.segB(Ly, ll, r.x) })
		if e != nil {
			return fmt.Errorf("layer %d segB: %w", l, e)
		}
		r.layers[l].gSegA, r.layers[l].gSegB = gA, gB
		if Ly.g4moe || Ly.isMoE {
			gC, e := r.stream.Capture(func() error { return r.segC(Ly, ll, r.x) })
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
		if Ly.isDeltaNet {
			// Gated-DeltaNet mixer: replaces segA + rope + attention + o-proj entirely (no KV
			// cache, no positional anything — the state IS the history). The FFN sub-block and
			// the router-readback gap below are the ordinary ones, so this rejoins the shared
			// path at segBFFN and everything after it is unchanged.
			//
			// Captured as ONE segment (gSegA) when graphs are on: with no rope/attention gap there
			// is nothing dynamic to split around, so this is the most graph-friendly layer kind in
			// the runner — 64 launches collapsing to one replay.
			if gA {
				if e := Ly.gSegA.Replay(); e != nil {
					return e
				}
				if r.graphsSync {
					if e := r.stream.Sync(); e != nil {
						return e
					}
				}
			} else {
				if e := r.deltaNetMixer(Ly, l); e != nil {
					return e
				}
				if e := r.segBFFN(Ly, l, r.x); e != nil {
					return e
				}
			}
			if e := r.layerTail(Ly, l, gC, r.x); e != nil {
				return e
			}
			continue
		}
		// segA: QKV proj + qk/v-norm (pre-RoPE, static).
		if gA {
			if e := Ly.gSegA.Replay(); e != nil {
				return e
			}
			if r.graphsSync {
				if err := r.stream.Sync(); err != nil {
					return err
				}
			}
		} else if e := r.segA(Ly, l); e != nil {
			return e
		}
		// --- dynamic gap: rope_kv + attention (bind pos/nKeys; attention's shared-mem grows with the
		// attended span — never graph-static). Same stream as the segments, so ordering is preserved.
		// fused rope(q)+rope(k)+kv_store(k)+kv_store(v): rhalf == hd/2 for full rotary, rotaryDim/2 for partial.
		if err := r.launch(r.ropeKV, g1cfg(r.nH*Ly.rhalf+Ly.nKV*Ly.rhalf+Ly.nKV*(Ly.hd-2*Ly.rhalf), 256),
			Arg(r.qB), Arg(r.kB), Arg(r.vB), Arg(Ly.invF), Arg(r.kc[l]), Arg(r.vc[l]),
			gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)),
			gpu.ArgValue(int32(pos)), gpu.ArgValue(int32(Ly.rhalf)), gpu.ArgValue(Ly.mscale)); err != nil {
			return err
		}
		nKeys := pos + 1
		// Sliding window (per layer: Mistral all-local, Mellum interleaves); shared sized to the attended span.
		nWin := nKeys
		if Ly.window > 0 && nKeys > int(Ly.window) {
			nWin = int(Ly.window)
		}
		// M-16: split-KV is REQUIRED, not merely preferred, once the single-block launch would
		// exceed the device's shared-memory limit. The `r.splitkvAttn` env gate and the
		// per-geometry perf threshold both describe when split-KV is FASTER; neither knows when
		// the alternative cannot run at all. Without this, -ctx 16384 on a model whose geometry
		// says splitkvNever (nH >= 24: Qwen2.5-7B, Llama-3-8B, phi3-mini) fails at position
		// 12,160 — or silently drops to the sequential prefill.
		mustSplit := splitKVRequired(nWin)
		if mustSplit && (!r.splitkvAttn || r.skScores == (Pipeline{})) {
			return fmt.Errorf("cuda: attention at %d attended keys needs %d B of shared memory, "+
				"past this device's %d B limit, and split-KV is unavailable (%s) — lower -ctx or "+
				"re-enable GOINFER_SPLITKV_ATTN (M-16)", nWin, attnShmemBytes(nWin),
				singleBlockAttnShmemLimit,
				map[bool]string{true: "kernel not loaded", false: "disabled by GOINFER_SPLITKV_ATTN"}[r.skScores == (Pipeline{})])
		}
		if r.splitkvAttn && r.skScores != (Pipeline{}) && (mustSplit || nWin >= r.splitkvMin(Ly.hd)) {
			// Campaign-A split-KV: high-occupancy, BIT-IDENTICAL to attn_batched(M=1) (proven by
			// TestSplitKV_bitIdentical) — fills the SMs the single-block kernel leaves idle at long ctx.
			// Gated PER LAYER on nWin (the EFFECTIVE attended span) against a per-geometry threshold, so
			// shallow decode keeps the cheaper single-block path. nWin not nKeys: a sliding-window layer
			// never attends more than `window` keys, so its cost is set by the window, not by position —
			// gating it on position made gemma3's windowed layers take the split path at a 512-key span
			// (its loss regime) at every depth past the window. Both arms are byte-identical, so a layer
			// flipping arms mid-request as nWin grows is safe by construction.
			if err := r.splitKVAttnDecode(l, pos); err != nil {
				return err
			}
		} else if r.prefillReady {
			// Coalesced M=1 decode attention: attn_batched with M=1 is BIT-IDENTICAL to the glue
			// `attention` (TestAttnBatched_bitIdentical) but reads K via float4 — 21.96%→98% bytes/sector.
			// ncu found the glue decode attention L1TEX-latency-bound at 2048 (~63% of the decode budget,
			// the 221→97 tok/s long-context deficit vs current Ollama); the coalesced read recovers it.
			// startPos=pos, M=1 → nKeys = pos+1; same GridX/block/shared/ctx-layout as the glue launch, so
			// decode stays byte-identical. glue `attention` (audited) is UNTOUCHED and is the fallback below.
			if err := r.launch(r.bAttn, LaunchConfig{GridX: uint32(r.nH), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((nWin + 128) * 4)},
				Arg(r.qB), Arg(r.kc[l]), Arg(r.vc[l]), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)), gpu.ArgValue(int32(pos)), gpu.ArgValue(r.attnScale), gpu.ArgValue(Ly.window), gpu.ArgValue(int32(1)), Arg(r.cctx), r.sinkArg(l)); err != nil {
				return err
			}
		} else {
			if err := r.launch(r.fAttn, LaunchConfig{GridX: uint32(r.nH), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((nWin + 128) * 4)},
				Arg(r.qB), Arg(r.kc[l]), Arg(r.vc[l]), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)), gpu.ArgValue(int32(nKeys)), gpu.ArgValue(r.attnScale), gpu.ArgValue(Ly.window), Arg(r.cctx)); err != nil {
				return err
			}
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
				if err := r.stream.Sync(); err != nil {
					return err
				}
			}
		} else if e := r.segB(Ly, l, r.x); e != nil {
			return e
		}
		// --- dynamic gap: clear the g4moe accumulator (H2D), then, if caching, read the routing back
		// to the host and DMA the routed experts into their VRAM slots (D2H+H2D). Host copies, not
		// graph-capturable; synchronous, so the slots are filled before segC replays.
		if e := r.layerTail(Ly, l, gC, r.x); e != nil {
			return e
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
	applySoftcap(r.logitsHost, r.finalSoftcap)
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

// SetHiddenCapture arms the resident hidden-state seam for the given target layer indices
// (layer OUTPUTS, ascending — the same convention decoder.Model.ForwardCapture uses, and
// the one DeepSpec/z-lab's `target_layer_ids` name). Pass nil to disarm.
//
// This is the resident counterpart of the CPU ForwardCapture seam that 05 paid for and
// that P10's block drafters need: a resident target cannot feed a hidden-state drafter
// without it. Cost is one sync + one hidden-sized download per TAPPED layer per token, so
// arm only the taps the drafter actually reads.
func (r *cudaResident) SetHiddenCapture(taps []int) error {
	if len(taps) == 0 {
		r.hidCapTaps, r.hidCapOut = nil, nil
		return nil
	}
	prev := -1
	for _, t := range taps {
		if t < 0 || t >= r.nLayers {
			return fmt.Errorf("cuda: hidden-capture tap %d out of range [0,%d)", t, r.nLayers)
		}
		if t <= prev {
			return fmt.Errorf("cuda: hidden-capture taps must be ascending and distinct, got %v", taps)
		}
		prev = t
	}
	r.hidCapTaps = append([]int(nil), taps...)
	r.hidCapOut = make([][]float32, len(taps))
	return nil
}

// HiddenCapture returns the most recent token's captured layer outputs, one row per tap in
// SetHiddenCapture order. The rows are owned by the caller (capVec allocates per token).
func (r *cudaResident) HiddenCapture() [][]float32 { return r.hidCapOut }

// dnetParams carries the model-level Gated-DeltaNet geometry (uniform across the linear layers).
// keyDim/valueDim/convDim are derived once rather than at every dispatch because the three are
// easy to conflate: convDim is 2*keyDim+valueDim (the conv runs over [q|k|v] together), and
// rep = nv/nk is the GVA factor mapping value heads to key heads.
type dnetParams struct {
	convK      int
	hk, hv     int
	nk, nv     int
	rep        int
	keyDim     int
	valueDim   int
	convDim    int
	stateElems int
	qScale     float32
}

// deltaNetMixer runs one Gated-DeltaNet layer's sequence mixer and folds its output into the
// residual — the recurrent replacement for segA + rope + attention + o-proj.
//
//	norm → in_proj_qkv → conv(ring) → l2norm(q,k) → gates → delta rule(state) → gated norm × silu(z)
//	     → out_proj + residual
//
// NOT GRAPH-CAPTURED. The three static segments exist so CUDA graphs can replay them; this path
// runs live. That costs nothing real — graphs measured 1.01× on this backend and are off by
// default — and it avoids capturing launches over buffers whose contents ARE the per-token state.
//
// The two small gate projections (dnB/dnA) run off the SAME quantized activation as the big ones.
// On the CPU they are f32 by deliberate choice: deltaNetWeights keeps inProjB/inProjA unquantized
// because they feed the write/decay gates, where the recurrence is most precision-sensitive. This
// is the one place the resident path is knowingly coarser than the reference, and the parity gate
// scores the gates as their own stage so the cost of that choice is visible rather than assumed.
func (r *cudaResident) deltaNetMixer(Ly *cudaLayer, l int) error {
	dp := r.dnet
	nullBias := ArgNull()
	if e := r.rms(r.x, Ly.preNorm, r.aq, r.aSc); e != nil {
		return e
	}
	if e := r.doG(Ly.dnQKV, r.aq, r.aSc, nullBias, r.dnMixed, 0); e != nil {
		return e
	}
	if e := r.doG(Ly.dnB, r.aq, r.aSc, nullBias, r.dnBt, 0); e != nil {
		return e
	}
	if e := r.doG(Ly.dnA, r.aq, r.aSc, nullBias, r.dnAt, 0); e != nil {
		return e
	}
	if e := r.doG(Ly.dnZ, r.aq, r.aSc, nullBias, r.dnZOut, 0); e != nil {
		return e
	}
	if e := r.launch(r.dnConv, g1cfg(dp.convDim, 256),
		Arg(r.dnMixed), Arg(Ly.dnConvW), Arg(Ly.dnWin), Arg(r.dnConvOut),
		gpu.ArgValue(int32(dp.convDim)), gpu.ArgValue(int32(dp.convK))); e != nil {
		return e
	}
	if e := r.launch(r.dnGates, g1cfg(dp.nv, 64),
		Arg(r.dnBt), Arg(r.dnAt), Arg(Ly.dnDtBias), Arg(Ly.dnNegExpA), Arg(r.dnHeadP),
		gpu.ArgValue(int32(dp.nv))); e != nil {
		return e
	}
	if e := r.launch(r.dnNorm, LaunchConfig{GridX: uint32(dp.nk), GridY: 1, GridZ: 1,
		BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 2 * 128 * 4},
		Arg(r.dnConvOut), Arg(r.dnQn), Arg(r.dnKn),
		gpu.ArgValue(int32(dp.nk)), gpu.ArgValue(int32(dp.hk)), gpu.ArgValue(int32(dp.keyDim)),
		gpu.ArgValue(dp.qScale)); e != nil {
		return e
	}
	// v is the conv output's third slice; vBase points at it rather than copying it out.
	if e := r.launch(r.dnRule, g1cfg(dp.valueDim, 128),
		Arg(r.dnQn), Arg(r.dnKn), Arg(r.dnConvOut), Arg(r.dnHeadP), Arg(Ly.dnState), Arg(r.dnCore),
		gpu.ArgValue(int32(dp.nv)), gpu.ArgValue(int32(dp.hk)), gpu.ArgValue(int32(dp.hv)),
		gpu.ArgValue(int32(dp.rep)), gpu.ArgValue(int32(2*dp.keyDim))); e != nil {
		return e
	}
	if e := r.launch(r.dnGNorm, LaunchConfig{GridX: uint32(dp.nv), GridY: 1, GridZ: 1,
		BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 4},
		Arg(r.dnCore), Arg(r.dnZOut), Arg(Ly.dnNormW), Arg(r.dnGated),
		gpu.ArgValue(int32(dp.nv)), gpu.ArgValue(int32(dp.hv)), gpu.ArgValue(r.eps)); e != nil {
		return e
	}
	// Quantize the gated output, then out_proj straight into the residual (accum=1), exactly as
	// the attention path's o-proj does.
	if e := r.launch(r.fQ, onecfg(256, 256*4),
		Arg(r.dnGated), gpu.ArgValue(int32(dp.valueDim)), Arg(r.dnGq), Arg(r.dnGSc)); e != nil {
		return e
	}
	return r.doG(Ly.dnOut, r.dnGq, r.dnGSc, nullBias, r.x, 1)
}

// layerTail is everything launchToken does AFTER the FFN pre-half: the g4moe accumulator clear,
// the C′ expert DMA, segC, and the two capture seams. Split out so the Gated-DeltaNet mixer path
// rejoins it instead of duplicating it — a DeltaNet layer's FFN can be MoE (qwen3_5_moe,
// qwen3_next) and would otherwise skip the router readback entirely.
func (r *cudaResident) layerTail(Ly *cudaLayer, l int, gC bool, x Buffer) error {
	if Ly.g4moe {
		// segC(l-1) writes AND reads g4x2 on r.stream (CU_STREAM_NON_BLOCKING); gpu.Upload runs the
		// zero-fill copy on the context's legacy null stream, which has NO ordering vs r.stream
		// (aikit gpu/cuda.go) — its full sync fixes upload→next-launch, not pending-launch→upload.
		// Without this, the DMA can land mid-segC(l-1) and zero the previous layer's expert
		// contribution before the join reads it: a data race on every g4moe layer after the first.
		// Sync r.stream so the clear is ordered after the prior layer's kernels (audit R-03).
		if e := r.stream.Sync(); e != nil {
			return e
		}
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
				if err := r.stream.Sync(); err != nil {
					return err
				}
			}
		} else if e := r.segC(Ly, l, x); e != nil {
			return e
		}
	}
	// hidden-state seam: this layer's OUTPUT residual, for a drafter that taps it.
	if len(r.hidCapTaps) > 0 {
		for slot, tap := range r.hidCapTaps {
			if tap == l {
				r.capVec(x, r.hidCapOut, slot, r.hidden)
				break
			}
		}
	}
	if r.layerCap { // DEBUG: snapshot the residual after this layer (divergence-localization probe)
		if err := r.stream.Sync(); err != nil {
			return err
		}
		h := make([]float32, r.hidden)
		if err := gpu.Download(x, h); err != nil {
			return err
		}
		r.layerCapBuf = append(r.layerCapBuf, h)
	}
	return nil
}
