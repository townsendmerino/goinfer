//go:build cuda

package cuda

import (
	"context"
	"errors"
	"fmt"
	"os"
	"reflect"
	"slices"
	"strings"
	"time"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/backend/cuda/internal/fitdecide"
	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/backend/cuda/internal/launchgeom"
	"github.com/townsendmerino/goinfer/backend/cuda/internal/wpack"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// cudaResident implements the decoder's resident-decode interfaces; the compile-time checks below keep that true.
var (
	_ decoder.ResidentForward    = (*cudaResident)(nil)
	_ decoder.VerifyPathReporter = (*cudaResident)(nil)
)

// cudaCtxCapDefault is the resident KV capacity in positions when nothing asks for more; the staged path handles
// longer. It is a DEFAULT, not a ceiling: decoder.Options.ResidentContext raises it (see resolveCtxCap). It stays
// 4096 so that a caller who did not ask never allocates deep-KV VRAM: raising it would silently multiply every
// resident model's KV footprint. 4096 is a round conservative figure, not tuned to VRAM headroom; a real card
// holds several times more (docs/code-notes/cuda.md#cudaCtxCapDefault), so raise -ctx and read checkKVFits' own
// error to find a deployment's real ceiling.
const cudaCtxCapDefault = 4096

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

// fitDefaultCtx is the candidate context resolveCtxCapFit asks Plan for on an UNPINNED request: the size of a
// coding agent's turn, not the model's full window. 8192 refused a real
// first agent request; 16384 holds it with headroom (docs/code-notes/cuda.md#fitDefaultCtx). It is a CANDIDATE,
// not a grant: Plan shrinks it to what the card holds and ctxForSlots shrinks it further until the requested KV
// slots all fit, so a card or model that cannot hold it lands where it always did. goinfer-chat fit's own -ctx
// default (internal/fitcmd/fit.go) uses the same figure so the dry run and the real load agree.
const fitDefaultCtx = 16384

// kvBytesForCap is the device bytes one resident KV slot takes at a given capacity (kvpool.go): the blocks its tables can map, every
// layer holding K and V at its row size (kvRowBytes) and an MLA layer one latent row, plus the slot's tables. The pools' null blocks are
// once per build, not per slot (kvPoolNullBytes). docs/benchmarks.md derives its deep-context sizing from this formula.
func kvBytesForCap(cap int, layers []cudaLayer) int64 {
	blocks := int64(kvBlocksFor(cap))
	var perBlock, windowed int64
	for i := range layers {
		if layers[i].kvShared {
			continue // aliases its source's cache: no bytes of its own
		}
		row := kvRowBytes(&layers[i])
		if layers[i].isMLA() {
			perBlock += kvBlock * row // MLA caches 1 latent row per pos (not 2x for K+V)
		} else if layers[i].kvWin {
			// a windowed layer maps window+slack positions however long the context (kvWindowBlocks)
			held := int64(kvWindowBlocks(cap, int(layers[i].window)+kvWindowSlack))
			windowed += 2 * row * kvBlock * held
		} else {
			perBlock += 2 * kvBlock * row // K+V
		}
	}
	return perBlock*blocks + windowed + kvGroups*4*blocks
}

// kvSlotAllocFailAtForTest, when > 0, makes the KV pools fail as a device OOM when sized for more than that many slots (test seam;
// zero in production), so a build keeps the slots that fit (kvAllocPoolsFit).
var kvSlotAllocFailAtForTest int

// cudaFreeVRAM is the free-VRAM probe checkKVFits prices the resident KV (every slot) against. A variable so
// TestCUDAKVSlots_pricedAgainstWhatIsLeft can stub it.
var cudaFreeVRAM = func(r *cudaResident) (uint64, error) {
	free, _, err := r.dev.Context().MemInfo()
	return free, err
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
	kvWin               bool   // this layer's K/V hold window+slack positions, not the context (kvwindow.go)
	kvHalf              bool   // this layer's K/V cache holds f16 (the _kv16 kernel twins); an MLA latent never does
	kvI8                bool   // this layer's K/V cache holds int8 with one f32 scale per (position, KV head) in ksc/vsc (the _kv8 twins and kv8_store)
	// Per-layer attention geometry. Uniform families set the model values on every layer; Gemma 4 varies them (local
	// head_dim 16 / global 512, K=V). launchToken reads ONLY these: the model-level hd/nKV/qDim/kvDim/rhalf are
	// deliberately absent from cudaResident so a launch site cannot bind the wrong (uniform) source.
	// qDim = nH*hd, kvDim = nKV*hd, rhalf = rotaryDim/2 (rotated pairs per head).
	hd, nKV, qDim, kvDim, rhalf int
	// kEqV (attention_k_eq_v, Gemma 4 global layers): this layer has NO v_proj — V is
	// v_norm(the raw pre-RoPE k_proj output), stored un-rotated in its OWN vCache (NOT aliased
	// to kCache; kvDim is NOT halved). launchToken derives V from k before rope_kv mutates it.
	kEqV bool
	// ffnI: this layer's dense FFN width (g.N), 0 on a layer with no dense FFN. Every dense-FFN launch reads it, not
	// r.inter, because an E-model's layers differ.
	ffnI int
	// PLE branch (Gemma 4 E-model): the per-layer input gate [P x hidden], projection [hidden x P] and the post-norm weight [hidden]. Zero otherwise.
	pleGate, pleProj cudaWQ
	postPLENorm      Buffer
	// kvShared / kvSrc (Gemma 4 E-models, num_kv_shared_layers): this layer owns no K/V. It projects Q only, rotates only Q, and attends over
	// layer kvSrc's cache, which r.kc[l]/r.vc[l] alias. kEqV and vNorm are false on it.
	kvShared bool
	kvSrc    int
	// vNorm: scale-less v_norm on this layer's V before it is stored. HF and the CPU apply it on EVERY Gemma 4 layer
	// that owns its K/V, K=V or not. Implies kEqV or a real v_proj.
	vNorm             bool
	preNorm, postNorm Buffer
	// Gemma sandwich norms (absent unless NormSandwich4): applied to the SUBLAYER OUTPUT before
	// the residual add, not to a GEMV input.
	postAttnNorm, postMLPNorm Buffer
	invF                      Buffer // per-layer RoPE inv-freq (local vs global base)
	// mscale: YaRN's attention_factor for THIS layer (decoder.Model.RopeMscaleLayer). 1.0 for every family without
	// YaRN and on a YaRN family's non-scaled layers (Mellum's sliding layers), so it is per-layer, not per-model.
	// Passed to rope_kv / rope_kv_batched, which fold it into cos/sin.
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

	// mixer and ffn are this layer's operations (cudaMixerOf, from the program): the sequence mixer, then a feed-forward block when ffn. launchToken
	// dispatches on mixer, and isDeltaNet and isMLA (methods below) read it, so nothing else decides a layer's kind.
	mixer decoder.ResidentMixer
	ffn   bool
	// walk is the decoded sublayer walk of this layer's program operations, driving sublayer norms, gating, QK-norm, residual modes, and FFN topology.
	walk program.LayerWalk
	// steps is the layer's fusion-table steps (cudaStepsOf), set at build; launchToken runs them in order (OP4cu-b2).
	steps []cudaStep
	// prefillSteps is the layer's batched prefill steps (cudaPrefillStepsOf), set at build; prefillCore runs them in order (OP4cu-d1).
	// Empty for a layer the batched pass has no blocks for, whose model prefillStaticDecline sends to the sequential path.
	prefillSteps []prefillStep
	// Gated-DeltaNet mixer layer. When mixer is MixerDeltaNet the sequence mixer is the recurrent delta rule (deltanet.cu)
	// instead of attention: no KV cache, no q/k/v/o, and TWO persistent state buffers updated in place per token.
	// dnState is [nv*hv*hk], stored TRANSPOSED relative to the CPU's [hk,hv] so each thread owns a contiguous row;
	// dnWin is the causal-conv ring, [(K-1)*convDim]. Both COMPOUND, so both are re-zeroed per generation (Reset),
	// unlike a KV cache, which the next sequence simply overwrites.
	dnQKV, dnZ, dnOut, dnB, dnA  cudaWQ  // the five DeltaNet projections
	dnConvW, dnDtBias, dnNegExpA Buffer  // conv taps, dt bias, precomputed -exp(A_log)
	dnNormW                      Buffer  // [hv] gated-RMSNorm weight, shared across heads
	dnBetaScale                  float32 // the write gate's scale: 2 where the layer's NegEigval widens it to [0, 2), else 1
	dnGNormEps                   float32 // the gated output norm's epsilon: the layer's ONormEps, else the model's NormEps
	dnWin, dnState               Buffer  // persistent: conv ring, recurrent matrix state
	// Mamba-2 mixer layer (Nemotron-H; mamba_resident.go): in_proj [projDim, hidden] and out_proj [hidden, dInner] as int8 with per-row scales, the conv
	// taps and bias, headP = {-exp(aLog), dtBias, D} per head, the gated-norm weight, and TWO persistent buffers updated in place per token: mbWin is the
	// conv window [(K-1)*convDim], mbSSM the SSM state [nHeads*headDim*dState]. Both compound, so resetState re-zeroes them per generation.
	mbIn, mbOut               cudaWQ
	mbConvW, mbConvB, mbHeadP Buffer
	mbNormW, mbWin, mbSSM     Buffer
	// recSlots is each KV slot's conv window and state, on a recurrent layer of a build with several slots. UseKVSlot points the
	// layer's dnWin and dnState, or mbWin and mbSSM, at the bound slot's pair.
	recSlots []recurrentPair
	// qGate marks a SOFTMAX layer of the same family: q_proj is double width ([query ‖ gate] per
	// head) and the context is scaled by sigmoid(gate) before o_proj. The weight stays fused
	// because it is quantized; the split happens on the activation.
	qGate bool

	// MLA (DeepSeek / Kimi, FeatMLA)
	mlaQA, mlaQB, mlaQ, mlaKVA cudaWQ
	mlaQANorm, mlaKVANorm      Buffer
	mlaWUK, mlaWUV             Buffer

	// CUDA-graph capture of this layer's three STATIC launch segments (r.graphs). segA = QKV proj +
	// qk/v-norm (pre-RoPE); segB = ctx-quant + o-proj + the MLP up to the router readback; segC =
	// the expert loop + join (nil for a dense layer — no readback gap). rope_kv, attention, the g4x2
	// zero-upload and the loadRoutedExperts D2H stay LIVE in the gaps (per-token pos/nKeys, or a host
	// round-trip — neither is graph-capturable). Captured once at build, replayed per token; each
	// replay reads the CURRENT buffer contents (TestCUDA_graphReplay), so the routing that changes
	// per token flows through unchanged. nil unless r.graphs.
	gSegA, gSegB, gSegC *gpu.Graph
}

// lookupKnob reads one operator knob from the model's snapshot. A resident built by hand in a test, with no
// model behind it, reads the live environment instead — decoder's nil-snapshot rule.
func (r *cudaResident) lookupKnob(name string) (string, bool) {
	if r.knob == nil {
		return os.LookupEnv(name)
	}
	return r.knob(name)
}

// knobValue is lookupKnob's value, "" when unset.
func (r *cudaResident) knobValue(name string) string { v, _ := r.lookupKnob(name); return v }

// cudaResident is the production resident decode runner: the parity-gated cgo-free forward, promoted from the test
// harness. All CUDA state is owned by a single LockOSThread-pinned executor goroutine; Forward routes one channel
// round-trip per token. BuildResident decides which families it takes (dense, MoE, MLA, Gated-DeltaNet, ...), with
// mixed int4/int8/f32 weights as the real q4_k_m checkpoint stores them.
type cudaResident struct {
	// prog is the model's program (decoder.Model.Program), shared and read-only, decoded into each layer's walk.
	prog *program.Program
	line string // the plan line (a mem line) the allocation helpers lease on: onLine sets it
	// recFloats is each recurrent layer's conv window and state in floats, read from stateSpecsOf when the layer is uploaded; zero on a
	// layer with none. Every recurrent buffer (live, further slots, checkpoint, reset) and every price of them reads it.
	recFloats []recurrentFloats

	// pk is the staging block streamW packs through; the build's setup job drops it when it ends.
	pk *packPool

	loadAbort <-chan struct{} // decoder.Model.LoadAbort: closed by the swap tripwire to stop the build; nil never fires
	reqCh     chan func() error
	ackCh     chan error

	// knob reads this model's operator knobs (decoder.Model.Knob): the snapshot taken at Load, with
	// Options.Knobs applied, never the live environment.
	knob func(name string) (string, bool)

	// device state — touched ONLY on the executor thread.
	dev    *Device
	stream Queue

	launchN int // diagnostic: per-forward dispatch count (graph-capturable-fraction bound)
	// launchViaExecutor sends launch through gocudrv's executor thread (Queue.Launch) instead of issuing it from this
	// thread (Queue.LaunchDirect). Same kernels and arguments either way; tests set it for the A/B between the two.
	launchViaExecutor bool
	launchErr         error // sticky first launch error within a launchToken call (reset per token)
	// resetErr holds the last Reset() failure. Reset cannot return one (the cross-backend interface returns nothing),
	// so Forward/ForwardN surface it.
	resetErr error
	setupErr error // first alloc/upload error during BuildResident's setup job

	layers    []cudaLayer
	lmW       cudaWQ
	finalNorm Buffer

	// Each subsystem's state, declared in its own file; every field is promoted.
	modelGeom
	planState
	decodeKernels
	decodeScratch
	argmaxState
	kvState
	kvWindowState
	flashDecodeState
	splitKVState
	prefillKernels
	prefillRun
	visionPrefill
	moeState
	expertStream
	gptOssState
	floatActState
	mlaState
	dnetState
	mambaState
	ckptState
	loraBind
	graphState
	captureState
	costState
	dbgState
}

// modelGeom is the model-level geometry and numerics cudaResident takes from the architecture at build: the widths, the norm and
// activation scheme, the softcaps and scales, and the per-family switches. Per-layer geometry is cudaLayer's.
type modelGeom struct {
	hidden, nLayers, inter, vocab int
	// Gemma 4 E-model: pleP is the per-layer-embedding width (0 otherwise), embLen the length of every embedding row a forward
	// takes (hidden, or hidden+nLayers*pleP), pleG the PLE gate's output scratch. r.x is embLen floats; the PLE inputs are its tail.
	pleP, embLen int
	pleG         Buffer
	eModel       bool // pleP > 0 or any KV-shared layer: the batched prefill, drafters and LoRA decline it by name
	// nH (query-head count) is the ONE model-level attention dimension: constant across a family's layers (Gemma 4
	// is 16 in both variants), so GQA tracks per-layer nKV via nH/Ly.nKV. hd/nKV/qDim/kvDim/rhalf are DELIBERATELY
	// per-layer only (cudaLayer): a uniform-source threading bug is then a compile error, not a silent byte-identical
	// pass on uniform models. Allocation-time maxima live as backend.go locals; the per-layer KV cache and UploadKV
	// read r.layers[l].kvDim.
	nH             int
	eps, attnScale float32
	finalSoftcap   float32 // Gemma final-logit softcap (30); 0 ⇒ none. Applied host-side in step().
	attnCap        float32 // Gemma 2 attention-score softcap (50); 0 ⇒ none. The covered kernels' capped variants take it as an argument (withCap).
	// attnTempBeta/attnTempOrigMaxPos (Ministral 3, FeatAttnTemp): the raw
	// params behind the post-RoPE query scale (decoder.Model.AttnTempParams), 0 for every family without it.
	// Recomputed into qTempScale per decode call (rope_kv's parameter) or passed raw into rope_kv_batched, which
	// must recompute it per row (position varies within one batched launch, unlike decode's single scalar per call).
	attnTempBeta, attnTempOrigMaxPos float64
	qkNorm                           bool  // arch needs per-head Q/K RMSNorm before RoPE
	qkNormWhole                      bool  // QK-norm reduces over the WHOLE q/k vector, not per head (Olmo 3/Olmo Hybrid) — qkNorm must also be true
	rmsAddOne                        bool  // (1+w) offset — false for Qwen3/Llama
	act                              int32 // gated MLP activation, decoder.ActKind (0=gelu-tanh, 1=silu)
	sandwich                         bool  // Gemma 4-norm sandwich: extra post-attn / post-MLP norms
	postOnly                         bool  // NO pre-norm at all (Olmo 3/Olmo Hybrid) — segA quantizes the raw residual; PostAttnNorm/PostMLPNorm still dispatch, same as sandwich's post half
	// Per-32 activation groups, LayerNorm, parallel-block and logit-scale fields (Cohere/Command-R, Cohere2).
	actG32        bool    // decoder.Options.ActQuantGroup == 32: actgroup.cu's per-32 kernels are bound into fRms/fQ/fSw/gemvW4/gemvW8, and every activation scale buffer holds 2·K/32 floats (actScaleLen: scales, then per-group sums for gemv_q4k_g32)
	layerNorm     bool    // arch.Norm==NormLayer — layernorm_quant (mean-centered, bias-free) instead of rmsnorm_quant at every norm site that feeds a GEMV; see the r.norm dispatcher
	parallelBlock bool    // FeatParallelBlock: ONE shared input norm feeds attn AND MLP independently (x_final = x_orig + attn_out + mlp_out) — segBFFN reuses segA's r.aq/r.aSc instead of re-normalizing r.x; no post-attn/post-MLP norm exists for this family
	logitScale    float32 // host-side final-logit multiplier (1/arch.LogitScale), applied in step(); 0 ⇒ none (FeatLogitScale)
	pairwiseRoPE  bool    // GPT-J pairwise rotation (Cohere/Cohere2/Aya/GLM-OCR): ropeKV/bRopeKV/bRopeKVMRoPE are bound to the rope_pairwise.ptx twins, not the NeoX kernels. Set from Model.BackendView().PairwiseRoPE() at build
}

// alloc/upload helpers, called ONLY inside the setup job (r.dev's context current on the executor thread).
// gpu.NewBufferLenOf PANICS on OOM (recorded into the Device ledger); the executor's runJob recover, NOT
// BuildResident's own defer (a different goroutine, which cannot reach an executor-thread panic), turns it into
// setupErr, so BuildResident declines gracefully (staged fallback) instead of proceeding with unusable buffers.
func (r *cudaResident) af(n int) Buffer {
	allocSeen(n * 4)
	return gpu.NewBufferLenOfOn[float32](r.dev, r.line, n)
}

// onLine sets the plan line the allocation helpers lease on, until the returned func restores the one before. Call it only on the
// resident's executor goroutine (inside a job; doOn from outside one), where the helpers read it, so the line needs no lock.
func (r *cudaResident) onLine(line string) func() {
	prev := r.line
	r.line = line
	return func() { r.line = prev }
}

// upOn uploads v on plan line line.
func (r *cudaResident) upOn(line string, v []float32) Buffer {
	defer r.onLine(line)()
	return r.up32(v)
}

// akv allocates n elements of layer l's K or V cache: f16 for a kvHalf layer, int8 for a kvI8 one, f32 otherwise. Buffer.Len is in elements
// either way.
func (r *cudaResident) akv(l, n int) Buffer {
	if r.layers[l].kvI8 {
		allocSeen(n)
		kvAllocSeen(r, n)
		return gpu.NewBufferLenOfOn[int8](r.dev, r.line, n)
	}
	if r.layers[l].kvHalf {
		allocSeen(n * 2)
		kvAllocSeen(r, n*2)
		return gpu.NewBufferLenOfOn[uint16](r.dev, r.line, n)
	}
	kvAllocSeen(r, n*4)
	return r.af(n)
}

// akvScales allocates an int8 layer's K or V scale pool: n f32 scales, one per (pool row, KV head).
func (r *cudaResident) akvScales(n int) Buffer {
	kvAllocSeen(r, n*4)
	return r.af(n)
}

// kvAllocSeenForTest, when set, is told the resident and the bytes of every K/V cache buffer akv allocates, from the branch that
// picks the element type. A test seam, nil in production; a delivered-bytes gate sums it for the resident a load returns, so another
// build during that load (a first load's probe) does not count.
var kvAllocSeenForTest func(r *cudaResident, bytes int)

func kvAllocSeen(r *cudaResident, bytes int) {
	if kvAllocSeenForTest != nil {
		kvAllocSeenForTest(r, bytes)
	}
}
func (r *cudaResident) ai(n int) Buffer {
	allocSeen(n * 4)
	return gpu.NewBufferLenOfOn[int32](r.dev, r.line, n)
}
func (r *cudaResident) au32(n int) Buffer {
	allocSeen(n * 4)
	return gpu.NewBufferLenOfOn[uint32](r.dev, r.line, n)
}

// allocSeenForTest, when set, is told the size in bytes of every buffer af, akv, ai and au32 allocate. A test seam, nil in
// production; it is called on whichever goroutine allocates (the executor), so a test reads what it gathered after r.do returns.
var allocSeenForTest func(bytes int)

func allocSeen(bytes int) {
	if allocSeenForTest != nil {
		allocSeenForTest(bytes)
	}
}

// recordUpload captures the FIRST alloc/upload error hit during BuildResident's setup job into r.setupErr. The
// up* helpers must not discard gpu.Upload's error: a failed weight upload would leave a ZEROED device buffer and
// the build would still return ok=true, a resident that decodes garbage. The setup job's last statement returns
// r.setupErr, which BuildResident turns into a graceful decline (staged/CPU fallback); later errors in the same
// doomed job are noise. Called only at load time.
func (r *cudaResident) recordUpload(e error) {
	if e != nil && r.setupErr == nil {
		r.setupErr = e
	}
}
func (r *cudaResident) up32(v []float32) Buffer {
	b := wbuf[float32](r, len(v))
	r.recordUpload(gpu.Upload(b, v))
	return b
}
func (r *cudaResident) upu32(v []uint32) Buffer {
	b := wbuf[uint32](r, len(v))
	r.recordUpload(gpu.Upload(b, v))
	return b
}
func (r *cudaResident) upu16(v []uint16) Buffer {
	b := wbuf[uint16](r, len(v))
	r.recordUpload(gpu.Upload(b, v))
	return b
}

// loadAborted reports whether the swap tripwire has closed the load's abort channel. Cheap enough for every layer and every upload block.
func (r *cudaResident) loadAborted() bool {
	select {
	case <-r.loadAbort:
		return true
	default:
		return false
	}
}

func (r *cudaResident) upW(h hostW) cudaWQ {
	if h.alt != nil {
		w := r.streamW(h)
		arena := r.wArena // the plan's walk does not count the int8 copy: it takes allocations of its own
		r.wArena = nil
		r.kquantAlt = append(r.kquantAlt, [2]cudaWQ{w, r.upW(*h.alt)})
		r.wArena = arena
		return w
	}
	if h.src != nil || h.readRows != nil { // planned, not packed: packed at upload
		return r.streamW(h)
	}
	w := cudaWQ{kind: h.kind, W: r.upu32(h.wpk), N: h.N, K: h.K}
	switch h.kind {
	case "int4":
		w.ws16 = r.upu16(h.ws16)
	case "q4k":
		// scales and minimums live inside the super-blocks
	default:
		w.ws = r.up32(h.ws)
	}
	return w
}

// upExperts uploads an expert stack, VRAM-slot-cached (C′) when cacheExperts is set and the weight is int4 (the
// only kind the resident MoE GEMVs accept), else fully VRAM-resident. The GEMVs read device memory only: a
// zero-copy direct read of the host stack was tried and mis-read by gemv_w4a8_moe at width
// ( NewMappedHostBuffer's doc).
func (r *cudaResident) upExperts(h hostW) cudaWQ {
	defer r.onLine(mem.LineExperts)()
	if r.cacheExperts && h.kind == "int4" {
		return r.cacheWQ(h)
	}
	if h.srcs != nil {
		r.recordUpload(fmt.Errorf("cuda: a planned expert stack reached an upload that is not the expert cache"))
		return cudaWQ{kind: h.kind, N: h.N, K: h.K}
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
	w.perExpertW, w.perExpertS = perExpertWords(h.N, h.K, r.nE)
	if h.srcs != nil {
		r.packStackPinned(&w, h)
		return w
	}
	w.srcW = r.mapBytes(u32bytes(h.wpk))
	w.srcS = r.mapBytes(u16bytes(h.ws16))
	// Device slot buffers (W/ws16) are NOT allocated here — allocSlots does it after the core +
	// KV are up, so it can size r.cacheSlots to the MEASURED free VRAM and never OOM.
	return w
}

// perExpertWords is one expert's share of an int4 stack of n rows over nE experts: uint32 weight words (K/8 a row) and uint16 group
// scales (K/32 a row).
func perExpertWords(n, k, nE int) (w, s int) {
	rows := n / nE
	return rows * (k / 8), rows * (k / 32)
}

// allocQuantumBytes is the driver's allocation granularity (gpu.AllocQuantumBytes, measured there).
const allocQuantumBytes = gpu.AllocQuantumBytes

// A compile error here means fitdecide.AllocQuantumBytes has drifted from the driver granularity.
const _ = uint(fitdecide.AllocQuantumBytes-gpu.AllocQuantumBytes) + uint(gpu.AllocQuantumBytes-fitdecide.AllocQuantumBytes)

// allocRoundSlack is what the driver adds to ONE buffer of n bytes: gpu.ChargedBytes's excess, the one rounding rule the plan and
// the ledger share (docs/code-notes/cuda.md#allocRoundSlack).
func allocRoundSlack(n int64) int64 { return gpu.ChargedBytes(n) - n }

// allocSlackSeenForTest, when set (tests only), is told at each real build the slack the plan priced and packedAllocSlack over the
// build's own planned buffers: they must be equal (TestPlanSlack_matchesTheBuild).
var allocSlackSeenForTest func(planned, built int64)

// recurrentStatePlannedForTest, when set (tests only), is told at each real build the recurrent state the context plan priced: it
// must equal what the build then charges for it (recurrentStateCharged; TestRecurrentStatePlanned_matchesTheBuild).
var recurrentStatePlannedForTest func(planned int64)

// recurrentStateAtKVFitForTest, when set (tests only), is told at each checkKVFits the bytes the device's ledger holds on the state
// line: the recurrent state is allocated at upload, before the KV fit's free-VRAM probe, so that probe already sees all of it.
var recurrentStateAtKVFitForTest func(charged int64)

// attnSoftcapDropForTest builds every resident without its attention-score softcap while the CPU keeps it: the planted defect of
// TestG2R_cuda's arm P, the real Gemma 2 gate. Tests only.
var attnSoftcapDropForTest bool

var hostWType = reflect.TypeFor[hostW]()

// packedAllocSlack is what the driver charges beyond the requested bytes of every weight buffer the build is about to upload
// (walkWeightBuffers): the dense weights share one arena (weightArena), whose alignment padding and single rounding are the slack;
// each expert-stack buffer is its own allocation, rounded when it is a quantum or more (allocRoundSlack) and packed into the pages
// the smaller ones leave part empty (pagePacker), after the arena, in upload order. Experts under C' are slot-cached rather than
// uploaded whole, so they are over-counted there, which only shrinks the planned context.
func packedAllocSlack(v any) int64 {
	var (
		pk                   pagePacker
		slack, arena, inside int64
		own                  []int64
	)
	walkWeightBuffers(v, func(n int64, isOwn bool) {
		switch {
		case n <= 0:
		case isOwn:
			own = append(own, n)
		default:
			arena += weightAligned(n)
			inside += n
		}
	})
	if arena > 0 {
		slack += arena - inside + allocRoundSlack(arena)
		pk.add(arena)
	}
	for _, n := range own {
		slack += allocRoundSlack(n)
		pk.add(n)
	}
	return slack + pk.waste()
}

// checkWeightsFit fails the load EARLY, right after the device exists and before any kernel compile or weight
// upload, when the model's FIXED weight bytes alone exceed free VRAM. Without it a dense model whose weights do
// not fit ran the whole compile+upload sequence and failed on whichever allocation first did not fit: a raw
// driver error, not a decline naming the numbers (checkKVFits' point, but it only runs after the weights are up).
//
// `need` is decoder.Model.ResidentDenseWeightBytesFor, deliberately NOT ResidentWeightBytesPaged(0): routed MoE
// experts have their own elastic sizing (capSlots), a bisection against LIVE free VRAM that runs after the dense
// weights are uploaded. Pricing experts at their full unpaged size here would decline a model whose experts are
// about to be capped down to something that fits.
func (r *cudaResident) checkWeightsFit(m *decoder.Model) error {
	free, _, err := r.dev.Context().MemInfo()
	site := "weights"
	if r.planning {
		site = "weightsOpen" // openForPlan's check, before the plan and before moe_route's pool is paid
	}
	if memReadSeenForTest != nil {
		memReadSeenForTest(site, int64(free), err == nil)
	}
	return fitdecide.WeightsFitDecision(m.ResidentDenseWeightBytesFor("cuda"), m.ExtraResidentBytes(), int64(free), err == nil, plantedMargin(site, r.ctxMargin()))
}

// checkCap guards the resident KV allocation. Every layer's cache is sized r.ctxCap*kvDim, so a write for
// absolute position p lands at kc[p*kvDim ...]; valid positions are [0, r.ctxCap). Writing past it
// (rope_kv/kv_store) is an out-of-bounds DEVICE write: silent memory corruption, and the attention launch's
// shared-mem request eventually exceeds the block limit. Nothing upstream clamps prompt+max_tokens to the cap, so
// return an error here; the decode loop stops on it (model.go) and the caller can fall back to the staged path,
// which handles longer contexts. r.ctxCap is always the capacity the caches were actually SIZED with, read from
// the field the allocation used.
func (r *cudaResident) checkCap(pos, n int) error {
	if pos < 0 || pos+n > r.ctxCap {
		return fmt.Errorf("cuda: KV position %d(+%d) exceeds resident context cap %d — raise it with the serve context setting (bounded by the model's own context window), or use the staged path for longer contexts", pos, n, r.ctxCap)
	}
	return nil
}

// ContextCap is the resident KV capacity in positions (queryable so callers can clamp max_tokens
// up front rather than discover the limit mid-generation).
func (r *cudaResident) ContextCap() int { return r.ctxCap }

// ResidentCaps is decoder.ResidentRuntimeSubsystem's. The batched prefill always reports a fast-kernel floor, 0 when the fast kernels are off.
func (r *cudaResident) ResidentCaps() decoder.ResidentCaps {
	return decoder.ResidentCaps{
		ContextCap:       r.ContextCap(),
		HasFastPrefill:   true,
		FastPrefillFloor: r.FastPrefillFloor(),
		TailMinRows:      r.PrefillTailMinRows(),
	}
}

// Forward runs one token at absolute position pos and returns logits[vocab].
func (r *cudaResident) Forward(embedding []float32, pos int) ([]float32, error) {
	if e := r.checkCap(pos, 1); e != nil {
		return nil, e
	}
	if pos == 0 {
		// Fresh sequence: re-zero the compounding Gated-DeltaNet state so it does not carry over from a prior Generate on this *Model. No-op for every other family.
		r.Reset()
		if r.resetErr != nil {
			return nil, r.resetErr // a failed re-zero would decode from the previous sequence
		}
	}
	var out []float32
	err := r.do(func() error {
		o, e := r.step(embedding, pos, pos)
		out = o
		return e
	})
	return out, err
}

// ForwardMRoPE is decoder.ResidentDecodeSubsystem: like Forward, but the rotation angle (ropePos) and the KV-cache/attention
// position (pos) are supplied separately: Qwen2.5-VL decode past an image block needs them to differ (see
// gemv_fwd.cu's rope_kv doc comment; decoder/rope.go's CPU reference is the ground truth this mirrors). Forward is
// the pos == ropePos case.
func (r *cudaResident) ForwardMRoPE(embedding []float32, pos, ropePos int) ([]float32, error) {
	if e := r.checkCap(pos, 1); e != nil {
		return nil, e
	}
	if pos == 0 {
		r.Reset() // fresh sequence — same reason as Forward
		if r.resetErr != nil {
			return nil, r.resetErr
		}
	}
	var out []float32
	err := r.do(func() error {
		o, e := r.step(embedding, pos, ropePos)
		out = o
		return e
	})
	return out, err
}

// ForwardNoLogits (ResidentDecodeSubsystem) runs the token's forward to build ONLY its resident K/V —
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
			return r.resetErr
		}
	}
	return r.do(func() error {
		if e := r.launchToken(embedding, pos, pos, false); e != nil {
			return e
		}
		return r.stream.Sync() // step()'s trailing sync is skipped here; drain so the KV write completes
	})
}

// ForwardN runs K tokens at consecutive positions. Correctness-first: sequential steps in a
// single executor round-trip (bit-identical to K Forward calls; amortizes the channel hop).
func (r *cudaResident) ForwardN(embeddings [][]float32, startPos int) ([][]float32, error) {
	if len(embeddings) == 0 {
		return nil, nil // no-op on empty, matching the cpu/webgpu ForwardN contract: prefillReady would
		// otherwise route into prefillCore and return a spurious "empty prompt" error.
	}
	if e := r.checkCap(startPos, len(embeddings)); e != nil {
		return nil, e
	}
	// Batched verify: the whole [cur, draft...] run in ONE weight-stationary pass (prefillCore, allLogits=true)
	// instead of len(embeddings) sequential decode steps: each weight is read once for all M positions, which is what
	// makes speculative decode a win on the resident CUDA path. Bit-identical to the sequential step loop below (every
	// batched kernel is the M=1 kernel with an M dimension + explicit FMA, gated by TestSpecDecodeCurve's
	// lossless-vs-sequential check). Falls back to the per-token loop for archs the batched path does not cover:
	// errPrefillDeclined only, a real compute error propagates. Spec verify runs M<=9, so the batched allocation is
	// tiny. A recurrent (Gated-DeltaNet) family always takes the loop: the weight-stationary pass has no notion of
	// recurrent state, which must advance one token at a time and in order, and the loop is the same r.step this
	// family's decode uses, so prefill and decode see the same state evolution.
	if r.batchedVerifyOK() {
		// context.Background(): ForwardN is the spec-decode verify, M<=9 rows, and its own
		// interface carries no context. Nothing here is long enough to want cancelling.
		if outs, _, err := r.prefillCore(context.Background(), embeddings, startPos, tailAllLogits, nil, nil, nil); err == nil {
			return outs, nil
		} else if !errors.Is(err, errPrefillDeclined) {
			return nil, err
		}
	}
	if startPos == 0 {
		r.Reset() // fresh sequence — the recurrent state compounds and is not positional
		if r.resetErr != nil {
			return nil, r.resetErr
		}
	}
	out := make([][]float32, len(embeddings))
	err := r.do(func() error {
		for i, emb := range embeddings {
			l, e := r.step(emb, startPos+i, startPos+i)
			if e != nil {
				return e
			}
			out[i] = slices.Clone(l) // each row kept, so copy off the reused host buf
		}
		return nil
	})
	return out, err
}

// VerifyPath (decoder.VerifyPathReporter) reports whether this resident's ForwardN executes
// a batched weight-stationary pass or falls back to a sequential per-row loop.
func (r *cudaResident) VerifyPath() (bool, string) {
	if !r.batchedVerifyOK() {
		return false, "sequential — arch/geometry not supported by batched verify"
	}
	return true, "batched weight-stationary CUDA pass"
}

// Reset re-zeroes the compounding recurrent state for a fresh generation; it is a no-op for every other family,
// because a KV cache needs no reset (the next sequence overwrites it positionally). A Gated-DeltaNet layer's
// {conv ring, matrix state} is NOT positional: it accumulates, and without this the next Generate on the same
// *Model decodes from the previous one's state. The state buffers are allocated zeroed, so the FIRST generation is
// correct either way, which is exactly what makes the omission invisible to a single-sequence test.
func (r *cudaResident) Reset() {
	if r.dnet == nil && r.mamba == nil {
		return
	}
	// Reset satisfies the cross-backend ResidentForward interface and returns nothing, so a failed re-zero cannot be
	// returned; it would be the same silent failure (the next sequence decodes from the PREVIOUS one's recurrent
	// state, invisible to a single-sequence test). Recorded in resetErr and surfaced by the next Forward/ForwardN, the
	// same shape gpu/residency.go uses.
	r.resetErr = r.do(func() error { return r.resetState() })
}

// resetState is Reset's body WITHOUT the executor hop, for callers already running on the executor
// thread. graphsSelfTest is one: it runs inside r.do, so calling Reset there posts to reqCh from
// the goroutine that services reqCh and deadlocks — a hang, not an error, and one that only appears
// once graphs are actually admitted for a recurrent model.
func (r *cudaResident) resetState() error {
	if k := r.ckptCopyOf(r.kvSlot); k >= 0 { // a new sequence: what the copy holds is no request's, so a later save takes it first
		r.ckptCopies[k].slot = -1
	}
	if e := r.resetMambaState(); e != nil {
		return e
	}
	if recurrentResetEverySlotForTest {
		if e := r.resetEverySlotForTest(); e != nil {
			return e
		}
	}
	r.DropFold() // rows marked against the state being zeroed must never fold into the new sequence
	if r.dnet == nil {
		return nil
	}
	zero := zeroFloats()
	for i := range r.layers {
		L := &r.layers[i]
		if !L.isDeltaNet() {
			continue
		}
		if e := gpu.Upload(L.dnWin, zero(r.recFloats[i].win)); e != nil {
			return e
		}
		if e := gpu.Upload(L.dnState, zero(r.recFloats[i].state)); e != nil {
			return e
		}
	}
	return nil
}

// Close releases the model's GPU memory and tears down the pinned executor (unpinning its OS thread).
//
// Freeing the DEVICE memory is the whole job: a resident model owns the weight buffers and the per-layer KV
// cache, gigabytes for a real checkpoint. A Close that only closed the channel leaks the entire model until the
// process exits: invisible in a one-model run, fatal for a model zoo, an /admin/models/unload, or a test binary
// loading models in sequence (it saturates an 8 GB card, after which every Alloc silently returns nil and the
// zero-filled buffers look like a parity bug rather than an OOM).
//
// Every buffer is freed EXPLICITLY rather than by leaning on context destruction. Releasing our primary-context
// reference only reclaims memory if the refcount reaches ZERO, and dev.Primary() hands out a refcounted
// per-device singleton, so any other holder (a second model in a zoo, another subsystem, a test's own probe
// context) keeps the context alive and the "freed" model's VRAM never comes back. TestResidentCloseFreesVRAM
// pins it.
//
// All of it runs ON the executor thread (that thread made the context current) and therefore before reqCh
// closes. Page-locked host memory goes first: it must be freed before the context.
//
// Returns error to satisfy io.Closer (see the assertion in backend.go); the native releases are best-effort and
// cannot meaningfully fail, so it always returns nil.
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
		if r.kvTabStage != nil {
			_ = r.kvTabStage.Close()
			r.kvTabStage = nil
		}
		// Host-mapped expert stacks are page-locked host memory too (not in the device ledger), so free them here alongside logitsPinned, before ReleaseObjects.
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
		// Release demoted host-mmap KV snapshots explicitly: each snapshot holds a back-pointer to r and a finalizer,
		// so leaving r.kvDemoted populated forms a finalized reference cycle that can prevent reclamation.
		for s := range r.kvDemoted {
			r.kvDropDemoted(s)
		}
		r.kvDemoted = nil
		// The Device OWNS every device allocation (weights, KV caches, scratch) in its ledger; ReleaseObjects frees all of
		// them in the correct order, then the modules + stream, then releases our primary-context ref.
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

// copyF32 copies n float32 from src to dst on the decode stream: a launch of kv_store at pos 0 (cache[i] = src[i]), so it is an ordinary kernel launch in the same stream order as the projections
// around it and, unlike gpu.CopyDevice (which synchronizes the context), it records into a CUDA graph. Exact: no arithmetic. The two buffers must not overlap.
func (r *cudaResident) copyF32(src, dst Buffer, n int) error {
	return r.launch(r.kvCopy, g1cfg(n, 256), Arg(src), Arg(dst), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(n)))
}

func g1cfg(n, b int) LaunchConfig {
	return LaunchConfig{GridX: uint32((n + b - 1) / b), GridY: 1, GridZ: 1, BlockX: uint32(b), BlockY: 1, BlockZ: 1}
}
func onecfg(b, sh int) LaunchConfig {
	return LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: uint32(b), BlockY: 1, BlockZ: 1, SharedMemBytes: uint32(sh)}
}

// actCfg is the launch of an activation kernel (r.fRms, r.fQ, r.fSw) over n elements, with block b and sh shared bytes. A per-vector
// kernel needs its whole-vector absmax in one block, so it gets one; a per-32 kernel (actG32) runs a warp a group at any grid and gives
// the same outputs at every grid (actgroup.cu), so it gets a block for every b/32 groups and its groups run in parallel.
func (r *cudaResident) actCfg(b, sh, n int) LaunchConfig {
	c := onecfg(b, sh)
	if r.actG32 {
		c.GridX = uint32(max(1, (n/32+b/32-1)/(b/32)))
	}
	return c
}

// glueQuantThreads is glu_quant's block size. The kernel is ONE block (its int8 scale needs the max over the whole
// intermediate vector), so its time is the serial per-thread loop over I elements, and 1024 threads divide it. Its
// only reduction is a MAX (exact, order-independent) and every element's value and packing are per-element, so the
// block size changes nothing but how the work is divided: the output is bit-identical (checked on real logits,
// docs/measurements/d7-decode-breakdown-2026-09-21.md).
const glueQuantThreads = 1024

func (r *cudaResident) launch(f Pipeline, cfg LaunchConfig, args ...KernelArg) error {
	if r.dbgFreeBeforeLaunch == 0 { // free VRAM at the FIRST launch, recording only
		if f0, _, e0 := r.dev.Context().MemInfo(); e0 == nil {
			r.dbgFreeBeforeLaunch = f0
		}
	}
	// Recording only: free VRAM immediately BEFORE the launch, the other side of the event from describeLaunchErr's
	// reading. That one is reached only after Launch returns non-nil, so it cannot distinguish "memory was released
	// before this launch was attempted" from "the failed attempt released it while unwinding". Recorded at every
	// launch it also yields the decrement from free-at-first-launch to free-at-the-failing-launch from one run.
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
	if r.cost != nil {
		r.countLaunch(f)
		if costPlantForTest == 1 && r.costNamed == 1 { // the planted second launch of the token's first kernel
			r.countLaunch(f)
			_ = r.stream.Launch(f, cfg, args...)
		}
	}
	var e error
	if r.launchViaExecutor {
		e = r.stream.Launch(f, cfg, args...)
	} else {
		e = r.stream.LaunchDirect(f, cfg, args...)
	}
	if e != nil {
		e = r.describeLaunchErr(f, e)
	}
	if e != nil && r.launchErr == nil {
		// Sticky: launchToken's dense hot chain discards many launch errors (`_ = r.launch(...)`), so a config error (bad
		// shared-mem size, bad args) would let the token "succeed" with stale buffers. Record the first here;
		// launchToken returns it. doG/rms funnel through launch too, so this covers the whole chain without touching every
		// call site.
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
	pt := reflect.TypeFor[Pipeline]()
	// A subsystem's fields sit in an embedded struct; the walk descends into it, so a pipeline is named by its promoted name.
	var find func(v reflect.Value) string
	find = func(v reflect.Value) string {
		t := v.Type()
		for i := 0; i < t.NumField(); i++ {
			sf := t.Field(i)
			if sf.Anonymous && sf.Type.Kind() == reflect.Struct {
				if n := find(v.Field(i)); n != "" {
					return n
				}
				continue
			}
			if sf.Type != pt {
				continue
			}
			fv := reflect.NewAt(pt, unsafe.Pointer(v.Field(i).UnsafeAddr())).Elem().Interface().(Pipeline)
			if fv == f {
				return sf.Name
			}
		}
		return ""
	}
	if n := find(reflect.ValueOf(r).Elem()); n != "" {
		return n
	}
	return "?"
}

// describeLaunchErr turns a bare driver status into something that names what ran out and what the operator can do
// about it. A launch that runs out of memory with the expert cache on otherwise reads as exactly
// `cuLaunchKernel: CUDA_ERROR_OUT_OF_MEMORY`: the API call, not the kernel, with no connection to the setting that
// caused it, and capSlots' decline floor (below topK) does not cover it
// (docs/code-notes/cuda.md#describeLaunchErr). Message content only: the error is wrapped with %w, so type and
// classification are unchanged.
func (r *cudaResident) describeLaunchErr(f Pipeline, e error) error {
	name := r.pipeName(f)
	// Reframe an out-of-memory launch against the expert cache when it is on: the slot count is by
	// far the largest operator-controlled VRAM consumer on this path. Deliberately does NOT suggest
	// a specific safe value — the computation that would produce one (capSlots) is the thing under
	// suspicion, and printing a number from it would launder a suspect figure into advice.
	if r.cacheExperts && mem.DeviceOutOfMemory(e) {
		// Name the REQUESTED and EFFECTIVE counts both. They differ when the cap fires, and naming
		// only the effective one sends a user who set 48 to lower it to 40 — which caps to the same
		// value, fails identically, and makes the advice look wrong.
		slots := fmt.Sprintf("%d slots/layer", r.cacheSlots)
		if r.cacheSlotsReq != r.cacheSlots {
			slots = fmt.Sprintf("%d slots/layer requested, capped to %d", r.cacheSlotsReq, r.cacheSlots)
		}
		// Deliberately NO free-VRAM reading here. This site is reached only after Launch has returned non-nil, so any
		// figure taken here is a POST-failure state that a reader cannot tell from a pre-launch one: its meaning depends
		// on where the probe sits, which an error string cannot carry. Free VRAM belongs to instrumentation, which can
		// state its own probe position.
		return fmt.Errorf("launch %s: out of device memory with the expert cache at %s — the slot "+
			"count is the likely cause; lower GOINFER_MOE_CACHE_SLOTS below the effective value "+
			"above and retry: %w", name, slots, e)
	}
	return fmt.Errorf("launch %s: %w", name, e)
}

// capVec copies the first n elements of a device vector to host into dst[l] (diagnostic sublayer capture: n is
// hidden for the o-proj/down contributions, qDim for the pre-o-proj context). Runs on the executor thread inside
// launchToken, so it syncs before the readback. It cannot return an error (its callers are capture hooks inside
// launchToken, with no error channel), so it records into setupErr, the field the build path uses, and leaves
// dst[l] nil rather than zero-filled: this feeds HiddenCapture(), and a zero row is a plausible-looking vector a
// speculative drafter would fuse from, while a nil row is something the caller can notice.
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

func (r *cudaResident) rmsDim(src, nrm Buffer, dim int, qOut Buffer, sOut Buffer) error {
	return r.launch(r.fRms, r.actCfg(256, (dim+256)*4, dim),
		Arg(src), Arg(nrm), gpu.ArgValue(int32(dim)), gpu.ArgValue(r.eps),
		gpu.ArgValue(r.addOneArg()), Arg(qOut), Arg(sOut))
}

func (r *cudaResident) rms(src, nrm Buffer, qOut Buffer, sOut Buffer) error {
	return r.rmsDim(src, nrm, r.hidden, qOut, sOut)
}

// layerNormQuant is rms's mean-centered twin (Cohere/Command-R, FeatLayerNorm): no addOne
// selector (bias-free, no Gemma-style (1+w) family uses this norm kind on this backend). Decode runs the batched paths' kernel at one
// row (bLayerNormQuantB), so a step, PrefillLastN or verify row normalizes exactly as decode does: layernorm_quant rounds in another
// order and moves an int8 code by a last bit (docs/measurements/fcoh-cohere-layernorm-2026-11/README.md).
func (r *cudaResident) layerNormQuant(src, nrm Buffer, qOut Buffer, sOut Buffer) error {
	return r.bLayerNormQuantB(src, nrm, r.hidden, 1, qOut, sOut)
}

// norm dispatches the family's pre-GEMV norm+quant into qOut/sOut — layerNormQuant (Cohere's
// bias-free mean-centered LayerNorm) when r.layerNorm, else the default rms — mirroring Metal's
// encodeNorm/decoder/model.go's normalize() dispatch on arch.Norm.
func (r *cudaResident) norm(src, nrm Buffer, qOut Buffer, sOut Buffer) error {
	if r.layerNorm {
		return r.layerNormQuant(src, nrm, qOut, sOut)
	}
	return r.rms(src, nrm, qOut, sOut)
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
	if wt.kind == "q4k" {
		// BuildResident admits q4k only with actG32 (the per-group sums in as[K/32:] come from quantG32).
		return r.launch(r.gemvQ4K, cfg, Arg(wt.W), Arg(a), Arg(as), bias,
			gpu.ArgValue(int32(wt.N)), gpu.ArgValue(int32(wt.K/256)), Arg(dst), gpu.ArgValue(accum))
	}
	if wt.kind == "q5k" || wt.kind == "q6k" {
		cfg.GridX = uint32(launchgeom.KQuantGemvGridX(wt.N))
		args := []KernelArg{Arg(wt.W), Arg(a), Arg(as), bias, gpu.ArgValue(int32(wt.N)), gpu.ArgValue(int32(wt.K / 256)), Arg(dst), gpu.ArgValue(accum)}
		if wt.kind == "q6k" {
			return r.launch(r.gemvQ6K, cfg, args...)
		}
		return r.launch(r.gemvQ5K, cfg, args...)
	}
	if wt.kind == "int4" {
		return r.launch(r.gemvW4, cfg, Arg(wt.W), Arg(a), Arg(wt.ws16), Arg(as), bias,
			gpu.ArgValue(int32(wt.N)), gpu.ArgValue(int32(wt.K/8)), gpu.ArgValue(int32(wt.K/32)), Arg(dst), gpu.ArgValue(accum))
	}
	return r.launch(r.gemvW8, cfg, Arg(wt.W), Arg(a), Arg(wt.ws), Arg(as), bias,
		gpu.ArgValue(int32(wt.N)), gpu.ArgValue(int32(wt.K/4)), Arg(dst), gpu.ArgValue(accum))
}

// sharedExpert adds layer Ly's always-on shared expert to the residual x, from the normed, quantized activation mq/mSc
// the routed experts read: a SwiGLU MLP at sharedInter, added after the routed sum as decoder/mlp.go adds it, ungated
// (GLM/DeepSeek) or scaled by sigmoid(SharedGate·h) (Qwen-MoE). It runs on one row and uses the single-row shared
// scratch, so a caller covering several rows calls it once per row, in stream order.
func (r *cudaResident) sharedExpert(Ly *cudaLayer, mq, mSc, x Buffer) error {
	nullBias := ArgNull()
	if e := r.doG(Ly.shGU, mq, mSc, nullBias, r.shGUout, 0); e != nil {
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
	if !Ly.walk.FFN.FFN.MoE.SharedUngated {
		if e := r.doG(Ly.shGateW, mq, mSc, nullBias, r.shGl, 0); e != nil {
			return e
		}
		gl, ungated = r.shGl, 0
	}
	if e := r.launch(r.fSharedCombine, g1cfg(r.hidden, 256),
		Arg(x), Arg(r.shDownOut), Arg(gl), gpu.ArgValue(int32(r.hidden)),
		gpu.ArgValue(ungated)); e != nil {
		return e
	}
	return nil
}

// launchToken issues one token's whole kernel chain, leaving logits[vocab] on the device.
// ropePos is the rope_kv kernel's rotation-angle position — equal to pos for every ordinary
// (non-m-RoPE) call; only Qwen2.5-VL decode past an image block needs them to differ (see
// gemv_fwd.cu's rope_kv doc comment). pos alone still drives KV storage/attention range.
func (r *cudaResident) launchToken(emb []float32, pos, ropePos int, head bool) error {
	r.launchErr = nil // reset the sticky launch-error accumulator for this token
	nullBias := ArgNull()
	// qTempScale (Ministral 3, FeatAttnTemp): computed ONCE per token (the same
	// value at every layer, unlike mscale, which is per-layer); mirrors decoder.Model.AttnTempScale. attnTempBeta==0
	// (every family without the feature) skips the division: attnTempOrigMaxPos is 0 for those, and pos/0 would poison
	// Q with NaN.
	qTempScale := r.qTempScaleAt(pos)
	if e := r.kvEnsure(pos, 1); e != nil {
		return e
	}
	if r.pleP > 0 && len(emb) != r.embLen {
		// An E-model row is [hidden ‖ nLayers*P]. A short row would leave the PLE tail stale from the previous token, and a plain hidden-sized
		// row (what every other model passes) would silently read the previous token's inputs: refuse by length, here, for every entry point.
		return fmt.Errorf("cuda: Gemma 4 E-model embedding row has %d floats, want %d (hidden %d + %d layers x PLE width %d)", len(emb), r.embLen, r.hidden, r.nLayers, r.pleP)
	}
	if e := r.uploadEmb(emb); e != nil {
		return e
	}
	if r.cacheProf {
		r.profT0 = time.Now()
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
		if legacyLayerForTest != nil {
			if e := legacyLayerForTest(r, Ly, l, pos, ropePos, qTempScale, gA, gB, gC); e != nil {
				return e
			}
			continue
		}
		if e := r.walkLayer(Ly, l, pos, ropePos, qTempScale, gA, gB, gC); e != nil {
			return e
		}
	}
	// Final norm + LM head — skipped for KV-only prefill (head=false): prompt[:-1] tokens need only
	// their K/V in the cache, and the head is a big-vocab matmul + ~1 MB readback + softcap. The layer
	// loop above already wrote this position's K/V identically, so decode stays byte-identical.
	if head {
		if e := r.norm(r.x, r.finalNorm, r.aq, r.aSc); e != nil {
			return e
		}
		if e := r.doG(r.lmW, r.aq, r.aSc, nullBias, r.logits, 0); e != nil {
			return e
		}
		r.profMark(&r.profHead)
	}
	return r.launchErr // surface any launch error discarded in the dense chain above
}

// uploadEmb copies the token's embedding row into r.x as a copy on r.stream: stream order puts it after every launch queued before it
// (the previous token's, which read r.x) and before every launch after it, with no device synchronization around it.
func (r *cudaResident) uploadEmb(emb []float32) error {
	return gpu.UploadPageableAsync(r.stream, r.x, emb)
}

// step returns full logits — the general contract (sampler / constrained decode / logprobs).
// Costs a vocab*4 B D2H every token (594 KB at a 151936 vocab). ropePos: see launchToken.
func (r *cudaResident) step(emb []float32, pos, ropePos int) ([]float32, error) {
	if e := r.launchToken(emb, pos, ropePos, true); e != nil {
		return nil, e
	}
	if e := r.stream.Sync(); e != nil {
		return nil, e
	}
	if e := gpu.ReadToHost(r.logits, r.logitsPinned); e != nil {
		return nil, e
	}
	// Final-logit softcap (Gemma 2/4): softcap·tanh(logits/softcap), applied ONCE to the logit vector after the LM
	// head, host-side exactly as the CPU path (forwardn.go / logitsFromHidden) and as FeatEmbedScale's √hidden is.
	// This is what FeatFinalLogitSoftcap declares on this backend; 0 for every non-softcapped family (no-op). Covers
	// Forward and ForwardN (both route through step).
	applySoftcap(r.logitsHost, r.finalSoftcap)
	// Cohere/Command-R's logits_scaling (FeatLogitScale): a plain host-side multiply, the same
	// shape and site as the softcap just above. 0 for every non-FeatLogitScale family (no-op).
	applyLogitScale(r.logitsHost, r.logitScale)
	return r.logitsHost, nil
}

// ForwardArgmax is the greedy fast path (decoder.ResidentDecodeSubsystem): reduce the argmax on-device
// and read back 4 B instead of the whole logits vector. Same kernel chain, same numerics —
// only the readback differs, so the id equals argmax(Forward(...)) exactly.
func (r *cudaResident) ForwardArgmax(embedding []float32, pos int) (int, error) {
	if e := r.checkCap(pos, 1); e != nil {
		return 0, e
	}
	var id int
	err := r.do(func() error {
		if e := r.launchToken(embedding, pos, pos, true); e != nil {
			return e
		}
		var e error
		id, e = r.argmaxPick(r.vocab)
		return e
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

	// A planned weight (planWeight) has no packed data: src is the weight, packed at upload (streamW), and nWpk, nWs, nWs16 are the element counts of
	// the three device buffers upW allocates for it.
	src              *linalg.WeightMat
	nWpk, nWs, nWs16 int
	// A planned row source (planRows) has no src: readRows widens a block of its rows into the staging block at upload.
	readRows func(r0, n int, dst []float32) error

	// A planned native Q5_K or Q6_K weight (planKQuant) reads its rows from the file: row(r) is destination row r's blocks, and
	// nBytes is the one device buffer's size, the file's bytes for the weight.
	row    func(r int) []byte
	nBytes int
	// alt is the same weight's int8 plan, uploaded beside the native one only under kquantKeepInt8ForTest (KS-O1's A/B).
	alt *hostW

	// A planned expert stack (planWeightStack) names its member weights here, in row order, with src nil: only cacheWQ consumes it.
	srcs []*linalg.WeightMat
}

// packWeight quantizes/repacks a projection into the resident device layout, or errors
// (so BuildResident can decline gracefully → staged fallback) for shapes it can't handle.
func packWeight(w *linalg.WeightMat) (hostW, error) {
	if _, err := planWeight(w); err != nil {
		return hostW{}, err
	}
	p, err := wpack.Pack(w)
	return hostWOf(p), err
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
	p, err := wpack.PackStack(func(w *linalg.WeightMat) error { _, err := planWeight(w); return err }, ws...)
	return hostWOf(p), err
}

// hostWOf is a packed weight as the resident holds it.
func hostWOf(p wpack.Packed) hostW {
	return hostW{kind: p.Kind, wpk: p.Wpk, ws: p.Ws, ws16: p.Ws16, N: p.N, K: p.K}
}

// SetHiddenCapture arms the resident hidden-state seam for the given target layer indices (layer OUTPUTS,
// ascending: the convention decoder.Model.ForwardCapture uses and DeepSpec/z-lab's `target_layer_ids` name). Pass
// nil to disarm. It is the resident counterpart of the CPU ForwardCapture seam that a hidden-state drafter needs
// (not yet wired into production; see cudaResident.hidCap). Cost is one sync + one hidden-sized download per
// TAPPED layer per token, so arm only the taps the drafter actually reads.
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
	r.hidCapTaps = slices.Clone(taps)
	r.hidCapOut = make([][]float32, len(taps))
	return nil
}

// HiddenCapture returns the most recent token's captured layer outputs, one row per tap in
// SetHiddenCapture order. The rows are owned by the caller (capVec allocates per token).
func (r *cudaResident) HiddenCapture() [][]float32 { return r.hidCapOut }

// actScaleLen is how many f32 activation scales a K-wide int8 activation carries: 1 per vector, or
// K/32 under per-32 activation quantization (actG32).
func (r *cudaResident) actScaleLen(k int) int {
	if r.actG32 {
		return 2 * k / 32 // scales, then aS·Σaq per group (quantG32; read by gemv_q4k_g32)
	}
	return 1
}

// wkCode is actgroup.cu's WK_* code for a per-32 weight kind (fused_rms_qkv_g32 / fused_rms_gu_g32).
func wkCode(kind string) int32 {
	switch kind {
	case "q6k":
		return 4
	case "q5k":
		return 3
	case "q4k":
		return 2
	case "int4":
		return 1
	default:
		return 0 // int8
	}
}

// g32Fusable reports whether every listed projection has a kind the fused per-32 kernels read.
func g32Fusable(kinds ...string) bool {
	for _, k := range kinds {
		if k != "int8" && k != "int4" && k != "q4k" && k != "q5k" && k != "q6k" {
			return false
		}
	}
	return true
}

// wkScale is a projection's scale argument for the fused per-32 kernels: the int8 row scales, the
// int4 f16 group scales, or null for the block kinds (their scales live in the super-blocks).
func wkScale(w cudaWQ) KernelArg {
	switch w.kind {
	case "q4k", "q5k", "q6k":
		return ArgNull()
	case "int4":
		return Arg(w.ws16)
	default:
		return Arg(w.ws)
	}
}

// fusedG32MaxHidden is the widest hidden size the fused per-32 kernels are used at: the largest size measured to
// win (H=1536). Wider measured sizes (Phi-3 at 3072, qwen2.5-7b at 3584) were slower fused than unfused,
// and the sizes between are unmeasured and stay unfused.
const fusedG32MaxHidden = 1536

// fusedG32Shmem is fused_rms_qkv_g32 / fused_rms_gu_g32's shared memory for hidden size H:
// normed[H] | red[256] | aq[H/4] int32 | aS[2·H/32].
func fusedG32Shmem(H int) uint32 { return uint32((H + 256 + H/4 + 2*(H/32)) * 4) }

// g4DropVNormForTest re-drops Gemma 4's v_norm on the layers that are not K=V, so a gate can show the fix is what
// moved the parity numbers. Set only by a test seam; never in production.
var g4DropVNormForTest bool

// pleLayerShiftForTest shifts every PLE input slice by this many layers (a planted off-by-one per-layer offset). 0 in production; set only by a test seam.
var pleLayerShiftForTest int

// Test seams that each re-plant one defect, so a gate can show it
// goes red. All false in production; only tests set them.
var (
	g4SkipPLEForTest           bool // (1) the PLE branch skipped
	g4KVSrcOffForTest          bool // (3) a shared layer's source one owning layer of its type off
	g4OneFFNWidthForTest       bool // (4) every dense layer takes layer 0's FFN width
	g4DropLayerScalarForTest   bool // (6) the dense layer scalar dropped
	g4KeepSharedKVStoreForTest bool // (5) a shared layer's K/V store NOT skipped (rope_kv keeps the real nKV)
)
