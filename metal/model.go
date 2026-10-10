//go:build darwin

// Resident Metal decoder: BuildResident uploads a model's weights to the device once, and each token then runs the
// full layer stack in one command buffer. cgo-free throughout (purego-objc plus dlopen of Metal).
package metal

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"slices"
	"sync"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

// metalCtxCapDefault is the resident KV capacity in positions when nothing asks for more (4096).
// The staged/CPU path handles longer unless explicitly requested via decoder.Options.ResidentContext.
const metalCtxCapDefault = decoder.MetalCtxDefault

// metalCtxCapMax is the ceiling on resident KV positions for this backend (32768). The decode attention
// kernels that keep scores in a 4096-key threadgroup buffer (attention, attention_f32 and attention_i8;
// attnScoreTileBound) tile past it with online softmax, which TestAttentionKernelsPastTileBound checks
// against a float64 reference. The exact prefill kernel does not tile; PrefillLast declines it past
// prefillExactAttnMaxKeys.
const metalCtxCapMax = decoder.MetalCtxCeiling

// attnScoreTileBound is the attention kernel's threadgroup score-buffer tile capacity:
// `threadgroup float sc[4096]` in kernels.go holds one score per key in the active tile.
const attnScoreTileBound = 4096

// resolveMetalCtxCap turns a request into the effective resident KV capacity, in the shape of cuda's resolveCtxCap: an
// unpinned load (req <= 0) gets metalCtxCapDefault; an explicit request up to metalCtxCapMax is honored, clamped to
// the model's window; a request above metalCtxCapMax is refused with the numbers.
// A context the load-time fit guard auto-pinned (the caller did not choose it) is a ceiling, not a request: it may
// lower the default, never raise it. Read as a request it allocated several times the default KV on a tight machine,
// or was refused, which moved the whole forward to the CPU.
func resolveMetalCtxCap(m *decoder.Model) (cap int, err error) {
	req := m.ResidentContextRequest()
	if req <= 0 {
		req = metalCtxCapDefault
		if c, ok := metalCtxCeiling.Load(m); ok {
			req = min(req, c.(int)) // S18: lowered by BuildResident before it declines (shrinkCtxToFit)
		}
		return req, nil
	}
	if !m.ResidentContextPinned() {
		req = min(req, metalCtxCapDefault)
	}
	if c, ok := metalCtxCeiling.Load(m); ok && !m.ResidentContextPinned() {
		req = min(req, c.(int)) // S18: lowered by BuildResident before it declines (shrinkCtxToFit)
	}
	if req > metalCtxCapMax {
		return 0, fmt.Errorf("metal: resident context %d positions exceeds this backend's hard "+
			"ceiling of %d (kernel score tile capacity %d with deep-context online softmax) — use the "+
			"staged/CPU path for a longer context, or request %d or fewer",
			req, metalCtxCapMax, attnScoreTileBound, metalCtxCapMax)
	}
	if modelCtx := m.Config().MaxPositions; modelCtx > 0 && req > modelCtx {
		return modelCtx, nil // clamp to the model's own window, same as CUDA's resolveCtxCap
	}
	return req, nil
}

// Threadgroup widths for the kernels that contain a cross-thread float sum reduction (a `red[tid]+=red[tid+st]` tree):
// rmsnorm sum-of-squares, softmax denominator, qk-norm.
//
// These are bit-identity-load-bearing, not performance knobs. Float add is non-associative, so a tree reduction's
// result depends on its width: change T and the last bits of the sum move. On Metal the threadgroup width is also what
// one would sweep for speed, so every such kernel's numerics are wired to its launch configuration.
//
// The existing gates will not catch a change here: paged versus non-paged compares the same kernel at the same width
// (both move together), and GPU-versus-CPU parity is a cosine tolerance. The coupling surfaces only when a new path
// computes the same reduction at a different width and is gated byte-exact. So pin the width here, keep every dispatch
// of these kernels bound to it, and make any alternate same-op kernel inherit it. A byte-exact fixture for such an op
// must use a context longer than the width. Max reductions and simd_sum (32, hardware-fixed) are exempt: order-exact.
//
// See docs/ollama-chase.md §2 (ground rules) and §A2-Metal; the divergence that taught it:
// docs/code-notes/metal.md#tgReduceNorm.
const (
	tgReduceNorm = 256 // rmsnorm_quant / rmsnorm_f32 / rmsnorm_quant_f16 / rmsnorm_f16 sum-of-squares
	tgReduceAttn = 128 // attention / attention_f32 / attention_prefill softmax denom; qk_norm / qk_norm_f16
)

// preciseMathCompile toggles fast-math OFF for the resident kernel library (Task 3 measurement; see
// the CompileLibrary call in BuildResident). Default false = current shipped behavior (fast-math on).
var preciseMathCompile bool

// prefillFeatures is what the f16 MMA prefill kernels (prefill.go) implement: a dense gated FFN (SiLU or GeGLU),
// per-head QK-norm, a per-layer rope table and window, Gemma's sandwich norms and (1+w) RMS offset, and the
// embed-scale and final-logit-softcap pair. Declaring that pair here is a capability statement only: embedResident
// scales the input embeddings before PrefillLast runs, and PrefillLast's final step applies the softcap.
// MoE: the attention half batches normally. By default the FFN half runs expert-major (one batched router GEMM and
// top-k over all rows, host grouping, then each active expert over its rows); GOINFER_MOE_EXPERT_MAJOR=0 keeps the
// row-by-row loop through the per-token decode MoE chain (PrefillLast's L.moe != nil branch). A paged MoE declines
// batched prefill. FeatMoEGatedShared comes along: it is the same encodeMoESharedExpert dispatch decode uses.
var prefillFeatures = map[decoder.ResidentFeature]bool{
	decoder.FeatQKNorm:            true,
	decoder.FeatSlidingWindow:     true,
	decoder.FeatPartialRotary:     true,
	decoder.FeatSandwichNorm:      true,
	decoder.FeatGatedGELU:         true,
	decoder.FeatRMSAddOne:         true,
	decoder.FeatEmbedScale:        true,
	decoder.FeatFinalLogitSoftcap: true,
	decoder.FeatMoE:               true,
	decoder.FeatMoEGatedShared:    true,
	decoder.FeatPerLayerRoPE:      true,
	decoder.FeatRopeMscale:        true, // YaRN attention_factor (Mellum, Olmo 3) — threaded via L.mscale to rope_f16
	decoder.FeatNoPE:              true, // SmolLM3 NoPE layers — invFreq is zero, exact identity in rope_f16
	decoder.FeatPostOnlyNorm:      true, // Olmo 3 / Olmo Hybrid — no pre-norm; sublayer outputs normed before residual
	decoder.FeatQKNormWhole:       true, // Olmo 3 / Olmo Hybrid — single QK reduction over the whole projected width
	// GPT-J pairwise rotation (GLM-OCR; docs/tasks/task-metal-pairwise-followups-2026-10.md Part A): rope_f16_pw and
	// rope_mrope_f16_pw, bound in place of the NeoX prefill kernels for a pairwise model (ropeF16Name).
	decoder.FeatPairwiseRoPE:  true,
	decoder.FeatPairwiseMRoPE: true,
	// Cohere / Command-R7B (Part B): a bias-free LayerNorm (layernorm_f16, layernorm_quant_f16), the parallel block (the
	// MLP reads the attention's shared input norm) and the host-side logit scale. A biased LayerNorm, or LayerNorm with
	// QK-norm, is declined in batchedPrefill.
	decoder.FeatLayerNorm:     true,
	decoder.FeatParallelBlock: true,
	decoder.FeatLogitScale:    true,
}

type residLayer struct {
	qkvW, qkvS, guW, guS, oW, oS, dW, dS Buffer // fused QKV + fused gate/up + o + down
	// The q4k lane (docs/tasks/task-metal-q4k-2026-10.md): each projection as its runs of same-kind rows (q4kLaneProj);
	// the xxxW/xxxS buffers above stay empty on the lane.
	qkvSegs, oSegs, guSegs, dSegs []q4kSeg
	qkvBias, preNorm, postNorm    Buffer
	qNorm, kNorm                  Buffer          // per-head QK-RMSNorm weights (Qwen3; zero if !qkNorm)
	moe                           *moeLayer       // non-nil ⇒ this layer's FFN is MoE (dense guW/dW unused)
	g4moe                         *gemma4MoeLayer // non-nil ⇒ Gemma-4 parallel dense‖MoE FFN (gemma4_moe.go)
	uLayerScalar                  Buffer          // Gemma 4 dense layer's output scalar (S1.0); zero Buffer when 1 or absent
	// ffnI is this dense layer's FFN width and uFFNI its uniform. Equal to the model's I (and r.uI) for every family
	// except a Gemma 4 E-model, whose KV-shared layers are double-wide (S1.3, docs/tasks/task-multimodal-support-2026-10.md).
	ffnI  int
	uFFNI Buffer
	// Gemma 4 E-model PLE branch (S1.5): input gate [P,H], projection [H,P], post norm [H]. Zero when the model has
	// no PLE.
	pleGW, pleGS, plePW, plePS, postPLENorm Buffer
	// kvShared: a Gemma 4 E-model layer that owns no K/V (S1.4). It projects Q only, stores nothing, and attends over
	// layer kvSrc's cache, which r.kc[l]/r.vc[l] (and every slot's) alias.
	kvShared                  bool
	kvSrc                     int
	postAttnNorm, postMLPNorm Buffer // Gemma sandwich norms on each sublayer OUTPUT (zero if !sandwich)
	invf                      Buffer // per-layer RoPE inv-freq (Gemma local 10k vs global 1M base)
	mscale                    Buffer // per-layer YaRN mscale (RopeMscaleLayer; 1.0 = no-op for every family without it)
	uWindow                   Buffer // per-layer attention window (0 = full causal; Gemma mixes local/global)
	window                    uint32 // R2: CPU-side twin of uWindow's value, for canUseAttnFA's dispatch-time guard (windows are out of scope for attention_fa — see its own doc comment)
	// GPT-2 (FeatLayerNorm/FeatNonGatedMLP/FeatOutBias): preNormBias/postNormBias are LayerNorm's
	// bias (unused for RMS families — layernorm_quant only dispatches when arch.Norm==NormLayer).
	// upW/upS is the SEPARATE (not gate-fused) up-projection a non-gated MLP uses instead of
	// guW/guS; oBias/upBias/downBias are the three projection biases FeatOutBias/FeatNonGatedMLP
	// need. All zero-value (unused, never dispatched) for every other family.
	preNormBias, postNormBias Buffer
	upW, upS                  Buffer
	oBias, upBias, downBias   Buffer
	// gpt-oss (FeatAttnSink, DECLARED for metal — decoder/features.go): attnSinks is the per-head learned sink logit
	// [nH]; uHasSink is a 0/1 flag mirroring layerNormBias's pattern. Unused for every other family.
	attnSinks, uHasSink Buffer
	geom                *attnGeom // per-layer attention geometry (hd/nKV/half/kEqV + uniforms); see geom.go

	// Gated-DeltaNet mixer layer (see metal/deltanet.go). delta non-nil marks this layer's sequence mixer as the
	// recurrent delta rule instead of attention: no KV cache, no q/k/v/o, no geom; r.kc[l]/r.vc[l] stay the zero
	// Buffer, and encodeLayer never dispatches pKv/pAttn for it.
	delta *deltaNetLayer
	// qGate marks a SOFTMAX layer of the SAME family: q_proj emits [query ‖ gate] per head at
	// double width and the attention context is scaled by sigmoid(gate) before o_proj. dnQw/dnQs
	// is that double-width Q projection, separate from the (still fused) K‖V in qkvW/qkvS — see
	// encodeAttention's qGate branch. Zero-value Buffers (never dispatched) for every other family.
	qGate      bool
	dnQw, dnQs Buffer
}

// modelKnob is one of m's operator knobs (decoder.Model.Knob): the Load-time snapshot with Options.Knobs applied,
// never the live environment (docs/tasks/task-env-config-2026-09.md, phase 4). "" when unset.
func modelKnob(m *decoder.Model, name string) string { v, _ := m.Knob(name); return v }

// knobValue reads one operator knob from the model this resident was built from. A resident built by hand in a
// test, with no model behind it, reads the live environment instead — decoder's nil-snapshot rule.
func (r *resident) knobValue(name string) string {
	if r.knob == nil {
		return os.Getenv(name)
	}
	v, _ := r.knob(name)
	return v
}

// resident is a Metal-resident decoder: BuildResident uploads the weights once, and per token only the embedding and
// position uniforms change.
type resident struct {
	// knob is the model's decoder.Model.Knob, for the knobs read per call (PrefillLast, the fast-prefill floor).
	knob func(name string) (string, bool)

	d                                                                  *Device
	q                                                                  Queue
	pRms, pQv, pGemv, pGemvResid, pRope, pRope2, pKv, pAttn, pSw, pRes Pipeline
	pLayerScale                                                        Pipeline // Gemma 4 dense layer scalar (S1.0)
	// Gemma 4 E-model PLE (S1.5): pleP is P (0 = no PLE); pleIn holds this token's [L·P] per-layer inputs, copied
	// from the embedding row's tail (decoder embedResidentInto); pleG/pleQ/pleSc the branch's P-wide scratch.
	pPLEGeluMul              Pipeline
	pleP                     int
	pleIn, pleG, pleQ, pleSc Buffer
	ctxCeilKey               *decoder.Model // S18: the metalCtxCeiling entry this resident's context came from; nil when none
	emodelLayerMajorRuns     int            // S9: prompts the E-model layer-major pass ran (prefill_emodel.go); test introspection
	uPleP                    Buffer
	g4VNorm                  bool     // Gemma 4: scale-less v_norm on every K/V-owning layer, K=V or not (S1.0)
	pSA, pSABias, pSAResid   Pipeline // Stage A gemv (K bounded by the M-11 threadgroup-memory guard, not a fixed constant)
	// The W4F16 decode lane: f16 activations, no int8 quantization, selected by GOINFER_METAL_DECODE_LANE=w4f16
	// (decodeLaneW4F16). The pipelines are always built; only dispatch is conditional (canUseF16Lane). It covers
	// plain dense QKV, o-proj and gate-up only: down-proj and every special case (MoE, paged, sandwich, postOnly,
	// parallelBlock, qGate, outBias, DeltaNet, nonGatedMLP) stay on the shipped W4A8 kernels.
	pRmsF16                         Pipeline // rmsnorm_f16_act: f32 residual in, half activation out, no quant
	pF32ToF16                       Pipeline // bare convert, for o-proj's input (no norm/weight there)
	pSAf16, pSAf16Bias, pSAf16Resid Pipeline
	decodeLaneW4F16                 bool
	axF16, mxF16, cxF16             Buffer // half-typed activation buffers for the f16 lane (QKV-in, gate/up-in, o-proj-in)
	// The q4k lane (docs/tasks/task-metal-q4k-2026-10.md): a model loaded at --quant q4k runs every projection on f32
	// activations (no activation quantization), over Q4_K super-blocks or int8 rows. axF32 is a projection's input (the
	// normed residual), swF32 the down-projection's (SwiGLU's output).
	q4kLane                                                    bool
	mropeAxis                                                  Buffer // S16: the m-RoPE prefill\'s per-pair axis table; empty when the model has none
	pQ4K, pQ4KResid, pQ4KBias, pW8F32, pW8F32Resid, pW8F32Bias Pipeline
	pRmsF32Out, pSwF32                                         Pipeline
	axF32, swF32                                               Buffer
	pArgFinish                                                 Pipeline // fused block-argmax lm head reduce
	pArgRowsPart                                               Pipeline // E-P08: a batched step row's partial argmaxes (argmax_rows_part)
	// The split-KV decode-attention lane, gridded by (kvHead, split) instead of by query head: on by default
	// (decodeAttnFA, from metalAttnFAEnabled; GOINFER_METAL_ATTN_FA=0 opts out). The pipelines are always built;
	// dispatch is conditional (canUseAttnFA). It covers hd==128 (or hd==64 through the block twin, attnFAHeadDimOK)
	// on dense GQA only: no window, sinks or f32 KV. Why, and why S must be sized for real occupancy (S=1 is
	// uniformly worse than the shipped kernel at every depth), is in attention_fa's comment in kernels.go and
	// docs/code-notes/metal.md#resident.pAttnFA.
	pAttnFA, pAttnFACombine Pipeline
	// E-P05: pAttnFA's and pAttnFACombine's multi-row forms for the batched step (batch_rows.go), and the length of one
	// row's partial region, attnFAPartial's.
	pAttnFARows, pAttnFACombineRows Pipeline
	// int8 slice 3b: a native int8 resident's batched-step projection, B rows per weight read (batch_rows.go)
	pW8Rows          [batchMaxSeqs + 1]Pipeline // by B
	attnFAPartialLen int
	decodeAttnFA     bool
	attnFAPartial    Buffer // [nKV][maxSplit][G][hd+2] f32 scratch, sized once for the widest layer
	attnFAMaxSplit   int
	// attnFASplitOverride, when > 0, replaces attnFASplitFor's split-count rule (the nKeys/32 and attnFAMaxSplit
	// caps still apply). Zero in production; set only by tests that sweep S. Read inside attnFASplitFor, the one
	// function both the dispatch grid and setPos's uAttnFANSplit use, and both for the same key count (the grid's
	// comes from planNKeys), so the two cannot disagree.
	attnFASplitOverride int
	// moeCap is D-G01's routing capture (metal/dg01_moe_gate_test.go). ZERO in production. With idx and wgt set, decode's
	// non-paged MoE FFN copies each layer's selected experts and weights into slot l (k words each), and the batched
	// prefill's row-by-row MoE (GOINFER_MOE_EXPERT_MAJOR=0) copies row m's into slot l*rows+m; the expert-major path,
	// which reads its routing back to the host anyway, hands it to major. Encoded at encode time, so a test sets it
	// before the first forward it captures.
	moeCap struct {
		idx, wgt Buffer
		rows     int
		major    func(l int, idx []uint32, wgt []float32)
	}
	// attnFAFloorOverride, when > 0, replaces attnFADepthFloor as the key count at which attention_fa takes over.
	// Zero in production; set only by tests (legacy-against-blk arms below the floor). Read through attnFAFloor by
	// attnPlanFor, canUseAttnFA and canUseAttnFAAt, so the single-token step and the batched step plan alike.
	attnFAFloorOverride int
	// promptStepOff, when true, keeps PrefillLast off the step route (E-P01, prefillByStep). FALSE in production — set
	// only by tests that measure the batched pass or the sequential decline on a resident that could take the route.
	promptStepOff bool
	// gemvRows (R18, docs/measurements/metal-decode-gemv-r18-2026-09-26.md): rows per simdgroup for the dense decode
	// layer's four int4 GEMVs — qkv, o, fused gate/up and down — set by buildResident through gemvRowsFor. When a field
	// is > 0 that GEMV dispatches its rows kernel (pSABiasRows / pSAResidRows / pSARows / pGemvResidStaged, grid
	// rows·32/R, tg 256; down stages its activations in K bytes of threadgroup memory); 0 keeps the shipped kernel.
	// The rows kernels are bit-identical to the shipped ones. Tests zero it to time or compare against the shipped
	// kernels (then flush the executor, stopExec).
	gemvRows                                             struct{ qkv, o, gu, down int }
	pSABiasRows, pSAResidRows, pSARows, pGemvResidStaged Pipeline
	// E-P02: the batched step's per-row qkv and gate|up GEMVs as one dispatch with each tile's B threadgroups adjacent
	// (mc3_gemv_w4a8_sa_{bias_}rows_adj<R>, the twin of pSABiasRows / pSARows), set with them; batch.go mc3AdjRowsOn.
	pSABiasRowsAdj, pSARowsAdj Pipeline
	// D-B04: R18's rows form at the int4 GEMV sites R18 never reached (gemvExt): the staged coal kernel, the SA rows
	// kernel and the staged residual kernel at R = 2 and 4, and the threadgroup memory a staged kernel may take.
	gemvExtCoal, gemvExtSA, gemvExtResid [5]Pipeline
	gemvExtTGMax                         int
	gemvExtRows                          [3]int // rows-form dispatches per kind since the build (tests read it)
	// attnFABlkSplit > 0: pAttnFA is the R17 block kernel (attention_fa_blk_g<G>), which runs at this fixed split
	// count instead of attention_fa's core-count rule — see attnFABlkSplit's const and the selection in
	// buildResident.
	attnFABlkSplit int
	// attnFALayer is the first layer attention_fa can run (the one attnFANKV was taken from); its head dim picks the
	// block kernel's hd = 128 form or B-P01's hd = 64 twin.
	attnFALayer int
	curNKeys    int // CPU-side twin of uNKeys' value, set by setPos
	// encNKeys, when > 0, is the key count of the command buffer being ENCODED: the pipelined executor sets it
	// around each encode, because it encodes before the job's setPos (and ahead, for the predicted next job).
	// 0 = the buffer runs at the position setPos last set — every synchronous path sets the position, then
	// encodes. Read through planNKeys by the two encode-time attention decisions (canUseAttnFA's depth gate,
	// attention_fa's split grid). See execLoop and attnPlan.
	encNKeys int
	// encFANSplit, when set, is the attention_fa split uniform the command buffer being encoded binds instead of
	// uAttnFANSplit: the greedy chain's per-token set (greedy_chain.go, C-B01). Zero outside its encode.
	encFANSplit Buffer
	// The greedy chain (greedy_chain.go, C-B01): its gather pipeline, its goroutine's channels, and its two alternating
	// uniform sets and token-id buffers.
	pEmbedGather Pipeline
	lmTied       bool // lmW/lmS are the embedding table itself (a tied LM head): the chain gathers from them, no copy
	chainReq     chan chainReq
	chainResp    chan chainResp
	chainDone    chan struct{}
	chainSets    [2]posUniforms
	chainTok     [2]Buffer
	chainServed  int // tokens chainNext has returned since the build (tests read it)
	// The untied head's gather table (C-B01): the int8 embedding table on the device, made the first time a chain is
	// asked for (chainEmbedTable), aliased from the .giw mapping where it can be. A tied head's gather reads lmW/lmS.
	// chainEmbBase is the memory guard's price of this build (weights, host copy, every KV slot), which the table is
	// priced on top of; chainEmbGuardOff is GOINFER_NO_RESIDENT_MEM_GUARD. chainGW/chainGS: the open chain's table.
	chainEmbMu           sync.Mutex
	chainEmbTried        bool
	chainEmbW, chainEmbS Buffer
	chainEmbWhy          string
	chainEmbBase         int64
	chainEmbGuardOff     bool
	chainGW, chainGS     Buffer
	// The sampled chain's own gumbel uniforms (C-P02): 1/temperature and the seed for the chain, the draw per uniform set,
	// and a logits row per set (the -1 recovery takes the argmax of a row the next buffer has not overwritten).
	chainInvT, chainK0, chainK1   Buffer
	chainD0, chainD1, chainLogits [2]Buffer
	chainNegInf                   Buffer // chainNegInfForTest's row; never made in production
	attnFANKV                     int    // cached at BuildResident: the (uniform, dense-GQA-only) nKV attention_fa-eligible layers share
	uAttnFAG, uAttnFANSplit       Buffer // shared scratch uniforms — SetU32'd ONLY from setPos (see setPos's own comment), never from the
	// per-layer dispatch site: the pipelined executor encodes t+1 while t runs on the GPU (execLoop), so a per-layer
	// CPU write would race the running kernel's read of the same buffer, which Metal's automatic hazard tracking
	// does not cover. TestAttentionFA_pipelinedEncodeRace pins it.
	pQKNorm              Pipeline // per-head QK-RMSNorm (Qwen3)
	pRmsF32              Pipeline // Gemma sandwich: in-place RMSNorm of a sublayer output
	pGemvW8, pGemvW8Amax Pipeline // int8 GEMV + fused block-argmax — the logit-critical LM head (see lmW)
	pCopyVec             Pipeline // copy_f32 for on-device embedding copy in batched forward
	pCopyU32             Pipeline // copy_u32: D-G01's routing capture (moeCap), tests only
	qkNorm               bool     // arch has QK-norm
	qkNormWhole          bool     // G5: QK-norm reduces over the WHOLE q/k vector, not per head (Olmo 3/Olmo Hybrid) — qkNorm must also be true
	sandwich             bool     // Gemma NormSandwich4: norm each sublayer output before the residual add
	postOnly             bool     // G5: NO pre-norm at all (Olmo 3/Olmo Hybrid) — quantize the raw residual; PostAttnNorm/PostMLPNorm still dispatch, same as sandwich's post half
	kvF32                bool     // f32 KV cache (Gemma sandwich path) — f16 rounding craters Gemma's sensitive contexts
	uAct                 Buffer   // gated-MLP activation ordinal (decoder.ActKind: 0=GELU-tanh, 1=SiLU)
	uAddOne, uWindow     Buffer   // qk_norm + sliding-window uniforms
	uZero                Buffer   // constant 0 (nH=0 / nHhd=0 / addOne=0 for the scale-less v_norm dispatch)
	vNormUnit            Buffer   // [maxHd] of 1.0 — unit weight so qk_norm (x·rms·w, addOne=0) = scale-less v_norm for K=V layers

	// GPT-2 (FeatLayerNorm/FeatNonGatedMLP/FeatLearnedPos/FeatOutBias).
	pLayerNorm, pActQuant            Pipeline          // layernorm_quant, act_quant
	pSABiasResid, pCoalBiasResid     Pipeline          // gemv_w4a8_sa_bias_resid (o-proj), gemv_w4a8_resid_bias (down-proj — always routed to the coal family, never the SA family)
	layerNorm, layerNormBias         bool              // arch.Norm==NormLayer; whether it carries a bias (GPT-2 yes, Cohere no)
	nonGatedMLP, outBias, learnedPos bool              // arch.NonGatedMLP / arch.OutBias / arch.LearnedPosEmbed
	posEmbed                         *linalg.WeightMat // [MaxPositions, H] learned position embedding table (learnedPos only)
	uLNHasBias                       Buffer            // layernorm_quant's hasBias uniform (r.layerNormBias as 0/1)

	// G5 (docs/tasks/task-gpu-paths-2026-09.md), the last row: Cohere/Command-R + Cohere2/Command-R7B.
	pairwiseRoPE  bool    // FeatPairwiseRoPE: pRope/pRope2 and the prefill's rope pipelines are the GPT-J pairwise twins (rope_pw, ...)
	parallelBlock bool    // FeatParallelBlock: ONE shared input norm feeds attn AND MLP independently (x_final = x_orig + attn_out + mlp_out) — encodeLayer reuses encodeAttention's r.aq/r.aSc instead of re-normalizing r.x; no post-attn/post-MLP norm exists for this family
	logitScale    float32 // host-side final-logit multiplier (1/arch.LogitScale), applied in finalizeLogits; 0 ⇒ none (FeatLogitScale)

	// R7b Mac half (docs/tasks/red-october.md): device-side temperature-only sampling by
	// Gumbel-max, decoder.ResidentSample (gumbel_sample.go). Always built (like attnFAPartial
	// above): SampleAvailable is a per-call gate (finalSoftcap/logitScale), not a build-time one.
	pGumbel1, pGumbel2 Pipeline
	gumbelNB           int    // stage-1 dispatch size: ceil(V / (4*GB_THREADS))
	gumbelBKey         Buffer // [gumbelNB] per-threadgroup winning keys
	gumbelBIdx         Buffer // [gumbelNB] per-threadgroup winning indices (int32)
	gumbelOut          Buffer // [1] the drawn id (int32), -1 if nothing was comparable
	// uGumbelV/uGumbelNB are immutable (V and gumbelNB never change for a built resident); the
	// other four are reused scratch, SetU32'd/Floats-written fresh before each draw — safe because
	// ForwardSample's command buffer is synchronous (Begin/End), the same argument uAttnFAG/
	// uAttnFANSplit's field comment makes for the per-layer attention_fa uniforms above.
	uGumbelV, uGumbelNB                        Buffer
	uGumbelInvT                                Buffer
	uGumbelK0, uGumbelK1, uGumbelD0, uGumbelD1 Buffer

	// gpt-oss (FeatAttnSink, DECLARED for metal — kernels wired end-to-end; TestGptOssResidentParity).
	attnSink                 bool    // arch.gptoss != nil
	gptossAlpha, gptossLimit float64 // clamped-SwiGLU constants (0 for every other family)

	// Compute-time LoRA (lora.go). pLoraDelta is always created in BuildResident and gated on the per-model state,
	// like every other optional pipeline here. It is one fused down+up dispatch: t[R] lives in the kernel's own
	// threadgroup memory. loraLayers is nil until SetAdapter binds one; the dispatch sites (encodeAttention,
	// encodeLayer) skip a projection when it is nil or that projection's delta is nil.
	pLoraDelta Pipeline
	loraLayers []residLoRALayer

	// loraCacheSrc and loraCached are the single-adapter device cache: the source layers a bind's device buffers
	// were built from, and those buffers, kept alive across SetAdapter(nil) so a rebind of the same adapter (one
	// chat session, many turns) skips the re-upload. SetAdapter's comment has the identity check.
	loraCacheSrc []decoder.ResidentAdapterLayer
	loraCached   []residLoRALayer

	qkv, gu Buffer // fused QKV out, fused gate/up out

	// Model-level (constant across a family's layers). Per-layer attention geometry —
	// hd/nKV/kvDim/half and their uniform buffers — lives on residLayer.geom (see geom.go);
	// it was REMOVED from here so a launch site cannot bind the uniform shape by mistake.
	H, nL, nH, I, V int
	// ctxCap is the resolved resident KV capacity in positions (resolveMetalCtxCap), set once in buildResident.
	// Every r.kc[l]/r.vc[l] is sized ctxCap*kvDim; checkCap (metal/backend.go) guards writes against it. Always <=
	// metalCtxCapMax (the kernel's hard ceiling); smaller when an explicit -ctx asked for less.
	ctxCap       int
	finalSoftcap float32 // Gemma final-logit softcap (30); 0 ⇒ none. Applied host-side in finalizeLogits (FeatFinalLogitSoftcap).
	// embedScale is Gemma's sqrt(hidden) token-embedding multiplier (FeatEmbedScale); 0 or 1 means none. The
	// id-taking entry points (Forward, ForwardArgmax) apply it right after the embedding lookup; ForwardEmb does
	// not, because its caller has already scaled (decoder.embedResident).
	embedScale float32
	embed      *linalg.WeightMat

	// attnTempBeta/attnTempOrigMaxPos (Ministral 3, FeatAttnTemp, G5 docs/tasks/task-gpu-paths-2026-09.md):
	// the raw params behind the post-RoPE query scale (Model.AttnTempParams) — 0 for every family
	// without it. uQTempScale (below) is the per-forward-call scale computed from these plus the
	// CURRENT position, mutated in place at every uPos.SetU32 call site the same way uPos itself is.
	attnTempBeta, attnTempOrigMaxPos float64

	layers                   []residLayer
	finalNorm, finalNormBias Buffer // finalNormBias: GPT-2 ln_f's LayerNorm bias (unused for RMS families)
	lmW, lmS                 Buffer
	kc, vc                   []Buffer
	ks, vs                   []Buffer // int8 KV scales (paddedCtxCap * nKV floats)
	// kvSlotBufs: MC1's resident KV slots (docs/tasks/task-concurrency-2026-09.md), each a full copy of the per-layer
	// kc/vc (and int8 ks/vs) buffers; kc/vc/ks/vs above are the BOUND slot's, and useKVSlot rebinds them. nil with one
	// slot. kvSlot is the bound index.
	kvSlotBufs []kvSlotBuf
	kvSlot     int
	// kvContig: the slots of each layer are one allocation (kc/vc are views into it at kvSlot*kvSlotBytes[l]) — MC3
	// S3. Host code indexing a KV buffer's contents must add kvHostOff: aikit's Floats/U16s views ignore a view's
	// offset and start at the allocation.
	kvContig    bool
	kvSlotBytes []int
	// batch: MC3's batched decode step over those slots (batch.go); nil when this resident cannot batch. preciseMath
	// is the build's fast-math choice, which the batched kernels must share to round like the kernels they reproduce.
	batch       *batchState
	preciseMath bool
	// MC3 S3's multi-row forms of the per-row kernels (batch_rows.go), for the batched step
	pRmsRows, pQvRows, pSwRows, pRope2Rows, pKvRows, pAttnRows Pipeline
	pQKNormRows                                                Pipeline           // E-P07: qk_norm over the step's rows (batch_rows.go)
	pMoeBatch                                                  [5]Pipeline        // D-P01: moe_batch_gemv at R = 2, 4 (prefill_g4batch.go)
	pMoeCombine                                                Pipeline           // D-P01: moe_batch_combine
	kvI8                                                       bool               // m.KVCacheI8() (CLI flag --kv i8)
	pKvI8, pAttnI8                                             Pipeline           // int8 KV store and attention pipelines
	moe                                                        *moeResident       // non-nil ⇒ MoE model (router + stacked experts); see moe.go
	g4moe                                                      *gemma4MoeResident // non-nil ⇒ Gemma-4 enable_moe_block (parallel dense‖MoE); see gemma4_moe.go
	pagedNoHead                                                bool               // a paged forward skips the final norm and LM head (ForwardEmbNoLogitsPipe); exec-thread only
	g4LayerMajorRuns                                           int                // prompts prefillG4Paged ran (4b); a test reads it to see the route was taken
	g4BatchGroups, g4BatchMaxCnt                               int                // D-P01: slot groups the batched phase 2 ran, and the most pairs one entry shared
	moeLayerMajorRuns                                          int                // prompts prefillMoEPaged ran; a test reads it to see the route was taken

	// prefillOK reports whether the f16 MMA prefill kernels (prefill.go) actually implement
	// this model's shape. They run a DENSE FFN out of L.guW/L.dW with a model-level rope +
	// window and a SiLU-only swiglu — so a model whose decode path diverges (MoE leaves the
	// dense FFN buffers unset entirely) MUST decline prefill and let the caller fall back to
	// the sequential Forward loop: correct, just a slower TTFT.
	prefillOK bool
	// prefillResidNonFinite reports that the last batched pass ended with a non-finite value in its last residual row: the
	// f16 residual overflowed (Gemma 3 4B's outgrows 65,504 by layer 6, docs/tasks/task-multimodal-support-2026-10.md S17).
	// batchedPrefillImg declines on it. poisonPrefillResidForTest writes an inf there first, for the test of that decline.
	prefillResidNonFinite, poisonPrefillResidForTest bool

	// w8: every dense body projection runs as W8A8 on its int8 weights instead of being re-quantized to int4
	// (docs/tasks/task-metal-int8-2026-10.md, slice 1). Set by buildResident (w8Eligible). The GEMV pipelines are then
	// the int8 twins, and the paths that only read int4 are off: the R18 rows kernels, the f16 lane, fast prefill and
	// the MC3 batched step.
	w8 bool
	// w8Attn: int4mix run natively (slice 4): the attention projections as W8A8 on their int8 weights, the FFN on its
	// int4 kernels. Set by buildResident (w8AttnEligible); never with w8.
	w8Attn bool

	x, aq, aSc, ctx, cq, cSc, oO, mq, mSc, dq, dSc, dO, logits Buffer
	invf, uH, uI, uNH, uScale, uEps                            Buffer // invf = model-level rope (prefill only); geometry uniforms live on residLayer.geom
	uPos, uNKeys, uRopePos, uQTempScale                        Buffer
	uQKWholeOne, uQKWholeHD                                    Buffer // G5 QKNormWhole: constant-1 nH/nKV replacement + the full nH*hd width
	part, tok, uP                                              Buffer // fused-argmax: tile partials, token out, tile count
	logitsHost                                                 []float32
	gpuStart, gpuEnd, kernStart, kernEnd                       float64      // last-Forward GPU timing (Step 0)
	prof                                                       pagedProfile // per-phase paging decomposition (accumulates; snapshot+diff over a timed window)
	alias                                                      *weightAlias // S6: set on a .giw-mapped model unless GOINFER_METAL_ALIAS=0; test/banner introspection
	residency                                                  ResidencySet // pinned working set (GOINFER_MOE_RESIDENCY, paged path); zero value if unused
	residencyBufs                                              []Buffer     // exactly the buffers added to `residency` (for the teardown-consistency gate)

	// Batched ForwardN buffers: pre-allocated up to batchCap tokens
	batchCap                                 int
	batchX                                   Buffer
	batchLogits                              Buffer
	batchUPos, batchUNKeys, batchUQTempScale []Buffer

	// The pipelined logits executor (encode-ahead): a persistent OS-thread-pinned goroutine that commits token t,
	// pre-encodes t+1 while the GPU runs t, then waits, hiding the host encode bubble. It starts lazily on the first
	// ForwardEmbPipe while execReq is nil. SetAdapter tears it down (stopExec) because its pre-encoded t+1 buffer
	// bakes in r.loraLayers at encode time; a bind, switch or clear between two ForwardEmbPipe calls would otherwise
	// commit that stale buffer under the new adapter state. ForwardEmbPipe and SetAdapter are never concurrent, so
	// the plain nil check needs no lock.
	execReq  chan execJob
	execAck  chan []float32
	execDone chan struct{} // closed when execLoop returns — Close must WAIT on this before freeing

	// execErr latches the first aborted-command-buffer error any forward path sees. waitUntilCompleted returns
	// cleanly even on a GPU fault, so every completion site records enc.Err() here and the adapter consumes it
	// (takeExecErr) after each forward, surfacing an error instead of the stale logits or token the aborted buffer
	// left behind. Plain field: the pipelined executor is single-goroutine and writes it before the execAck send
	// (which happens-before the adapter's read); the synchronous paths run on the caller's locked thread.
	execErr error

	pf *prefillState // lazily-compiled f16 MMA prefill pipelines (opt-in)
	// pfErr latches a compile or pipeline-creation failure from ensurePrefill, so r.pf staying nil does not make
	// every later PrefillLast retry the full MSL compile and panic again.
	pfErr error

	// Gated-DeltaNet mixer (deltanet_kernels.go — own module, nothing else here is recurrent).
	// dnet nil ⇒ dense model; every field below is loaded/allocated only when it is non-nil.
	dnet                                          *dnetParams
	pDnConv, pDnGates, pDnNorm, pDnRule, pDnGNorm Pipeline
	pDnQSplit, pDnAttnGate                        Pipeline // this family's fused double-width q_proj + output gate
	// D-B01: the mixer over M prompt rows (deltanet_kernels.go's _seq/_rows forms; prefill_deltanet.go)
	pDnConvSeq, pDnGatesRows, pDnNormRows, pDnRuleSeq, pDnGNormRows, pDnProjW8Rows Pipeline
	dnMixed, dnConvOut, dnQn, dnKn, dnHeadP                                        Buffer // per-token scratch, sized from dnet (model-level, uniform across layers)
	dnBt, dnAt, dnZOut, dnCore, dnGated                                            Buffer
	dnGq, dnGSc                                                                    Buffer // int8 activation + scale for the gated output's out_proj GEMV
	dnQg, dnAGate                                                                  Buffer // qGate scratch: [2*maxNHhd] fused [query‖gate] q_proj output, [maxNHhd] the split gate
	uDnConvDim, uDnK, uDnNv, uDnNk, uDnHk, uDnHv                                   Buffer // DeltaNet geometry uniforms (model-level, uniform across layers)
	uDnKeyDim, uDnQScale, uDnRep, uDnVBase                                         Buffer
	uDnValueDim                                                                    Buffer // dnet.valueDim — the out_proj GEMV's K (input width)
}

// recordExecErr latches the first command-buffer abort a forward path observes (audit C-09).
func (r *resident) recordExecErr(err error) {
	if err != nil && r.execErr == nil {
		r.execErr = err
	}
}

// takeExecErr returns and clears the latched command-buffer error (audit C-09). The adapter calls
// it after each forward and returns the error in place of the (stale) logits.
func (r *resident) takeExecErr() error {
	err := r.execErr
	r.execErr = nil
	return err
}

type execJob struct {
	emb     []float32
	pos     int
	ropePos int
	noHead  bool
}

func byteBuf(d *Device, n int) Buffer {
	return d.NewBufferBytes(n)
}

// q4kSeg is one run of a q4k-lane projection's output rows that share a weight kind: Q4_K super-blocks (s unused) or
// int8 rows with per-row scales s, written at row off of the projection's output (uN holds rows).
type q4kSeg struct {
	q4k       bool
	w, s, uN  Buffer
	rows, off int
}

// q4kLaneProj uploads a projection, or a fused group with its parts' rows in order, as runs of consecutive parts of one
// kind: a GGUF q4_k_m file mixes them inside a group (Qwen2.5's attn_v is Q6_K, so int8, in half the layers, beside
// Q4_K attn_q/attn_k).
func q4kLaneProj(d *Device, wms ...*linalg.WeightMat) []q4kSeg {
	var segs []q4kSeg
	off := 0
	for i := 0; i < len(wms); {
		j, q4k := i+1, wms[i].Kind() == "q4k"
		for j < len(wms) && (wms[j].Kind() == "q4k") == q4k {
			j++
		}
		rows := 0
		for _, w := range wms[i:j] {
			rows += w.Rows()
		}
		w, s := q4kLaneBuf(d, wms[i:j]...)
		segs = append(segs, q4kSeg{q4k: q4k, w: w, s: s, uN: NewBufferU32(d, uint32(rows)), rows: rows, off: off})
		off += rows
		i = j
	}
	return segs
}

// q4kLaneBuf uploads parts of one kind for the q4k lane (docs/tasks/task-metal-q4k-2026-10.md): GGUF Q4_K super-blocks
// verbatim when every part is Q4_K (the scale buffer unused), else int8 rows with their per-row scales. Any other
// kind, or a mix, panics; buildResident recovers that into a decline.
func q4kLaneBuf(d *Device, wms ...*linalg.WeightMat) (Buffer, Buffer) {
	if _, ok := wms[0].Q4K(); ok {
		var all []byte
		for _, w := range wms {
			raw, ok := w.Q4K()
			if !ok {
				panic(fmt.Sprintf("metal: q4k lane: a fused group mixes Q4_K with %q", w.Kind()))
			}
			all = append(all, raw...)
		}
		return NewBufferInt8(d, unsafe.Slice((*int8)(unsafe.Pointer(&all[0])), len(all))), Buffer{}
	}
	var q []int8
	var sc []float32
	for _, w := range wms {
		q8, s8, _, ok := w.Int8()
		if !ok {
			panic(fmt.Sprintf("metal: q4k lane: weight kind %q is neither q4k nor int8", w.Kind()))
		}
		q, sc = append(q, q8...), append(sc, s8...)
	}
	return NewBufferInt8(d, q), NewBufferFloats(d, sc)
}

func int8Buf(d *Device, w *linalg.WeightMat) (Buffer, Buffer, error) {
	q8, sc, _, ok := w.Int8()
	if !ok {
		return Buffer{}, Buffer{}, fmt.Errorf("metal: weight kind %q is not int8 (load Options{Quant:\"int8int8\"})", w.Kind())
	}
	return NewBufferInt8(d, q8), NewBufferFloats(d, sc), nil
}

// int4DirectWords converts a decoder int4 WeightMat's packed nibbles and f32 group scales straight into Metal's W4A8
// buffers (uint32 words plus f16 scales), with no int8 intermediate. aikit's group=32 packing (nib = q+8, byte k/2
// low/high) and Metal's packW4A8Row (element k to word k/8, bit 4*(k%8)) are the same bytes on little-endian, so the
// nibbles copy verbatim; only the group scales narrow f32 to f16. Returns ok=false if the weight is not group-32 int4.
// Consuming the int4 directly avoids the double quantization (f32 to int8 to int4) of BuildResident's default path,
// which Gemma's low-magnitude attention contexts amplify into a large context error (metal/gemma_sublayer_test.go).
func int4DirectWords(w *linalg.WeightMat) (words []uint32, scales []uint16, ok bool) {
	q4, q4s, group, ok := decoder.Int4F32(w)
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

// bytesToU32 reinterprets a little-endian byte slice as uint32 words (len must be a multiple of 4) with one bulk copy:
// b already holds the target word bytes, and w is freshly allocated, so it is 4-aligned, unlike the mmap-backed
// sources int4DirectBytesOnly exists to avoid reinterpreting.
func bytesToU32(b []byte) []uint32 {
	w := make([]uint32, len(b)/4)
	if len(w) == 0 {
		return w
	}
	copy(unsafe.Slice((*byte)(unsafe.Pointer(&w[0])), len(w)*4), b)
	return w
}

// int4DirectBytesOnly returns a canonical group-32 int4 WeightMat's packed nibble bytes aliased straight from the
// mmap: no bytesToU32 reconstruction and no per-stage []uint32 allocation. They are byte for byte the words
// int4DirectWords builds (little-endian), so a byte copy into a uint32 slot buffer reproduces them exactly
// (copyBytesToU32Buf). The paged MoE stage functions (buildMoELayer, buildGemma4MoELayer) pair it with the WeightMat's
// own Int4ScalesF16, rather than re-deriving f16 scales from an f32 copy on every page-in or caching a duplicate of
// what the mapping already holds.
func int4DirectBytesOnly(w *linalg.WeightMat) (q4 []byte, ok bool) {
	b, _, group, ok := w.Int4F16()
	if !ok || group != 32 {
		return nil, false
	}
	return b, true
}

// parallelF32ToF16 converts src (f32 group scales) to dst (f16 bits) across up to 8 workers. Every element is
// independent and f32ToF16 is deterministic, so the split is byte-identical to the serial loop. It exists for the
// expert-paging stage path, where the conversion runs once per expert per stage and is arithmetic-dominated; small
// inputs (the one-time non-paged build) stay serial, where goroutine spawn would not pay.
func parallelF32ToF16(dst []uint16, src []float32) {
	n := len(src)
	workers := min(runtime.GOMAXPROCS(0), 8)
	if n < 8192 || workers <= 1 {
		for i, s := range src {
			dst[i] = f32ToF16(s)
		}
		return
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for i := lo; i < hi; i++ {
				dst[i] = f32ToF16(src[i])
			}
		}(lo, hi)
	}
	wg.Wait()
}

// maxThreadgroupStageBytes returns the largest threadgroup staging allocation (bytes) any resident dispatch will
// request. The SA-GEMV and MoE kernels stage the contraction row into threadgroup memory at 2 bytes per element
// (DispatchTG tgBytes = 2*K); the widest staged K is hidden (qkv, gate-up), qWidth = nH*hd (o-proj), the MoE expert
// intermediate moeInter or g4moeInter (the expert down-proj stages inter), or dnValueDim (DeltaNet's out-projection,
// which deltanet.go dispatches with dp.valueDim*2 bytes). The dense down-proj uses the non-staging pGemv, so the dense
// intermediate is deliberately not counted. A separate function so the budget arithmetic is unit-testable.
func maxThreadgroupStageBytes(hidden, qWidth, moeInter, g4moeInter, dnValueDim int) int {
	return 2 * max(max(max(hidden, qWidth), max(moeInter, g4moeInter)), dnValueDim)
}

// int4Buf uploads a WeightMat as W4A8 (int4, group=32) plus f16 group scales. An int4 weight (a Quant:"int4" load) is
// consumed directly, with no int8 step; an int8 weight is re-quantized through the validated packer. One-time at build.
func int4Buf(d *Device, w *linalg.WeightMat) (Buffer, Buffer, error) { return int4BufA(d, nil, w) }

// int4BufA is int4Buf with an optional weightAlias (S6): when a is non-nil and w's nibbles live in the
// .giw mapping, the nibbles are bound in place instead of copied; the f16 scales are built exactly as the
// copy path builds them.
func int4BufA(d *Device, a *weightAlias, w *linalg.WeightMat) (Buffer, Buffer, error) {
	// The W4A8 layout and every GEMV kernel assume K is a multiple of the group (32): rows pack K/8 words and K/32
	// scales with no partial-group handling. A weight with K%32 != 0 would pack a truncated last group with a row
	// stride the kernel disagrees with, silently wrong or a panic at K<32. Decline, so BuildResident falls back to
	// CPU.
	if k := w.Cols(); k%32 != 0 {
		return Buffer{}, Buffer{}, fmt.Errorf("metal: W4A8 pack needs K%%32==0 (group=32), got K=%d — declining to CPU (audit M-10)", k)
	}
	if nib, ok := a.nibbles(d, w); ok {
		if sc, ok := a.scales16(d, []*linalg.WeightMat{w}); ok { // v14 metal-target file: f16 scales in place too
			return nib, sc, nil
		}
		_, q4s, _, _ := decoder.Int4F32(w)
		scales := make([]uint16, len(q4s))
		for i, s := range q4s {
			scales[i] = f32ToF16(s) // the same conversion int4DirectWords applies
		}
		a.addCopy(int64(2 * len(scales)))
		if a != nil {
			a.f16Converted++
		}
		return nib, NewBufferU16s(d, scales), nil
	}
	if words, scales, ok := int4DirectWords(w); ok {
		a.addCopy(int64(4*len(words) + 2*len(scales)))
		return NewBufferUint32s(d, words), NewBufferU16s(d, scales), nil
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
	return NewBufferUint32s(d, words), NewBufferU16s(d, scales), nil
}

// int4Concat re-quantizes and row-concatenates several same-K WeightMats into ONE W4A8
// buffer — the fusion enabler (combined QKV → one GEMV, combined gate/up → one GEMV).
func int4Concat(d *Device, wms ...*linalg.WeightMat) (Buffer, Buffer) {
	// Pre-size from the final shape (N*K/8 words and N*K/32 f16 scales per group-32 tensor, the formula int4BufA's
	// int8 branch allocates by) instead of growing two nil slices by append.
	var totalWords, totalScales int
	for _, w := range wms {
		n, k := w.Rows(), w.Cols()
		totalWords += n * k / 8
		totalScales += n * k / 32
	}
	words := make([]uint32, 0, totalWords)
	scales := make([]uint16, 0, totalScales) // f16 group scales (L1)
	for _, w := range wms {
		// K%32==0 is a hard W4A8 invariant (group=32); see int4BufA. int4Concat has no error return and runs only on the
		// build path, so it panics; buildResident's recover turns that into a clean CPU decline, as with its wrong-kind
		// panic below.
		if k := w.Cols(); k%32 != 0 {
			panic(fmt.Sprintf("metal: W4A8 concat needs K%%32==0 (group=32), got K=%d (audit M-10)", k))
		}
		if dw, ds, ok := int4DirectWords(w); ok { // int4-direct: consume nibbles verbatim
			words = append(words, dw...)
			scales = append(scales, ds...)
			continue
		}
		q8, sc, _, ok := w.Int8()
		if !ok {
			// int4DirectWords already declined (the weight has no canonical int4 bytes) and this is not int8 either. The
			// common cause is the repacked-only int4 policy (wantsCanonicalInt4): a tensor built with Backend "cpu", or
			// loaded generically and handed to BuildResident out of band, may have no canonical bytes left. Name that
			// condition specifically; a tensor that is neither int8 nor int4 at all (a programming error) gets the generic
			// message.
			if w.IsInt4() {
				panic(fmt.Sprintf("metal: int4 tensor has no canonical bytes (layout %s-only): "+
					"model was loaded for a CPU-only backend; load with Options.Backend set to "+
					"\"metal\" to keep canonical bytes for residency", w.Int4Layout()))
			}
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
	return NewBufferUint32s(d, words), NewBufferU16s(d, scales)
}

// nativeInt8 turns the native int8 path on (w8Eligible). Tests turn it off for the int4 re-quant arm; it is not an
// option or an environment variable. Grades: docs/tasks/task-metal-int8-2026-10.md.
var nativeInt8 = true

// nativeInt4Mix turns on int4mix's native path (w8AttnEligible: attention W8A8, FFN int4). Tests turn it off for the
// re-quant arm. Grades: docs/tasks/task-metal-int8-2026-10.md, "Slice 4: int4mix".
var nativeInt4Mix = true

// nativeInt8MoE admits a generic resident MoE to the native int8 path (w8Eligible; int8 expert GEMVs gemv_w8a8_moe*);
// tests turn it off. Grades: docs/tasks/task-metal-int8-2026-10.md, "Slice 4: MoE int8".
var nativeInt8MoE = true

// w8FastMath keeps fast math for a native int8 model (w8PreciseMath off): test-only, gate S's fast-math arm, which
// prices the owner's precise-math decision.
var w8FastMath = false

// w8Eligible reports whether m runs on the native int8 path (r.w8): every dense body projection is int8-kind
// (int8int8, or weight-only int8, which CUDA and WebGPU also run against int8 activations) with K a multiple of 4,
// and the model is one slice 1 of docs/tasks/task-metal-int8-2026-10.md covers: no MoE, no Gemma-4 MoE, no DeltaNet
// (whose families also carry the gated-softmax layers). Anything else keeps the int4 re-quantization. A K=V layer
// has no v_proj and fails the check, so it is never fused as int8.
func w8Eligible(m *decoder.Model, r *resident) bool {
	if !nativeInt8 || r.g4moe != nil || r.dnet != nil {
		return false
	}
	// slice 4: a generic resident MoE (Qwen-style experts, not gpt-oss's biased ones, not paged: the pool stages int4)
	if r.moe != nil && (!nativeInt8MoE || r.moe.isGptOss || r.moe.paged) {
		return false
	}
	return w8Weights(m)
}

// w8Weights is w8Eligible's weight half, which needs only the model: every dense body projection int8-kind with K a
// multiple of 4. An MoE or DeltaNet layer has empty dense projections and fails it.
func w8Weights(m *decoder.Model) bool {
	w := m.Weights()
	if len(w.Layers) == 0 {
		return false
	}
	nonGated := m.NonGatedMLPResident()
	for l := range w.Layers {
		lw := &w.Layers[l]
		mats := []*linalg.WeightMat{&lw.QProj, &lw.KProj, &lw.VProj, &lw.OProj}
		switch {
		case len(lw.Experts) > 0: // an MoE layer: its routed experts and shared expert carry the FFN (slice 4)
			for e := range lw.Experts {
				mats = append(mats, &lw.Experts[e].Gate, &lw.Experts[e].Up, &lw.Experts[e].Down)
			}
			if lw.SharedExpert.Up.Rows() > 0 {
				mats = append(mats, &lw.SharedExpert.Gate, &lw.SharedExpert.Up, &lw.SharedExpert.Down)
			}
		case nonGated:
			mats = append(mats, &lw.UpProj, &lw.DownProj)
		default:
			mats = append(mats, &lw.GateProj, &lw.UpProj, &lw.DownProj)
		}
		for _, wm := range mats {
			if _, _, _, ok := wm.Int8(); !ok || wm.Rows() == 0 || wm.Cols()%4 != 0 {
				return false
			}
		}
	}
	return true
}

// w8AttnEligible reports whether m runs int4mix natively (r.w8Attn, docs/tasks/task-metal-int8-2026-10.md slice 4):
// every layer's attention projections int8-kind and its FFN int4-kind (int4mix's per-tensor split), on the plain
// dense path whose qkv and o-proj run pSABias and pSAResid alone. Sandwich, post-only and parallel-block layers run the
// o-proj on pSA, which the FFN shares; non-gated MLPs and output biases run other handles; MoE, DeltaNet and K=V layers
// are not slice 1's shapes. Any of those keeps the int4 re-quantization.
func w8AttnEligible(m *decoder.Model, r *resident) bool {
	if !nativeInt8 || !nativeInt4Mix || r.moe != nil || r.g4moe != nil || r.dnet != nil || r.sandwich || r.postOnly || r.parallelBlock ||
		r.nonGatedMLP || r.outBias || r.layerNorm {
		return false
	}
	return w8AttnWeights(m)
}

// w8AttnWeights is w8AttnEligible's weight half: attention int8 (K a multiple of 4), gate, up and down int4.
func w8AttnWeights(m *decoder.Model) bool {
	w := m.Weights()
	if len(w.Layers) == 0 {
		return false
	}
	for l := range w.Layers {
		lw := &w.Layers[l]
		for _, wm := range []*linalg.WeightMat{&lw.QProj, &lw.KProj, &lw.VProj, &lw.OProj} {
			if _, _, _, ok := wm.Int8(); !ok || wm.Rows() == 0 || wm.Cols()%4 != 0 {
				return false
			}
		}
		for _, wm := range []*linalg.WeightMat{&lw.GateProj, &lw.UpProj, &lw.DownProj} {
			if wm.Kind() != "int4" || wm.Rows() == 0 {
				return false
			}
		}
	}
	return true
}

// f32Projection names the first dense body projection still at f32, "" when there is none (every projection int4 or
// int8). A model's router and other f32-kept tensors are not body projections and do not count.
func f32Projection(m *decoder.Model) string {
	w := m.Weights()
	for l := range w.Layers {
		lw := &w.Layers[l]
		for _, c := range []struct {
			name string
			wm   *linalg.WeightMat
		}{{"q_proj", &lw.QProj}, {"o_proj", &lw.OProj}, {"up_proj", &lw.UpProj}, {"down_proj", &lw.DownProj}} {
			if c.wm.Rows() > 0 && c.wm.Kind() == "f32" {
				return fmt.Sprintf("layer %d %s", l, c.name)
			}
		}
	}
	return ""
}

// w8PreciseMath compiles the library of a model headed for the native int8 path without fast math: at int8int8 any
// difference from the CPU is amplified by the activation quantization, and fast math was a large part of gate F3's gap
// (docs/tasks/task-metal-int8-2026-10.md). It is decided before the build from the model alone, so a model that passes
// w8Weights but is MoE (Gemma 4's parallel dense||MoE) is excluded here too.
func w8PreciseMath(m *decoder.Model) bool {
	if !nativeInt8 || w8FastMath || m.HasGemma4MoEResident() {
		return false
	}
	if _, _, _, _, _, _, _, _, _, _, moe := m.MoEResidentParams(); moe {
		// slice 4's MoE, as w8Eligible will admit it: not gpt-oss, not paged
		if _, _, gptoss := m.GptOssActResident(); !nativeInt8MoE || gptoss || metalMoESlotsRequest(m) != "" {
			return false
		}
		return w8Weights(m)
	}
	return w8Weights(m) || nativeInt4Mix && w8AttnWeights(m) // int4mix's attention runs the same int8 kernels
}

// int8Concat row-concatenates same-K int8 WeightMats into ONE int8 buffer and its per-row f32 scales: int4Concat's
// twin for a w8 model's fused QKV and gate|up. Codes and scales are copied, not re-quantized, so the fused GEMV
// computes exactly what the separate ones would. Slice 1 copies them even from a .giw mapping (the single tensors
// alias through int8BufA); the copy is counted in the alias stats.
func int8Concat(d *Device, a *weightAlias, wms ...*linalg.WeightMat) (Buffer, Buffer) {
	var nCodes, nRows int
	for _, w := range wms {
		nCodes += w.Rows() * w.Cols()
		nRows += w.Rows()
	}
	codes := make([]int8, 0, nCodes)
	scales := make([]float32, 0, nRows)
	for _, w := range wms {
		q8, sc, _, ok := w.Int8()
		if !ok {
			panic(fmt.Sprintf("metal: int8Concat weight kind %q is not int8", w.Kind())) // buildResident's recover declines
		}
		codes = append(codes, q8[:w.Rows()*w.Cols()]...)
		scales = append(scales, sc[:w.Rows()]...)
	}
	a.addCopy(int64(len(codes) + 4*len(scales)))
	return NewBufferInt8(d, codes), NewBufferFloats(d, scales)
}

// buildResident builds a Metal resident decoder from a loaded Model: it uploads the weights, creates the pipelines and
// allocates the KV slots. It returns an error, which the caller treats as a decline to the CPU path, when the model's
// shape or the device cannot run it resident.
func buildResident(m *decoder.Model) (res *resident, err error) {
	// Entry point: convert the build-time panics (pipeline-compile failure, non-int4 expert weights, buffer OOM in
	// model.go, moe.go and gemma4_moe.go) into the error this signature promises, so a direct caller gets a decline,
	// not a process-killing panic. Registered first, so it runs last, after the cleanup defer below has released the
	// partial device state. The backend wrapper keeps its own recover() as defence in depth.
	defer func() {
		if p := recover(); p != nil {
			res, err = nil, fmt.Errorf("metal: BuildResident panicked: %v", p)
		}
	}()
	// Pin the thread and hold one autorelease pool for the whole build: CompileLibrary, NewComputePipeline and every
	// NewBuffer*/nsString create autoreleased temporaries that would otherwise leak on this unpinned, pool-less
	// thread.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	pool := NewARPool()
	defer pool.Drain()
	// MC1's slot count is priced before any buffer below exists. Its budget, metalMemoryCeiling, reads the live
	// available memory; read after the weights were allocated, that figure already excludes the resident's own
	// buffers while the base it is compared with (residentNeedBytes) still counts them, so the weights would be
	// counted twice and too few slots granted.
	kvSlots := metalKVSlots(m)
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		return nil, err
	}
	// Every buffer and objc object below lands on d's ledgers. On any early return or panic (a shape-check error, a
	// buildMoE/int4/int8-kind error, or a mustBuf/pipe OOM panic that backend.go recovers into a clean CPU decline),
	// release it all: otherwise a declined build leaks gigabytes while serve continues on CPU. Cleared once
	// construction completes.
	ok := false
	defer func() {
		if !ok {
			d.ReleaseAll()
			d.ReleaseObjects()
		}
	}()
	// preciseMathCompile (a test and measurement toggle) turns fast-math off, which removes the compiler's
	// contraction, reassociation and transcendental discretion: the axis that makes within-machine bit-identity
	// fragile to an OS toolchain update. The default is fast-math: precise costs a few percent of decode speed and
	// does not improve CPU parity (docs/ollama-chase.md §A2-Metal). GOINFER_PRECISE_MATH is a documented opt-in for
	// bits robust to a toolchain update; the snapshot golden otherwise detects such drift.
	compile := d.CompileLibrary
	preciseMath := preciseMathCompile || modelKnob(m, "GOINFER_PRECISE_MATH") != "" || w8PreciseMath(m)
	if preciseMath {
		compile = d.CompileLibraryPrecise
	}
	lib, err := compile(allKernels+mc3RowsKernels, MSL3_1) // + MC3 S3's multi-row forms, derived from allKernels (batch_rows.go)
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
	r := &resident{knob: m.Knob, d: d, H: H, nL: nL, nH: nH, I: I, V: V, preciseMath: preciseMath}
	r.chainEmbBase = residentNeedBytes(m) + int64(kvSlots-1)*residentKVBytes(m)
	r.chainEmbGuardOff = modelKnob(m, "GOINFER_NO_RESIDENT_MEM_GUARD") != ""
	if r.ctxCap, err = resolveMetalCtxCap(m); err != nil {
		return nil, err
	}
	r.pRms, r.pQv, r.pGemv = pipe("rmsnorm_quant"), pipe("quant_vec"), pipe("gemv_w4a8_coal")
	r.pGemvResid = pipe("gemv_w4a8_resid")
	r.pSA, r.pSABias, r.pSAResid = pipe("gemv_w4a8_sa"), pipe("gemv_w4a8_sa_bias"), pipe("gemv_w4a8_sa_resid")
	r.pRmsF16 = pipe("rmsnorm_f16_act")
	r.pF32ToF16 = pipe("f32_to_f16")
	r.pSAf16, r.pSAf16Bias, r.pSAf16Resid = pipe("gemv_w4f16_sa"), pipe("gemv_w4f16_sa_bias"), pipe("gemv_w4f16_sa_resid")
	r.decodeLaneW4F16 = modelKnob(m, "GOINFER_METAL_DECODE_LANE") == "w4f16"
	if r.q4kLane = m.Quant() == "q4k"; r.q4kLane {
		r.decodeLaneW4F16 = false
		r.pQ4K, r.pQ4KResid, r.pQ4KBias = pipe("gemv_q4k_f32"), pipe("gemv_q4k_f32_resid"), pipe("gemv_q4k_f32_bias")
		r.pW8F32, r.pW8F32Resid, r.pW8F32Bias = pipe("gemv_w8_f32"), pipe("gemv_w8_f32_resid"), pipe("gemv_w8_f32_bias")
		r.pRmsF32Out, r.pSwF32 = pipe("rmsnorm_f32_out"), pipe("swiglu_f32")
	}
	r.pAttnFA, r.pAttnFACombine = pipe("attention_fa"), pipe("attention_fa_combine")
	r.pAttnFARows, r.pAttnFACombineRows = pipe("mc3_attention_fa_rows"), pipe("mc3_attention_fa_combine_rows")
	r.decodeAttnFA = metalAttnFAEnabled(modelKnob(m, "GOINFER_METAL_ATTN_FA"))
	r.pArgFinish = pipe("argmax_finish")
	r.pArgRowsPart = pipe("argmax_rows_part") // E-P08: the batched step's greedy rows (batch.go)
	r.pEmbedGather = pipe("embed_gather_i8")
	// gemv_w4a8_bias and gemv_w4a8_sa_amax are not built: nothing dispatches them (ForwardArgmax uses the int8
	// pGemvW8Amax head; the profiler builds gemv_w4a8_bias locally). If gemv_w4a8_sa_amax is wired, it needs an
	// N/row>=N guard (see the note on the kernel in kernels.go): it is a reduction, so mask the logit to -INF, do
	// not early-return past the barrier.
	r.pRope, r.pRope2, r.pKv, r.pAttn = pipe("rope"), pipe("rope2"), pipe("kv_store"), pipe("attention")
	// Pairwise rope (docs/tasks/task-metal-pairwise-rope-2026-10.md): a GPT-J pairwise family (Cohere, Command-R7B, Aya,
	// GLM-OCR) binds the pairwise twins into the same fields, so no dispatch site changes; every other family keeps these.
	if r.pairwiseRoPE = m.PairwiseRoPEResident(); r.pairwiseRoPE {
		r.pRope, r.pRope2 = pipe("rope_pw"), pipe("rope2_pw")
	}
	r.pSw, r.pRes = pipe("swiglu_quant"), pipe("residual")
	r.pLayerScale = pipe("layer_scale")
	if P := m.Gemma4PLEDimResident(); P > 0 {
		r.pPLEGeluMul, r.pleP = pipe("ple_gelu_mul"), P
		r.pleIn, r.pleG, r.pleQ, r.pleSc = d.NewBufferLen(nL*P), d.NewBufferLen(P), byteBuf(d, P), d.NewBufferLen(1)
		r.uPleP = NewBufferU32(d, uint32(P))
	}
	r.pRmsRows, r.pQvRows = pipe("mc3_rmsnorm_quant_rows"), pipe("mc3_quant_vec_rows")
	r.pSwRows, r.pRope2Rows = pipe("mc3_swiglu_quant_rows"), pipe("mc3_rope2_rows")
	r.pKvRows, r.pAttnRows = pipe("mc3_kv_store_rows"), pipe("mc3_attention_rows")
	r.pQKNormRows = pipe("mc3_qk_norm_rows")
	r.pMoeBatch[2], r.pMoeBatch[4] = pipe("moe_batch_gemv2"), pipe("moe_batch_gemv4")
	r.pMoeCombine = pipe("moe_batch_combine")
	r.pGemvW8, r.pGemvW8Amax = pipe("gemv_w8a8_coal"), pipe("gemv_w8a8_amax")
	if r.q4kLane { // the same buffer order and launch shapes, f32 activations: every head dispatch site runs unchanged
		r.pGemvW8, r.pGemvW8Amax = pipe("gemv_w8f32_head"), pipe("gemv_w8f32_amax")
	}
	r.pCopyVec = pipe("copy_f32")
	r.pCopyU32 = pipe("copy_u32")
	// R7b Mac half: device Gumbel-max sampler over r.logits (the same buffer pGemvW8 writes for
	// forwardLogits/ForwardEmbPipe) — see gumbel_sample.go. Always built, sized once for V.
	r.pGumbel1, r.pGumbel2 = pipe("gumbel_stage1"), pipe("gumbel_stage2")
	r.gumbelNB = gumbelBlocks(V)
	r.gumbelBKey, r.gumbelBIdx = d.NewBufferLen(r.gumbelNB), d.NewBufferLen(r.gumbelNB)
	r.gumbelOut = d.NewBufferLen(1)
	r.uGumbelV, r.uGumbelNB = NewBufferU32(d, uint32(V)), NewBufferU32(d, uint32(r.gumbelNB))
	r.uGumbelInvT = NewBufferFloats(d, []float32{1})
	r.uGumbelK0, r.uGumbelK1 = NewBufferU32(d, 0), NewBufferU32(d, 0)
	r.uGumbelD0, r.uGumbelD1 = NewBufferU32(d, 0), NewBufferU32(d, 0)
	r.pQKNorm, r.pRmsF32 = pipe("qk_norm"), pipe("rmsnorm_f32")
	r.qkNorm = m.HasQKNorm()
	r.qkNormWhole = m.QKNormWholeResident() // G5 (docs/tasks/task-gpu-paths-2026-09.md): Olmo 3/Olmo Hybrid
	r.sandwich = m.SandwichNormResident()
	r.postOnly = m.PostOnlyNormResident() // G5: Olmo 3/Olmo Hybrid
	r.pLayerNorm, r.pActQuant = pipe("layernorm_quant"), pipe("act_quant")
	r.pLoraDelta = pipe("lora_delta") // G3/P-11: compute-time LoRA, down+up fused into one kernel
	r.pSABiasResid, r.pCoalBiasResid = pipe("gemv_w4a8_sa_bias_resid"), pipe("gemv_w4a8_resid_bias")
	r.layerNorm = m.LayerNormResident()
	r.parallelBlock = m.ParallelBlockResident() // G5: Cohere/Command-R + Cohere2/Command-R7B
	r.logitScale, _ = m.LogitScaleResident()    // ok=false ⇒ 1, already finalizeLogits's no-op value
	r.nonGatedMLP = m.NonGatedMLPResident()
	r.outBias = m.OutBiasResident()
	r.learnedPos = m.LearnedPosResident()
	if alpha, limit, isGptOss := m.GptOssActResident(); isGptOss {
		r.attnSink = true
		r.gptossAlpha, r.gptossLimit = float64(alpha), float64(limit)
	}
	// The f32 KV path (kv_store_f32, attention_f32) exists but is off: a matched-input confirmer showed f16 versus
	// f32 KV storage is not what degrades Gemma's contexts. What was found, and the open claim about position-0 K/V,
	// is in docs/code-notes/metal.md#buildResident.kvF32.
	r.kvF32 = false
	if r.kvF32 {
		r.pKv, r.pAttn = pipe("kv_store_f32"), pipe("attention_f32")
	}
	r.kvI8 = m.KVCacheI8()
	if r.kvI8 {
		r.pKvI8, r.pAttnI8 = pipe("kv_store_i8"), pipe("attention_i8")
	}
	// Gated-MLP activation: ordinals ARE decoder.ActKind's iota (0=GELU-tanh, 1=SiLU), so this
	// passes straight through to glu_act. Gemma is GeGLU; everything else admitted is SwiGLU.
	r.uAct = NewBufferU32(d, uint32(m.GatedActResident()))
	addOne := uint32(0)
	if m.RMSAddOne() {
		addOne = 1
	}
	r.uAddOne = NewBufferU32(d, addOne)
	win := max(
		// 0 = full causal; Mistral is all-local with this window
		m.SlidingWindowResident(), 0)
	r.uWindow = NewBufferU32(d, uint32(win))
	if r.moe, err = buildMoE(d, m, pipe, H); err != nil { // nil for a dense (or gemma4-MoE) model; error ⇒ decline
		return nil, err
	}
	if r.g4moe, err = buildGemma4MoE(d, m, pipe, H, nL); err != nil { // Gemma-4 enable_moe_block; nil otherwise; error ⇒ decline
		return nil, err
	}
	if r.dnet, err = buildDeltaNet(d, m, pipe); err != nil { // Gated-DeltaNet mixer; nil otherwise; error ⇒ decline
		return nil, err
	}
	if r.dnet != nil {
		dp := r.dnet
		r.pDnConv, r.pDnGates = pipe("delta_conv"), pipe("delta_gates")
		r.pDnNorm, r.pDnRule, r.pDnGNorm = pipe("delta_norm"), pipe("delta_rule"), pipe("delta_gnorm")
		r.pDnQSplit, r.pDnAttnGate = pipe("delta_qsplit"), pipe("delta_attn_gate")
		r.pDnConvSeq, r.pDnGatesRows, r.pDnNormRows = pipe("delta_conv_seq"), pipe("delta_gates_rows"), pipe("delta_norm_rows")
		r.pDnRuleSeq, r.pDnGNormRows, r.pDnProjW8Rows = pipe("delta_rule_seq"), pipe("delta_gnorm_rows"), pipe("delta_proj_w8_rows")
		r.dnMixed, r.dnConvOut = d.NewBufferLen(dp.convDim), d.NewBufferLen(dp.convDim)
		r.dnBt, r.dnAt = d.NewBufferLen(dp.nv), d.NewBufferLen(dp.nv)
		r.dnHeadP = d.NewBufferLen(dp.nv * 2)
		r.dnQn, r.dnKn = d.NewBufferLen(dp.keyDim), d.NewBufferLen(dp.keyDim)
		r.dnCore, r.dnZOut, r.dnGated = d.NewBufferLen(dp.valueDim), d.NewBufferLen(dp.valueDim), d.NewBufferLen(dp.valueDim)
		r.dnGq, r.dnGSc = d.NewBufferBytes(dp.valueDim), d.NewBufferLen(1)
		r.uDnConvDim, r.uDnK = NewBufferU32(d, uint32(dp.convDim)), NewBufferU32(d, uint32(dp.convK))
		r.uDnNv, r.uDnNk = NewBufferU32(d, uint32(dp.nv)), NewBufferU32(d, uint32(dp.nk))
		r.uDnHk, r.uDnHv = NewBufferU32(d, uint32(dp.hk)), NewBufferU32(d, uint32(dp.hv))
		r.uDnKeyDim, r.uDnQScale = NewBufferU32(d, uint32(dp.keyDim)), NewBufferFloats(d, []float32{dp.qScale})
		r.uDnRep, r.uDnVBase = NewBufferU32(d, uint32(dp.rep)), NewBufferU32(d, uint32(2*dp.keyDim))
		r.uDnValueDim = NewBufferU32(d, uint32(dp.valueDim))
	}
	// Native int8 (docs/tasks/task-metal-int8-2026-10.md): a dense model whose body projections are all int8 runs
	// them as W8A8 instead of re-quantizing them to int4. The int4-only paths cannot read those buffers, so they are
	// off for it: the f16 lane here, the R18 rows kernels and the MC3 step below. The batched prefill pass reads
	// them through its W8 tiles (gemm_w8f16_*).
	r.w8 = w8Eligible(m, r)
	if r.w8 && r.moe != nil { // slice 4: the experts' GEMVs on their int8 weights (buildMoELayer uploads them as int8)
		r.moe.w8 = true
		r.moe.pGU, r.moe.pDownWacc = pipe("gemv_w8a8_moe"), pipe("gemv_w8a8_moe_wacc")
		r.moe.guR, r.moe.downR = 0, 0
		if moeExpertRowsOn { // the int8 expert GEMVs' rows form (gemv_w8a8_moe_rows), where the row counts admit it
			if R := gemvRowsFor(2*r.moe.inter, 4); R > 0 {
				r.moe.pGU, r.moe.guR = pipe(fmt.Sprintf("gemv_w8a8_moe_rows%d", R)), R
			}
			if R := gemvRowsFor(r.H, 4); R > 0 {
				r.moe.pDownWacc, r.moe.downR = pipe(fmt.Sprintf("gemv_w8a8_moe_wacc_rows%d", R)), R
			}
		}
		r.moe.kSlots = false // D-P03's k-slot kernels read int4
	}
	r.w8Attn = !r.w8 && w8AttnEligible(m, r)
	if r.q4kLane {
		r.w8, r.w8Attn = false, false
	}
	if r.w8Attn {
		r.decodeLaneW4F16 = false // the f16 lane reads int4 attention weights
	}
	if why := f32Projection(m); why != "" {
		// An f32 load reaches here only when f32 was asked for; this backend runs int4 or int8 weights and does not
		// quantize them behind the request (docs/tasks/task-metal-int8-2026-10.md, slice 4). A clear decline instead of
		// int4Concat's panic on an f32 kind.
		return nil, fmt.Errorf("weights loaded at f32 (%s); metal runs int4 or int8 weights: load with -quant int4 or int8int8 to use the GPU", why)
	}
	if r.w8 {
		r.decodeLaneW4F16 = false
	}
	// prefillOK is derived, not hand-listed: the f16 prefill kernels implement exactly the features in
	// prefillFeatures, so a model needing more (MoE never packs the dense FFN buffers at all) declines prefill and
	// falls back to the sequential Forward loop. Admission by prefillFeatures alone is not enough, so these are
	// declined explicitly:
	// - Per-layer geometry. PrefillLast reads r.layers[0].geom once and reuses it for every layer's rope and
	//   attention dims. That is right for every uniform-geometry family (Gemma 3's dual-base RoPE keeps head_dim
	//   uniform) but not for Gemma 4's local/global split (head_dim 256 versus 512). Check !m.HasPerLayerGeometry()
	//   directly: decoder.Model.PerLayerGeomOK("metal") answers whether Metal's decode path supports per-layer
	//   geometry, so it would wrongly clear Gemma 4.
	// - Gemma 4's enable_moe_block (parallel dense||MoE FFN, residLayer.g4moe, encodeGemma4MoEFFN) is a third FFN
	//   shape that PrefillLast's L.moe != nil branch does not cover. Geometry does not catch it, so it is checked
	//   directly.
	// - A paged generic MoE: its per-layer expGuW/expGuS/expDW/expDS buffers are zero-value (the weights live in the
	//   slot pool), but both of PrefillLast's MoE paths (the row loop's non-paged encodeMoERoute/encodeMoEExperts pair
	//   and the expert-major pass) read those stacked buffers unconditionally.
	// - An int8 KV cache (-kv i8). The prefill kernels write K/V with kv_store_f16 (half per element at pos*kvDim)
	//   and PrefillLast's attention reads half, but the int8 cache is one byte per element with separate scale
	//   buffers: every position would land in the wrong layout, and positions at or past ctxCap/2 would write past
	//   the buffer. Such a model takes the sequential path, whose decode kernels write and read the int8 cache.
	// - A Gated-DeltaNet hybrid takes the pass only with dnetPrefillOn, and only in the shape prefill_deltanet.go
	//   implements: Qwen3.5's pre-norm layers, no LayerNorm bias. Olmo Hybrid (postOnly) stays sequential.
	missing := m.MissingResidentFeatures(prefillFeatures)
	dnetOK := r.dnet == nil
	if r.dnet != nil && dnetPrefillOn && !r.postOnly {
		dnetOK = true
		for i := range r.layers {
			if r.layers[i].preNormBias != (Buffer{}) {
				dnetOK = false
			}
		}
		if dnetOK {
			missing = slices.DeleteFunc(missing, func(f decoder.ResidentFeature) bool { return f == decoder.FeatDeltaNet })
		}
	}
	if r.q4kLane && (r.sandwich || r.postOnly || r.parallelBlock || r.outBias || r.layerNorm || r.moe != nil || r.g4moe != nil ||
		r.dnet != nil || r.pleP > 0 || r.attnSink || r.learnedPos || r.kvI8 || r.qkNorm) {
		return nil, fmt.Errorf("metal: --quant q4k runs on Metal for the plain dense pre-norm shape only (Phi-3, Llama); this family has a variant the q4k lane does not cover")
	}
	r.prefillOK = !r.q4kLane && len(missing) == 0 && !m.HasPerLayerGeometry() &&
		!m.HasGemma4MoEResident() && !(r.moe != nil && r.moe.paged) && dnetOK && !r.kvI8 &&
		!r.attnSink // D-C01: gpt-oss's sink and clamped, biased SwiGLU are in no prefill kernel; explicit, so a feature-map edit cannot admit it
	r.q = d.NewCommandQueue()

	w := m.Weights()
	r.embed = &w.Embed
	if r.learnedPos {
		r.posEmbed = m.PosEmbedResident()
	}
	// Whether THIS family's LayerNorm carries a bias (GPT-2) or not (Cohere) — a per-arch
	// constant, so layer 0's presence/absence decides it for every layer.
	if r.layerNorm && len(w.Layers) > 0 {
		r.layerNormBias = len(w.Layers[0].PreAttnNormBias) > 0
	}
	if r.layerNormBias {
		r.uLNHasBias = NewBufferU32(d, 1)
	} else {
		r.uLNHasBias = NewBufferU32(d, 0)
	}
	alias := newWeightAlias(m) // on for a .giw-mapped model unless GOINFER_METAL_ALIAS=0 (S6)
	if r.g4moe != nil {
		r.g4moe.alias = alias
	}
	if r.moe != nil {
		r.moe.alias = alias
	}
	// The native int8 path uploads int8 weights as int8: every projection on r.w8, the attention ones on r.w8Attn
	// (int4mix, whose FFN stays int4). Anything else re-quantizes an int8 weight to int4.
	upload := func(d *Device, a *weightAlias, wm *linalg.WeightMat) (Buffer, Buffer, error) {
		if _, _, _, ok := wm.Int8(); ok && (r.w8 || r.w8Attn) {
			return int8BufA(d, a, wm)
		}
		return int4BufA(d, a, wm)
	}
	mk := func(wm *linalg.WeightMat) (Buffer, Buffer) {
		if r.q4kLane { // the lane uploads its own segments (q4kLaneProj, below)
			return Buffer{}, Buffer{}
		}
		q, s, e := upload(d, alias, wm)
		if e != nil {
			panic(e)
		}
		return q, s
	}
	// fuse uploads a fused group (QKV, gate|up): int4ConcatA, or int8Concat on the native int8 path.
	fuse := func(wms ...*linalg.WeightMat) (Buffer, Buffer) {
		if r.q4kLane {
			return Buffer{}, Buffer{}
		}
		if _, _, _, ok := wms[0].Int8(); ok && (r.w8 || r.w8Attn) {
			return int8Concat(d, alias, wms...)
		}
		return int4ConcatA(d, alias, wms...)
	}
	r.layers = make([]residLayer, nL)
	r.kc, r.vc = make([]Buffer, nL), make([]Buffer, nL)
	r.kvSlotBytes = make([]int, nL)
	if r.kvI8 {
		r.ks, r.vs = make([]Buffer, nL), make([]Buffer, nL)
	}
	// Per-layer attention geometry, deduped by value (geom.go). Uniform families resolve every
	// layer to one geom; Gemma 4's local/global interleave resolves to two. maxNHhd/maxKvDim size
	// the shared per-token scratch to the widest layer (== the model shape for a uniform family).
	geomCache := map[[4]int]*attnGeom{}
	var maxNHhd, maxKvDim, maxHd int
	var maxAttnFAPartialElems int // R2: max over layers of nKV*G*(hd+2), G=nH/nKV — see attnFAPartial's own comment
	_, _, _, _, _, attnGate, dnetOK := m.Qwen35ResidentParams()
	for l := range nL {
		lw := &w.Layers[l]
		var L residLayer
		var kEqV bool // false for isDelta (no geometry) and qGate (this family never has K=V); set below otherwise
		isDelta := dnetOK && m.Qwen35LinearLayer(l)
		if isDelta {
			// Gated-DeltaNet mixer layer: no attention projections, no KV cache, no geometry at
			// all — see metal/deltanet.go and residLayer.delta's doc. L.geom stays nil and
			// r.kc[l]/r.vc[l] stay zero-value; encodeLayer never calls encodeAttention for this
			// layer, so nothing downstream dereferences them.
			DL, e := buildDeltaNetLayer(d, m, l, r.dnet, mk)
			if e != nil {
				return nil, e
			}
			L.delta = DL
		} else if dnetOK && attnGate {
			// The same family's gated softmax layer (qwen3_5, qwen3_5_moe, qwen3_next), not every dnetOK family: Olmo
			// Hybrid's full-attention layer is olmo3's plain scheme (AttnGate=false) and takes the ordinary branch below.
			// Its q/k/v/o live off qattn, not lw.QProj/KProj/VProj/OProj, and q_proj is double width: [query || gate] per
			// head, interleaved, not two concatenated blocks; treated as an ordinary q_proj it yields plausible logits
			// from the wrong tensor. So the fused QKV path does not apply: K||V still fuse into L.qkvW (2-way), and Q is a
			// separate double-width projection (L.dnQw/dnQs), split on the activation at encode time (delta_qsplit),
			// because slicing rows out of a quantized int4 bundle with per-group scales is real surgery.
			qP, kP, vP, oP, qN, kN := m.Qwen35AttnWeights(l)
			if len(qN) == 0 || len(kN) == 0 {
				return nil, fmt.Errorf("metal: layer %d: qwen35 softmax layer has empty q_norm/k_norm "+
					"(%d/%d) — the loader did not populate them for this container", l, len(qN), len(kN))
			}
			L.qGate = true
			L.dnQw, L.dnQs = mk(qP)
			L.qkvW, L.qkvS = int4ConcatA(d, alias, kP, vP) // K‖V fused (Q handled separately above)
			L.oW, L.oS = mk(oP)
			L.qNorm, L.kNorm = NewBufferFloats(d, qN), NewBufferFloats(d, kN)
		} else {
			// The ordinary plain-attention layer: every layer that is neither DeltaNet nor qGate, including a dnetOK
			// family's full-attention layer when AttnGate is false (Olmo Hybrid). lw.QProj/KProj/VProj/OProj/QNorm/KNorm
			// are read normally.
			//
			// K=V (attention_k_eq_v, Gemma 4 global layers): no v_proj; V = v_norm(the raw k_proj output). The V slot is
			// fused with k_proj (the fused projection yields qkv=[Q|K_raw|K_raw]); at encode time (encodeTrunkInto) the V
			// slot is scale-less-v_norm'd and left un-roped, while the K slot gets k_norm and RoPE. This is a build-time
			// weight-layout difference, so the value-independent ForwardEmbPipe pre-encode stays correct (see geom.kEqV).
			// False (and a no-op) for every family without K=V layers.
			kEqV = m.VFromKResident(l)
			if src := m.KVSrcAtResident(l); src != l {
				// Gemma 4 E-model KV-shared layer (S1.4): the checkpoint has no k_proj/v_proj/k_norm here; it attends
				// over layer src's keys and values. Q-only projection; kEqV does not apply (no K of its own).
				if src < 0 || src >= l {
					return nil, fmt.Errorf("metal: layer %d: KV source %d is not an earlier layer", l, src)
				}
				if g4KVSrcOffForTest { // G2 defect (3): the previous owning layer of the same attention type, if any
					for j := src - 1; j >= 0; j-- {
						if m.KVSrcAtResident(j) == j && m.LayerIsLocalResident(j) == m.LayerIsLocalResident(src) {
							src = j
							break
						}
					}
				}
				L.kvShared, L.kvSrc, kEqV = true, src, false
				L.qkvW, L.qkvS = fuse(&lw.QProj)
			} else if kEqV {
				L.qkvW, L.qkvS = int4Concat(d, &lw.QProj, &lw.KProj, &lw.KProj) // V slot = raw k_proj
			} else {
				L.qkvW, L.qkvS = fuse(&lw.QProj, &lw.KProj, &lw.VProj) // fused QKV
			}
			L.oW, L.oS = mk(&lw.OProj)
			if r.outBias {
				L.oBias = NewBufferFloats(d, lw.OBias)
			}
			if r.qkNorm { // Qwen3 per-head Q/K norm weights [hd] — qGate layers set these above instead
				// QKNormWhole (Olmo 3/Olmo Hybrid, G5): the resident kernel launch that reproduces
				// this collapses the grid to one Q block + one K block sized from a SINGLE hd
				// argument (encodeAttention), which is only correct when nH==nKV (MHA). Decline
				// rather than silently mis-normalize a hypothetical future GQA family.
				if r.qkNormWhole {
					if nKVL := m.KVHeadsAtResident(l); r.nH != nKVL {
						return nil, fmt.Errorf("metal: layer %d: arch claims QKNormWhole but nH=%d != nKV=%d — the resident whole-vector kernel launch requires MHA", l, r.nH, nKVL)
					}
				}
				L.qNorm = NewBufferFloats(d, lw.QNorm)
				if L.kvShared {
					L.kNorm = L.qNorm // bound, never read: a shared layer's qk_norm grid covers the Q heads only
				} else {
					L.kNorm = NewBufferFloats(d, lw.KNorm)
				}
			}
		}
		if !isDelta {
			// attention always dispatches L.attnSinks/L.uHasSink (unlike L.invf, which is only bound
			// when the rope dispatch itself runs) — every family needs a VALID buffer bound, gated by
			// the flag, so non-gpt-oss gets a real but unused 1-element dummy, not a zero Buffer.
			if r.attnSink {
				L.attnSinks, L.uHasSink = NewBufferFloats(d, lw.AttnSinks), NewBufferU32(d, 1)
			} else {
				L.attnSinks, L.uHasSink = NewBufferFloats(d, []float32{0}), NewBufferU32(d, 0)
			}
		}
		g4b, isG4MoE := decoder.Gemma4MoEResidentBundle{}, false
		if r.g4moe != nil {
			g4b, isG4MoE = m.Gemma4MoEResidentLayer(l)
		}
		switch {
		case isG4MoE: // Gemma-4 enable_moe_block: parallel dense‖MoE FFN (its own norms/router/experts)
			L.g4moe = buildGemma4MoELayer(d, m, &g4b, r.g4moe)
		case r.moe != nil && len(lw.Experts) > 0: // generic MoE layer: stacked experts instead of a dense FFN
			L.moe = buildMoELayer(d, m, l, lw, r.moe)
		case r.nonGatedMLP: // GPT-2/Nemotron relu²: single up-proj (no gate) + biases
			L.upW, L.upS = mk(&lw.UpProj)
			L.dW, L.dS = mk(&lw.DownProj)
			L.upBias = NewBufferFloats(d, lw.UpBias)
			L.downBias = NewBufferFloats(d, lw.DownBias)
		default: // dense FFN (also GLM/DeepSeek's FirstKDense prefix layers, and gemma4 dense layers)
			L.guW, L.guS = fuse(&lw.GateProj, &lw.UpProj) // fused gate/up
			L.dW, L.dS = mk(&lw.DownProj)
			L.ffnI = lw.GateProj.Rows()
		}
		if r.q4kLane {
			// The q4k lane covers the plain dense shape (Phi-3, Llama, Qwen2): pre-norm attention (q/k/v bias allowed), a
			// SwiGLU FFN.
			if isDelta || L.moe != nil || L.g4moe != nil || L.qGate || L.kvShared || kEqV || r.nonGatedMLP {
				return nil, fmt.Errorf("metal: --quant q4k runs on Metal for dense attention+SwiGLU layers only (layer %d is not)", l)
			}
			L.qkvSegs = q4kLaneProj(d, &lw.QProj, &lw.KProj, &lw.VProj)
			L.oSegs, L.dSegs = q4kLaneProj(d, &lw.OProj), q4kLaneProj(d, &lw.DownProj)
			L.guSegs = q4kLaneProj(d, &lw.GateProj, &lw.UpProj)
		}
		// postOnly (Olmo 3, Olmo Hybrid) is a model-level flag, but Olmo Hybrid's DeltaNet layers reach NormPre2
		// through NormPlacementLinear and carry real pre-norm weights regardless; isDelta is already resolved above,
		// so gate on the per-layer truth, not the model-level flag.
		postOnlyHere := r.postOnly && !isDelta
		if !postOnlyHere {
			// A postOnly (non-delta) layer has NO pre-norm weight at all (lw.PreAttnNorm/
			// PreMLPNorm are 0-length — NewBufferFloats on an empty slice is a build-time error,
			// not a no-op), so this is skipped entirely; segA/encodeLayer quantize the raw
			// residual instead (see encodeAttention/encodeLayer's own postOnly branch).
			L.preNorm = NewBufferFloats(d, lw.PreAttnNorm)
			// parallelBlock (Cohere/Command-R) has no pre-MLP norm tensor at all: the MLP reuses the shared input norm the
			// attention branch computed (encodeLayer's r.aq/r.aSc reuse), so lw.PreMLPNorm is empty (see
			// Model.ParallelBlockResident), with the same NewBufferFloats hazard postOnly's comment above names.
			if L.g4moe == nil && !r.parallelBlock { // dense/generic FFN entry norm (PreMLPNorm); g4moe carries its five norms in the bundle
				L.postNorm = NewBufferFloats(d, lw.PreMLPNorm)
			}
			if r.layerNorm && r.layerNormBias {
				L.preNormBias = NewBufferFloats(d, lw.PreAttnNormBias)
				if L.g4moe == nil && !r.parallelBlock {
					L.postNormBias = NewBufferFloats(d, lw.PreMLPNormBias)
				}
			}
		}
		// qk-norm weights are set above, in whichever of the isDelta/qGate/default branches applies: a qGate layer's
		// qattn-sourced qNorm/kNorm must not be overwritten by the generic lw.QNorm/lw.KNorm, which this family does
		// not populate.
		// Gemma sandwich / Olmo 3 postOnly norms on each sublayer OUTPUT — postOnly is the
		// post-norm half of sandwich, minus the pre half (see PostOnlyNormResident's own
		// comment). Required to be present when the arch declares either — a silently-missing
		// one would DROP the norm rather than error. A g4moe layer still runs the ATTENTION
		// sandwich (postAttnNorm) but its FFN block is the parallel dense‖MoE, which carries its
		// own post-norms (postFFN1/postFFN2/postFFN) in the bundle and never touches
		// postMLPNorm — exactly as decoder/forward_gemma4.go skips PostMLPNorm for
		// enable_moe_block layers. So require/build postMLPNorm only for the non-g4moe FFN path.
		if r.sandwich || postOnlyHere {
			if len(lw.PostAttnNorm) != H {
				return nil, fmt.Errorf("metal: layer %d declares sandwich/postOnly norms but PostAttnNorm is not len==hidden(%d) (got %d)",
					l, H, len(lw.PostAttnNorm))
			}
			L.postAttnNorm = NewBufferFloats(d, lw.PostAttnNorm)
			if L.g4moe == nil {
				if len(lw.PostMLPNorm) != H {
					return nil, fmt.Errorf("metal: layer %d declares sandwich/postOnly norms but PostMLPNorm is not len==hidden(%d) (got %d)",
						l, H, len(lw.PostMLPNorm))
				}
				L.postMLPNorm = NewBufferFloats(d, lw.PostMLPNorm)
			}
		}
		// Gemma 4's per-layer output scalar on a dense layer (h *= layer_scalar after the FFN residual;
		// decoder/forward_gemma4.go, HF Gemma4DecoderLayer). The g4moe join applies its own. 1 and absent (0) are
		// no-ops, as on the CPU, so no dispatch is encoded for them.
		if L.g4moe == nil && m.IsGemma4Resident() {
			if s := m.Gemma4DenseLayerScalarAtResident(l); s != 0 && s != 1 {
				L.uLayerScalar = NewBufferFloats(d, []float32{s})
			}
		}
		if r.pleP > 0 { // Gemma 4 E-model PLE branch weights (S1.5)
			if lw.PLEGate.Rows() != r.pleP || lw.PLEProj.Rows() != H || len(lw.PostPLENorm) != H {
				return nil, fmt.Errorf("metal: layer %d: PLE weights %dx? / %dx? / %d, want P=%d and H=%d", l, lw.PLEGate.Rows(), lw.PLEProj.Rows(), len(lw.PostPLENorm), r.pleP, H)
			}
			L.pleGW, L.pleGS = mk(&lw.PLEGate)
			L.plePW, L.plePS = mk(&lw.PLEProj)
			L.postPLENorm = NewBufferFloats(d, lw.PostPLENorm)
		}
		if !isDelta {
			// Per-layer RoPE table (Gemma local 10k vs global 1M base) and per-layer window. The rope
			// KERNEL is unchanged — all per-layer variation rides in the table contents AND the width
			// (geom.half). RopeInvFreqLayerResident is the resident per-layer table (== the generic
			// RopeInvFreqLayer for every uniform family; Gemma 4's global proportional-rotary tail is
			// what makes it differ). Its length is this layer's rhalf (rotated pairs/head). A DeltaNet
			// layer has no attention geometry at all (see residLayer.delta's doc) — skip this whole
			// block for it: no RoPE table, no geom (L.geom stays nil), no KV cache allocated.
			invf := m.RopeInvFreqLayerResident(l)
			if len(invf) > 0 { // GPT-2 (FeatLearnedPos): no RoPE at all, invf is empty — NewBufferFloats panics on empty; L.invf stays unused (rope dispatch is skipped)
				L.invf = NewBufferFloats(d, invf)
				L.mscale = NewBufferFloats(d, []float32{float32(m.RopeMscaleLayer(l))}) // 1.0 for every family without YaRN
			}
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
			if attnFAHeadDimOK(L.geom.hd, r.nH, L.geom.nKV) && attnFAGroupOK(r.nH, L.geom.nKV) {
				g := r.nH / L.geom.nKV
				if e := L.geom.nKV * g * (L.geom.hd + 2); e > maxAttnFAPartialElems {
					maxAttnFAPartialElems = e
				}
				// Cache nKV here: attention_fa is dense-GQA-only, so every eligible layer shares one nKV (canUseAttnFA's own
				// gate is this same condition), and setPos can compute G and nSplit once per decode step instead of the
				// per-layer dispatch site doing it during encode-ahead (see uAttnFAG/uAttnFANSplit's field comment for the
				// race that would be).
				if r.attnFANKV == 0 {
					r.attnFANKV, r.attnFALayer = L.geom.nKV, l
				}
			}
			lw2 := uint32(0) // 0 = full causal; only local layers carry the window
			if m.LayerIsLocalResident(l) {
				lw2 = uint32(win)
			}
			L.uWindow = NewBufferU32(d, lw2)
			L.window = lw2
			if !L.qGate {
				// combined qkv bias (zeros where absent) so the fused pSABias GEMV epilogue is
				// uniform, sized for the full [Q|K|V] concat pSABias expects. qGate's K‖V is only
				// a 2-way concat (Q is separate — see above), so it dispatches via the no-bias pSA
				// kernel instead (this family is bias-free everywhere anyway: lw.QBias is nil for
				// it same as for a qGate layer, and Qwen35AttnWeights returns no bias either) —
				// L.qkvBias would be the wrong width for it and is simply unused.
				qb, kb, vb := lw.QBias, lw.KBias, lw.VBias
				if qb == nil {
					qb, kb, vb = make([]float32, r.nH*L.geom.hd), make([]float32, L.geom.kvDim), make([]float32, L.geom.kvDim)
				}
				L.qkvBias = NewBufferFloats(d, append(append(append([]float32{}, qb...), kb...), vb...))
			}
			// attention_prefill_fused's key loop reads whole 8-row simdgroup tiles and masks the ragged remainder after
			// the load (by position, not by skipping the read), so when nKeysMax lands on the cache's last, ragged tile
			// the load can run up to 7 rows past ctxCap*kvDim. Round the allocation up to a multiple of 8 rows (r.ctxCap,
			// the checked capacity, is unchanged) so that read lands in the buffer's own padding.
			paddedCtxCap := (r.ctxCap + 7) / 8 * 8
			kvBytes := paddedCtxCap * L.geom.kvDim * 2 // f16 KV: 2 bytes/elem (halves the cache)
			if r.kvI8 {
				kvBytes = paddedCtxCap * L.geom.kvDim * 1 // int8 KV: 1 byte/elem
				r.ks[l] = d.NewBufferLen(paddedCtxCap * L.geom.nKV)
				r.vs[l] = d.NewBufferLen(paddedCtxCap * L.geom.nKV)
			} else if r.kvF32 {
				kvBytes = paddedCtxCap * L.geom.kvDim * 4 // Gemma: f32 KV — see r.kvF32
			}
			// MC3 S3: with several resident KV slots (and not int8 KV), a layer's slots are ONE allocation, slot s at
			// byte s*kvBytes (views: Buffer.At), so a batched step's single dispatch can reach every row's slot through
			// one buffer plus a per-row offset — aikit binds at most 16 buffers per dispatch, too few to pass 8 slots'
			// K and V separately. Each slot keeps its own padded region, so C-01's over-read stays inside it.
			allocSlots := 1
			if kvSlots > 1 && !r.kvI8 {
				allocSlots = kvSlots
				r.kvContig = true
			}
			if L.kvShared {
				// S1.4: no cache of its own. Its geometry must be its source's (same attention type, so the same
				// head_dim and KV heads); the buffers are aliased below, once every source exists.
				if sg := r.layers[L.kvSrc].geom; sg == nil || *sg != *L.geom {
					return nil, fmt.Errorf("metal: layer %d shares layer %d's KV but their attention geometries differ", l, L.kvSrc)
				}
				if r.kvI8 {
					r.ks[l], r.vs[l] = r.ks[L.kvSrc], r.vs[L.kvSrc]
				}
				r.kc[l], r.vc[l] = r.kc[L.kvSrc], r.vc[L.kvSrc]
				r.layers[l] = L
				continue
			}
			r.kvSlotBytes[l] = kvBytes
			r.kc[l] = byteBuf(d, kvBytes*allocSlots)
			r.vc[l] = byteBuf(d, kvBytes*allocSlots)
		}
		r.layers[l] = L
	}
	// MC1: further resident KV slots, each a copy of the per-layer buffers just allocated (kvSlots: the model's
	// request, clamped to what the memory guard allows, priced at the top of this build; 1 for a family with
	// recurrent state).
	if n := kvSlots; n > 1 {
		r.kvSlotBufs = make([]kvSlotBuf, n)
		r.kvSlotBufs[0] = kvSlotBuf{r.kc, r.vc, r.ks, r.vs}
		for s := 1; s < n; s++ {
			b := kvSlotBuf{kc: make([]Buffer, nL), vc: make([]Buffer, nL)}
			if r.ks != nil {
				b.ks, b.vs = make([]Buffer, nL), make([]Buffer, nL)
			}
			for l := range nL {
				if L := &r.layers[l]; L.kvShared { // S1.4: alias this slot's source buffers (src < l, already set)
					b.kc[l], b.vc[l] = b.kc[L.kvSrc], b.vc[L.kvSrc]
					if r.ks != nil {
						b.ks[l], b.vs[l] = b.ks[L.kvSrc], b.vs[L.kvSrc]
					}
					continue
				}
				if r.kc[l] != (Buffer{}) && r.kvContig {
					b.kc[l], b.vc[l] = r.kc[l].At(s*r.kvSlotBytes[l]), r.vc[l].At(s*r.kvSlotBytes[l])
				} else if r.kc[l] != (Buffer{}) {
					b.kc[l], b.vc[l] = byteBuf(d, r.kc[l].Len()), byteBuf(d, r.vc[l].Len())
				}
				if r.ks != nil && r.ks[l] != (Buffer{}) {
					b.ks[l], b.vs[l] = d.NewBufferLen(r.ks[l].Len()), d.NewBufferLen(r.vs[l].Len())
				}
			}
			r.kvSlotBufs[s] = b
		}
	}
	r.finalNorm = NewBufferFloats(d, w.FinalNorm)
	if r.layerNorm && r.layerNormBias {
		r.finalNormBias = NewBufferFloats(d, w.FinalNormBias)
	}
	lm := &w.LMHead
	if lm.Rows() == 0 {
		lm = &w.Embed // tied
		r.lmTied = true
	}
	// The LM head is logit-critical and must stay int8: at int4 it flips the argmax and tanks the cosine (a tied
	// head dots every logit against it), which is why the decoder pins the embedding and LM head at int8 even in
	// int4 mode (decoder/weightmat.go). Worst for a tied head such as Gemma's, 262k x 2560.
	if r.lmW, r.lmS, err = int8BufA(d, alias, lm); err != nil {
		return nil, err
	}

	// The gate|up and down scratch must fit the widest FFN width: the dense I, or (for MoE) the
	// larger of the expert and shared-expert intermediate dims.
	guDim, maxFFNI := I, I // maxFFNI: the widest dense FFN (a Gemma 4 E-model's double-wide shared layers, S1.3)
	for l := range r.layers {
		maxFFNI = max(maxFFNI, r.layers[l].ffnI)
	}
	guDim = maxFFNI
	if r.moe != nil {
		guDim = max(guDim, max(r.moe.inter, r.moe.sharedInter))
	}
	if r.g4moe != nil { // Gemma-4 dense‖MoE: the gate|up/down scratch must fit BOTH the dense branch and the experts
		guDim = max(guDim, max(r.g4moe.denseInter, r.g4moe.moeInter))
	}
	// The SA-GEMV decode kernels (gemv_w4a8_sa, _sa_bias, _sa_resid and the MoE variants) derive the output row
	// from the runtime threadgroup size and, unlike gemv_w4a8_sa_bk, carry no `row >= N` guard. An output width
	// N%8 != 0 therefore makes the tail threadgroup rewrite an already-written row while the true tail rows stay
	// uninitialised scratch: plausible-looking wrong logits, no error. The attention widths (qDim = nH*hd, kvDim =
	// nKV*hd) are structurally %8 (hd is 64 or 128), so the risk is the model-level FFN widths below: decline any
	// that is not a multiple of 8, and the CPU path serves it. The deeper fix (an N parameter and a `row >= N`
	// guard in the SA-family and MoE kernels) would touch every dispatch binding and is not done.
	//
	// Vocab is not checked: it never routes through an SA-family kernel. The LM head is pinned int8 and dispatches
	// only gemv_w8a8_coal (forwardLogits) or gemv_w8a8_amax (ForwardArgmax). gemv_w8a8_coal addresses its row
	// directly via threadgroup_position_in_grid, correct whatever a threadgroup's uniformity. gemv_w8a8_amax uses
	// the same hazardous tgs-derived formula as the SA family, so ForwardArgmax routes a non-%8 vocab around it
	// (full logits plus host argmax) instead of declining the family; GPT-2 (50257) is the first family this
	// reaches.
	bad8 := func(name string, n int) error {
		if n%8 != 0 {
			return fmt.Errorf("metal: %s width %d is not a multiple of 8 — SA-GEMV tail-write hazard (audit C-10); use the CPU path", name, n)
		}
		return nil
	}
	// Include the attention widths too: the fused-QKV projection dispatches (nH*hd + 2*nKV*hd) rows through
	// pSABias, so both the q-width (nH*hd) and the kv-width (nKV*hd), and hence their sum, must be %8. "hd is 64
	// or 128" is not guaranteed for every admitted arch; one with hd%8 != 0 declines instead of corrupting.
	widthChecks := []error{bad8("hidden", H), bad8("intermediate", I)}
	// Check each layer's geom, not the maxima: a two-geom arch (Gemma 4 local/global) whose smaller q or kv width
	// is non-%8 while the larger is %8 would pass a max-only check yet corrupt that layer's SA-GEMV. Duplicate
	// widths across uniform layers are harmless.
	for l := range r.layers {
		if r.layers[l].delta != nil {
			continue // Gated-DeltaNet layer: no attention geometry at all — L.geom is nil by design
		}
		g := r.layers[l].geom
		widthChecks = append(widthChecks,
			bad8(fmt.Sprintf("layer %d attn q-width (nH·hd)", l), r.nH*g.hd),
			bad8(fmt.Sprintf("layer %d attn kv-width (nKV·hd)", l), g.kvDim))
		if fi := r.layers[l].ffnI; fi != 0 && fi != I {
			widthChecks = append(widthChecks, bad8(fmt.Sprintf("layer %d intermediate", l), fi))
		}
	}
	if r.moe != nil {
		widthChecks = append(widthChecks, bad8("MoE expert intermediate", r.moe.inter), bad8("MoE shared-expert intermediate", r.moe.sharedInter))
	}
	if r.g4moe != nil {
		widthChecks = append(widthChecks, bad8("gemma-4 dense intermediate", r.g4moe.denseInter), bad8("gemma-4 MoE intermediate", r.g4moe.moeInter))
	}
	for _, e := range widthChecks {
		if e != nil {
			return nil, e
		}
	}
	// The SA-GEMV and MoE kernels stage the contraction row into threadgroup memory (see
	// maxThreadgroupStageBytes). A dispatch whose threadgroup memory exceeds the device tile limit aborts the
	// command buffer, and the host would then read stale logits. Decline here so the caller falls back to CPU.
	// (Mixtral's inter=14336 is 28672 B, already 87% of a 32 KiB budget; inter >= 16384 exceeds it. The dense
	// down-proj is not counted: R18's staged kernel checks its own I-byte fit below, else coal.)
	moeInter, g4Inter := 0, 0
	if r.moe != nil {
		moeInter = r.moe.inter
	}
	if r.g4moe != nil {
		g4Inter = r.g4moe.moeInter
	}
	dnValueDim := 0
	if r.dnet != nil {
		dnValueDim = r.dnet.valueDim
	}
	if tg, lim := maxThreadgroupStageBytes(H, maxNHhd, moeInter, g4Inter, dnValueDim), d.MaxThreadgroupMemoryLength(); tg > lim {
		return nil, fmt.Errorf("metal: threadgroup staging needs %d B (2×K) > device tile-memory max %d B — declining to CPU (audit M-11)", tg, lim)
	}
	// R18: the dense decode GEMVs at R rows per simdgroup — 4 for qkv, gate/up and down, 2 for o: R18's confirmed
	// configuration (docs/measurements/metal-decode-gemv-r18-2026-09-26.md) with qkv moved to 4 by R18b's
	// (metal-decode-gemv-r18b-2026-09-26.md), whose kernel form changed what the best qkv width is. Each is taken only where every layer's row
	// count fills whole threadgroups (gemvRowsFor), and down only where its K = I bytes of staged activations fit the
	// threadgroup memory; anything else keeps the shipped kernel.
	{
		qkvR := 4
		for _, L := range r.layers {
			if L.geom != nil { // a DeltaNet layer has no attention geometry (and no qkv GEMV)
				rows := nH*L.geom.hd + 2*L.geom.kvDim
				if L.kvShared {
					rows = nH * L.geom.hd // Q only (S1.4)
				}
				qkvR = min(qkvR, gemvRowsFor(rows, 4))
			}
		}
		guR := gemvRowsFor(2*I, 4)
		for _, L := range r.layers { // every dense width must tile at R (halving R keeps the widths already tiled)
			if L.ffnI != 0 {
				guR = gemvRowsFor(2*L.ffnI, guR)
			}
		}
		r.gemvRows.qkv, r.gemvRows.o, r.gemvRows.gu = qkvR, gemvRowsFor(H, 2), guR
		// down stages K = the layer's FFN width of int8 activations: the widest must fit, and every width be %4.
		downOK := maxFFNI <= d.MaxThreadgroupMemoryLength()
		for _, L := range r.layers {
			if fi := max(L.ffnI, I); fi%4 != 0 {
				downOK = false
			}
		}
		if downOK {
			r.gemvRows.down = gemvRowsFor(H, 4)
		}
		rowsPipe := func(base string, R int) Pipeline {
			if R == 0 {
				return Pipeline{}
			}
			return pipe(fmt.Sprintf("%s%d", base, R))
		}
		r.pSABiasRows = rowsPipe("gemv_w4a8_sa_bias_rows", r.gemvRows.qkv)
		r.pSAResidRows = rowsPipe("gemv_w4a8_sa_resid_rows", r.gemvRows.o)
		r.pSARows = rowsPipe("gemv_w4a8_sa_rows", r.gemvRows.gu)
		r.pSABiasRowsAdj = rowsPipe("mc3_gemv_w4a8_sa_bias_rows_adj", r.gemvRows.qkv)
		r.pSARowsAdj = rowsPipe("mc3_gemv_w4a8_sa_rows_adj", r.gemvRows.gu)
		r.pGemvResidStaged = rowsPipe("gemv_w4a8_resid_staged", r.gemvRows.down)
		for _, R := range []int{2, 4} {
			r.gemvExtCoal[R] = pipe(fmt.Sprintf("gemv_w4a8_coal_staged%d", R))
			r.gemvExtSA[R] = pipe(fmt.Sprintf("gemv_w4a8_sa_rows%d", R))
			r.gemvExtResid[R] = pipe(fmt.Sprintf("gemv_w4a8_resid_staged%d", R))
		}
		r.gemvExtTGMax = d.MaxThreadgroupMemoryLength()
	}
	if r.w8Attn {
		// int4mix (slice 4): the attention projections' handles swap to their int8 twins (same arguments, same launch
		// shape); the FFN keeps its int4 kernels and R18 rows forms. w8AttnEligible admits only the families whose
		// qkv and o-proj run exactly these two handles.
		r.gemvRows.qkv, r.gemvRows.o = 0, 0
		r.pSABias, r.pSAResid = pipe("gemv_w8a8_sa_bias"), pipe("gemv_w8a8_sa_resid")
		for B := 2; B <= batchMaxSeqs; B++ {
			r.pW8Rows[B] = pipe(fmt.Sprintf("mc3_gemv_w8a8_rows%d", B))
		}
	}
	if r.w8 {
		// Each int8 twin takes its int4 kernel's arguments in the same order at the same launch shape, so swapping
		// the handles is the whole dispatch change. The R18 rows kernels read int4 only: off.
		r.gemvRows = struct{ qkv, o, gu, down int }{}
		r.pGemv, r.pGemvResid, r.pCoalBiasResid = pipe("gemv_w8a8_body"), pipe("gemv_w8a8_body_resid"), pipe("gemv_w8a8_body_resid_bias")
		r.pSA, r.pSABias, r.pSAResid = pipe("gemv_w8a8_sa"), pipe("gemv_w8a8_sa_bias"), pipe("gemv_w8a8_sa_resid")
		r.pSABiasResid = pipe("gemv_w8a8_sa_bias_resid")
		for B := 2; B <= batchMaxSeqs; B++ { // the batched step's projections, B rows per weight read (batch_rows.go)
			r.pW8Rows[B] = pipe(fmt.Sprintf("mc3_gemv_w8a8_rows%d", B))
		}
	}
	r.x = d.NewBufferLen(H)
	r.aq, r.aSc = byteBuf(d, H), d.NewBufferLen(1)
	if r.q4kLane { // the head's slot 0 holds the f32 final norm on this lane (gemv_w8f32_head)
		r.aq = byteBuf(d, 4*H)
	}
	r.qkv = d.NewBufferLen(maxNHhd + 2*maxKvDim) // fused [q | k | v], sized to the widest layer
	r.gu = d.NewBufferLen(2 * guDim)             // fused [gate | up]
	r.ctx, r.cq, r.cSc = d.NewBufferLen(maxNHhd), byteBuf(d, maxNHhd), d.NewBufferLen(1)
	// attnFAPartial is independent of context depth (nSplit is capped, not depth-proportional; see
	// canUseAttnFA/attnFASplitFor), so it is sized once here, not per token. It is zero-size on a model this
	// kernel can never engage for (no layer attnFAHeadDimOK admits): canUseAttnFA's hd guard then always declines,
	// so the zero buffer is never dispatched into.
	r.attnFAMaxSplit = 32
	// For the group sizes it was graded at (G = 6 and 7), attention_fa's first pass is the block-of-32 kernel
	// attention_fa_blk at the fixed split count attnFABlkSplit (same grid shape, partial layout and combine). G =
	// 2 to 8 take it too while attnFABlkAnyG; every other group size keeps attention_fa and its core-count rule.
	if r.attnFANKV > 0 {
		switch g := r.nH / r.attnFANKV; {
		case r.layers[r.attnFALayer].geom.hd == 64: // B-P01: attnFAHeadDimOK admitted hd = 64 only where blk64 exists
			r.pAttnFA = pipe(fmt.Sprintf("attention_fa_blk64_g%d", g))
			r.pAttnFARows = pipe(fmt.Sprintf("mc3_attention_fa_blk64_rows_g%d", g))
			r.attnFABlkSplit = attnFABlkSplit
		case g == 6 || g == 7, attnFABlkAnyG && g >= 2 && g <= 8:
			r.pAttnFA = pipe(fmt.Sprintf("attention_fa_blk_g%d", g))
			r.pAttnFARows = pipe(fmt.Sprintf("mc3_attention_fa_blk_rows_g%d", g))
			r.attnFABlkSplit = attnFABlkSplit
		}
	}
	if maxAttnFAPartialElems > 0 {
		r.attnFAPartialLen = maxAttnFAPartialElems * r.attnFAMaxSplit
		r.attnFAPartial = d.NewBufferLen(r.attnFAPartialLen)
		r.uAttnFAG, r.uAttnFANSplit = NewBufferU32(d, 0), NewBufferU32(d, 0)
	}
	r.oO, r.mq, r.mSc = d.NewBufferLen(H), byteBuf(d, H), d.NewBufferLen(1)
	// R1 W4F16 lane: half-typed twins of aq/mq/cq (2 bytes/elem, no separate scale buffer — the
	// f16 activation values are already the true values). Always allocated ("one binary carries
	// both arms"); only used when decodeLaneW4F16 dispatches into them.
	r.axF16, r.mxF16, r.cxF16 = byteBuf(d, 2*H), byteBuf(d, 2*H), byteBuf(d, 2*maxNHhd)
	if r.q4kLane {
		r.axF32, r.swF32 = d.NewBufferLen(H), d.NewBufferLen(guDim)
	}
	r.dq, r.dSc, r.dO = byteBuf(d, guDim), d.NewBufferLen(1), d.NewBufferLen(H)
	r.logits = d.NewBufferLen(V)
	// Ceil, not floor: ForwardArgmax dispatches V*32 threads = ceil(V/8) threadgroups and the amax kernel writes
	// part[tgid] unconditionally, so a floor V/8 would under-size r.part (an 8-byte write past it, on UMA into an
	// adjacent buffer) and leave the last tile out of uP's reduce, so the greedy token could differ from
	// argmax(Forward).
	nTiles := (V + 7) / 8 // one (maxLogit,rowIdx) partial per threadgroup (8 rows)
	r.part, r.tok, r.uP = d.NewBufferLen(nTiles*2), d.NewBufferLen(1), NewBufferU32(d, uint32(nTiles))
	// Model-level rope table for the (uniform-only) prefill path; decode uses each layer's L.invf.
	// GPT-2 (FeatLearnedPos): no RoPE at all, so this is empty — NewBufferFloats panics on empty;
	// r.invf stays unused (prefillOK is false for a learned-pos family — see prefillFeatures).
	if invf0 := m.RopeInvFreq(); len(invf0) > 0 {
		r.invf = NewBufferFloats(d, invf0)
	}
	r.uH, r.uI = NewBufferU32(d, uint32(H)), NewBufferU32(d, uint32(I))
	for l := range r.layers {
		L := &r.layers[l]
		if L.ffnI == 0 || L.ffnI == I || g4OneFFNWidthForTest { // the seam: G2 defect (4)
			L.ffnI, L.uFFNI = I, r.uI
		} else {
			L.uFFNI = NewBufferU32(d, uint32(L.ffnI))
		}
	}
	r.uNH = NewBufferU32(d, uint32(nH)) // query heads: constant across a family, so model-level
	r.uScale, r.uEps = NewBufferFloats(d, []float32{m.AttnScale()}), NewBufferFloats(d, []float32{m.NormEps()})
	r.uPos, r.uNKeys, r.uRopePos = NewBufferU32(d, 0), NewBufferU32(d, 1), NewBufferU32(d, 0)
	r.attnTempBeta, r.attnTempOrigMaxPos = m.AttnTempParams()
	r.uQTempScale = NewBufferFloats(d, []float32{1}) // identity; overwritten per call alongside uPos
	if r.qkNormWhole {
		// Olmo 3/Olmo Hybrid (G5): encodeAttention's whole-vector qk_norm dispatch needs a
		// "one reduction group" nH/nKV and the FULL width nH*hd instead of the per-head hd —
		// both MHA-uniform-geometry-only (BuildResident already declines otherwise above), so
		// one pair of model-level buffers suffices; no per-layer variant needed.
		r.uQKWholeOne = NewBufferU32(d, 1)
		r.uQKWholeHD = NewBufferU32(d, uint32(nH*m.HeadDimAtResident(0)))
	}
	// Scale-less v_norm plumbing for K=V layers (Gemma 4 globals): a unit-1.0 weight and a shared
	// zero (nH=0 / nHhd=0 / addOne=0) so qk_norm reduces to x·rms·1 on the V slot. Sized to the
	// widest head; harmless (a few KB) for models with no K=V layer.
	r.uZero = NewBufferU32(d, 0)
	ones := make([]float32, maxHd)
	for i := range ones {
		ones[i] = 1
	}
	r.vNormUnit = NewBufferFloats(d, ones)
	// HF and the CPU apply Gemma 4's scale-less v_norm on EVERY layer that owns its K/V, not only on K=V layers
	// (S1.0); a K=V layer's V slot holds the raw k_proj output, any other's the v_proj output, and both take it.
	r.g4VNorm = m.IsGemma4Resident()
	if r.dnet != nil {
		// qGate scratch (Gated-DeltaNet family's softmax layers): sized to the widest layer's
		// qDim, same maxNHhd every other per-token scratch buffer uses — this family's softmax
		// layers are uniform, so in practice one size, but sizing from the max stays consistent
		// with how every other per-token buffer in this build is sized.
		r.dnQg = d.NewBufferLen(2 * maxNHhd)
		r.dnAGate = d.NewBufferLen(maxNHhd)
	}
	r.finalSoftcap = m.FinalLogitSoftcapResident() // Gemma 4: 30 (host-side softcap); 0 for every other family
	r.embedScale = float32(m.EmbedScaleResident()) // Gemma: √hidden; 0 for every non-scaled family (G-02)
	r.logitsHost = make([]float32, V)
	r.ensureBatchCap(16)

	// Residency set (on by default when supported and paged; GOINFER_MOE_RESIDENCY=0 opts out). The paged path
	// submits per layer, and the pread stage CPU-writes the slot buffers each token, dirtying their residency so
	// the driver re-validates them on every phase-2 commit. Pinning the slot pool resident holds it across those
	// writes. Slots only: pinning anything more (weights, KV, scratch) regresses phase 1 in proportion to the
	// pin-set size, read/write-agnostic; pinning helps only the pread-invalidated buffers. Footprint: the slot
	// pool is N x MoE-layers x per-expert-bytes, permanently requested resident, so re-measure the win if N grows
	// or the box is under other load. Capability-gated (macOS 15+); older OSes keep the correct, slower per-submit
	// path. It gates on any paged MoE, generic (moe.go) or Gemma 4 (g4moe): both have pread-invalidated slot
	// buffers.
	paged := (r.g4moe != nil && r.g4moe.paged) || (r.moe != nil && r.moe.paged)
	if paged && modelKnob(m, "GOINFER_MOE_RESIDENCY") != "0" && ResidencySetsSupported() {
		rs, rerr := d.NewResidencySet()
		if rerr != nil {
			fmt.Fprintf(os.Stderr, "metal: residency set unavailable (%v) — per-submit validation stands\n", rerr)
		} else {
			slots := r.slotBuffers()      // GPU-read-only (CPU-written by pread)
			written := r.writtenBuffers() // GPU-WRITTEN per token: KV cache + intermediates/scratch
			kv := r.kvBuffers()
			scratch := r.scratchBuffers()
			// pinned accumulates exactly what addAll() adds, so r.residencyBufs is the true pinned list in every
			// slot-scoped arm, not just the default (the teardown-consistency gate reads it).
			var pinned []Buffer
			addAll := func(bs []Buffer) {
				for _, b := range bs {
					rs.Add(b)
					pinned = append(pinned, b)
				}
			}
			switch modelKnob(m, "GOINFER_MOE_RESIDENCY_SCOPE") {
			case "slots":
				addAll(slots)
			case "slots+kv": // bisect: does the KV cache cause the phase-1 regression?
				addAll(slots)
				addAll(kv)
			case "slots+scratch": // bisect: do the intermediates cause it?
				addAll(slots)
				addAll(scratch)
			case "readonly", "slots+weights": // diagnostic: pin all buffers EXCEPT the written set
				rs.AddAllDeviceBuffersExcept(d, written) // whole-device: not slot-scoped, pinned stays nil
			case "all": // diagnostic: pin every device buffer (regresses phase 1 — set-size overhead)
				rs.AddAllDeviceBuffers(d) // whole-device: not slot-scoped, pinned stays nil
			default: // SHIP DEFAULT: slots only. The five-arm bisect showed pinning anything BEYOND the
				// pread-invalidated slot buffers regresses phase 1 in proportion to the pinned set size
				// (read/write-agnostic), so slots-only is the sole net win (phase 2 idle 9→0.44 ms/CB).
				addAll(slots)
				_ = kv
				_ = scratch
			}
			r.residencyBufs = pinned
			rs.Commit()
			rs.RequestResidency()
			// Attach the set per encoder, only on phase 2's command buffers (the callers of encodeG4Phase2Paged and
			// encodeMoEExpertsPaged, via Encoder.UseResidencySet, aikit gpu/v0.33.1+), the buffer category that reads it,
			// not at the queue (r.q.AddResidencySet): at the queue it rides on every command buffer, and phase 1 carries
			// the pinned slots in its referenced set though it never touches them.
			r.residency = rs
		}
	}

	r.alias = alias
	if line := alias.summary(); line != "" {
		fmt.Fprint(os.Stderr, line)
	}
	// S16: the m-RoPE prefill's per-pair axis table, from the decoder's own layout rule; nil for a model without a
	// three-way m-RoPE section, whose PrefillMRoPELast then declines.
	if g := r.prefillGeom(); g != nil && r.prefillOK {
		if ax := m.MRopeAxisResident(g.half); ax != nil {
			u := make([]uint32, len(ax))
			for i, v := range ax {
				u[i] = uint32(v)
			}
			r.mropeAxis = NewBufferUint32s(d, u)
		}
	}
	r.buildBatch() // MC3: the batched decode step, when this resident can run one (batch.go)
	ok = true      // construction complete — the resident owns everything; Close (not the defer) frees it
	return r, nil
}

// loadEmbedRow dequantizes token id's embedding into the shared input buffer and applies the arch's embedding scale
// (Gemma's sqrt(hidden), FeatEmbedScale; a no-op, <= 1, for every other family): the complete
// token-id-to-layer-0-input step. It exists so the two id-taking entry points (Forward, ForwardArgmax) cannot disagree
// with production, where decoder.embedResident applies the scale on the ForwardEmb path. The snapshot golden drives
// exactly these two methods, so a regression confined to this seam would be invisible to the Metal suite's one
// absolute gate.
func (r *resident) loadEmbedRow(id, pos int) {
	dst := r.x.Floats()
	r.embed.Row(id, dst) // CPU dequant embedding into the shared buffer
	if r.embedScale > 1 {
		for i := range dst {
			dst[i] *= r.embedScale
		}
	}
	r.addLearnedPos(pos)
}

// addLearnedPosTo adds wpe[pos] to dst: GPT-2's learned absolute position embedding, host-side (CPU dequant, like the
// token embedding lookup) since it happens once per token before any GPU dispatch, mirroring decoder/model.go's
// ResidentForward (arch.LearnedPosEmbed: h[i] += wpe[pos][i]). A no-op for every family without FeatLearnedPos.
// addLearnedPos applies it to the input embedding already in r.x.
func (r *resident) addLearnedPosTo(dst []float32, pos int) {
	if !r.learnedPos {
		return
	}
	pe := make([]float32, len(dst))
	r.posEmbed.Row(pos, pe)
	for i := range dst {
		dst[i] += pe[i]
	}
}

func (r *resident) addLearnedPos(pos int) {
	r.addLearnedPosTo(r.x.Floats(), pos)
}

// ensureBatchCap ensures r.batchX, r.batchLogits, and per-token uniform buffers have capacity for at least n tokens.
func (r *resident) ensureBatchCap(n int) {
	if n <= r.batchCap {
		return
	}
	c := 16
	for c < n {
		c *= 2
	}
	r.batchX = r.d.NewBufferLen(c * r.H)
	r.batchLogits = r.d.NewBufferLen(c * r.V)
	for i := len(r.batchUPos); i < c; i++ {
		r.batchUPos = append(r.batchUPos, NewBufferU32(r.d, 0))
		r.batchUNKeys = append(r.batchUNKeys, NewBufferU32(r.d, 1))
		r.batchUQTempScale = append(r.batchUQTempScale, NewBufferFloats(r.d, []float32{1}))
	}
	r.batchCap = c
}

// setPos writes pos/nKeys/qTempScale into their uniform buffers in place — called at every
// forward entry point right before encoding or committing, mirroring uPos/uNKeys's existing
// per-call-mutation pattern. qTempScale (Ministral 3, FeatAttnTemp) is the SAME formula
// decoder.Model.AttnTempScale computes host-side; recomputed here rather than round-tripping
// through the decoder package because resident has no *decoder.Model reference, only the raw
// params fetched once at build time (attnTempBeta/attnTempOrigMaxPos). attnTempBeta==0 (every
// family without this feature) always gives exactly 1 without evaluating the division at all —
// same guard as decoder/attention.go's sequential path, load-bearing: attnTempOrigMaxPos is 0
// for those families, and pos/0 would poison every Q with NaN otherwise.
func (r *resident) setPos(pos int, ropePos ...int) {
	rp := pos
	if len(ropePos) > 0 {
		rp = ropePos[0]
	}
	r.curNKeys = pos + 1
	r.writePosUniforms(posUniforms{r.uPos, r.uNKeys, r.uRopePos, r.uQTempScale, r.uAttnFANSplit}, pos, rp)
}

// posUniforms is one set of the per-token uniforms a decode command buffer binds: setPos writes the resident's own
// set, and the greedy chain (greedy_chain.go) alternates two of its own so a buffer can be committed while the one
// before it still reads its set.
type posUniforms struct{ uPos, uNKeys, uRopePos, uQTempScale, uFANSplit Buffer }

// writePosUniforms writes token pos's values into set u: the one formula for both setPos and the chain.
func (r *resident) writePosUniforms(u posUniforms, pos, rp int) {
	u.uPos.SetU32(uint32(pos))
	u.uNKeys.SetU32(uint32(pos + 1))
	u.uRopePos.SetU32(uint32(rp))
	// uAttnFAG and uAttnFANSplit are written here, not at the per-layer dispatch site (see their field comment):
	// this happens before this buffer commits, not during the next buffer's encode-ahead, the same argument
	// uPos/uNKeys/uRopePos above rely on. attnFANKV == 0 means no layer is attention_fa-eligible: the buffers stay
	// unused (canUseAttnFA gates on it too), but skip the divide by zero regardless.
	if r.attnFANKV > 0 {
		r.uAttnFAG.SetU32(uint32(r.nH / r.attnFANKV))
		u.uFANSplit.SetU32(uint32(r.attnFASplitFor(pos+1, r.attnFANKV)))
	}
	scale := float32(1)
	if r.attnTempBeta != 0 {
		scale = float32(1 + r.attnTempBeta*math.Log1p(math.Floor(float64(pos)/r.attnTempOrigMaxPos)))
	}
	u.uQTempScale.Floats()[0] = scale
}

// Forward runs token `id` at absolute position `pos` and returns logits[V]. The whole
// layer stack + LM head is encoded into ONE command buffer, one commit/wait.
func (r *resident) Forward(id, pos int) []float32 {
	// Pin to one OS thread for the whole call: the NSAutoreleasePool (begin/end) is
	// per-OS-thread, and Go can migrate goroutines mid-call — draining a pool on a
	// different thread than it was pushed is UB (intermittent SIGSEGV). Same discipline
	// the CUDA backend's LockOSThread executor uses.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	r.loadEmbedRow(id, pos)
	return r.forwardLogits(pos)
}

// ForwardEmb is Forward given a precomputed embedding[H] (the decoder.ResidentForward shape) instead of a token id: it
// copies the embedding into the shared input buffer and skips the internal lookup. This is the production path:
// decoder.embedResident does the lookup and applies any embed scale before calling in. Numerically identical to
// Forward(id, pos) when emb is the scaled embedding row (Forward applies the scale in loadEmbedRow).
func (r *resident) ForwardEmb(emb []float32, pos int) []float32 {
	return r.ForwardEmbMRoPE(emb, pos, pos)
}

// ForwardEmbMRoPE is ForwardEmb with decoupled rotation position (ropePos) and KV cache position (pos).
func (r *resident) ForwardEmbMRoPE(emb []float32, pos, ropePos int) []float32 {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	r.loadEmb(emb)
	r.addLearnedPos(pos)
	if r.g4moe != nil && r.g4moe.paged { // synchronous expert paging: per-layer submit+wait, staged experts
		return r.forwardLogitsPaged(pos, ropePos)
	}
	if r.moe != nil && r.moe.paged { // generic MoE's twin (moe.go) — same mechanism, generalized
		return r.forwardLogitsMoEPaged(pos, ropePos)
	}
	return r.forwardLogits(pos, ropePos)
}

// forwardHiddenNoHead encodes the trunk (all layers plus final norm) for one token at absolute position pos, writing
// this position's K/V into the resident cache as ForwardEmb does, but never dispatching the LM head:
// decoder.ResidentHiddenLast never needs logits, and the head is the most expensive matmul in a forward. want is false
// for every position but the last of a HiddenLast sequence: those still need their K/V written (attention over the
// whole sequence, matching hiddenLastBatched's causal chain) but their hidden state is never read, so nil comes back
// without the int8-to-float32 dequant loop.
// The returned hidden state is exactly what feeds the head on the decode path (encodeNorm's rmsnorm_quant or
// layernorm_quant output in r.aq/r.aSc), dequantized; it matches forwardHeadForTest's act.
func (r *resident) forwardHiddenNoHead(emb []float32, pos int, want bool) ([]float32, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	r.loadEmb(emb)
	r.addLearnedPos(pos)
	r.setPos(pos)
	e := r.q.Begin()
	r.encodeTrunkInto(e) // layers → final norm → r.aq/r.aSc; no head dispatch
	e.End()
	r.recordExecErr(e.Err()) // C-09
	if !want {
		return nil, nil
	}
	q, sc := r.aq.Int8s(), r.aSc.Floats()[0]
	out := make([]float32, r.H)
	for i := range out {
		out[i] = float32(q[i]) * sc
	}
	return out, nil
}

// forwardLogits encodes the trunk + full lm head and reads back logits[V]. Caller must hold
// the OS thread and have filled r.x with the input embedding.
func (r *resident) forwardLogits(pos int, ropePos ...int) []float32 {
	rp := pos
	if len(ropePos) > 0 {
		rp = ropePos[0]
	}
	r.setPos(pos, rp)
	e := r.q.Begin()
	r.encodeTrunkInto(e)                                                           // 28 layers → final norm → r.aq/r.aSc
	e.Dispatch(r.pGemvW8, (r.V)*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.logits, r.uH) // full lm head (int8 — logit-critical)
	e.End()
	r.recordExecErr(e.Err()) // C-09
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
func (r *resident) finalizeLogits() {
	copy(r.logitsHost, r.logits.Floats())
	if r.finalSoftcap > 0 {
		softcapParallel(r.logitsHost, r.finalSoftcap)
	}
	// Cohere/Command-R's logits_scaling (FeatLogitScale): a plain host-side multiply, the same
	// site and shape as the softcap just above. 0 or 1 for every non-FeatLogitScale family
	// (Model.LogitScaleResident's own no-op value) — a single serial pass, since a multiply is
	// memory- not compute-bound at any vocab size this repo has seen (unlike softcap's tanh).
	if r.logitScale != 0 && r.logitScale != 1 {
		for i, v := range r.logitsHost {
			r.logitsHost[i] = v * r.logitScale
		}
	}
}

// softcapParallel applies Gemma's final-logit softcap sc*tanh(x/sc) in place, across cores. Every element is
// independent and math.Tanh is deterministic, so the split is byte-identical to the serial form (disjoint writes, no
// reduction or ordering; TestMetalSoftcapParallel_bitIdentical). Serial below the goroutine-spawn threshold (as
// parallelF32ToF16).
// It does not skip on greedy decode, although argmax is softcap-invariant: finalizeLogits calls it whenever
// r.finalSoftcap > 0, on every execLoop token whatever the sampling mode. The paths that skip it never read
// r.logitsHost (ForwardArgmax, and the greedy chain, which greedyChainWhyNot refuses for an embed-scaled family), so a
// Gemma family pays this 262k-wide tanh loop on every decode token regardless of temperature.
func softcapParallel(logits []float32, softcap float32) {
	sc := softcap
	n := len(logits)
	workers := min(runtime.GOMAXPROCS(0), 8)
	if n < 8192 || workers <= 1 {
		for j, v := range logits {
			logits[j] = sc * float32(math.Tanh(float64(v/sc)))
		}
		return
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for j := lo; j < hi; j++ {
				logits[j] = sc * float32(math.Tanh(float64(logits[j]/sc)))
			}
		}(lo, hi)
	}
	wg.Wait()
}

func (r *resident) ensureExec() {
	if r.execReq == nil {
		r.execReq = make(chan execJob)
		r.execAck = make(chan []float32)
		r.execDone = make(chan struct{})
		go r.execLoop()
	}
}

// ForwardEmbPipe is ForwardEmb through the pipelined executor (encode-ahead) — the production
// decode path. Returns logits[V] (reused buffer; consume before the next call). Synchronous to
// the caller (one job in, one logits out), but the executor overlaps the next token's encode
// with this token's GPU execution.
func (r *resident) ForwardEmbPipe(emb []float32, pos int) []float32 {
	return r.ForwardEmbMRoPEPipe(emb, pos, pos)
}

// ForwardEmbMRoPEPipe is ForwardEmbMRoPE through the pipelined executor (encode-ahead).
func (r *resident) ForwardEmbMRoPEPipe(emb []float32, pos, ropePos int) []float32 {
	if r.g4moe != nil && r.g4moe.paged {
		// Paging tears each MoE layer into two submits with a host readback between — the encode-ahead
		// executor (one static command buffer/token) cannot express it. Fall back to the synchronous
		// paged path; there is no pipelining to lose (Step-0: paging is submit-bound, not encode-bound).
		return r.ForwardEmbMRoPE(emb, pos, ropePos)
	}
	if r.moe != nil && r.moe.paged { // generic MoE's twin — same reasoning, same fallback
		return r.ForwardEmbMRoPE(emb, pos, ropePos)
	}
	r.ensureExec()
	r.execReq <- execJob{emb: emb, pos: pos, ropePos: ropePos, noHead: false}
	return <-r.execAck
}

// ForwardEmbNoLogitsPipe is ForwardNoLogits through the pipelined executor (encode-ahead) —
// overlapping token t+1's trunk encode with token t's GPU execution while skipping the LM head
// dispatch and logits readback.
func (r *resident) ForwardEmbNoLogitsPipe(emb []float32, pos int) {
	if (r.g4moe != nil && r.g4moe.paged) || (r.moe != nil && r.moe.paged) {
		// The synchronous paged path (no executor, ForwardEmbMRoPEPipe's reason), with the final norm and LM head
		// skipped: a prompt token's logits are discarded, and on M26 the head reads ~0.74 GB of int8 a token
		// (task-m26-mac-2026-10.md). The layers, and so the KV, are the same calls either way.
		r.pagedNoHead = true
		r.ForwardEmb(emb, pos)
		r.pagedNoHead = false
		return
	}
	r.ensureExec()
	r.execReq <- execJob{emb: emb, pos: pos, ropePos: pos, noHead: true}
	<-r.execAck
}

// encodeTrunkCB builds an uncommitted command buffer with only the trunk (no LM head),
// autoreleasing into the executor's long-lived pool.
func (r *resident) encodeTrunkCB() *Encoder {
	e := r.q.BeginNP()
	r.encodeTrunkInto(e)
	e.FinishEncoding()
	return e
}

// encodeLogitsCB builds a complete, un-committed command buffer (trunk + full lm head) with no
// per-call pool — it autoreleases into the executor's long-lived pool.
func (r *resident) encodeLogitsCB() *Encoder {
	e := r.q.BeginNP()
	r.encodeTrunkInto(e)
	e.Dispatch(r.pGemvW8, (r.V)*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.logits, r.uH)
	e.FinishEncoding()
	return e
}

// execLoop is the pinned executor: pipeline commit(t), pre-encode(t+1), wait(t). One shared autorelease pool, drained
// every drainEvery tokens, with a one-token non-overlapped hiccup so no un-committed command buffer is live across the
// drain (which keeps the pool LIFO-safe).
//
// A buffer is encoded for a key count, not just a head mode: the attention plan (attnPlan) is baked in at encode time.
// Each buffer is encoded for the job it will run (the current job's own key count when encoded fresh, pos+2 when
// pre-encoded for the predicted next job), and a pre-encoded buffer whose plan is not the arriving job's is dropped
// uncommitted and re-encoded. So a new request, the attnFADepthFloor crossing, or a toggle can never run another
// position's plan (TestExecutorAttnPlan; docs/measurements/metal-decode-attn-r17-2026-09-25.md).
func (r *resident) execLoop() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	defer close(r.execDone) // Close waits here: no command buffer is live once this returns
	const drainEvery = 64
	pool := NewARPool()
	var cur *Encoder
	var curNoHead bool
	var curPlan attnPlan
	count := 0
	encodeFor := func(noHead bool, nKeys int) *Encoder {
		r.encNKeys = nKeys
		defer func() { r.encNKeys = 0 }()
		if noHead {
			return r.encodeTrunkCB()
		}
		return r.encodeLogitsCB()
	}
	for job := range r.execReq {
		want := r.attnPlanFor(job.pos + 1)
		if cur != nil && (curNoHead != job.noHead || curPlan != want) {
			// Pre-encoded for a different head mode or attention plan than this job needs: drop it
			// uncommitted, with the pool it was encoded into.
			pool.Drain()
			pool = NewARPool()
			cur = nil
			count = 0
		}
		if cur == nil {
			cur, curNoHead, curPlan = encodeFor(job.noHead, job.pos+1), job.noHead, want
		}
		r.loadEmb(job.emb)       // this token's embedding + pos (set at commit time, not encode)
		r.addLearnedPos(job.pos) // GPT-2: += wpe[pos], same commit-time placement as the copy above
		r.setPos(job.pos, job.ropePos)
		cur.Commit()

		count++
		drain := count%drainEvery == 0
		var next *Encoder
		var nextNoHead bool
		var nextPlan attnPlan
		if !drain {
			next, nextNoHead, nextPlan = encodeFor(job.noHead, job.pos+2), job.noHead, r.attnPlanFor(job.pos+2)
		}
		cur.WaitDone()
		r.recordExecErr(cur.Err()) // C-09: set BEFORE the execAck send so the adapter's read is ordered
		r.gpuStart, r.gpuEnd, r.kernStart, r.kernEnd = cur.GPUStart(), cur.GPUEnd(), cur.KernStart(), cur.KernEnd()
		if !job.noHead {
			r.finalizeLogits()
		}

		if drain { // no un-committed cb live now → safe to drain the shared pool
			pool.Drain()
			pool = NewARPool()
			cur = nil
		} else {
			cur, curNoHead, curPlan = next, nextNoHead, nextPlan
		}
		if job.noHead {
			r.execAck <- nil
		} else {
			r.execAck <- r.logitsHost
		}
	}
	pool.Drain()
}

// stopExec shuts down the executor goroutine (if started) and blocks until it has returned, which is what makes
// freeing safe: closing the channel only signals, the loop may still be waiting on an in-flight command buffer, and
// releasing a buffer it references is a use-after-free.
func (r *resident) stopExec() {
	r.stopChain() // an open greedy chain has command buffers in flight on r.q too (greedy_chain.go)
	if r.execReq != nil {
		close(r.execReq)
		r.execReq = nil
		<-r.execDone
	}
}

// slotBuffers returns the paged MoE slot-pool buffers, g4moe's and the generic MoE's: GPU-read-only (phase 2 reads
// them; the pread stage CPU-writes their contents), so safe to pin resident. Both pools hold the buffer category the
// residency-set bisect found the win in (the pread-invalidated slot buffers; see buildResident).
func (r *resident) slotBuffers() []Buffer {
	var out []Buffer
	for l := range r.layers {
		// A pool's storage is one contiguous buffer per field (not N per-slot objects), so pinning it is one 4-buffer
		// add per pool regardless of N.
		if p := r.layers[l].g4moe; p != nil && p.pool != nil {
			out = append(out, p.pool.guW, p.pool.guS, p.pool.dW, p.pool.dS)
		}
		if p := r.layers[l].moe; p != nil && p.pool != nil {
			out = append(out, p.pool.guW, p.pool.guS, p.pool.dW, p.pool.dS)
		}
	}
	return out
}

// kvBuffers returns the per-layer KV cache buffers — GPU-WRITTEN by attention every token.
func (r *resident) kvBuffers() []Buffer {
	out := make([]Buffer, 0, len(r.kc)+len(r.vc)+len(r.ks)+len(r.vs))
	out = append(out, r.kc...)
	out = append(out, r.vc...)
	out = append(out, r.ks...)
	out = append(out, r.vs...)
	return out
}

// scratchBuffers returns the per-token GPU-WRITTEN intermediates (attention/MLP staging, MoE scratch,
// logits, and the per-token uniform buffers). NOT the KV cache (see kvBuffers).
func (r *resident) scratchBuffers() []Buffer {
	out := []Buffer{
		r.x, r.aq, r.aSc, r.ctx, r.cq, r.cSc, r.oO, r.mq, r.mSc, r.dq, r.dSc, r.dO,
		r.logits, r.qkv, r.gu, r.part, r.tok, r.uPos, r.uNKeys, r.uRopePos,
	}
	if g := r.g4moe; g != nil {
		out = append(out, g.rLogits, g.rIdx, g.rWgt, g.g4x1, g.g4x2, g.g4rn)
	}
	return out
}

// writtenBuffers is every GPU-written buffer (KV cache + scratch) — the set that MUST be excluded when
// pinning "all read-only" buffers (pinning a written member regresses the writing phase).
func (r *resident) writtenBuffers() []Buffer {
	return append(r.kvBuffers(), r.scratchBuffers()...)
}

// kvSlotBuf is one resident KV slot's per-layer buffers (MC1).
type kvSlotBuf struct{ kc, vc, ks, vs []Buffer }

// kvHostOff is where the bound slot's KV starts, in elements of elemBytes, within layer l's buffer contents as the host
// sees them (Floats/U16s start at the allocation, not at a view's offset). 0 unless the slots are one allocation.
func (r *resident) kvHostOff(l, elemBytes int) int {
	if !r.kvContig {
		return 0
	}
	return r.kvSlot * r.kvSlotBytes[l] / elemBytes
}

// kvSlotCount is how many resident KV slots were allocated (at least 1).
func (r *resident) kvSlotCount() int { return max(1, len(r.kvSlotBufs)) }

// useKVSlot binds resident KV slot i: every later encode reads and writes its buffers. The executor is stopped first,
// because a command buffer it pre-encoded for the next token baked in the previous slot's buffers (execLoop encodes
// token t+1 while t runs).
func (r *resident) useKVSlot(i int) error {
	if i < 0 || i >= r.kvSlotCount() {
		return fmt.Errorf("metal: KV slot %d out of range (%d allocated)", i, r.kvSlotCount())
	}
	if len(r.kvSlotBufs) == 0 || i == r.kvSlot {
		return nil
	}
	r.stopExec()
	b := r.kvSlotBufs[i]
	r.kc, r.vc, r.ks, r.vs = b.kc, b.vc, b.ks, b.vs
	r.kvSlot = i
	return nil
}

// Close stops the executor, waits for it, then releases every MTLBuffer this resident allocated, so it must be called
// once the model is unloaded: purego has no ARC and Metal has no context-destroy to reclaim in bulk, and cmd/serve loads
// and unloads models at runtime, so an unclosed resident leaks its weights, per-layer KV and MoE experts until the
// process exits. It returns error to satisfy io.Closer and match cuda/gpu's Close() error; teardown cannot fail, so it
// always returns nil. Idempotent: ReleaseAll empties the ledger, so a second Close is a no-op.
//
// Close must not run while a request still holds the model: that is a use-after-free (on CUDA, a driver SIGSEGV that
// kills the server). Serve's unload drains in-flight holders first (internal/serveapp/admin.go, handleAdminUnload;
// docs/completed/task-admin-unload-drain.md); keep that ordering for any new caller.
func (r *resident) Close() error {
	if r.ctxCeilKey != nil {
		metalCtxCeiling.Delete(r.ctxCeilKey) // S18: a ceiling must not keep its model alive past the resident
		r.ctxCeilKey = nil
	}
	r.stopExec()
	if r.g4moe != nil && r.g4moe.giwFile != nil {
		_ = r.g4moe.giwFile.Close() // the pread-staging fd (GOINFER_MOE_PREAD)
		r.g4moe.giwFile = nil
	}
	if r.moe != nil && r.moe.giwFile != nil {
		_ = r.moe.giwFile.Close() // the generic-MoE path's twin (moe.go)
		r.moe.giwFile = nil
	}
	if r.d != nil {
		r.d.ReleaseAll()     // every MTLBuffer
		r.d.ReleaseObjects() // command queue, ~40 pipelines, 1-2 libraries, and the MTLDevice (M24b)
	}
	return nil
}

// LastGPUTimes returns, for the last Forward, the GPU-busy window and the kernel window
// (incl scheduling) in seconds — GPUEnd-GPUStart and kernelEnd-kernelStart from the command
// buffer. wall - gpuBusy is the per-token host bubble (Step 0 of the headroom decision tree).
func (r *resident) LastGPUTimes() (gpuBusy, kernTotal float64) {
	return r.gpuEnd - r.gpuStart, r.kernEnd - r.kernStart
}

// ForwardArgmax runs the identical trunk but replaces the full lm head and logits readback with the fused block-argmax
// (per-tile (maxLogit, rowIdx), then argmax_finish, then a 4-byte token). It returns argmax(Forward's logits), first
// max wins, without materializing the logit vector. Production greedy decode does not call it: Metal has no
// ResidentGreedy implementation, so greedy runs through the greedy chain (ResidentGreedyChain, greedy_chain.go) when
// it is available and otherwise through the full-logits ForwardEmbPipe with a host argmax. Only tests and gates call
// this.
func (r *resident) ForwardArgmax(id, pos int) uint32 {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	r.loadEmbedRow(id, pos) // includes the arch embed scale — see loadEmbedRow (G-02)
	if r.V%8 != 0 {
		// gemv_w8a8_amax derives its output row from tgid*(tgs>>5)+sgid, as the SA-family decode kernels do (see
		// buildResident's width checks): dispatchThreads: makes the last threadgroup non-uniform whenever the dispatch
		// total is not a multiple of the threadgroup size, and that formula silently miscomputes for a non-uniform
		// group, rewriting an already-written row while the true tail row is never computed. Padding the weight is not
		// a safe workaround here: a padding row's score depends on the activation, which is unknown at load time, so
		// it could win the argmax. Route a non-%8 vocab (GPT-2's 50257 is the first) through the full-logits path
		// instead: gemv_w8a8_coal addresses its row directly from threadgroup_position_in_grid, which Metal guarantees
		// correct whatever the threadgroup's uniformity. Slower, but correct.
		return uint32(argmaxF32(r.forwardLogits(pos)))
	}
	// setPos, not a direct uPos/uNKeys write: the same gap as the paged MoE forwards (moe.go, gemma4_moe.go). It
	// matters only for FeatAttnTemp, which no family combines with this dispatch today, so it is currently a
	// no-op, as in the paged case.
	r.setPos(pos)
	e := r.q.Begin()
	r.encodeTrunkInto(e)
	e.Dispatch(r.pGemvW8Amax, (r.V)*32, 256, r.aq, r.aSc, r.lmW, r.lmS, r.part, r.uH) // tile partials (int8 head)
	e.Dispatch(r.pArgFinish, 256, 256, r.part, r.tok, r.uP)                           // reduce tiles → token
	e.End()
	r.recordExecErr(e.Err()) // C-09
	return r.tok.U32()
}

// argmaxF32 is decoder/sampler.go's argmax, mirrored here to avoid a package dependency for one
// function: first-max-wins (strict >), the same tie-break metal's own argmax_finish kernel uses
// (cv>v||(cv==v&&ci<idx) — lower index wins a tie) and decoder's CPU path uses.
func argmaxF32(v []float32) int {
	best, bi := v[0], 0
	for i, x := range v[1:] {
		if x > best {
			best, bi = x, i+1
		}
	}
	return bi
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

// encodeTrunkWith encodes all decoder layers + the final norm into e using the specified
// uPos, uNKeys, and uQTempScale uniform buffers.
func (r *resident) encodeTrunkWith(e *Encoder, uPos, uNKeys, uQTempScale Buffer, uRopePos ...Buffer) {
	rp := uPos
	if len(uRopePos) > 0 && uRopePos[0] != (Buffer{}) {
		rp = uRopePos[0]
	}
	for l := 0; l < r.nL; l++ {
		r.encodeLayerWith(e, l, uPos, uNKeys, uQTempScale, rp)
	}
	if r.q4kLane { // f32 into r.aq: the swapped head pipelines read it as float
		e.Dispatch(r.pRmsF32Out, tgReduceNorm, tgReduceNorm, r.x, r.finalNorm, r.aq, r.uH, r.uEps, r.uAddOne)
		return
	}
	r.encodeNorm(e, r.x, r.finalNorm, r.finalNormBias, r.aq, r.aSc)
}

// encodeTrunkInto encodes all decoder layers + the final norm into e, leaving the quantized
// final hidden state in r.aq/r.aSc ready for an lm head. It ONLY records dispatches (referencing
// the shared buffers) — it does NOT set uPos/uNKeys or fill r.x; the caller sets those before
// commit. This value-independence is what lets the executor pre-encode token t+1 while token t
// runs (encode-ahead) — with one exception, the attention plan (attnPlan: attention_fa on or off
// and its split grid depend on the key count), which is encoded for planNKeys and which the
// executor checks against each job before committing.
func (r *resident) encodeTrunkInto(e *Encoder) {
	r.encodeTrunkWith(e, r.uPos, r.uNKeys, r.uQTempScale, r.uRopePos)
}

// encodeNorm dispatches the family's pre-GEMV norm+quant into aq/aSc — layernorm_quant (GPT-2's
// mean-centered LayerNorm, with bias) when r.layerNorm, else the default rmsnorm_quant —
// mirroring decoder/model.go's normalize() dispatch on arch.Norm. Used at every norm site that
// feeds a GEMV (pre-attn, pre-MLP, final); the Gemma sandwich post-norms are a different
// contract (rmsnorm_f32, in-place on the residual, RMS-only families) and don't route through here.
func (r *resident) encodeNorm(e *Encoder, x, w, bias, aq, aSc Buffer) {
	if r.layerNorm {
		e.Dispatch(r.pLayerNorm, tgReduceNorm, tgReduceNorm, x, w, bias, aq, aSc, r.uH, r.uEps, r.uLNHasBias)
	} else {
		e.Dispatch(r.pRms, tgReduceNorm, tgReduceNorm, x, w, aq, aSc, r.uH, r.uEps, r.uAddOne)
	}
}

// encodeLayer encodes one decoder layer (attention block + FFN block) into e using default uniform buffers.
func (r *resident) encodeLayer(e *Encoder, l int) {
	r.encodeLayerWith(e, l, r.uPos, r.uNKeys, r.uQTempScale, r.uRopePos)
}

// encodeLayerWith encodes one decoder layer (attention block + FFN block) into e with parameterized uniforms.
func (r *resident) encodeLayerWith(e *Encoder, l int, uPos, uNKeys, uQTempScale Buffer, uRopePos ...Buffer) {
	r.encodeLayerResidualWith(e, l, r.x, uPos, uNKeys, uQTempScale, uRopePos...)
}

// encodeLayerResidualWith encodes one decoder layer into e with parameterized residual buffer x and uniforms.
func (r *resident) encodeLayerResidualWith(e *Encoder, l int, x Buffer, uPos, uNKeys, uQTempScale Buffer, uRopePos ...Buffer) {
	rp := uPos
	if len(uRopePos) > 0 && uRopePos[0] != (Buffer{}) {
		rp = uRopePos[0]
	}
	L := &r.layers[l]
	if L.delta != nil {
		// Gated-DeltaNet mixer: replaces the whole attention sub-block (norm through o-proj) —
		// no KV cache, no positional anything, the state IS the history. The FFN sub-block below
		// is the ordinary one for this layer (dense or MoE), unchanged.
		r.encodeDeltaNetMixer(e, L)
	} else {
		r.encodeAttentionResidualWith(e, l, x, uPos, uNKeys, uQTempScale, rp)
	}
	// --- ffn block (dense SwiGLU/GeGLU, generic MoE, or Gemma-4 parallel dense‖MoE) ---
	if L.g4moe != nil {
		r.encodeGemma4MoEFFN(e, L)
	} else if L.moe != nil {
		r.encodeMoEFFNWithX(e, L, x)
		if r.moeCap.idx != (Buffer{}) {
			r.encodeMoECapture(e, l)
		}
	} else if r.nonGatedMLP {
		// GPT-2: up, act, down with no gate: a single up-proj (K=hidden, covered by buildResident's threadgroup-memory
		// check) feeding act_quant (glu_act with no multiply), then the coal-family down-proj (K=intermediate, always
		// past the SA cap) fused with its bias and the residual add.
		r.encodeNorm(e, x, L.postNorm, L.postNormBias, r.mq, r.mSc)
		e.DispatchTG(r.pSABias, r.I*32, 256, r.H*2, L.upW, L.upS, r.mq, r.mSc, r.gu, L.upBias, r.uH)
		e.Dispatch(r.pActQuant, 256, 256, r.gu, r.dq, r.dSc, r.uI, r.uAct)
		e.Dispatch(r.pCoalBiasResid, r.H*32, 32, L.dW, L.dS, r.dq, r.dSc, x, L.downBias, r.uI)
	} else {
		// postOnly is model-level, but this FFN half is SHARED with DeltaNet layers (the mixer
		// above only replaces the attention half) — Olmo Hybrid's DeltaNet layers reach NormPre2
		// via NormPlacementLinear instead and carry a REAL pre-MLP norm (L.postNorm), so gate on
		// the per-layer truth, matching cuda/resident.go segBFFN's postOnlyHere.
		postOnlyHere := r.postOnly && L.delta == nil
		if r.q4kLane { // f32 activations throughout: norm, fused gate|up, SwiGLU, down + residual
			e.Dispatch(r.pRmsF32Out, tgReduceNorm, tgReduceNorm, x, L.postNorm, r.axF32, r.uH, r.uEps, r.uAddOne)
			r.q4kGemv(e, L.guSegs, false, Buffer{}, r.axF32, r.gu, r.uH)
			e.Dispatch(r.pSwF32, L.ffnI, 256, r.gu, r.gu.At(L.ffnI*4), r.swF32, L.uFFNI, r.uAct)
			r.q4kGemv(e, L.dSegs, true, Buffer{}, r.swF32, x, L.uFFNI)
			return
		}
		f16Lane := r.canUseF16Lane(l)
		gq, gSc := r.mq, r.mSc
		switch {
		case r.parallelBlock:
			// Cohere/Command-R: reuse encodeAttention's shared input norm (r.aq/r.aSc) — the MLP
			// consumes the SAME normed+quantized activation the attention branch already computed,
			// not a fresh norm of the post-attention residual (Model.ParallelBlockResident's
			// comment). No dispatch needed here at all.
			gq, gSc = r.aq, r.aSc
		case postOnlyHere:
			// Olmo 3/Olmo Hybrid: no pre-MLP norm either — quantize the raw residual directly.
			e.Dispatch(r.pQv, 256, 256, x, r.mq, r.mSc, r.uH)
		case f16Lane:
			e.Dispatch(r.pRmsF16, tgReduceNorm, tgReduceNorm, x, L.postNorm, r.mxF16, r.uH, r.uEps, r.uAddOne)
		default:
			r.encodeNorm(e, x, L.postNorm, L.postNormBias, r.mq, r.mSc)
		}
		if f16Lane {
			e.DispatchTG(r.pSAf16, (2*L.ffnI)*32, 256, r.H*2, L.guW, L.guS, r.mxF16, r.gu, r.uH) // fused gate|up
		} else {
			p, n := saRowsPick(r.pSA, r.pSARows, 2*L.ffnI, r.gemvRows.gu)
			e.DispatchTG(p, n, 256, r.H*2, L.guW, L.guS, gq, gSc, r.gu, r.uH) // fused gate|up
		}
		if r.loraLayers != nil {
			// G3: added into r.gu BEFORE the activation (r.pSw) below — matching applyLoRA's CPU
			// order exactly ("delta into gate/up before the activation", decoder/mlp.go). gq/gSc
			// is whichever input the base gate/up projection just used (r.aq/r.aSc for
			// parallelBlock, r.mq/r.mSc otherwise) — the same input applyLoRA takes on CPU.
			LA := &r.loraLayers[l]
			r.applyResidentLoRA(e, LA.gate, gq, gSc, r.gu)
			r.applyResidentLoRA(e, LA.up, gq, gSc, r.gu.At(L.ffnI*4))
		}
		e.Dispatch(r.pSw, 256, 256, r.gu, r.gu.At(L.ffnI*4), r.dq, r.dSc, L.uFFNI, r.uAct) // gate @0, up @I
		if r.sandwich || postOnlyHere || r.parallelBlock {
			e.Dispatch(r.pGemv, r.H*32, 32, L.dW, L.dS, r.dq, r.dSc, r.dO, L.uFFNI) // down → scratch
			if r.loraLayers != nil {
				// Added BEFORE the post-MLP norm (sandwich/postOnly) or the deferred residual add
				// (parallelBlock, which has none) — same "delta before any subsequent norm" order.
				r.applyResidentLoRA(e, r.loraLayers[l].down, r.dq, r.dSc, r.dO)
			}
			// parallelBlock has no post-MLP norm at all (L.postMLPNorm is never built for it — see
			// the build loop), so it skips straight to the deferred residual add.
			if r.sandwich || postOnlyHere {
				e.Dispatch(r.pRmsF32, tgReduceNorm, tgReduceNorm, r.dO, L.postMLPNorm, r.uH, r.uEps, r.uAddOne)
			}
			e.Dispatch(r.pRes, r.H, 256, x, r.dO)
		} else {
			if R := r.gemvRows.down; R > 0 { // R18: staged activations, R rows per simdgroup
				e.DispatchTG(r.pGemvResidStaged, r.H*32/R, 256, L.ffnI, L.dW, L.dS, r.dq, r.dSc, x, L.uFFNI) // down + residual
			} else {
				e.Dispatch(r.pGemvResid, r.H*32, 32, L.dW, L.dS, r.dq, r.dSc, x, L.uFFNI) // down + residual
			}
			if r.loraLayers != nil {
				r.applyResidentLoRA(e, r.loraLayers[l].down, r.dq, r.dSc, x)
			}
		}
	}
	if L.pleGW != (Buffer{}) && !g4SkipPLEForTest {
		r.encodePLE(e, l, x)
	}
	if L.uLayerScalar != (Buffer{}) && !g4DropLayerScalarForTest {
		e.Dispatch(r.pLayerScale, r.H, 256, x, L.uLayerScalar) // Gemma 4 dense layer: h *= layer_scalar (S1.0)
	}
}

// q4kGemv dispatches one q4k-lane projection over the f32 activation ax (K = uK), one dispatch per segment into its
// rows of out: gemv_q4k_f32 for Q4_K super-blocks, gemv_w8_f32 for int8 rows with per-row scales. resid accumulates
// into out; a non-empty bias is added per row (resid and bias are never both set).
func (r *resident) q4kGemv(e *Encoder, segs []q4kSeg, resid bool, bias, ax, out, uK Buffer) {
	for _, sg := range segs {
		o := out.At(sg.off * 4)
		switch {
		case sg.q4k && resid:
			e.Dispatch(r.pQ4KResid, sg.rows*32, 256, sg.w, ax, o, uK, sg.uN)
		case sg.q4k && bias != (Buffer{}):
			e.Dispatch(r.pQ4KBias, sg.rows*32, 256, sg.w, ax, o, uK, sg.uN, bias.At(sg.off*4))
		case sg.q4k:
			e.Dispatch(r.pQ4K, sg.rows*32, 256, sg.w, ax, o, uK, sg.uN)
		case resid:
			e.Dispatch(r.pW8F32Resid, sg.rows*32, 256, sg.w, sg.s, ax, o, uK, sg.uN)
		case bias != (Buffer{}):
			e.Dispatch(r.pW8F32Bias, sg.rows*32, 256, sg.w, sg.s, ax, o, uK, sg.uN, bias.At(sg.off*4))
		default:
			e.Dispatch(r.pW8F32, sg.rows*32, 256, sg.w, sg.s, ax, o, uK, sg.uN)
		}
	}
}

// encodePLE is a Gemma 4 E-model layer's Per-Layer-Embedding branch (S1.5), after the FFN residual and before the
// layer scalar, as decoder/forward_gemma4.go and HF's Gemma4DecoderLayer order it:
// x += post_per_layer_input_norm( PLEProj · ( gelu_tanh(PLEGate · x) * pleIn[l] ) ). PLEGate reads the raw residual
// (no pre-norm), quantized like every other W4A8 input; the projection lands in the FFN's dO scratch, free by now.
func (r *resident) encodePLE(e *Encoder, l int, x Buffer) {
	L := &r.layers[l]
	P := r.pleP
	e.Dispatch(r.pQv, 256, 256, x, r.mq, r.mSc, r.uH)
	e.DispatchTG(r.pSA, P*32, 256, r.H*2, L.pleGW, L.pleGS, r.mq, r.mSc, r.pleG, r.uH)
	e.Dispatch(r.pPLEGeluMul, P, 64, r.pleG, r.pleIn.At(l*P*4), r.uPleP)
	e.Dispatch(r.pQv, 256, 256, r.pleG, r.pleQ, r.pleSc, r.uPleP)
	e.DispatchTG(r.pSA, r.H*32, 256, P*2, L.plePW, L.plePS, r.pleQ, r.pleSc, r.dO, r.uPleP)
	e.Dispatch(r.pRmsF32, tgReduceNorm, tgReduceNorm, r.dO, L.postPLENorm, r.uH, r.uEps, r.uAddOne)
	e.Dispatch(r.pRes, r.H, 256, x, r.dO)
}

// loadEmb stages one embedding row for a single-token forward: the hidden state into r.x and, for a Gemma 4
// E-model, the row's PLE tail into r.pleIn (S1.5). The backend's entry points check the row's length
// (metalResident.embLen) before it gets here.
func (r *resident) loadEmb(emb []float32) {
	copy(r.x.Floats(), emb[:r.H])
	if r.pleP > 0 {
		copy(r.pleIn.Floats(), emb[r.H:])
	}
}

// gemvRowsFor is the rows-per-simdgroup count for an R18 GEMV of `rows` outputs: the largest of want, want/2, ..., 2
// whose threadgroups of 8·R rows (tg 256 = 8 simdgroups) tile rows exactly, or 0 (keep the shipped kernel). The rows
// kernels index rows from full threadgroups — a partial last one under dispatchThreads reports a smaller
// threads_per_threadgroup — so an inexact tiling must not reach them.
func gemvRowsFor(rows, want int) int {
	for R := want; R >= 2; R /= 2 {
		if rows%(8*R) == 0 {
			return R
		}
	}
	return 0
}

// saRowsPick returns the SA-family pipeline and grid for `rows` outputs: the R-rows kernel at rows·32/R threads when R
// is set (resident.gemvRows), else the shipped one-row-per-simdgroup kernel at rows·32.
func saRowsPick(shipped, rowsKernel Pipeline, rows, R int) (Pipeline, int) {
	if R > 0 {
		return rowsKernel, rows * 32 / R
	}
	return shipped, rows * 32
}

// canUseF16Lane reports whether layer l's attention block and FFN gate/up can use the W4F16 decode lane
// (docs/tasks/red-october.md) instead of the shipped W4A8 path. It covers the plainest dense layer only: every special
// case (Gemma sandwich, Olmo postOnly/qkNorm-whole, Cohere parallelBlock, GPT-2 non-gated MLP or FeatOutBias, qGate's
// double-width Q, MoE/Gemma-4 MoE, DeltaNet, compute-time LoRA) keeps the int8 path, since each changes which kernel
// family or epilogue is dispatched and none is ported. Down-proj (the coal family, a different kernel shape) is out of
// scope regardless: encodeLayer's default branch never checks this for r.pGemvResid.
func (r *resident) canUseF16Lane(l int) bool {
	if !r.decodeLaneW4F16 {
		return false
	}
	if r.sandwich || r.postOnly || r.parallelBlock || r.nonGatedMLP || r.outBias || r.layerNorm {
		return false
	}
	if r.loraLayers != nil {
		return false
	}
	L := &r.layers[l]
	return L.moe == nil && L.g4moe == nil && L.delta == nil && !L.qGate
}

// attnFACoreCount is the GPU core count attention_fa's split count targets (kvHead x S >= 2x the core count): 14, the
// M1 Pro it was tuned on. It sets only the legacy kernel's split (the block kernel uses attnFABlkSplit), so changing
// it changes the legacy kernel's bits. It is hardcoded, not device-queried (aikit's Device has no core-count
// accessor), although the kernel is default-on on every chip via metalAttnFAEnabled; a port to other Apple GPU core
// counts would need it read from the device, not assumed.
const attnFACoreCount = 14

// attnFABlkSplit is the split count the block kernel (attention_fa_blk) runs at. Its split response is not monotone
// and not the legacy kernel's; S = 16 was best or near-best at the depths and models measured, and was the setting of
// the confirmation run and the fidelity decision (docs/measurements/metal-decode-attn-r17-2026-09-25.md). The nKeys/32
// cap never binds above attnFADepthFloor.
const attnFABlkSplit = 16

// attnFADepthFloor is where attention_fa takes over from the shipped kernel: 1024 keys
// (docs/tasks/task-metal-audit-2026-10.md, B-P03). Below it the two kernels were level. S=1 (no split) is not a
// shallow-depth fallback within attention_fa: it measured uniformly worse than the shipped kernel at every depth, so
// below the floor canUseAttnFA declines entirely and the shipped kernel runs.
// A revert to 1536 on the evidence of TestSpecNgram_copyOnStepVerify was a test artifact (its spec arm reused the
// cache while the cold plain arm prefilled it, and the f16 prefill is not bit-identical to decode). Before moving the
// floor, read docs/code-notes/metal.md#attnFADepthFloor.
const attnFADepthFloor = 1024

// attnFAMaxG is the most query heads per KV head attention_fa holds: its per-thread arrays are sized
// ATTN_FA_MAXG in kernels.go (TestAttnFAMaxG_matchesKernel ties the two). A wider group would index past
// them, so attnFAGroupOK keeps such a layer on the shipped kernel (F-C03, docs/audit-metal-2026-09-30.md).
// The group size comes from the checkpoint, not the family (Llama 3.1 405B has 16), so this is a guard,
// not a list.
const attnFAMaxG = 8

// attnFAGroupOK reports whether a layer with nH query heads over nKV KV heads fits attention_fa's arrays.
func attnFAGroupOK(nH, nKV int) bool { return nKV > 0 && nH%nKV == 0 && nH/nKV <= attnFAMaxG }

// attnFAHeadDimOK reports whether attention_fa can run a layer of head dim hd: 128 (attention_fa and the block kernel),
// or 64 where the block kernel's hd = 64 twin is instantiated for the group size, G = 7 (B-P01). attention_fa itself
// is hd = 128 only, so an hd = 64 layer must never reach it without the twin.
func attnFAHeadDimOK(hd, nH, nKV int) bool {
	if hd == 128 {
		return true
	}
	return hd == 64 && nKV > 0 && nH%nKV == 0 && nH/nKV == 7 && attnFABlk64On
}

// gemvExtOn routes the int4 GEMVs R18 never reached (DeltaNet's qkv and z projections, a shared expert's gate|up and
// down) through the rows-per-simdgroup kernels. Each rows kernel is bit-identical to the one it replaces (same
// lane-strided words, per-word sum and simd_sum; TestGemvExt_bitIdentical). Grade:
// docs/tasks/task-metal-audit-2026-10.md, "D-B04".
var gemvExtOn = true

// gemvExtKind names the shipped kernel family a gemvExt site runs: the plain coal projection, the SA projection, or
// the coal projection with a residual add.
type gemvExtKind int

const (
	gemvExtKCoal gemvExtKind = iota
	gemvExtKSA
	gemvExtKResid
)

// gemvExt encodes one int4 GEMV of `rows` outputs over K inputs at a site R18 never reached: its rows-per-simdgroup
// twin when gemvExtOn and the shape allows it (rows fill whole threadgroups, K bytes of staged activations fit, an
// int4 resident), else exactly the dispatch the site always made. args are the shipped kernel's buffers in order.
func (r *resident) gemvExt(e *Encoder, kind gemvExtKind, rows, K int, args ...Buffer) {
	R := 0
	if gemvExtOn && !r.w8 && K%4 == 0 {
		R = gemvRowsFor(rows, 4)
	}
	stage := K
	if kind == gemvExtKSA {
		stage = 2 * K // the SA family stages K shorts
	}
	if R > 0 && stage > r.gemvExtTGMax {
		R = 0
	}
	if R > 0 {
		r.gemvExtRows[kind]++
	}
	switch {
	case R > 0 && kind == gemvExtKCoal:
		e.DispatchTG(r.gemvExtCoal[R], rows*32/R, 256, stage, args...)
	case R > 0 && kind == gemvExtKSA:
		e.DispatchTG(r.gemvExtSA[R], rows*32/R, 256, stage, args...)
	case R > 0 && kind == gemvExtKResid:
		e.DispatchTG(r.gemvExtResid[R], rows*32/R, 256, stage, args...)
	case kind == gemvExtKCoal:
		e.Dispatch(r.pGemv, rows*32, 32, args...)
	case kind == gemvExtKSA:
		e.DispatchTG(r.pSA, rows*32, 256, 2*K, args...)
	default:
		e.Dispatch(r.pGemvResid, rows*32, 32, args...)
	}
}

// attnFABlkAnyG selects the block kernel for the dense group sizes other than the graded 6 and 7 (G = 2, 3, 4, 5 and
// 8). The kernel agrees with g7 head for head (TestAttnFABlk_anyGMatchesG7). Grade:
// docs/tasks/task-metal-audit-2026-10.md, "B-P02".
var attnFABlkAnyG = true

// attnFABlk64On admits hd = 64 layers to attention_fa through the block kernel's hd = 64 twin. The twin is not
// bit-identical to the per-query-head kernel, which is why it was graded on fidelity rather than identity. Grade:
// docs/tasks/task-metal-audit-2026-10.md, "B-P01".
var attnFABlk64On = true

// fanSplitBuf is the attention_fa split-count uniform the command buffer being encoded binds: the greedy chain's own
// set's while it encodes one of its buffers (encFANSplit), otherwise the resident's, which setPos writes.
func (r *resident) fanSplitBuf() Buffer {
	if r.encFANSplit != (Buffer{}) {
		return r.encFANSplit
	}
	return r.uAttnFANSplit
}

// planNKeys is the key count the command buffer being encoded will run at: the executor's encNKeys while it encodes,
// otherwise the position setPos last set.
func (r *resident) planNKeys() int {
	if r.encNKeys > 0 {
		return r.encNKeys
	}
	return r.curNKeys
}

// attnPlan is the part of an encoded decode command buffer that depends on its key count or on a runtime
// toggle: whether attention_fa is dispatched (the depth gate), its split grid, and the f16 lane. Everything
// else an encode bakes in is fixed for the resident's life, and the per-token values (position, key count,
// the split uniform) are written at commit time by setPos. The pipelined executor compares a pre-encoded
// buffer's plan with the arriving job's and re-encodes on a mismatch (execLoop). nSplit is 0 when
// attention_fa is not dispatched, so the key count's effect on the cap below the floor never forces a
// re-encode. Pipeline swaps (tests) are not part of it: a test that swaps a pipeline flushes (stopExec).
type attnPlan struct {
	fa      bool
	nSplit  int
	f16Lane bool
}

// attnFAFloor is the key count at which attention_fa takes over: attnFADepthFloor unless a test set
// attnFAFloorOverride.
func (r *resident) attnFAFloor() int {
	if r.attnFAFloorOverride > 0 {
		return r.attnFAFloorOverride
	}
	return attnFADepthFloor
}

// attnPlanFor is the plan a decode command buffer running at nKeys keys needs.
func (r *resident) attnPlanFor(nKeys int) attnPlan {
	p := attnPlan{f16Lane: r.decodeLaneW4F16}
	if r.decodeAttnFA && r.attnFAPartial != (Buffer{}) && r.attnFANKV > 0 && nKeys >= r.attnFAFloor() {
		p.fa, p.nSplit = true, r.attnFASplitFor(nKeys, r.attnFANKV)
	}
	return p
}

// canUseAttnFA reports whether attention_fa may replace the shipped kernel for layer l in the command buffer being
// encoded (at planNKeys keys): dense GQA only, at a head dim attnFAHeadDimOK admits (the kernel's cooperative-load
// tiling is fixed to 32 lanes x half4). Sinks, windows and the f32-KV twin are out of scope, for the reasons of
// canUseF16Lane's family exclusions.
func (r *resident) canUseAttnFA(l int) bool {
	if !r.decodeAttnFA || r.attnFAPartial == (Buffer{}) {
		return false
	}
	if r.sandwich || r.postOnly || r.parallelBlock || r.attnSink || r.kvI8 {
		return false
	}
	if r.loraLayers != nil {
		return false
	}
	L := &r.layers[l]
	if L.moe != nil || L.g4moe != nil || L.delta != nil || L.qGate || L.window != 0 {
		return false
	}
	g := L.geom
	return g != nil && attnFAHeadDimOK(g.hd, r.nH, g.nKV) && attnFAGroupOK(r.nH, g.nKV) && r.planNKeys() >= r.attnFAFloor()
}

// attnFASplitFor picks S so kvHead*S clears 2x attnFACoreCount (R2's own registered rule),
// capped at r.attnFAMaxSplit (the partial buffer's own allocation) and at nKeys/32 (a floor on
// keys-per-split — an S so large most splits see one or zero keys wastes the second dispatch's
// combine pass for no parallelism gain).
func (r *resident) attnFASplitFor(nKeys, nKV int) int {
	want := (2*attnFACoreCount + nKV - 1) / nKV
	if r.attnFABlkSplit > 0 {
		want = r.attnFABlkSplit
	}
	if r.attnFASplitOverride > 0 {
		want = r.attnFASplitOverride
	}
	if splitCap := max(nKeys/32, 1); want > splitCap {
		want = splitCap
	}
	if want > r.attnFAMaxSplit {
		want = r.attnFAMaxSplit
	}
	if want < 1 {
		want = 1
	}
	return want
}

// encodeAttention records one layer's attention block (through the o-proj + residual/sandwich norm).
// Split from encodeLayer so the paged Gemma-4 MoE forward can put [attention + dense + router] in one
// command buffer, submit+wait, read the router idx, stage experts, then encode [experts + join] in a
// second — the value-dependent seam paging forces. Byte-identical to the old inline attention block.
func (r *resident) encodeAttention(e *Encoder, l int) {
	r.encodeAttentionWith(e, l, r.uPos, r.uNKeys, r.uQTempScale, r.uRopePos)
}

func (r *resident) encodeAttentionWith(e *Encoder, l int, uPos, uNKeys, uQTempScale Buffer, uRopePos ...Buffer) {
	r.encodeAttentionResidualWith(e, l, r.x, uPos, uNKeys, uQTempScale, uRopePos...)
}

func (r *resident) encodeAttentionResidualWith(e *Encoder, l int, x Buffer, uPos, uNKeys, uQTempScale Buffer, uRopePos ...Buffer) {
	rp := uPos
	if len(uRopePos) > 0 && uRopePos[0] != (Buffer{}) {
		rp = uRopePos[0]
	}
	L := &r.layers[l]
	g := L.geom
	nHhd := r.nH * g.hd
	qkvRows := nHhd + 2*g.kvDim
	if L.kvShared {
		qkvRows = nHhd // S1.4: Q only; K/V come from layer L.kvSrc's cache, which r.kc[l]/r.vc[l] alias
	}
	kOff, vOff := nHhd*4, (nHhd+g.kvDim)*4 // byte offsets of k, v within the fused qkv buffer
	// --- attention block (7 dispatches in the baseline dense case: norm, fused QKV+bias, merged Q+K RoPE, KV
	// store, attention, o-proj input quant, fused o-proj+residual; 8 at attention_fa depths, whose combine is a
	// second attention dispatch; qGate, qkNorm, kEqV, sandwich and LoRA each add their own) ---
	// postOnly (Olmo 3, Olmo Hybrid): no pre-norm at all, so quantize the raw residual (quant_vec, the symmetric
	// int8 quantizer the ctx-before-o-proj step also uses) instead. encodeAttention is never called for a DeltaNet
	// layer (encodeLayer routes those to encodeDeltaNetMixer), so the model-level flag is safe here, unlike
	// encodeLayer's shared FFN half below.
	f16Lane := r.canUseF16Lane(l)
	if r.q4kLane {
		e.Dispatch(r.pRmsF32Out, tgReduceNorm, tgReduceNorm, x, L.preNorm, r.axF32, r.uH, r.uEps, r.uAddOne)
	} else if r.postOnly {
		e.Dispatch(r.pQv, 256, 256, x, r.aq, r.aSc, r.uH)
	} else if f16Lane {
		// R1: f16 activation, no quantization — bypasses encodeNorm (which always produces the
		// int8 aq/aSc pair) since preNorm here has no bias on the plain dense path canUseF16Lane
		// admits (a biased pre-norm would need layernorm_quant's shape, not rmsnorm's).
		e.Dispatch(r.pRmsF16, tgReduceNorm, tgReduceNorm, x, L.preNorm, r.axF16, r.uH, r.uEps, r.uAddOne)
	} else {
		r.encodeNorm(e, x, L.preNorm, L.preNormBias, r.aq, r.aSc)
	}
	if L.qGate {
		// This family's softmax layer: q_proj is DOUBLE WIDTH ([query ‖ gate] per head,
		// interleaved) and bias-free, so it dispatches separately via the no-bias SA kernel
		// into dnQg, then delta_qsplit extracts Q into r.qkv's normal Q slot (offset 0, same as
		// every other family) and the gate into dnAGate. K‖V is a 2-way fused (still bias-free)
		// projection straight into r.qkv's K/V slots (kOff) — same buffer, same layout, just two
		// dispatches building it instead of one. Both MUST land before qk_norm below runs.
		e.DispatchTG(r.pSA, 2*nHhd*32, 256, r.H*2, L.dnQw, L.dnQs, r.aq, r.aSc, r.dnQg, r.uH)
		e.Dispatch(r.pDnQSplit, nHhd, 256, r.dnQg, r.qkv, r.dnAGate, g.uNHhd, g.uHd)
		e.DispatchTG(r.pSA, 2*g.kvDim*32, 256, r.H*2, L.qkvW, L.qkvS, r.aq, r.aSc, r.qkv.At(kOff), r.uH)
	} else if r.q4kLane {
		r.q4kGemv(e, L.qkvSegs, false, L.qkvBias, r.axF32, r.qkv, r.uH) // the combined bias, zeros where absent
	} else if f16Lane {
		e.DispatchTG(r.pSAf16Bias, qkvRows*32, 256, r.H*2, L.qkvW, L.qkvS, r.axF16, L.qkvBias, r.qkv, r.uH)
	} else {
		p, n := saRowsPick(r.pSABias, r.pSABiasRows, qkvRows, r.gemvRows.qkv)
		e.DispatchTG(p, n, 256, r.H*2, L.qkvW, L.qkvS, r.aq, r.aSc, r.qkv, L.qkvBias, r.uH)
		if r.loraLayers != nil {
			// G3: compute-time LoRA — added on top of the base q/k/v projection, into the SAME
			// r.qkv slots it just wrote, before anything downstream (qk_norm/RoPE) reads them.
			// Not reached for the qGate branch above (see lora.go's file comment).
			LA := &r.loraLayers[l]
			r.applyResidentLoRA(e, LA.q, r.aq, r.aSc, r.qkv)
			if !L.kvShared {
				r.applyResidentLoRA(e, LA.k, r.aq, r.aSc, r.qkv.At(kOff))
				r.applyResidentLoRA(e, LA.v, r.aq, r.aSc, r.qkv.At(vOff))
			}
		}
	}
	if r.qkNorm { // Qwen3: per-head Q/K RMSNorm before RoPE
		// QKNormWhole (Olmo 3, Olmo Hybrid): the same kernel with the grid collapsed to one Q block and one K block,
		// via the constant-1 and whole-width buffers built in buildResident: qk_norm's block h<nH reduces [base, +hd)
		// with base = head*hd, so nH=1, hd=nH_orig*hd spans the whole contiguous Q vector for h=0 and the whole K
		// vector for h=nH=1, matching decoder/attention.go's rmsNorm(q,QNorm,1,nH*hd,...). g.uNHhd (the K offset
		// within the fused qkv buffer) does not depend on the reinterpreted reduction width. Correct only for MHA
		// (nH==nKV); buildResident declines otherwise.
		qkNH, qkNKV, qkHD := r.uNH, g.uNKV, g.uHd
		if r.qkNormWhole {
			qkNH, qkNKV, qkHD = r.uQKWholeOne, r.uQKWholeOne, r.uQKWholeHD
		}
		grid := (r.nH + g.nKV) * tgReduceAttn
		if r.qkNormWhole {
			grid = 2 * tgReduceAttn
		}
		if L.kvShared {
			grid = r.nH * tgReduceAttn // Q heads only: the K slot holds nothing on a shared layer
		}
		e.Dispatch(r.pQKNorm, grid, tgReduceAttn, r.qkv, L.qNorm, L.kNorm, qkNH, qkNKV, qkHD, g.uNHhd, r.uEps, r.uAddOne)
	}
	if !L.kvShared && (g.kEqV || (r.g4VNorm && !g4DropVNormForTest)) {
		// Gemma 4, every K/V-owning layer (S1.0): scale-less v_norm on the V slot. For K=V (Gemma 4 globals)
		// the V slot holds the RAW k_proj output ([Q|K|K] fusion); otherwise the v_proj output. Apply
		// scale-less v_norm to it — qk_norm over the V slot (qkv.At(vOff)) with nH=0 so every
		// head takes the K branch at base 0+head*hd, a UNIT weight, and addOne=0 → x·rms·1. Runs
		// AFTER qk_norm (which touched the K slot, not V) and BEFORE RoPE (which never touches V),
		// so V = v_norm(raw k), un-rotated — exactly the CPU path (copy(v,k) pre-k_norm; then
		// rmsNormNoWeight(v)). See TestVNorm_scaleless.
		e.Dispatch(r.pQKNorm, g.nKV*tgReduceAttn, tgReduceAttn, r.qkv.At(vOff), r.vNormUnit, r.vNormUnit, r.uZero, g.uNKV, g.uHd, r.uZero, r.uEps, r.uZero)
	}
	if !r.learnedPos { // GPT-2: no RoPE at all — position rides the learned embedding added at input
		// One merged dispatch for both Q and K (rope2, kernels.go) instead of two: gid<qTotal
		// addresses Q at offset 0, gid>=qTotal addresses K at offset g.uNHhd (the fused qkv
		// buffer's kOff, in elements) — V (at vOff) is untouched either way.
		ropeN := r.nH*g.half + g.nKV*g.half
		if L.kvShared {
			ropeN = r.nH * g.half // Q only: rope2's gid < qTotal branch
		}
		e.Dispatch(r.pRope2, ropeN, 64, r.qkv, L.invf, g.uHd, rp, g.uQtotal, g.uKtotal, g.uHalf, L.mscale, g.uNHhd, uQTempScale)
	}
	// S1.4: a shared layer must never write its source's cache. The seam is G2's defect (5): the store not skipped,
	// so the shared layer writes its (unprojected) K/V slots over its source's row.
	storeKV := !L.kvShared || g4KeepSharedKVStoreForTest
	if r.kvI8 {
		if storeKV {
			e.Dispatch(r.pKvI8, g.nKV, 1, r.qkv.At(kOff), r.qkv.At(vOff), r.kc[l], r.vc[l], r.ks[l], r.vs[l], g.uNKV, g.uHd, uPos)
		}
		e.Dispatch(r.pAttnI8, r.nH*tgReduceAttn, tgReduceAttn, r.qkv, r.kc[l], r.vc[l], r.ks[l], r.vs[l], r.ctx, r.uNH, g.uNKV, g.uHd, uNKeys, r.uScale, L.uWindow, L.attnSinks, L.uHasSink)
	} else {
		if storeKV {
			e.Dispatch(r.pKv, g.kvDim, 64, r.qkv.At(kOff), r.qkv.At(vOff), r.kc[l], r.vc[l], g.uKvDim, uPos)
		}
		if r.canUseAttnFA(l) {
			// nSplit here sizes the dispatch grid only, which is fixed once a command buffer is encoded. It is not written
			// into r.uAttnFANSplit here: setPos does that once per decode step, before this buffer commits (see
			// uAttnFAG/uAttnFANSplit's field comment for the race a per-layer SetU32 was). The two agree by construction:
			// both are attnFASplitFor of the key count this buffer runs at (planNKeys here, the job's own position in
			// setPos), and the executor re-encodes a pre-encoded buffer whose plan does not match the arriving job
			// (execLoop). Reading the key count at encode time instead would give the previous job's: one token late at
			// the floor, and the previous request's depth on a new request's first step, with a grid sized for the old
			// depth and a uniform for the new.
			nSplit := r.attnFASplitFor(r.planNKeys(), g.nKV)
			if os.Getenv("GOINFER_ATTNFA_DEBUG") == "1" {
				fmt.Fprintf(os.Stderr, "[ATTNFA] l=%d nKeys=%d nSplit=%d nKV=%d G=%d hd=%d partialLen=%d\n",
					l, r.planNKeys(), nSplit, g.nKV, r.nH/g.nKV, g.hd, r.attnFAPartial.Len())
			}
			shmBytes := 128 * 6 * (r.nH / g.nKV) * 4
			e.DispatchTG(r.pAttnFA, g.nKV*nSplit*128, 128, shmBytes, r.qkv, r.kc[l], r.vc[l], r.attnFAPartial,
				g.uNKV, r.uAttnFAG, uNKeys, r.uScale, L.uWindow, r.fanSplitBuf())
			e.Dispatch(r.pAttnFACombine, r.nH*g.hd, g.hd, r.attnFAPartial, r.ctx, r.uAttnFAG, g.uHd, r.fanSplitBuf())
		} else {
			e.Dispatch(r.pAttn, r.nH*tgReduceAttn, tgReduceAttn, r.qkv, r.kc[l], r.vc[l], r.ctx, r.uNH, g.uNKV, g.uHd, uNKeys, r.uScale, L.uWindow, L.attnSinks, L.uHasSink)
		}
	}
	if L.qGate { // ctx *= sigmoid(gate), before o-proj — matches the CPU qwen35Attention
		e.Dispatch(r.pDnAttnGate, nHhd, 256, r.ctx, r.dnAGate, g.uNHhd)
	}
	if r.q4kLane {
		// o-proj reads r.ctx (f32) directly: no conversion, no quantization.
	} else if f16Lane {
		e.Dispatch(r.pF32ToF16, nHhd, 256, r.ctx, r.cxF16)
	} else {
		e.Dispatch(r.pQv, 256, 256, r.ctx, r.cq, r.cSc, g.uNHhd)
	}
	if r.sandwich || r.postOnly || r.parallelBlock {
		// Gemma sandwich / Olmo 3 postOnly: the sublayer OUTPUT is normed BEFORE the residual
		// add, which the fused _resid epilogue can't express — project into the (otherwise
		// dead) oO scratch, norm it, then add. Cohere parallelBlock shares the "project into oO,
		// defer the add" shape but has NO post-attn norm at all (L.postAttnNorm is never built for
		// it — see the build loop), so it skips the norm dispatch entirely.
		e.DispatchTG(r.pSA, r.H*32, 256, r.nH*g.hd*2, L.oW, L.oS, r.cq, r.cSc, r.oO, g.uNHhd)
		if r.loraLayers != nil {
			// G3: added into the RAW o-proj output, BEFORE the sandwich/postOnly norm below —
			// matching applyLoRA's CPU order (delta added to `out`, caller norms afterward).
			r.applyResidentLoRA(e, r.loraLayers[l].o, r.cq, r.cSc, r.oO)
		}
		if r.sandwich || r.postOnly {
			e.Dispatch(r.pRmsF32, tgReduceNorm, tgReduceNorm, r.oO, L.postAttnNorm, r.uH, r.uEps, r.uAddOne)
		}
		e.Dispatch(r.pRes, r.H, 256, x, r.oO)
	} else if r.outBias { // GPT-2/gpt-oss: o-proj carries an additive bias, fused with the residual add
		e.DispatchTG(r.pSABiasResid, r.H*32, 256, r.nH*g.hd*2, L.oW, L.oS, r.cq, r.cSc, x, L.oBias, g.uNHhd)
		if r.loraLayers != nil {
			r.applyResidentLoRA(e, r.loraLayers[l].o, r.cq, r.cSc, x)
		}
	} else if r.q4kLane {
		r.q4kGemv(e, L.oSegs, true, Buffer{}, r.ctx, x, g.uNHhd) // o-proj + residual
	} else if f16Lane {
		e.DispatchTG(r.pSAf16Resid, r.H*32, 256, r.nH*g.hd*2, L.oW, L.oS, r.cxF16, x, g.uNHhd) // o-proj + residual
	} else {
		p, n := saRowsPick(r.pSAResid, r.pSAResidRows, r.H, r.gemvRows.o)
		e.DispatchTG(p, n, 256, r.nH*g.hd*2, L.oW, L.oS, r.cq, r.cSc, x, g.uNHhd) // o-proj + residual
		if r.loraLayers != nil {
			r.applyResidentLoRA(e, r.loraLayers[l].o, r.cq, r.cSc, x)
		}
	}
}

// ForwardBatch runs N embeddings at consecutive positions starting at startPos,
// encoding all N token forward passes into a SINGLE Metal command buffer and
// single compute command encoder using a LAYER-MAJOR dispatch schedule (weights
// streamed once per layer across all N tokens), returning all N logits vectors.
func (r *resident) ForwardBatch(embeddings [][]float32, startPos int) ([][]float32, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	n := len(embeddings)
	if n == 0 {
		return nil, nil
	}

	// This path drives encodeAttentionResidualWith with its own per-token r.batchUNKeys[m] buffers, never through
	// setPos, so r.curNKeys (planNKeys, canUseAttnFA's depth gate) would otherwise stay stale from whatever a
	// prior single-token call set. Force it to 0 for the whole batch rather than dispatch attention_fa off a stale
	// depth: attention_fa targets decode (M=1), so the shipped kernel is the intended choice here too. The
	// executor does not read this zero: it encodes each job for its own key count (execLoop).
	r.curNKeys = 0

	// Drain and stop the pipelined executor so we have exclusive, synchronous access to the queue
	// and no in-flight or pre-encoded command buffers conflict with our batch.
	r.stopExec()

	r.ensureBatchCap(n)

	// Populate batch input embeddings and per-token uniforms up front.
	for m, emb := range embeddings {
		if len(emb) != r.H {
			return nil, fmt.Errorf("metal: embedding[%d] len %d != hidden %d", m, len(emb), r.H)
		}
		dst := r.batchX.Floats()[m*r.H : (m+1)*r.H]
		copy(dst, emb)
		pos := startPos + m
		if r.learnedPos {
			r.addLearnedPosTo(dst, pos)
		}
		r.batchUPos[m].SetU32(uint32(pos))
		r.batchUNKeys[m].SetU32(uint32(pos + 1))
		scale := float32(1)
		if r.attnTempBeta != 0 {
			scale = float32(1 + r.attnTempBeta*math.Log1p(math.Floor(float64(pos)/r.attnTempOrigMaxPos)))
		}
		r.batchUQTempScale[m].Floats()[0] = scale
	}

	// Encode all N tokens in LAYER-MAJOR order into a SINGLE command buffer and single compute encoder.
	// Layer l's weights stay hot in GPU cache/memory across all N tokens instead of streaming the
	// full multi-GB model weights N separate times.
	e := r.q.Begin()
	for l := 0; l < r.nL; l++ {
		for m := range n {
			r.encodeLayerResidualWith(e, l, r.batchX.At(m*r.H*4), r.batchUPos[m], r.batchUNKeys[m], r.batchUQTempScale[m])
		}
	}
	for m := range n {
		r.encodeNorm(e, r.batchX.At(m*r.H*4), r.finalNorm, r.finalNormBias, r.aq, r.aSc)
		e.Dispatch(r.pGemvW8, (r.V)*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.batchLogits.At(m*r.V*4), r.uH)
	}
	e.End()

	r.recordExecErr(e.Err())
	if err := r.takeExecErr(); err != nil {
		return nil, err
	}
	r.gpuStart, r.gpuEnd, r.kernStart, r.kernEnd = e.GPUStart(), e.GPUEnd(), e.KernStart(), e.KernEnd()

	// Read back and finalize logits for each token.
	out := make([][]float32, n)
	bLogits := r.batchLogits.Floats()
	for m := range n {
		row := make([]float32, r.V)
		copy(row, bLogits[m*r.V:(m+1)*r.V])
		if r.finalSoftcap > 0 {
			softcapParallel(row, r.finalSoftcap)
		}
		if r.logitScale != 0 && r.logitScale != 1 {
			for j, v := range row {
				row[j] = v * r.logitScale
			}
		}
		out[m] = row
	}

	return out, nil
}

// S1.0 test seams (docs/tasks/task-multimodal-support-2026-10.md): re-drop each Gemma 4 fix, so a gate can show it
// reproduces the before-fix numbers. Never set outside tests.
var (
	g4DropLayerScalarForTest bool
	g4DropVNormForTest       bool
)

// S1 G2 planted defects (docs/tasks/task-multimodal-support-2026-10.md): each re-introduces one E-model mistake on the
// Metal side only, so G1 (Metal resident against the CPU) must turn red under it. Never set outside tests.
var (
	g4KeepSharedKVStoreForTest bool // (5) a shared layer stores its K/V over its source's cache
	g4SkipPLEForTest           bool // (1) the PLE branch skipped
	g4KVSrcOffForTest          bool // (3) a shared layer reads the previous owning layer of its type, not its source
	g4OneFFNWidthForTest       bool // (4) every dense layer at the model's one intermediate width
)
