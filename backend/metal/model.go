//go:build darwin

// Resident Metal decoder: BuildResident uploads a model's weights to the device once, and each token then runs the
// full layer stack in one command buffer. cgo-free throughout (purego-objc plus dlopen of Metal).
package metal

import (
	"fmt"
	"os"
	"runtime"
	"unsafe"

	"github.com/townsendmerino/goinfer/internal/runtime/mem"
	"github.com/townsendmerino/goinfer/internal/runtime/program"

	"github.com/townsendmerino/aikit/linalg"
	gpu "github.com/townsendmerino/goinfer/backend/metal/internal/gpudev"
	"github.com/townsendmerino/goinfer/backend/metal/kernels"
	"github.com/townsendmerino/goinfer/decoder"
)

// metalCtxCapDefault is the resident KV capacity in positions an unpinned load never goes under of its own accord (4096), and
// what it takes when fit-by-default does not apply (metalFitFor). An explicit decoder.Options.ResidentContext asks for more.
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

// resolveMetalCtxCap turns a request into the effective resident KV capacity, in the shape of cuda's resolveCtxCapFit: an
// unpinned load gets the fit-by-default context (metalFitFor: up to decoder.MetalFitDefaultCtx when memory holds it, never
// under metalCtxCapDefault); an explicit request up to metalCtxCapMax is honored, clamped to the model's window; a request
// above metalCtxCapMax is refused with the numbers.
// A context the load-time fit guard auto-pinned (the caller did not choose it) is a ceiling, not a request: metalFitFor
// takes it as a limit that lowers the choice and never raises it. Read as a request it allocated several times the
// default KV on a tight machine, or was refused, which moved the whole forward to the CPU.
func resolveMetalCtxCap(m *decoder.Model, plan metalCtxPlan) (cap int, err error) {
	req := m.ResidentContextRequest()
	if req <= 0 || !m.ResidentContextPinned() {
		ctx := metalFitFor(m, plan).Ctx
		if plan.ceil > 0 {
			ctx = min(ctx, plan.ceil) // lowered by BuildResident before it declines (shrinkCtxToFit)
		}
		return ctx, nil
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
	decoder.FeatAttnLogitSoftcap:  true, // compiled into attention_prefill; a capped model takes no fused or steel kernel (prefillFusedOn)
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
	// GPT-J pairwise rotation: rope_f16_pw and
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
	// The q4k lane: each projection as its runs of same-kind rows (q4kLaneProj);
	// the xxxW/xxxS buffers above stay empty on the lane.
	qkvSegs, oSegs, guSegs, dSegs []q4kSeg
	qkvBias, preNorm, postNorm    Buffer
	qNorm, kNorm                  Buffer          // per-head QK-RMSNorm weights (Qwen3; zero if !qkNorm)
	moe                           *moeLayer       // non-nil ⇒ this layer's FFN is MoE (dense guW/dW unused)
	g4moe                         *gemma4MoeLayer // non-nil ⇒ Gemma-4 parallel dense‖MoE FFN (gemma4_moe.go)
	uLayerScalar                  Buffer          // Gemma 4 dense layer's output scalar; zero Buffer when 1 or absent
	layerScaleRead                bool            // the build read the layer's output scalar (any value), so its steps end with the layer scale
	// ffnI is this dense layer's FFN width and uFFNI its uniform. Equal to the model's I (and r.uI) for every family
	// except a Gemma 4 E-model, whose KV-shared layers are double-wide.
	ffnI  int
	uFFNI Buffer
	// Gemma 4 E-model PLE branch: input gate [P,H], projection [H,P], post norm [H]. Zero when the model has
	// no PLE.
	pleGW, pleGS, plePW, plePS, postPLENorm Buffer
	// kvShared: a Gemma 4 E-model layer that owns no K/V. It projects Q only, stores nothing, and attends over
	// layer kvSrc's cache, which r.kc[l]/r.vc[l] (and every slot's) alias.
	kvShared                  bool
	kvSrc                     int
	kvg                       int    // the KV group whose table this layer's attention reads: 0 the primary group (resident.kvGroup), 1 the window group (resident.kvWin); a shared layer takes its source's
	postAttnNorm, postMLPNorm Buffer // Gemma sandwich norms on each sublayer OUTPUT (zero if !sandwich)
	invf                      Buffer // per-layer RoPE inv-freq (Gemma local 10k vs global 1M base)
	mscale                    Buffer // per-layer YaRN mscale (RopeMscaleLayer; 1.0 = no-op for every family without it)
	uWindow                   Buffer // per-layer attention window (0 = full causal; Gemma mixes local/global)
	window                    uint32 // CPU-side twin of uWindow's value, for canUseAttnFA's dispatch-time guard (windows are out of scope for attention_fa — see its own doc comment)
	// GPT-2 (FeatLayerNorm/FeatNonGatedMLP/FeatOutBias): preNormBias/postNormBias are LayerNorm's
	// bias (unused for RMS families — layernorm_quant only dispatches when arch.Norm==NormLayer).
	// upW/upS is the SEPARATE (not gate-fused) up-projection a non-gated MLP uses instead of
	// guW/guS; oBias/upBias/downBias are the three projection biases FeatOutBias/FeatNonGatedMLP
	// need. All zero-value (unused, never dispatched) for every other family.
	preNormBias, postNormBias Buffer
	upW, upS                  Buffer
	oBias, upBias, downBias   Buffer
	downBiasSet               bool // the non-gated MLP's down projection has a bias of its own (the batched pass declines it)
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

	// mamba non-nil marks a Mamba-2 mixer layer (mamba.go): like delta, no KV cache, no q/k/v/o, no geom. A layer with
	// neither and a nil geom has no mixer at all (decoder.MixerNone: Nemotron-H's MLP blocks).
	mamba *mambaLayer
	// mla non-nil marks an MLA layer (mla.go): its KV cache is the latent rows in kc (vc aliases it), and it has no geom.
	mla *mlaLayer
	// ffn is whether a feed-forward block follows the mixer (decoder.Model.ResidentLayerOps); false on Nemotron-H's
	// Mamba and attention blocks.
	ffn bool
	// steps is the layer's per-token body, in order (buildLayerSteps), stepOps each step's operations in the program, and walk the
	// decoded sublayer walk of the layer's operations.
	steps   []layerStep
	stepOps [][]program.Op
	walk    program.LayerWalk
}

// recurrent reports whether the layer's mixer keeps per-sequence state instead of a KV cache: every path that cannot
// carry that state (a batched step, a KV copy, a layer-major prefill) declines on it.
func (L *residLayer) recurrent() bool { return L.delta != nil || L.mamba != nil }

// recurrentModel reports whether any layer has a recurrent mixer (DeltaNet or Mamba-2): the model-level form of
// residLayer.recurrent, for the paths that decline such a model whole.
func (r *resident) recurrentModel() bool { return r.dnet != nil || r.mamba != nil }

// recurrentPrefillOK reports whether the batched prefill implements this resident's recurrent mixers: none, DeltaNet in the
// shape prefill_deltanet.go covers (dnetPrefillOn, pre-norm layers, no LayerNorm bias; Olmo Hybrid's postOnly stays
// sequential), or Mamba-2 in the shape prefill_mamba.go covers (mambaPrefillOn, the widths below, pre-norm layers).
func (r *resident) recurrentPrefillOK() bool {
	switch {
	case r.mamba != nil:
		// The batched Mamba-2 prefill (prefill_mamba.go): the int8 GEMMs read K in 32-wide slabs, and mamba_conv_seq holds the window in 8 registers.
		p := r.mamba
		if !mambaPrefillOn || r.dnet != nil || r.postOnly || p.dConv < 2 || p.dConv > 8 || r.H%32 != 0 || p.dInner%32 != 0 {
			return false
		}
	case r.dnet == nil:
		return true
	case !dnetPrefillOn || r.postOnly:
		return false
	}
	for i := range r.layers {
		if r.layers[i].preNormBias != (Buffer{}) {
			return false
		}
	}
	return true
}

// q4kLaneShapeOK reports whether the q4k lane covers this resident's shape: the plain dense pre-norm one (Phi-3, Llama).
func (r *resident) q4kLaneShapeOK() bool {
	return !(r.sandwich || r.postOnly || r.parallelBlock || r.outBias || r.layerNorm || r.moe != nil || r.g4moe != nil ||
		r.recurrentModel() || r.pleP > 0 || r.attnSink || r.learnedPos || r.kvI8 || r.kvF32 || r.qkNorm)
}

// modelKnob is one of m's operator knobs (decoder.Model.Knob): the Load-time snapshot with Options.Knobs applied,
// never the live environment. "" when unset.
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

// pleState holds Gemma 4 E-model Per-Layer-Embedding (PLE) state: pleP is P (0 = no PLE); pleIn holds this
// token's [L*P] per-layer inputs, copied from the embedding row's tail; pleG/pleQ/pleSc hold branch scratch.
type pleState struct {
	pPLEGeluMul              Pipeline
	pleP                     int
	pleIn, pleG, pleQ, pleSc Buffer
	uPleP                    Buffer
}

// resident is a Metal-resident decoder: BuildResident uploads the weights once, and per token only the embedding and
// position uniforms change. Subsystem-specific pipelines, scratch buffers, and uniforms live in embedded subsystem
// structs defined in their owning files (deltanet.go, mamba.go, mla.go, gumbel_sample.go, greedy_chain.go,
// recurrent_checkpoint.go, lora.go, kvpool.go, forward.go, encode.go).
type resident struct {
	// knob is the model's decoder.Model.Knob, for the knobs read per call (PrefillLast, the fast-prefill floor).
	knob func(name string) (string, bool)
	// prog is the model's program (decoder.Model.Program), read-only; nothing is encoded from it yet.
	prog *program.Program

	d                                                                  *Device
	q                                                                  Queue
	pRms, pQv, pGemv, pGemvResid, pRope, pRope2, pKv, pAttn, pSw, pRes Pipeline
	pLayerScale                                                        Pipeline // Gemma 4 dense layer scalar
	pSA, pSABias, pSAResid                                             Pipeline // Stage A gemv
	pArgFinish                                                         Pipeline // fused block-argmax lm head reduce
	pArgRowsPart                                                       Pipeline // batched step row partial argmaxes (argmax_rows_part)
	pW8Rows                                                            [batchMaxSeqs + 1]Pipeline
	pQKNorm, pRmsF32                                                   Pipeline // per-head QK-RMSNorm; Gemma sandwich in-place RMSNorm
	pGemvW8, pGemvW8Amax                                               Pipeline // int8 GEMV + fused block-argmax LM head
	pCopyVec, pCopyU32                                                 Pipeline // copy_f32 (batched emb); copy_u32 (routing capture)
	pLayerNorm, pActQuant                                              Pipeline // GPT-2 layernorm_quant, act_quant
	pSABiasResid, pCoalBiasResid                                       Pipeline // biased residual GEMVs (o-proj, down-proj)
	pNgActRows                                                         Pipeline // non-gated MLP f16 activation (prefill)
	pRmsRows, pQvRows, pSwRows, pRope2Rows, pKvRows, pAttnRows         Pipeline // batched-step multi-row kernels (batch_rows.go)
	pQKNormRows                                                        Pipeline // qk_norm over batched-step rows
	pMoeBatch                                                          [5]Pipeline
	pMoeCombine, pMoeCombineAcc                                        Pipeline

	// Embedded subsystem state (each struct lives in its owning domain file).
	pleState
	recurrentCkpt
	w4f16Lane
	q4kState
	attnFAState
	gemvRowsState
	chainState
	gumbelState
	loraState
	kvPoolState
	batchFwdState
	execState
	dnetState
	mambaState
	mlaState

	// Model geometry and feature flags.
	H, nL, nH, I, V    int
	ctxCap             int // resolved resident KV capacity in positions (resolveMetalCtxCap)
	ctxCeil            int // build plan ceil (metalCtxPlan), 0 for none
	g4VNorm            bool
	qkNorm             bool
	qkNormWhole        bool
	sandwich           bool
	postOnly           bool
	layerNorm          bool
	layerNormBias      bool
	nonGatedMLP        bool
	outBias            bool
	learnedPos         bool
	pairwiseRoPE       bool
	parallelBlock      bool
	attnSink           bool
	w8, w8Attn         bool
	preciseMath        bool
	prefillOK          bool
	moeStep            bool
	recurrentStep      bool
	gemma4Step         bool
	pagedNoHead        bool
	ksd7Old            bool
	promptStepOff      bool
	curNKeys           int
	encNKeys           int
	encLayer           int
	stepHeadroom       int64
	stepDecline        string
	finalSoftcap       float32
	attnSoftcap        float32
	embedScale         float32
	logitScale         float32
	gptossAlpha        float64
	gptossLimit        float64
	attnTempBeta       float64
	attnTempOrigMaxPos float64

	// Host/model weights, layers, and MoE/prefill/batch owners.
	embed                    *linalg.WeightMat
	posEmbed                 *linalg.WeightMat // [MaxPositions, H] learned position embedding (learnedPos only)
	layers                   []residLayer
	finalNorm, finalNormBias Buffer
	lmW, lmS                 Buffer
	moe                      *moeResident
	g4moe                    *gemma4MoeResident
	batch                    *batchState
	pf                       *prefillState
	pfErr                    error
	decode                   decodeShape

	// Per-token scratch and shared uniforms.
	qkv, gu                                                    Buffer
	mropeAxis                                                  Buffer
	x, aq, aSc, ctx, cq, cSc, oO, mq, mSc, dq, dSc, dO, logits Buffer
	invf, uH, uI, uNH, uScale, uEps                            Buffer
	uPos, uNKeys, uRopePos, uQTempScale                        Buffer
	uAct, uAddOne, uWindow, uZero, vNormUnit                   Buffer
	uLNHasBias                                                 Buffer
	uQKWholeOne, uQKWholeHD                                    Buffer
	part, tok, uP                                              Buffer
	logitsHost                                                 []float32

	// Profiling, alias, residency set, and test/introspection counters.
	gpuStart, gpuEnd, kernStart, kernEnd             float64
	prof                                             pagedProfile
	alias                                            *weightAlias
	residency                                        ResidencySet
	residencyBufs                                    []Buffer
	emodelLayerMajorRuns                             int
	g4LayerMajorRuns, g4BatchGroups, g4BatchMaxCnt   int
	moeBatchGroups, moeLayerMajorRuns                int
	prefillResidNonFinite, poisonPrefillResidForTest bool
	moeForce                                         struct{ idx, wgt Buffer }
	moeCap                                           struct {
		idx, wgt Buffer
		rows     int
		major    func(l int, idx []uint32, wgt []float32)
	}
}

// int4Buf uploads a WeightMat as W4A8 (int4, group=32) plus f16 group scales. An int4 weight (a Quant:"int4" load) is
// consumed directly, with no int8 step; an int8 weight is re-quantized through the validated packer. One-time at build.
func int4BufOn(line string, d *Device, w *linalg.WeightMat) (Buffer, Buffer, error) {
	return int4BufAOn(line, d, nil, w)
}

// int4Buf is int4BufOn on gpu.UnlinedLine: tests call it; production code names its plan line.
func int4Buf(d *Device, w *linalg.WeightMat) (Buffer, Buffer, error) {
	return int4BufOn(gpu.UnlinedLine, d, w)
}

// int4BufA is int4Buf with an optional weightAlias: when a is non-nil and w's nibbles live in the.
// giw mapping, the nibbles are bound in place instead of copied; the f16 scales are built exactly as the
// copy path builds them.
func int4BufAOn(line string, d *Device, a *weightAlias, w *linalg.WeightMat) (Buffer, Buffer, error) {
	// The W4A8 layout and every GEMV kernel assume K is a multiple of the group (32): rows pack K/8 words and K/32
	// scales with no partial-group handling. A weight with K%32 != 0 would pack a truncated last group with a row
	// stride the kernel disagrees with, silently wrong or a panic at K<32. Decline, so BuildResident falls back to
	// CPU.
	if k := w.Cols(); k%32 != 0 {
		return Buffer{}, Buffer{}, fmt.Errorf("metal: W4A8 pack needs K%%32==0 (group=32), got K=%d — declining to CPU", k)
	}
	if nib, ok := a.nibblesOn(line, d, w); ok {
		if sc, ok := a.scales16On(line, d, []*linalg.WeightMat{w}); ok { // the scales are in the mapping: bound in place too
			return nib, sc, nil
		}
		_, q4s16, _, _ := w.Int4F16()
		sb := gpu.NewBufferLenOfOn[uint16](d, line, len(q4s16))
		copy(sb.U16s(), q4s16) // the scales as stored: int4DirectWords' f32 round trip returns these bits
		a.addCopy(int64(2 * len(q4s16)))
		if a != nil {
			a.f16Converted++
		}
		return nib, sb, nil
	}
	if _, _, ok := w4a8Size(w); !ok {
		return Buffer{}, Buffer{}, fmt.Errorf("metal: weight kind %q is not int8 or int4", w.Kind())
	}
	nw, ns, _ := w4a8Size(w)
	wb, sb := gpu.NewBufferLenOfOn[uint32](d, line, nw), gpu.NewBufferLenOfOn[uint16](d, line, ns)
	if w4a8Into(w, wb.U32s(), sb.U16s()) {
		a.addCopy(int64(4*nw + 2*ns))
	}
	return wb, sb, nil
}

// w4a8Size is the uint32 words and f16 group scales w's W4A8 upload takes, ok=false when w is neither group-32 int4 nor
// int8.
func w4a8Size(w *linalg.WeightMat) (words, scales int, ok bool) {
	if _, _, group, ok := w.Int4F16(); ok && group == 32 {
		n, k := w.Rows(), w.Cols()
		return n * k / 8, n * k / 32, true
	}
	if _, _, _, ok := w.Int8(); ok {
		n, k := w.Rows(), w.Cols()
		return n * (k / 8), n * (k / 32), true
	}
	return 0, 0, false
}

// w4a8Into writes w's W4A8 words and f16 group scales into words and scales (views of the device buffers, sized by
// w4a8Size) with no whole-tensor staging copy: a group-32 int4 weight's nibbles and binary16 scales verbatim
// (int4DirectWords' bytes, whose f32 round trip of a scale returns its stored bits; reported true), else its int8 rows
// re-quantized one row at a time (false). It must be called on a weight w4a8Size accepts.
func w4a8Into(w *linalg.WeightMat, words []uint32, scales []uint16) (direct bool) {
	if q4, q4s16, group, ok := w.Int4F16(); ok && group == 32 {
		if len(q4) > 0 {
			copy(unsafe.Slice((*byte)(unsafe.Pointer(&words[0])), len(q4)), q4)
		}
		copy(scales, q4s16)
		return true
	}
	q8, sc, _, _ := w.Int8()
	N, K := w.Rows(), w.Cols()
	row, sd := make([]float32, K), make([]float32, K/32) // one row's scratch, reused: nothing allocated per row
	for n := range N {
		for k, q := range q8[n*K : (n+1)*K] {
			row[k] = float32(q) * sc[n] // the int8 weight, dequantized to f32 for the W4A8 re-pack
		}
		kernels.PackW4A8RowInto(row, words[n*(K/8):(n+1)*(K/8)], sd)
		f16Into(scales[n*(K/32):(n+1)*(K/32)], sd)
	}
	return false
}

// f16Into narrows src to f16 bits in dst (len(dst) >= len(src)).
func f16Into(dst []uint16, src []float32) {
	for i, s := range src {
		dst[i] = f32ToF16(s)
	}
}

// int4Concat re-quantizes and row-concatenates several same-K WeightMats into ONE W4A8
// buffer — the fusion enabler (combined QKV → one GEMV, combined gate/up → one GEMV).
func int4ConcatOn(line string, d *Device, wms ...*linalg.WeightMat) (Buffer, Buffer) {
	// The buffers are allocated at their final size (N*K/8 words and N*K/32 f16 scales per group-32 tensor) and every part
	// is written straight into them: a whole-projection staging copy is garbage the build does not collect.
	var totalWords, totalScales int
	for _, w := range wms {
		// K%32==0 is a hard W4A8 invariant (group=32); see int4BufA. int4Concat has no error return and runs only on the
		// build path, so it panics; buildResident's recover turns that into a clean CPU decline, as with its wrong-kind
		// panic below.
		if k := w.Cols(); k%32 != 0 {
			panic(fmt.Sprintf("metal: W4A8 concat needs K%%32==0 (group=32), got K=%d", k))
		}
		nw, ns, ok := w4a8Size(w)
		if !ok {
			// Neither group-32 int4 nor int8. The common cause is the repacked-only int4 policy (wantsCanonicalInt4): a
			// tensor built with Backend "cpu", or loaded generically and handed to BuildResident out of band, may have no
			// canonical bytes left. Name that condition specifically; a tensor that is neither int8 nor int4 at all (a
			// programming error) gets the generic message.
			if w.IsInt4() {
				panic(fmt.Sprintf("metal: int4 tensor has no canonical bytes (layout %s-only): "+
					"model was loaded for a CPU-only backend; load with Options.Backend set to "+
					"\"metal\" to keep canonical bytes for residency", w.Int4Layout()))
			}
			panic(fmt.Sprintf("metal: int4Concat weight kind %q not int8 or int4", w.Kind()))
		}
		totalWords, totalScales = totalWords+nw, totalScales+ns
	}
	wb, sb := gpu.NewBufferLenOfOn[uint32](d, line, totalWords), gpu.NewBufferLenOfOn[uint16](d, line, totalScales) // f16 group scales
	words, scales := wb.U32s(), sb.U16s()
	for _, w := range wms {
		nw, ns, _ := w4a8Size(w)
		w4a8Into(w, words[:nw], scales[:ns]) // int4-direct consumes nibbles verbatim; int8 re-quantizes row by row
		words, scales = words[nw:], scales[ns:]
	}
	return wb, sb
}

// int4Concat is int4ConcatOn on gpu.UnlinedLine: tests call it; production code names its plan line.
func int4Concat(d *Device, wms ...*linalg.WeightMat) (Buffer, Buffer) {
	return int4ConcatOn(gpu.UnlinedLine, d, wms...)
}

// buildResident builds a Metal resident decoder from a loaded Model: it uploads the weights, creates the pipelines and
// allocates the KV slots. It returns an error, which the caller treats as a decline to the CPU path, when the model's
// shape or the device cannot run it resident. The work is residentBuild's phases (resident_build.go), run in the order the
// calls below list; the order is the order of the allocations the memory gate and the lease census read.
func buildResident(m *decoder.Model, plan metalCtxPlan) (res *resident, err error) {
	// Entry point: convert the build-time panics (pipeline-compile failure, non-int4 expert weights, buffer OOM in
	// model.go, moe.go and gemma4_moe.go) into the error this signature promises, so a direct caller gets a decline,
	// not a process-killing panic. Registered first, so it runs last, after the cleanup defer below has released the
	// partial device state. The backend wrapper keeps its own recover() as defence in depth.
	defer func() {
		if p := recover(); p != nil {
			mem.RepanicBug(p)
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
	// The KV slot count is priced before any buffer below exists. Its budget, metalMemoryCeiling, reads the live
	// available memory; read after the weights were allocated, that figure already excludes the resident's own
	// buffers while the base it is compared with (residentNeedBytes) still counts them, so the weights would be
	// counted twice and too few slots granted.
	// The layer shape the encoder dispatches on comes from the program; a program whose layers ask for two placements
	// at once declines here, before any device state exists.
	shape, err := program.ShapeOf(m.Program())
	if err != nil {
		return nil, fmt.Errorf("metal: %w", err)
	}
	kvSlots := metalKVSlots(m, plan)
	stepHeadroom := metalStepHeadroom(m, plan, kvSlots)
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
	b := &residentBuild{m: m, plan: plan, d: d, shape: shape, kvSlots: kvSlots, stepHeadroom: stepHeadroom}
	defer func() { _ = b.kq.Close() }() // nil-safe: only the q4k lane opens one
	if err := b.compileKernels(); err != nil {
		return nil, err
	}
	r := b.r
	b.bindPipelines()
	if err := b.buildMixers(); err != nil {
		return nil, err
	}
	if err := b.selectLanes(); err != nil {
		return nil, err
	}
	if err := b.uploadLayers(); err != nil {
		return nil, err
	}
	if err := b.allocKVTables(); err != nil {
		return nil, err
	}
	if err := b.uploadHead(); err != nil {
		return nil, err
	}
	if err := b.checkWidths(); err != nil {
		return nil, err
	}
	b.selectGemvRows()
	if err := b.sizeDecodeScratch(); err != nil {
		return nil, err
	}
	b.pinResidencySet()
	if err := b.finish(); err != nil {
		return nil, err
	}
	ok = true // construction complete — the resident owns everything; Close (not the defer) frees it
	return r, nil
}

// Test seams: re-drop each Gemma 4 fix (the layer scalar, v_norm), so a gate can show it
// reproduces the before-fix numbers. Never set outside tests.
var (
	g4DropLayerScalarForTest bool
	g4DropVNormForTest       bool
)

// Planted defects: each re-introduces one E-model mistake on the Metal side only, so
// the gate of the Metal resident against the CPU must turn red under it. Never set outside tests.
var (
	g4KeepSharedKVStoreForTest bool // (5) a shared layer stores its K/V over its source's cache
	g4SkipPLEForTest           bool // (1) the PLE branch skipped
	g4KVSrcOffForTest          bool // (3) a shared layer reads the previous owning layer of its type, not its source
	g4OneFFNWidthForTest       bool // (4) every dense layer at the model's one intermediate width
)
