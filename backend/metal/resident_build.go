//go:build darwin

package metal

import (
	"fmt"
	"slices"
	"strings"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// residentBuild is buildResident's state: what each build phase reads from the phases before it. A phase is a method that
// runs in build order and returns the error that declines the build; buildResident calls them in sequence, and the device's
// ledgers (buildResident's deferred release) own whatever a phase allocated when a later one fails.
type residentBuild struct {
	m                *decoder.Model
	bv               decoder.BackendModelView
	d                *Device
	r                *resident
	shape            program.Shape
	pipe             func(name string) Pipeline
	kvSlots          int
	stepHeadroom     int64
	attnSoftcap      float32
	addOne           uint32
	win              int
	kq               *decoder.KQuantReader // the file's Q5_K and Q6_K blocks for the q4k lane (q4kLaneProj); nil uploads their int8 copies
	w                *decoder.Weights
	alias            *weightAlias
	maxNHhd          int // the widest layer's attention sizes, which the shared per-token scratch is cut to
	maxKvDim         int
	maxHd            int
	maxAttnFAPartial int // R2: attnFAPartialFloats at its widest layer; see attnFAPartial's own comment
}

// compileKernels compiles the kernel library and creates the resident, with its dimensions, context cap and the memory-guard readings
// the later phases size from. It sets b.bv, b.pipe and b.r.
func (b *residentBuild) compileKernels() error {
	m, d, kvSlots, stepHeadroom := b.m, b.d, b.kvSlots, b.stepHeadroom
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
	bv := m.BackendView()
	attnSoftcap := bv.AttnLogitSoftcap()
	if attnSoftcapDropForTest {
		attnSoftcap = 0
	}
	libSrc, err := withAttnSoftcap(allKernels+mc3RowsKernels+kvF32Kernels+kvRowsKernels, attnSoftcap) // + MC3 S3's multi-row forms (batch_rows.go), attention_f32 (kv_f32.go) and the step's int8 and f32 KV forms (kv_rows.go), derived from allKernels
	if ksc2OldDownForTest {
		libSrc = "#define KSC2_OLD_DOWN 1\n" + libSrc
	}
	if p := mambaSrcPlantForTest; err == nil && p[0] != "" {
		libSrc = strings.Replace(libSrc, p[0], p[1], 1)
	}
	if err != nil {
		return err
	}
	lib, err := compile(libSrc, MSL3_1)
	if err != nil {
		return err
	}
	pipe := func(name string) Pipeline {
		p, e := d.NewComputePipeline(lib, name)
		if e != nil {
			panic(fmt.Sprintf("metal pipeline %s: %v", name, e))
		}
		return p
	}
	H, nL, nH, _, _, I, V := bv.Dims() // model-level hd/nKV dropped — geometry is per-layer (geom.go)
	r := &resident{knob: m.Knob, prog: m.Program(), d: d, H: H, nL: nL, nH: nH, I: I, V: V, preciseMath: preciseMath, stepHeadroom: stepHeadroom,
		moeStep: m.ResidentMoEStep(), recurrentStep: m.ResidentRecurrentStepOn(kvSlots), gemma4Step: m.ResidentGemma4StepOn(kvSlots), ckptSlot: -1}
	r.chainEmbBase = residentNeedBytes(m) + int64(kvSlots-1)*metalSlotBytes(m) // every slot's KV and recurrent state
	r.chainEmbGuardOff = modelKnob(m, "GOINFER_NO_RESIDENT_MEM_GUARD") != ""
	if r.ctxCap, err = resolveMetalCtxCap(m); err != nil {
		return err
	}
	b.bv, b.pipe, b.r, b.attnSoftcap = bv, pipe, r, attnSoftcap
	return nil
}

// bindPipelines creates the decode, batched-step and head pipelines every build takes, and the lane and flag fields they pick
// (the q4k lane, the pairwise rope, the KV precision), from the shape the program resolved.
func (b *residentBuild) bindPipelines() {
	m, r, bv, pipe, shape, V := b.m, b.r, b.bv, b.pipe, b.shape, b.r.V
	attnSoftcap := b.attnSoftcap
	r.pRms, r.pQv, r.pGemv = pipe("rmsnorm_quant"), pipe("quant_vec"), pipe("gemv_w4a8_coal")
	r.pGemvResid = pipe("gemv_w4a8_resid")
	r.pSA, r.pSABias, r.pSAResid = pipe("gemv_w4a8_sa"), pipe("gemv_w4a8_sa_bias"), pipe("gemv_w4a8_sa_resid")
	r.pRmsF16 = pipe("rmsnorm_f16_act")
	r.pF32ToF16 = pipe("f32_to_f16")
	r.pSAf16, r.pSAf16Bias, r.pSAf16Resid = pipe("gemv_w4f16_sa"), pipe("gemv_w4f16_sa_bias"), pipe("gemv_w4f16_sa_resid")
	r.decodeLaneW4F16 = modelKnob(m, "GOINFER_METAL_DECODE_LANE") == "w4f16"
	if r.q4kLane = m.Quant() == "q4k"; r.q4kLane {
		r.decodeLaneW4F16 = false
		r.q4kR, r.w8F32R = q4kGemvRows, w8F32GemvRows
		qr, wr := fmt.Sprint(q4kGemvRows), fmt.Sprint(w8F32GemvRows)
		r.pQ4K, r.pQ4KResid, r.pQ4KBias = pipe("gemv_q4k_f32_rows"+qr), pipe("gemv_q4k_f32_resid_rows"+qr), pipe("gemv_q4k_f32_bias_rows"+qr)
		r.pW8F32, r.pW8F32Resid, r.pW8F32Bias = pipe("gemv_w8_f32_rows"+wr), pipe("gemv_w8_f32_resid_rows"+wr), pipe("gemv_w8_f32_bias_rows"+wr)
		r.pRmsF32Out, r.pSwF32 = pipe("rmsnorm_f32_out"), pipe("swiglu_f32")
		r.pRmsF32OutRows, r.pSwF32Rows = pipe("mc3_rmsnorm_f32_out_rows"), pipe("mc3_swiglu_f32_rows")
		for k, p := range map[string]*[3]Pipeline{"q5k": &r.pQ5K, "q6k": &r.pQ6K} {
			R := map[string]int{"q5k": q5kGemvRows, "q6k": q6kGemvRows}[k]
			*p = [3]Pipeline{pipe(fmt.Sprintf("gemv_%s_f32_rows%d", k, R)), pipe(fmt.Sprintf("gemv_%s_f32_resid_rows%d", k, R)), pipe(fmt.Sprintf("gemv_%s_f32_bias_rows%d", k, R))}
		}
		for B := 1; B <= batchMaxSeqs; B++ {
			r.pQ4KRows[B], r.pW8F32Rows[B] = pipe(fmt.Sprintf("mc3_gemv_q4k_f32_rows%d", B)), pipe(fmt.Sprintf("mc3_gemv_w8_f32_rows%d", B))
			r.pQ5KRows[B], r.pQ6KRows[B] = pipe(fmt.Sprintf("mc3_gemv_q5k_f32_rows%d", B)), pipe(fmt.Sprintf("mc3_gemv_q6k_f32_rows%d", B))
		}
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
	// Pairwise rope (docs/completed/task-metal-pairwise-rope-2026-10.md): a GPT-J pairwise family (Cohere, Command-R7B, Aya,
	// GLM-OCR) binds the pairwise twins into the same fields, so no dispatch site changes; every other family keeps these.
	if r.pairwiseRoPE = bv.PairwiseRoPE(); r.pairwiseRoPE {
		r.pRope, r.pRope2 = pipe("rope_pw"), pipe("rope2_pw")
	}
	r.pSw, r.pRes = pipe("swiglu_quant"), pipe("residual")
	r.pLayerScale = pipe("layer_scale")
	if P := bv.Gemma4PLEDim(); P > 0 {
		r.pPLEGeluMul, r.pleP = pipe("ple_gelu_mul"), P
	}
	r.pRmsRows, r.pQvRows = pipe("mc3_rmsnorm_quant_rows"), pipe("mc3_quant_vec_rows")
	r.pSwRows, r.pRope2Rows = pipe("mc3_swiglu_quant_rows"), pipe("mc3_rope2_rows")
	r.pKvRows, r.pAttnRows = pipe("mc3_kv_store_rows"), pipe("mc3_attention_rows")
	r.pQKNormRows = pipe("mc3_qk_norm_rows")
	r.pMoeBatch[2], r.pMoeBatch[4] = pipe("moe_batch_gemv2"), pipe("moe_batch_gemv4")
	r.pMoeCombine, r.pMoeCombineAcc = pipe("moe_batch_combine"), pipe("moe_batch_combine_acc")
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
	r.pQKNorm, r.pRmsF32 = pipe("qk_norm"), pipe("rmsnorm_f32")
	r.qkNorm = shape.QKNorm
	r.qkNormWhole = shape.QKNormWhole // G5 (docs/completed/task-gpu-paths-2026-09.md): Olmo 3/Olmo Hybrid
	r.sandwich = shape.Sandwich
	r.attnSoftcap = attnSoftcap
	r.postOnly = shape.PostOnly // G5: Olmo 3/Olmo Hybrid
	r.pLayerNorm, r.pActQuant = pipe("layernorm_quant"), pipe("act_quant")
	r.pLoraDelta = pipe("lora_delta") // G3/P-11: compute-time LoRA, down+up fused into one kernel
	r.pSABiasResid, r.pCoalBiasResid = pipe("gemv_w4a8_sa_bias_resid"), pipe("gemv_w4a8_resid_bias")
	r.layerNorm = shape.LayerNorm
	r.parallelBlock = shape.Parallel  // G5: Cohere/Command-R + Cohere2/Command-R7B
	r.logitScale, _ = bv.LogitScale() // ok=false ⇒ 1, already finalizeLogits's no-op value
	r.nonGatedMLP = shape.NonGated
	if r.nonGatedMLP {
		r.pNgActRows = pipe("nongated_act_rows_f16")
	}
	r.outBias = shape.OutBias
	r.learnedPos = shape.LearnedPos
	r.attnSink = bv.HasAttnSinks()
	if alpha, limit, isGptOss := bv.GptOssAct(); isGptOss {
		r.gptossAlpha, r.gptossLimit = float64(alpha), float64(limit)
	}
	// The f32 KV path (kv_store_f32, attention_f32) runs when the resolver delivers f32: on an explicit --kv f32, never for auto
	// (a matched-input confirmer showed f16 versus f32 KV storage is not what degrades Gemma's contexts,
	// docs/code-notes/metal.md#buildResident.kvF32). Every fast path declines it by name.
	// The resolver's answer, not the raw request (decoder.ResolveKV). It names the GQA K/V cache; an MLA model's latent is f32
	// whatever it says (metal/mla.go), so neither flag applies there.
	kvDelivered := m.KVDeliveredFor("metal")
	if _, _, _, _, _, _, _, _, isMLA := m.MLAResidentParams(); isMLA {
		kvDelivered = ""
	}
	r.kvF32 = kvDelivered == "f32"
	if r.kvF32 {
		r.pKv, r.pAttn = pipe("kv_store_f32"), pipe("attention_f32")
		r.pKvRows, r.pAttnRows = pipe("mc3_kv_store_f32_rows"), pipe("mc3_attention_f32_rows")
	}
	r.kvI8 = kvDelivered == "i8"
	if r.kvI8 {
		r.pKvI8, r.pAttnI8 = pipe("kv_store_i8"), pipe("attention_i8")
		r.pKvI8Rows, r.pAttnI8Rows = pipe("mc3_kv_store_i8_rows"), pipe("mc3_attention_i8_rows")
	}
}

// buildMixers builds the model-level state of the MoE, Gemma 4 MoE, Gated-DeltaNet, Mamba-2 and MLA mixers, and their pipelines, for the
// families that have them (each is nil otherwise), and reads the gated-MLP activation and sliding window the decode shape takes. It sets b.addOne
// and b.win.
func (b *residentBuild) buildMixers() error {
	m, d, r, pipe := b.m, b.d, b.r, b.pipe
	H, nL := r.H, r.nL
	var err error
	// Gated-MLP activation: ordinals ARE decoder.ActKind's iota (0=GELU-tanh, 1=SiLU), so this
	// passes straight through to glu_act. Gemma is GeGLU; everything else admitted is SwiGLU.
	addOne := uint32(0)
	if m.RMSAddOne() {
		addOne = 1
	}
	win := max(m.SlidingWindowResident(), 0)              // 0 = full causal; Mistral is all-local with this window
	if r.moe, err = buildMoE(d, m, pipe, H); err != nil { // nil for a dense (or gemma4-MoE) model; error ⇒ decline
		return err
	}
	if r.g4moe, err = buildGemma4MoE(d, m, pipe, H, nL); err != nil { // Gemma-4 enable_moe_block; nil otherwise; error ⇒ decline
		return err
	}
	if r.dnet, err = buildDeltaNet(d, m, pipe); err != nil { // Gated-DeltaNet mixer; nil otherwise; error ⇒ decline
		return err
	}
	if r.mamba, err = buildMamba(m); err != nil { // Mamba-2 mixer; nil otherwise; error ⇒ decline
		return err
	}
	if r.mamba != nil {
		r.buildMambaPipes(pipe)
	}
	if r.mla, err = buildMLA(m, r.nH, H); err != nil { // MLA; nil otherwise; error ⇒ decline
		return err
	}
	if r.mla != nil {
		r.pMlaStore, r.pMlaHeadMV, r.pMlaQRope, r.pMlaAttn = pipe("mla_store"), pipe("mla_head_mv"), pipe("mla_qrope"), pipe("mla_attn")
		r.mlaS = buildMLAScratch(d, m)
	}
	if r.dnet != nil {
		r.pDnConv, r.pDnGates = pipe("delta_conv"), pipe("delta_gates")
		r.pDnNorm, r.pDnRule, r.pDnGNorm = pipe("delta_norm"), pipe("delta_rule"), pipe("delta_gnorm")
		r.pDnQSplit, r.pDnAttnGate = pipe("delta_qsplit"), pipe("delta_attn_gate")
		r.pDnConvSeq, r.pDnGatesRows, r.pDnNormRows = pipe("delta_conv_seq"), pipe("delta_gates_rows"), pipe("delta_norm_rows")
		r.pDnRuleSeq, r.pDnGNormRows, r.pDnProjW8Rows = pipe("delta_rule_seq"), pipe("delta_gnorm_rows"), pipe("delta_proj_w8_rows")
		r.pDnRuleSeqTG, r.pDnConvSeqReg = pipe("delta_rule_seq_tg"), pipe("delta_conv_seq_reg")
	}
	b.addOne, b.win = addOne, win
	return nil
}

// selectLanes decides which weight lane the build takes (native int8, int4-with-int8-attention, the q4k lane) and whether the batched prefill
// pass is admitted, and declines a model none of them can run. It opens the q4k lane's K-quant reader into b.kq, which buildResident closes.
func (b *residentBuild) selectLanes() error {
	m, r, pipe := b.m, b.r, b.pipe
	var err error
	// Native int8 (docs/completed/task-metal-int8-2026-10.md): a dense model whose body projections are all int8 runs
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
		if !q4kNativeOffForTest {
			if b.kq, err = m.OpenKQuant(); err != nil {
				return err
			}
		}
	}
	if r.w8Attn {
		r.decodeLaneW4F16 = false // the f16 lane reads int4 attention weights
	}
	if why := f32Projection(m); why != "" {
		// An f32 load reaches here only when f32 was asked for; this backend runs int4 or int8 weights and does not
		// quantize them behind the request (docs/completed/task-metal-int8-2026-10.md, slice 4). A clear decline instead of
		// int4Concat's panic on an f32 kind.
		return fmt.Errorf("weights loaded at f32 (%s); metal runs int4 or int8 weights: load with -quant int4 or int8int8 to use the GPU", why)
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
	dnetOK := r.recurrentPrefillOK()
	if r.dnet != nil && dnetOK {
		missing = slices.DeleteFunc(missing, func(f decoder.ResidentFeature) bool { return f == decoder.FeatDeltaNet })
	}
	if r.mamba != nil && dnetOK {
		missing = slices.DeleteFunc(missing, func(f decoder.ResidentFeature) bool { return f == decoder.FeatSSM })
	}
	if r.nonGatedMLP && !slices.ContainsFunc(m.Weights().Layers, func(lw decoder.LayerWeights) bool { return len(lw.DownBias) > 0 }) {
		// KS-D8: the pass's non-gated MLP takes a relu² activation and no down-projection bias (another activation stays
		// missing on its own feature). This runs before r.layers is built, so it reads the weights, not residLayer.downBiasSet.
		missing = slices.DeleteFunc(missing, func(f decoder.ResidentFeature) bool {
			return f == decoder.FeatNonGatedMLP || f == decoder.FeatNonGatedReLU2
		})
	}
	if r.q4kLane && !r.q4kLaneShapeOK() {
		return fmt.Errorf("metal: --quant q4k runs on Metal for the plain dense pre-norm shape only (Phi-3, Llama); this family has a variant the q4k lane does not cover")
	}
	r.ksd7Old = ksd7OldDequantForTest
	r.prefillOK = !r.q4kLane && len(missing) == 0 && !m.HasPerLayerGeometry() &&
		!(r.moe != nil && r.moe.paged) && dnetOK && !r.kvF32 &&
		!r.attnSink && // D-C01: gpt-oss's sink and clamped, biased SwiGLU are in no prefill kernel; explicit, so a feature-map edit cannot admit it
		prefillStepsDecline(m.Program()) == "" // a step with no batched block: MLA's latent attention, Gemma 4's MoE block

	return nil
}

// uploadLayers creates the command queue, uploads the embedding, every layer's weights and norms, and allocates each layer's KV pool,
// and checks that every layer's declared KV rows are what the build holds. It sets b.w, b.alias and the widest-layer sizes the scratch is cut to.
func (b *residentBuild) uploadLayers() error {
	m, d, r, bv := b.m, b.d, b.r, b.bv
	H, nL := r.H, r.nL
	kvSlots, win, kq := b.kvSlots, b.win, b.kq
	r.q = d.NewCommandQueue()

	w := bv.Weights()
	r.embed = &w.Embed
	if r.learnedPos {
		r.posEmbed = bv.PosEmbed()
	}
	// Whether THIS family's LayerNorm carries a bias (GPT-2) or not (Cohere) — a per-arch
	// constant, so layer 0's presence/absence decides it for every layer.
	if r.layerNorm && len(w.Layers) > 0 {
		r.layerNormBias = len(w.Layers[0].PreAttnNormBias) > 0
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
			return int8BufAOn(mem.LineDense, d, a, wm)
		}
		return int4BufAOn(mem.LineDense, d, a, wm)
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
			return int8ConcatOn(mem.LineDense, d, alias, wms...)
		}
		return int4ConcatAOn(mem.LineDense, d, alias, wms...)
	}
	r.layers = make([]residLayer, nL)
	r.kc, r.vc = make([]Buffer, nL), make([]Buffer, nL)
	r.kvSlotBytes = make([]int, nL)
	r.kvNullBytes = make([]int, nL)
	if r.kvI8 {
		r.ks, r.vs = make([]Buffer, nL), make([]Buffer, nL)
	}
	// Per-layer attention geometry, deduped by value (geom.go). Uniform families resolve every
	// layer to one geom; Gemma 4's local/global interleave resolves to two. maxNHhd/maxKvDim size
	// the shared per-token scratch to the widest layer (== the model shape for a uniform family).
	geomCache := map[[4]int]*attnGeom{}
	var maxNHhd, maxKvDim, maxHd int
	var maxAttnFAPartial int // R2: attnFAPartialFloats at its widest layer; see attnFAPartial's own comment
	_, _, _, _, _, attnGate, dnetOK := m.Qwen35ResidentParams()
	for l := range nL {
		lw := &w.Layers[l]
		lv := bv.Layer(l)
		var L residLayer
		var kEqV bool // false for isDelta (no geometry) and qGate (this family never has K=V); set below otherwise
		mixer, hasFFN := lv.Ops()
		L.ffn = hasFFN
		isDelta := dnetOK && mixer == decoder.MixerDeltaNet
		isAttn := mixer == decoder.MixerAttention
		if isDelta {
			// Gated-DeltaNet mixer layer: no attention projections, no KV cache, no geometry at
			// all — see metal/deltanet.go and residLayer.delta's doc. L.geom stays nil and
			// r.kc[l]/r.vc[l] stay zero-value; encodeLayer never calls encodeAttention for this
			// layer, so nothing downstream dereferences them.
			DL, e := buildDeltaNetLayer(d, m, l, r.dnet, mk)
			if e != nil {
				return e
			}
			L.delta = DL
		} else if mixer == decoder.MixerMamba {
			// Mamba-2 mixer layer (mamba.go): no attention projections, no KV cache, no geometry, as for DeltaNet.
			if r.mamba == nil {
				return fmt.Errorf("metal: layer %d is a Mamba-2 layer but the model gives no Mamba geometry", l)
			}
			ML, e := buildMambaLayer(d, m, l, H, r.mamba)
			if e != nil {
				return e
			}
			L.mamba = ML
		} else if mixer == decoder.MixerMLA {
			// MLA (mla.go): the latent rows are this layer's KV cache, in kc with its slots as for attention (vc aliases kc).
			if r.mla == nil {
				return fmt.Errorf("metal: layer %d is an MLA layer but the model gives no MLA geometry", l)
			}
			ML, e := buildMLALayer(d, m, l, H, r.mla)
			if e != nil {
				return e
			}
			L.mla = ML
			lat, e := kvCheckRows(l, m.LayerBuffers(l), kvLayerWant{latent: r.mla.latDim})
			if e != nil {
				return e
			}
			r.kvNullBytes[l] = decoder.KVBlockRows * lat * 4
			kvBytes := kvBlocksFor(r.ctxCap) * r.kvNullBytes[l]
			r.kvContig = true
			r.kvSlotBytes[l] = kvBytes
			r.kc[l] = byteBufOn(mem.LineKV, d, r.kvNullBytes[l]+kvBytes*kvSlots)
			r.vc[l] = r.kc[l]
		} else if mixer == decoder.MixerNone {
			// A feed-forward-only block (Nemotron-H's MLP): no mixer to build.
		} else if dnetOK && attnGate {
			// The same family's gated softmax layer (qwen3_5, qwen3_5_moe, qwen3_next), not every dnetOK family: Olmo
			// Hybrid's full-attention layer is olmo3's plain scheme (AttnGate=false) and takes the ordinary branch below.
			// Its q/k/v/o live off qattn, not lw.QProj/KProj/VProj/OProj, and q_proj is double width: [query || gate] per
			// head, interleaved, not two concatenated blocks; treated as an ordinary q_proj it yields plausible logits
			// from the wrong tensor. So the fused QKV path does not apply: K||V still fuse into L.qkvW (2-way), and Q is a
			// separate double-width projection (L.dnQw/dnQs), split on the activation at encode time (delta_qsplit),
			// because slicing rows out of a quantized int4 bundle with per-group scales is real surgery.
			qP, kP, vP, oP, qN, kN := lv.Qwen35AttnWeights()
			if len(qN) == 0 || len(kN) == 0 {
				return fmt.Errorf("metal: layer %d: qwen35 softmax layer has empty q_norm/k_norm "+
					"(%d/%d) — the loader did not populate them for this container", l, len(qN), len(kN))
			}
			L.qGate = true
			L.dnQw, L.dnQs = mk(qP)
			L.qkvW, L.qkvS = int4ConcatAOn(mem.LineDense, d, alias, kP, vP) // K‖V fused (Q handled separately above)
			L.oW, L.oS = mk(oP)
			L.qNorm, L.kNorm = NewBufferFloatsOn(mem.LineDense, d, qN), NewBufferFloatsOn(mem.LineDense, d, kN)
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
			kEqV = lv.VFromK()
			if src := lv.KVSrc(); src != l {
				// Gemma 4 E-model KV-shared layer (S1.4): the checkpoint has no k_proj/v_proj/k_norm here; it attends
				// over layer src's keys and values. Q-only projection; kEqV does not apply (no K of its own).
				if src < 0 || src >= l {
					return fmt.Errorf("metal: layer %d: KV source %d is not an earlier layer", l, src)
				}
				if g4KVSrcOffForTest { // G2 defect (3): the previous owning layer of the same attention type, if any
					for j := src - 1; j >= 0; j-- {
						if bv.Layer(j).KVSrc() == j && bv.Layer(j).IsLocal() == bv.Layer(src).IsLocal() {
							src = j
							break
						}
					}
				}
				L.kvShared, L.kvSrc, kEqV = true, src, false
				L.qkvW, L.qkvS = fuse(&lw.QProj)
			} else if kEqV {
				L.qkvW, L.qkvS = int4ConcatOn(mem.LineDense, d, &lw.QProj, &lw.KProj, &lw.KProj) // V slot = raw k_proj
			} else {
				L.qkvW, L.qkvS = fuse(&lw.QProj, &lw.KProj, &lw.VProj) // fused QKV
			}
			L.oW, L.oS = mk(&lw.OProj)
			if r.outBias {
				L.oBias = NewBufferFloatsOn(mem.LineDense, d, lw.OBias)
			}
			if r.qkNorm { // Qwen3 per-head Q/K norm weights [hd] — qGate layers set these above instead
				// QKNormWhole (Olmo 3/Olmo Hybrid, G5): the resident kernel launch that reproduces
				// this collapses the grid to one Q block + one K block sized from a SINGLE hd
				// argument (encodeAttention), which is only correct when nH==nKV (MHA). Decline
				// rather than silently mis-normalize a hypothetical future GQA family.
				if r.qkNormWhole {
					if nKVL := lv.KVHeads(); r.nH != nKVL {
						return fmt.Errorf("metal: layer %d: arch claims QKNormWhole but nH=%d != nKV=%d — the resident whole-vector kernel launch requires MHA", l, r.nH, nKVL)
					}
				}
				L.qNorm = NewBufferFloatsOn(mem.LineDense, d, lw.QNorm)
				if L.kvShared {
					L.kNorm = L.qNorm // bound, never read: a shared layer's qk_norm grid covers the Q heads only
				} else {
					L.kNorm = NewBufferFloatsOn(mem.LineDense, d, lw.KNorm)
				}
			}
		}
		if isAttn {
			// attention always dispatches L.attnSinks/L.uHasSink (unlike L.invf, which is only bound
			// when the rope dispatch itself runs) — every family needs a VALID buffer bound, gated by
			// the flag, so non-gpt-oss gets a real but unused 1-element dummy, not a zero Buffer.
			if r.attnSink {
				L.attnSinks = NewBufferFloatsOn(mem.LineDense, d, lw.AttnSinks)
			} else {
				L.attnSinks = NewBufferFloatsOn(mem.LineDense, d, []float32{0})
			}
		}
		g4b, isG4MoE := decoder.Gemma4MoEResidentBundle{}, false
		if r.g4moe != nil {
			g4b, isG4MoE = lv.Gemma4MoEBundle()
		}
		switch {
		case !hasFFN: // the layer is its mixer alone (Nemotron-H's Mamba and attention blocks)
		case isG4MoE: // Gemma-4 enable_moe_block: parallel dense‖MoE FFN (its own norms/router/experts)
			L.g4moe = buildGemma4MoELayer(d, m, &g4b, r.g4moe)
		case r.moe != nil && len(lw.Experts) > 0: // generic MoE layer: stacked experts instead of a dense FFN
			L.moe = buildMoELayer(d, m, l, lw, r.moe)
		case r.nonGatedMLP: // GPT-2/Nemotron relu²: single up-proj (no gate) + biases (zero where the family has none)
			L.upW, L.upS = mk(&lw.UpProj)
			L.dW, L.dS = mk(&lw.DownProj)
			L.upBias = NewBufferFloatsOn(mem.LineDense, d, orZeros(lw.UpBias, lw.UpProj.Rows()))
			L.downBias = NewBufferFloatsOn(mem.LineDense, d, orZeros(lw.DownBias, lw.DownProj.Rows()))
			L.downBiasSet = len(lw.DownBias) > 0
		default: // dense FFN (also GLM/DeepSeek's FirstKDense prefix layers, and gemma4 dense layers)
			L.guW, L.guS = fuse(&lw.GateProj, &lw.UpProj) // fused gate/up
			L.dW, L.dS = mk(&lw.DownProj)
			L.ffnI = lw.GateProj.Rows()
		}
		if r.q4kLane {
			// The q4k lane covers the plain dense shape (Phi-3, Llama, Qwen2): pre-norm attention (q/k/v bias allowed), a
			// SwiGLU FFN.
			if !isAttn || !hasFFN || L.moe != nil || L.g4moe != nil || L.qGate || L.kvShared || kEqV || r.nonGatedMLP {
				return fmt.Errorf("metal: --quant q4k runs on Metal for dense attention+SwiGLU layers only (layer %d is not)", l)
			}
			L.qkvSegs = q4kLaneProjOn(mem.LineDense, d, kq, &lw.QProj, &lw.KProj, &lw.VProj)
			L.oSegs, L.dSegs = q4kLaneProjOn(mem.LineDense, d, kq, &lw.OProj), q4kLaneProjOn(mem.LineDense, d, kq, &lw.DownProj)
			L.guSegs = q4kLaneProjOn(mem.LineDense, d, kq, &lw.GateProj, &lw.UpProj)
		}
		// postOnly (Olmo 3, Olmo Hybrid) is a model-level flag, but Olmo Hybrid's DeltaNet layers reach NormPre2
		// through NormPlacementLinear and carry real pre-norm weights regardless; isDelta is already resolved above,
		// so gate on the per-layer truth, not the model-level flag.
		postOnlyHere := r.postOnly && isAttn
		if !postOnlyHere {
			// A postOnly (non-delta) layer has NO pre-norm weight at all (lw.PreAttnNorm/
			// PreMLPNorm are 0-length — NewBufferFloats on an empty slice is a build-time error,
			// not a no-op), so this is skipped entirely; segA/encodeLayer quantize the raw
			// residual instead (see encodeAttention/encodeLayer's own postOnly branch).
			L.preNorm = NewBufferFloatsOn(mem.LineDense, d, lw.PreAttnNorm)
			// parallelBlock (Cohere/Command-R) has no pre-MLP norm tensor at all: the MLP reuses the shared input norm the
			// attention branch computed (encodeLayer's r.aq/r.aSc reuse), so lw.PreMLPNorm is empty (see
			// Model.ParallelBlockResident), with the same NewBufferFloats hazard postOnly's comment above names.
			switch {
			case !hasFFN: // no FFN, no pre-MLP norm
			case mixer == decoder.MixerNone: // a feed-forward-only block's one norm is its input norm (Nemotron-H)
				L.postNorm = L.preNorm
			case L.g4moe == nil && !r.parallelBlock: // dense/generic FFN entry norm (PreMLPNorm); g4moe carries its five norms in the bundle
				L.postNorm = NewBufferFloatsOn(mem.LineDense, d, lw.PreMLPNorm)
			}
			if r.layerNorm && r.layerNormBias {
				L.preNormBias = NewBufferFloatsOn(mem.LineDense, d, lw.PreAttnNormBias)
				if hasFFN && mixer != decoder.MixerNone && L.g4moe == nil && !r.parallelBlock {
					L.postNormBias = NewBufferFloatsOn(mem.LineDense, d, lw.PreMLPNormBias)
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
				return fmt.Errorf("metal: layer %d declares sandwich/postOnly norms but PostAttnNorm is not len==hidden(%d) (got %d)",
					l, H, len(lw.PostAttnNorm))
			}
			L.postAttnNorm = NewBufferFloatsOn(mem.LineDense, d, lw.PostAttnNorm)
			if L.g4moe == nil {
				if len(lw.PostMLPNorm) != H {
					return fmt.Errorf("metal: layer %d declares sandwich/postOnly norms but PostMLPNorm is not len==hidden(%d) (got %d)",
						l, H, len(lw.PostMLPNorm))
				}
				L.postMLPNorm = NewBufferFloatsOn(mem.LineDense, d, lw.PostMLPNorm)
			}
		}
		// Gemma 4's per-layer output scalar on a dense layer (h *= layer_scalar after the FFN residual;
		// decoder/forward_gemma4.go, HF Gemma4DecoderLayer). The g4moe join applies its own. 1 and absent (0) are
		// no-ops, as on the CPU, so no dispatch is encoded for them.
		if L.g4moe == nil && bv.IsGemma4() {
			L.layerScaleRead = true
			if s := lv.Gemma4DenseScalar(); s != 0 && s != 1 {
				L.uLayerScalar = NewBufferFloatsOn(mem.LineDense, d, []float32{s})
			}
		}
		if r.pleP > 0 { // Gemma 4 E-model PLE branch weights (S1.5)
			if lw.PLEGate.Rows() != r.pleP || lw.PLEProj.Rows() != H || len(lw.PostPLENorm) != H {
				return fmt.Errorf("metal: layer %d: PLE weights %dx? / %dx? / %d, want P=%d and H=%d", l, lw.PLEGate.Rows(), lw.PLEProj.Rows(), len(lw.PostPLENorm), r.pleP, H)
			}
			L.pleGW, L.pleGS = mk(&lw.PLEGate)
			L.plePW, L.plePS = mk(&lw.PLEProj)
			L.postPLENorm = NewBufferFloatsOn(mem.LineDense, d, lw.PostPLENorm)
		}
		if isAttn {
			// Per-layer RoPE table (Gemma local 10k vs global 1M base) and per-layer window. The rope
			// KERNEL is unchanged — all per-layer variation rides in the table contents AND the width
			// (geom.half). lv.RopeInvFreq() is the resident per-layer table (== the generic
			// RopeInvFreqLayer for every uniform family; Gemma 4's global proportional-rotary tail is
			// what makes it differ). Its length is this layer's rhalf (rotated pairs/head). A DeltaNet
			// layer has no attention geometry at all (see residLayer.delta's doc) — skip this whole
			// block for it: no RoPE table, no geom (L.geom stays nil), no KV cache allocated.
			invf := lv.RopeInvFreq()
			if len(invf) > 0 { // GPT-2 (FeatLearnedPos): no RoPE at all, invf is empty — NewBufferFloats panics on empty; L.invf stays unused (rope dispatch is skipped)
				L.invf = NewBufferFloatsOn(mem.LineDense, d, invf)
				L.mscale = NewBufferFloatsOn(mem.LineDense, d, []float32{float32(lv.RopeMscale())}) // 1.0 for every family without YaRN
			}
			// Per-layer attention geometry — the whole point of the 9c seam. Uniform families resolve
			// every layer to the same {hd, nKV, half, kEqV}; Gemma 4 varies hd/nKV/kEqV between its
			// local and global layers. geomFor dedups by value so a uniform model shares one geom.
			L.geom = r.geomFor(geomCache, lv.HeadDim(), lv.KVHeads(), len(invf), kEqV)
			if L.geom.hd > maxHd {
				maxHd = L.geom.hd
			}
			if L.geom.kvDim > maxKvDim {
				maxKvDim = L.geom.kvDim
			}
			if r.nH*L.geom.hd > maxNHhd {
				maxNHhd = r.nH * L.geom.hd
			}
			if e := attnFAPartialFloats(L.geom.hd, r.nH, L.geom.nKV); e > 0 {
				maxAttnFAPartial = max(maxAttnFAPartial, e)
				// Cache nKV here: attention_fa is dense-GQA-only, so every eligible layer shares one nKV (canUseAttnFA's own
				// gate is this same condition), and setPos can compute G and nSplit once per decode step instead of the
				// per-layer dispatch site doing it during encode-ahead (see uAttnFAG/uAttnFANSplit's field comment for the
				// race that would be).
				if r.attnFANKV == 0 {
					r.attnFANKV, r.attnFALayer = L.geom.nKV, l
				}
			}
			lw2 := uint32(0) // 0 = full causal; only local layers carry the window
			if lv.IsLocal() {
				lw2 = uint32(win)
			}
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
				L.qkvBias = NewBufferFloatsOn(mem.LineDense, d, append(append(append([]float32{}, qb...), kb...), vb...))
			}
			// A layer's slots are one pool of 64-row blocks, reached through a table (r.kvTab), so a batched step's one dispatch
			// reaches every row's blocks. attention_prefill_fused reads whole 8-row tiles and masks the ragged remainder after the
			// load; a tile lies inside a block, so that read stays in the block. An int8 cache keeps one f32 scale per position and
			// head beside its one-byte codes, block for block.
			blocks := kvBlocksFor(r.ctxCap)
			want := kvLayerWant{kvDim: L.geom.kvDim, nKV: L.geom.nKV, shared: L.kvShared}
			kw, e := kvCheckRows(l, m.LayerBuffers(l), want)
			if e != nil {
				return e
			}
			r.kvNullBytes[l] = decoder.KVBlockRows * kw * r.kvElemBytes()
			if L.kvShared {
				r.kvNullBytes[l] = r.kvNullBytes[L.kvSrc]
			}
			kvBytes := blocks * r.kvNullBytes[l]
			allocSlots := kvSlots
			r.kvContig = true
			if L.kvShared {
				// S1.4: no cache of its own. Its geometry must be its source's (same attention type, so the same
				// head_dim and KV heads); the buffers are aliased below, once every source exists.
				if sg := r.layers[L.kvSrc].geom; sg == nil || *sg != *L.geom {
					return fmt.Errorf("metal: layer %d shares layer %d's KV but their attention geometries differ", l, L.kvSrc)
				}
				if r.kvI8 {
					r.ks[l], r.vs[l] = r.ks[L.kvSrc], r.vs[L.kvSrc]
				}
				r.kc[l], r.vc[l] = r.kc[L.kvSrc], r.vc[L.kvSrc]
				r.layers[l] = L
				continue
			}
			r.kvSlotBytes[l] = kvBytes
			r.kc[l] = byteBufOn(mem.LineKV, d, r.kvNullBytes[l]+kvBytes*allocSlots)
			r.vc[l] = byteBufOn(mem.LineKV, d, r.kvNullBytes[l]+kvBytes*allocSlots)
			if r.kvI8 {
				scaleFloats := (1 + blocks*allocSlots) * decoder.KVBlockRows * L.geom.nKV
				r.ks[l] = d.NewBufferLenOn(mem.LineKV, scaleFloats)
				r.vs[l] = d.NewBufferLenOn(mem.LineKV, scaleFloats)
			}
		}
		r.layers[l] = L
	}
	for l := range r.layers { // a layer with no pool of its own declares no row: the plan prices what the build holds
		if r.kc[l] == (Buffer{}) && !r.layers[l].kvShared {
			if _, e := kvCheckRows(l, m.LayerBuffers(l), kvLayerWant{}); e != nil {
				return e
			}
		}
	}
	b.w, b.alias = w, alias
	b.maxNHhd, b.maxKvDim, b.maxHd, b.maxAttnFAPartial = maxNHhd, maxKvDim, maxHd, maxAttnFAPartial
	return nil
}
