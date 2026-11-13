//go:build darwin

package metal

import (
	"fmt"
	"slices"
	"strings"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// residentBuild is buildResident's state: what each build phase reads from the phases before it. A phase is a method that
// runs in build order and returns the error that declines the build; buildResident calls them in sequence, and the device's
// ledgers (buildResident's deferred release) own whatever a phase allocated when a later one fails.
type residentBuild struct {
	m            *decoder.Model
	bv           decoder.BackendModelView
	d            *Device
	r            *resident
	shape        program.Shape
	pipe         func(name string) Pipeline
	kvSlots      int
	stepHeadroom int64
	attnSoftcap  float32
	addOne       uint32
	win          int
	kq           *decoder.KQuantReader // the file's Q5_K and Q6_K blocks for the q4k lane (q4kLaneProj); nil uploads their int8 copies
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
