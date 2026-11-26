//go:build cuda

package cuda

import (
	"fmt"
	"strconv"

	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/logs"
)

// bindGlueKernels compiles the decode modules every model takes (gemv_fwd, glue, the fused projections and their rows-per-warp
// variants) and binds their pipelines, with aikit's quantized GEMVs and the pairwise rope twin where the model rotates pairwise.
func (s *buildState) bindGlueKernels(r *cudaResident, bv decoder.BackendModelView) error {
	gmod, e := r.compileKV(r.kvPTX8(gemvFwdPTX, gemvFwdKV16PTX, gemvFwdKV8PTX))
	if e != nil {
		return e
	}
	glmod, e := r.compileKV(r.attnPTX(gluePTX, glueKV16PTX, glueKV8PTX, glueCapPTX, glueKV16CapPTX))
	if e != nil {
		return e
	}
	// The generic quantized GEMVs come from aikit (gpu.QuantGEMVPTX), not gemv_fwd.ptx; gemv_fwd.cu keeps only the
	// LLM-specific kv_store / rope_kv, and decode stays bit-identical.
	qgemv, e := r.dev.NewQuantGEMV()
	if e != nil {
		return e
	}
	r.gemvW4, r.gemvW8 = qgemv.W4A8, qgemv.W8A8
	// kv_store and rope are not bound: the fused rope_kv below subsumes both (TestPipelineLint_boundKernelsAreLaunched flags
	// a pipeline bound and never launched). GPT-J pairwise rotation (Cohere/Cohere2/Aya, GLM-OCR) binds the pairwise twin of
	// rope_kv (cuda/rope_pairwise.cu, same argument list and launch geometry) into the same field, so no launch site
	// changes. A load failure declines: the NeoX kernel on a pairwise model is silently wrong from position 1, so there is
	// no fallback.
	r.pairwiseRoPE = bv.PairwiseRoPE()
	if r.pairwiseRoPE {
		pwmod, pe := r.compileKV(r.kvPTX(ropePairwisePTX, ropePairwiseKV16PTX))
		if pe != nil {
			return fmt.Errorf("pairwise rope: compile rope_pairwise.ptx: %w", pe)
		}
		if r.ropeKV, e = r.dev.NewComputePipeline(pwmod, "rope_kv_pw"); e != nil {
			return e
		}
	} else if r.ropeKV, e = r.dev.NewComputePipeline(gmod, "rope_kv"); e != nil {
		return e
	}
	// kv_store at pos 0 is an exact device copy (copyF32), used by K=V layers to fill vB from kB. It is the one use of kv_store, which the fused rope_kv still subsumes for the KV cache.
	if r.kvCopy, e = r.dev.NewComputePipeline(gmod, "kv_store"); e != nil {
		return e
	}
	qmod, e2 := r.dev.CompileLibrary(fusedQKVPTX)
	if e2 != nil {
		return e2
	}
	if r.fQKV, e = r.dev.NewComputePipeline(qmod, "fused_rms_qkv"); e != nil {
		return e
	}
	if r.fGU, e = r.dev.NewComputePipeline(qmod, "fused_rms_gu"); e != nil {
		return e
	}
	if r.fQKN, e = r.dev.NewComputePipeline(qmod, "qk_norm"); e != nil {
		return e
	}
	// rows-per-warp variant of fused_rms_gu (own module; see fused_gu_rows.cu). A load failure only leaves the original kernel in use.
	if gmod2, ge2 := r.dev.CompileLibrary(fusedGURowsPTX); ge2 == nil {
		var le error
		if r.fGURows, le = r.dev.NewComputePipeline(gmod2, "fused_rms_gu_rows"); le != nil {
			r.fGURows = Pipeline{}
		}
	}
	r.readSMShape() // device shape for wave-sized fused-projection grids (waveRowsPerWarp); unreadable -> zero -> the static rules
	// rows-per-warp variant of fused_rms_qkv (own module; see fused_qkv_rows.cu). A load failure only leaves the original kernel in use.
	if rmod, re2 := r.dev.CompileLibrary(fusedQKVRowsPTX); re2 == nil {
		var le error
		if r.fQKVRows, le = r.dev.NewComputePipeline(rmod, "fused_rms_qkv_rows"); le != nil {
			r.fQKVRows = Pipeline{}
		}
	}
	fns := []struct {
		dst  *Pipeline
		name string
	}{
		{&r.fRms, "rmsnorm_quant"}, {&r.fRmsF32, "rmsnorm_f32"}, {&r.fQ, "quant_vec"},
		{&r.fAttn, "attention"}, {&r.fSw, "glu_quant"}, {&r.fRes, "residual"},
		{&r.fLN, "layernorm_quant"},
	}
	for _, f := range fns {
		if *f.dst, e = r.dev.NewComputePipeline(glmod, f.name); e != nil {
			return e
		}
	}
	return nil
}

// bindSideKernels binds the per-32 activation kernels under actG32, then the argmax, top-k, Gumbel and LoRA modules, and allocates
// LoRA's shared scratch.
func (s *buildState) bindSideKernels(r *cudaResident) error {
	var e error
	// Per-32 activations: bind actgroup.cu's kernels INTO the per-vector pipeline fields. Each has
	// its per-vector sibling's exact argument list and launch shape, so every launch site runs the
	// per-32 kernel unedited; the paths that would still read a per-vector scale declined above,
	// fusion is off, and batched prefill declines (prefillStaticDecline).
	if r.actG32 {
		agmod, e3 := r.dev.CompileLibrary(actGroupPTX)
		if e3 != nil {
			return fmt.Errorf("actgroup module: %w", e3)
		}
		for _, f := range []struct {
			dst  *Pipeline
			name string
		}{
			{&r.fRms, "rmsnorm_quant_g32"}, {&r.fQ, "quant_vec_g32"}, {&r.fSw, "glu_quant_g32"},
			{&r.gemvW4, "gemv_w4a8_g32"}, {&r.gemvW8, "gemv_w8a8_g32"}, {&r.gemvQ4K, "gemv_q4k_g32"},
			{&r.gemvQ5K, "gemv_q5k_g32"}, {&r.gemvQ6K, "gemv_q6k_g32"},
			{&r.fQKVg32, "fused_rms_qkv_g32"}, {&r.fGUg32, "fused_rms_gu_g32"},
		} {
			if *f.dst, e = r.dev.NewComputePipeline(agmod, f.name); e != nil {
				return fmt.Errorf("actgroup %s: %w", f.name, e)
			}
		}
	}
	// argmax_reduce lives in its own module (argmax.ptx), off glue.ptx, so a kernel fix does not force a glue.ptx
	// regeneration. moe.ptx, glue.ptx and gemv_fwd.ptx are the audited artifacts, pinned at NVRTC 12.6.85
	// (cuda/testdata/REGEN.md); every other production PTX, including argmax.ptx, is built at whatever NVRTC was on hand
	// when it was added. See cuda/argmax.cu.
	amod, e2 := r.dev.CompileLibrary(argmaxPTX)
	if e2 != nil {
		return e2
	}
	if r.fArg, e = r.dev.NewComputePipeline(amod, "argmax_reduce"); e != nil {
		return e
	}
	if r.fArgRows, e = r.dev.NewComputePipeline(amod, "argmax_rows"); e != nil {
		return e
	}
	// topk_compact / topk_finish / topk_z: their own module, same isolation as argmax_reduce above.
	tmod, e2 := r.dev.CompileLibrary(topkPTX)
	if e2 != nil {
		return e2
	}
	for _, f := range []struct {
		dst  *Pipeline
		name string
	}{{&r.fTopKCompact, "topk_compact"}, {&r.fTopKFinish, "topk_finish"}, {&r.fTopKZ, "topk_z"}} {
		if *f.dst, e = r.dev.NewComputePipeline(tmod, f.name); e != nil {
			return e
		}
	}
	// gumbel_stage1/2: own module, same isolation.
	gumod, e2 := r.dev.CompileLibrary(gumbelPTX)
	if e2 != nil {
		return e2
	}
	if r.fGumbel1, e = r.dev.NewComputePipeline(gumod, "gumbel_stage1"); e != nil {
		return e
	}
	if r.fGumbel2, e = r.dev.NewComputePipeline(gumod, "gumbel_stage2"); e != nil {
		return e
	}
	// Compute-time LoRA: own module, loaded unconditionally (cheap, and whether an
	// adapter will arrive is not known here).
	lmod, e2 := r.dev.CompileLibrary(loraPTX)
	if e2 != nil {
		return e2
	}
	if r.fLoraDown, e = r.dev.NewComputePipeline(lmod, "lora_delta_down"); e != nil {
		return e
	}
	if r.fLoraUp, e = r.dev.NewComputePipeline(lmod, "lora_delta_up"); e != nil {
		return e
	}
	r.loraT = r.af(loraRMax)
	return nil
}

// bindPrefillKernels binds the batched prefill modules and sets prefillReady when every required one loaded, then the image-block,
// span and m-RoPE prefill kernels the model can take. A load failure here declines a prefill path, never the build.
func (s *buildState) bindPrefillKernels(r *cudaResident, bv decoder.BackendModelView) {
	// Batched prefill kernels (weight-stationary M=len path): own module, prefill_batched.ptx, with the audited PTX
	// untouched. bGemvB dispatches int4 to bRN (gemv_w4a8_rn) unconditionally; there is no gemv_w4a8_batched module.
	{
		if pbmod, e3 := r.compileKV(r.attnPTX(prefillBatchedPTX, prefillBatchedKV16PTX, prefillBatchedKV8PTX, prefillBatchedCapPTX, prefillBatchedKV16CapPTX)); e3 == nil {
			ok := true
			load := func(dst *Pipeline, mod gpu.Library, name string) {
				if p, le := r.dev.NewComputePipeline(mod, name); le == nil {
					*dst = p
				} else {
					ok = false
				}
			}
			if rnmod, e4 := r.dev.CompileLibrary(gemvRNPTX); e4 == nil {
				load(&r.bRN, rnmod, "gemv_w4a8_rn")
			} else {
				ok = false
			}
			// int8 batched GEMV: own module, own file; the audited PTX is untouched.
			if w8mod, e5 := r.dev.CompileLibrary(gemvW8BatchedPTX); e5 == nil {
				load(&r.bW8, w8mod, "gemv_w8a8_batched")
			} else {
				ok = false
			}
			load(&r.bRms, pbmod, "rmsnorm_quant_batched")
			load(&r.bQKN, pbmod, "qk_norm_batched")
			load(&r.bNormF32, pbmod, "rmsnorm_f32_batched")
			if r.pairwiseRoPE {
				// GPT-J pairwise twin of rope_kv_batched (same signature); see the decode bind above.
				if pwmod, pe := r.compileKV(r.kvPTX(ropePairwisePTX, ropePairwiseKV16PTX)); pe == nil {
					load(&r.bRopeKV, pwmod, "rope_kv_batched_pw")
				} else {
					ok = false
				}
			} else {
				load(&r.bRopeKV, pbmod, "rope_kv_batched")
			}
			load(&r.bAttn, pbmod, "attn_batched")
			// A ragged step's twins of rope_kv_batched and attn_batched (stepAttnRows). Optional: without them every row takes
			// decode's own gap. The pairwise rope has no ragged twin.
			if !r.pairwiseRoPE {
				var e1, e2 error
				r.bRopeKVRagged, e1 = r.dev.NewComputePipeline(pbmod, "rope_kv_ragged")
				r.bAttnRagged, e2 = r.dev.NewComputePipeline(pbmod, "attn_ragged")
				if e1 != nil || e2 != nil {
					r.bRopeKVRagged, r.bAttnRagged = Pipeline{}, Pipeline{}
				}
			}
			load(&r.bQuant, pbmod, "quant_vec_batched")
			load(&r.bSw, pbmod, "glu_quant_batched")
			load(&r.bRes, pbmod, "residual_batched")
			// The expert-major MoE prefill's fold. Optional: without it that path is ineligible and the per-row loop runs.
			if fmod, e7 := r.dev.CompileLibrary(moeFoldPTX); e7 == nil {
				var e8 error
				if r.bMoeFold, e8 = r.dev.NewComputePipeline(fmod, "moe_fold_wacc"); e8 != nil {
					r.bMoeFold = Pipeline{}
				}
			}
			// The grouped expert kernels. Optional: without both, expert-major keeps its per-pair launches.
			if rmod, e7 := r.dev.CompileLibrary(moeRowsPTX); e7 == nil {
				var e8, e9 error
				r.bMoeRows, e8 = r.dev.NewComputePipeline(rmod, "gemv_w4a8_moe_rows")
				r.bGluRows, e9 = r.dev.NewComputePipeline(rmod, "glu_quant_rows")
				if e8 != nil || e9 != nil {
					r.bMoeRows, r.bGluRows = Pipeline{}, Pipeline{}
				}
			}
			if lnmod, e6 := r.dev.CompileLibrary(layernormQuantPTX); e6 == nil {
				load(&r.bLN, lnmod, "layernorm_quant_batched")
			}
			r.prefillReady = ok
		}
	}
	// Gemma 3 image-block prefill attention (decoder.ResidentPrefillSubsystem): own module, prefill_batched.ptx untouched. A
	// load failure is not fatal: bAttnImg stays zero and PrefillImageLast declines to the CPU-prefill-then-UploadKV bridge.
	// It depends on r.prefillReady, as L2 does.
	if r.prefillReady && !r.kvI8 { // no int8 twin: an int8 cache leaves the image and span prefills declined
		if imod, e8 := r.compileKV(r.kvPTX(attnImgPrefillPTX, attnImgPrefillKV16PTX)); e8 == nil {
			// &r.bAttnImg (not a bare tuple assign): TestPipelineLint_boundKernelsAreLaunched's
			// static scan recognizes `&r.<field>` or a direct NewComputePipeline RHS as a bind
			// site, matching the loadF/loadG idiom L2/L3 already use below.
			loadImg := func(dst *Pipeline, name string) {
				if pl, pe := r.dev.NewComputePipeline(imod, name); pe == nil {
					*dst = pl
					r.imgPrefillReady = true
				}
			}
			loadImg(&r.bAttnImg, "attn_img_batched")
			var pe error
			if r.bAttnSpan, pe = r.dev.NewComputePipeline(imod, "attn_span_batched"); pe == nil {
				r.spanPrefillReady = true
			} else {
				r.bAttnSpan = Pipeline{}
			}
		}
	}
	// Qwen2.5-VL m-RoPE batched prefill (decoder.ResidentPrefillSubsystem): own module, prefill_batched.ptx untouched. Loaded
	// only for an m-RoPE model, since it needs the model's MRopeSection for sec0/sec1 (unlike bAttnImg above, which loads
	// unconditionally). A load failure or a non-m-RoPE model is not fatal: PrefillMRoPELast declines to the
	// CPU-prefill-then-UploadKV bridge.
	if r.prefillReady {
		// GLM-OCR's pairwise m-RoPE kernel writes f32 rows, so an f16 cache never compiles or loads it, whatever the policy says: the
		// batched m-RoPE prefill stays unready and PrefillMRoPELast declines to the CPU prefill and upload (TestKV16_pairwiseMRoPENeverF16).
		if sec := bv.MRopeSection(); len(sec) == 3 && !(r.pairwiseRoPE && r.kvF16) && !r.kvI8 {
			// Pairwise m-RoPE (GLM-OCR) rotates (2d, 2d+1): its own kernel in rope_pairwise.ptx. A
			// model that needs it and cannot load it must NOT fall back to the NeoX m-RoPE kernel.
			mropePTX, mropeName := r.kvPTX(ropeMRopePrefillPTX, ropeMRopePrefillKV16PTX), "rope_kv_mrope_batched"
			if r.pairwiseRoPE { // GLM-OCR, which keeps f32 KV (decoder.CUDAKVF16Decline), so the f32 module's kernel
				mropePTX, mropeName = ropePairwisePTX, "rope_kv_mrope_batched_pw"
			}
			if mmod, e9 := r.dev.CompileLibrary(mropePTX); e9 == nil {
				loadMRope := func(dst *Pipeline, name string) {
					if pl, pe := r.dev.NewComputePipeline(mmod, name); pe == nil {
						*dst = pl
						r.mropePrefillReady = true
					}
				}
				loadMRope(&r.bRopeKVMRoPE, mropeName)
				if r.mropePrefillReady {
					r.mropeTakesMode = !r.pairwiseRoPE
					r.mropeSec0, r.mropeSec1 = int32(sec[0]), int32(sec[0]+sec[1])
					if bv.MRopeInterleaved() {
						if r.pairwiseRoPE {
							// Pairwise rotation over the interleaved layout is a combination no checkpoint registers (decoder.ropeAt),
							// and rope_kv_mrope_batched_pw has no such mode: decline rather than rotate by the wrong rule.
							r.mropePrefillReady = false
						}
						r.mropeMode, r.mropeSec0, r.mropeSec1 = 1, int32(3*sec[1]), int32(3*sec[2])
					}
				}
			}
		}
	}
}

// bindFastPrefill selects the fast prefill levers (L2's fused attention, L3's tensor-core GEMM) and binds their kernels; a lever
// whose module does not load stays on the exact path.
func (s *buildState) bindFastPrefill(r *cudaResident, m *decoder.Model) {
	// L2 fused prefill attention: default on above fastPrefillFloor;
	// GOINFER_CUDA_FAST_PREFILL=0 or Options.ExactPrefill keeps the exact path. Own module, prefill_batched.ptx untouched. A
	// load failure is not fatal: bAttnFused* stay zero and every selection site falls back to attn_batched, the exact path.
	r.fastAttn, r.fastGemm = fastPrefillEnabled(r.knobValue("GOINFER_CUDA_FAST_PREFILL"))
	if m.ExactPrefill() { // Options.ExactPrefill: this model's prefill stays on the exact path
		r.fastAttn, r.fastGemm = false, false
	}
	if (r.fastAttn || r.fastGemm) && r.prefillReady {
		// The tensor-core prefill attention loads its cache's build: f32, f16 or int8 (the _kv8 twins stage K and V as f16 from the int8 cache and
		// its scales, docs/measurements/cuda-kv-i8-2026-11/s6-registration.md); its launches pass the layer's scale pools (kvScaled).
		if fmod, e6 := r.compileKV(r.kvPTX8(attnFusedPTX, attnFusedKV16PTX, attnFusedKV8PTX)); r.fastAttn && e6 == nil {
			loadF := func(dst *Pipeline, name string) {
				if pl, pe := r.dev.NewComputePipeline(fmod, name); pe == nil {
					*dst = pl
				}
			}
			loadF(&r.bAttnFused64, "attn_fused_hd64")
			loadF(&r.bAttnFused128, "attn_fused_hd128")
			if bmod, eb := r.compileKV(r.kvPTX8(attnFusedBMPTX, attnFusedBMKV16PTX, attnFusedBMKV8PTX)); eb == nil {
				loadF2 := func(dst *Pipeline, name string) {
					if pl, pe := r.dev.NewComputePipeline(bmod, name); pe == nil {
						*dst = pl
					}
				}
				loadF2(&r.bAttnBM32x64hd64, "attn_fused_bm32_hd64")
				loadF2(&r.bAttnBM32x64hd128, "attn_fused_bm32_hd128")
				loadF2(&r.bAttnBM32x32hd64, "attn_fused_bm32n32_hd64")
				loadF2(&r.bAttnBM32x32hd128, "attn_fused_bm32n32_hd128")
				loadF2(&r.bAttnBM128hd64, "attn_fused_bm128_hd64")
				loadF2(&r.bAttnBM128hd128, "attn_fused_bm128_hd128")
				if v := r.knobValue("GOINFER_CUDA_ATTN_FUSED_TILE"); v == "64x64" {
					r.attnTile = -1
				} else if v == "128x64" {
					r.attnTile = 3
				} else if v == "32x64" {
					r.attnTile = 1
				} else if v == "32x32" {
					r.attnTile = 2
				}
			}
		}
		// L3 tensor-core GEMM: own module, own half of the gate. Bound through an &r.<field> loader closure, like every pipeline
		// here: TestPipelineLint_boundKernelsAreLaunched reads a plain `r.F = pl` as a launch with no binding, i.e. a
		// nil-pipeline dispatch.
		if gmod, e7 := r.dev.CompileLibrary(gemmMMAPTX); r.fastGemm && e7 == nil {
			loadG := func(dst *Pipeline, name string) {
				if pl, pe := r.dev.NewComputePipeline(gmod, name); pe == nil {
					*dst = pl
				}
			}
			loadG(&r.bGemmMMA, "gemm_w4a8_mma")
		}
	}
}

// bindGptOss binds gpt-oss's own expert epilogue and router, for that family only. A load failure is not fatal.
func (s *buildState) bindGptOss(r *cudaResident, bv decoder.BackendModelView) {
	// gpt-oss's clamped interleaved-SwiGLU expert epilogue: own module, so the audited glue.ptx/moe.ptx stay untouched.
	// Loaded only for that family; launchGluSplitExpert branches on this pipeline being populated. A load failure is not
	// fatal.
	if alpha, limit, isGptOss := bv.GptOssAct(); isGptOss {
		if gmod, ge := r.dev.CompileLibrary(gptOssActPTX); ge == nil {
			// Loader closure, not a plain assignment: see the pipeline-lint note at the L3 load above.
			loadG := func(dst *Pipeline, name string) {
				if pl, pe := r.dev.NewComputePipeline(gmod, name); pe == nil {
					*dst = pl
				}
			}
			loadG(&r.gptOssSw, "glu_quant_gptoss")
			// gpt-oss's router (route_gptoss): moe_route's "bias" steers expert selection only and takes the weight from the
			// unbiased score, which is right for DeepSeek/GLM and wrong here, where softmax runs over the selected biased logits.
			// Running gpt-oss through moe_route gives plausible mixing weights that are not this model's, a silent quality loss.
			loadG(&r.gptOssRoute, "route_gptoss")
			r.gptOssAlpha, r.gptOssLimit = alpha, limit
		}
	}
}

// bindSplitKV binds the split-KV decode attention and its ragged twins, allocates its scratch and reads its settings. It needs
// prefillReady, since it replaces that path's single-row launch.
func (s *buildState) bindSplitKV(r *cudaResident, bv decoder.BackendModelView, nLayers int) {
	if r.prefillReady {
		if skmod, e2 := r.compileKV(r.attnPTX(decodeSplitKVPTX, decodeSplitKVKV16PTX, decodeSplitKVKV8PTX, decodeSplitKVCapPTX, decodeSplitKVKV16CapPTX)); e2 == nil {
			skOK := true
			loadSK := func(dst *Pipeline, name string) {
				if p, le := r.dev.NewComputePipeline(skmod, name); le == nil {
					*dst = p
				} else {
					skOK = false
				}
			}
			loadSK(&r.skScores, "splitkv_scores")
			loadSK(&r.skSoftmax, "splitkv_softmax")
			loadSK(&r.skVsum, "splitkv_vsum")
			if skOK {
				// A ragged step's twins (splitkv_*_ragged). A load failure leaves them zero, so the step's split-KV rows take
				// decode's own three launches row by row.
				var e1, e2, e3 error
				r.skScoresRagged, e1 = r.dev.NewComputePipeline(skmod, "splitkv_scores_ragged")
				r.skSoftmaxRagged, e2 = r.dev.NewComputePipeline(skmod, "splitkv_softmax_ragged")
				r.skVsumRagged, e3 = r.dev.NewComputePipeline(skmod, "splitkv_vsum_ragged")
				if e1 != nil || e2 != nil || e3 != nil {
					r.skScoresRagged, r.skSoftmaxRagged, r.skVsumRagged = Pipeline{}, Pipeline{}, Pipeline{}
				}
				r.skScoreBuf = r.af(r.nH * r.ctxCap)
				r.skInvBuf = r.af(r.nH)
				// Flash-decode V-sum SPIKE, NOT bit-identical. Loaded and allocated only when a test
				// sets vsumSplitForTest, so a stock binary carries neither the pipelines nor the
				// scratch.
				if v := vsumSplitForTest; v > 1 {
					// maxHd comes from the MODEL, not r.layers: r.layers is not populated until
					// later in BuildResident, so reading it here silently yielded 0 and the
					// resulting 0-byte allocation panicked the executor and declined the whole
					// resident path to CPU. And hidden/nH is not a substitute — qwen3_5's
					// head_dim is 256 where hidden/heads is not.
					maxHd := 0
					for l := range nLayers {
						if h := bv.Layer(l).HeadDim(); h > maxHd {
							maxHd = h
						}
					}
					if maxHd <= 0 {
						// Never attempt the allocation on a degenerate size: that is the exact
						// failure above, and it takes the entire resident path down with it.
						logs.Warnf("[cuda] the V-sum split is ignored: no positive head dim")
					} else {
						loadSK(&r.skVsumPartial, "splitkv_vsum_partial")
						loadSK(&r.skVsumCombine, "splitkv_vsum_combine")
						if r.skVsumPartial != (Pipeline{}) && r.skVsumCombine != (Pipeline{}) {
							r.skPartialBuf = r.af(r.nH * maxHd * v)
							r.skVsumSplit = v
						}
					}
				}
				// Default ON (bit-identical; gated per layer at runtime on the effective attended span
				// nWin ≥ splitkvThreshold(nH, hd), so geometries and depths it loses on are unaffected).
				// GOINFER_SPLITKV_ATTN=0 force-disables it (A/B / rollback); GOINFER_SPLITKV_MIN_KEYS
				// overrides the per-geometry threshold so the crossover is re-measurable without a
				// rebuild (0 ⇒ always split — the force-on arm).
				r.splitkvAttn = r.knobValue("GOINFER_SPLITKV_ATTN") != "0"
				if v, err := strconv.Atoi(r.knobValue("GOINFER_SPLITKV_MIN_KEYS")); err == nil && v >= 0 {
					r.skMinKeys = v
				}
			}
		}
	}
}

// bindMoEKernels binds the routed-expert module for a MoE model and Gemma 4's router_f32 module, and sizes the MoE-branch
// capture when it is on.
func (s *buildState) bindMoEKernels(r *cudaResident, nLayers int) error {
	var e error
	// MoE module: loaded only for a routed model, so a dense one JITs nothing extra.
	if r.moe {
		mmod, e2 := r.dev.CompileLibrary(moePTX)
		if e2 != nil {
			return e2
		}
		for _, f := range []struct {
			dst  *Pipeline
			name string
		}{
			{&r.fRoute, "moe_route"}, {&r.fRouterGemv, "gemv_f32_a8"},
			{&r.fMoEGemv, "gemv_w4a8_moe"}, {&r.fMoEWacc, "gemv_w4a8_moe_wacc"},
			{&r.fMoEWaccBias, "gemv_w4a8_moe_wacc_bias"},
			{&r.fSharedCombine, "shared_gate_combine"},
		} {
			if *f.dst, e = r.dev.NewComputePipeline(mmod, f.name); e != nil {
				return e
			}
		}
	}
	// router_f32 module: Gemma 4's own kernels, kept off the audited moe.ptx. gemma4Dense (any gemma4 checkpoint, not only
	// enable_moe_block ones) compiles it too: scale_vec (fScaleVec) is segB's dense-tail per-layer-output-scalar kernel,
	// which every dense gemma4 layer needs. The three router-specific kernels stay gemma4Moe-only.
	if r.gemma4Moe || r.gemma4Dense {
		rmod, e2 := r.dev.CompileLibrary(routerF32PTX)
		if e2 != nil {
			return e2
		}
		if r.fScaleVec, e = r.dev.NewComputePipeline(rmod, "scale_vec"); e != nil {
			return e
		}
		if r.gemma4Moe {
			for _, f := range []struct {
				dst  *Pipeline
				name string
			}{
				{&r.fRouterF32, "gemv_f32_f32"}, {&r.fScaleWgt, "scale_wgt_by_expert"},
				{&r.fRmsNW, "rmsnorm_nw"},
			} {
				if *f.dst, e = r.dev.NewComputePipeline(rmod, f.name); e != nil {
					return e
				}
			}
			if r.g4cap {
				r.g4capRn, r.g4capWgt = make([][]float32, nLayers), make([][]float32, nLayers)
				r.g4capX1, r.g4capX2 = make([][]float32, nLayers), make([][]float32, nLayers)
			}
		}
	}
	return nil
}
