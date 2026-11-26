//go:build cuda

package cuda

import (
	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// moeMLPPre issues the pre-readback half of one MoE FFN block (decoder/mlp.go's moeMLP is the CPU reference): the
// shared normed activation (mq/mSc) and the router (logits → top-k idx/wgt, left on the device). It ends exactly
// where the cacheExperts path must read rIdx back to the host, so this half is graph-static (segB) and the
// loadRoutedExperts D2H stays live in the gap between segB and segC. moeMLPPost issues the rest. The SAME normed
// activation feeds router AND experts; the router logits are f32 (see cudaResident.moe).
func (r *cudaResident) moeMLPPre(Ly *cudaLayer, x Buffer) error {
	// Explicit rmsnorm, NOT the fused fGU path: the fused kernel folds the norm into the dense
	// gate/up GEMV and never writes r.mq, but the router needs that quantized activation too.
	if e := r.emitInputNorm(Ly.walk.FFN.InputNorm.Form, x, Ly.postNorm, r.mq, r.mSc); e != nil {
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
	// gpt-oss routes through its OWN kernel: moe_route takes the mixing weight from the UNBIASED score (bias steers
	// selection only), while gpt-oss softmaxes over the SELECTED BIASED logits. Same selection, different weights,
	// which is why the wrong one produces plausible output rather than an error.
	if Ly.walk.FFN.FFN.Form == program.FfnGptOss {
		if e := r.launch(r.gptOssRoute, onecfg(1, 0),
			Arg(r.rLogits), Arg(Ly.routerB), Arg(r.rIdx), Arg(r.rWgt),
			gpu.ArgValue(int32(r.nE)), gpu.ArgValue(int32(r.topK))); e != nil {
			return e
		}
		return r.recordRouted()
	}
	// Argument ORDER is easy to transpose silently here: group-routed families (DeepSeek/Kimi; testdata/deepseek-tiny sets
	// n_group=2, topk_group=1) pass nGroup then topkGroup. TestMoERoute drives the kernel with its own argument list, so it
	// validates the kernel's math, not this call's order, and a transposition once passed TestMLAResidentParityCUDA while
	// it compared only the first generated token (a discrete-selection bug need not show in the first tokens); it now
	// compares the full sequence (cuda/mla_resident_test.go). Typed launch wrappers make a transposition a COMPILE error but
	// cannot check that a value is the right way round; see the reciprocal trap at decoder/features.go's cuda entry and
	// docs/parity-coverage-policy.md § "Relevant".
	if e := r.launch(r.fRoute, onecfg(1, 0),
		Arg(r.rLogits), Arg(Ly.routerB), Arg(r.rIdx), Arg(r.rWgt),
		gpu.ArgValue(int32(r.nE)), gpu.ArgValue(int32(r.topK)), gpu.ArgValue(r.moeSigmoid),
		gpu.ArgValue(r.moeNormTopK), gpu.ArgValue(r.moeScale),
		gpu.ArgValue(int32(r.nGroup)), gpu.ArgValue(int32(r.topkGroup))); e != nil {
		return e
	}
	return r.recordRouted()
}

// launchGluSplit runs the fused gate||up SwiGLU (glu_quant): dscratch[k]=act(gu[k])*gu[inter+k], then
// symmetric-int8-quantizes it into (outQ,outSc). The stacked-expert GEMV lays gate and up CONTIGUOUS in one
// buffer, so this passes the same pointer twice with gOff=0, uOff=inter: the ONE place that gate/up split
// convention lives, shared by every MoE call site (routed, shared, gemma-4). Keep it the sole gate/up-split
// dispatch: a gOff/uOff swap silently computes silu(up)*gate, which the e2e MoE parity gate CANNOT catch on
// random-weight experts (the two are near-equal in magnitude), so TestMoeSwigluWiring exercises this exact helper
// with crafted gate != up and asserts gate-first.
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

// oBiasArg returns layer Ly's attention output-projection bias, or null when the family has none. Centralized for
// the reason sinkArg is: the DECODE and the PREFILL o_proj launches must not disagree, and a bias applied on one
// path only reads as drift partway through a sequence, once decode takes over from prefill. It needs no dedicated
// kernel: both GEMV pairs fold the bias into the value BEFORE the accumulate select (aikit gemv_quant.cu:
// `val = fma(facc, aScale, bias?bias[n]:0); dst[n] = accum ? dst[n]+val : val`, and goinfer's batched
// gemv_w4a8_rn.cu identically), so bias-plus-residual is one instruction on this backend.
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
	return r.launch(r.fSw, r.actCfg(glueQuantThreads, glueQuantThreads*4, inter),
		Arg(gu), Arg(gu), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(inter)),
		gpu.ArgValue(int32(inter)), gpu.ArgValue(r.act),
		Arg(outQ), Arg(outSc), Arg(outScr))
}

// moeMLPPost issues the post-readback half of one MoE FFN block (mirrors decoder/mlp.go's moeMLP): the sequential
// expert loop, then the always-on shared expert, weight-accumulating into the residual x. Every launch has the
// same geometry whatever the router picked: each expert is selected by ARITHMETIC on r.expIdx() (the weight-row
// index inside the kernel), not by binding a different buffer, which is what keeps the dispatch chain static
// (graph-capturable). The final GEMV weight-accumulates into x, so the per-expert combine and the residual add are
// one instruction (no scratch, no combine pass). The cacheExperts readback (if any) has already filled the slots
// when this runs.
func (r *cudaResident) moeMLPPost(Ly *cudaLayer, x Buffer) error {
	gu := 2 * r.moeInter
	for j := 0; j < r.topK; j++ {
		if e := r.waitMiss(j); e != nil {
			return e
		}
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
		// down-proj, weight-accumulating into the residual: x += wgt[j] * (Down_e · act). gpt-oss adds its per-expert down
		// bias INSIDE that product via its own kernel; every other family keeps the untouched wacc, so their numerics are
		// unchanged.
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
	if Ly.walk.FFN.FFN.MoE.SharedIntermediateDim > 0 {
		return r.sharedExpert(Ly, r.mq, r.mSc, x)
	}
	return nil
}

// gemma4MoeMLPPre/Post issue Gemma-4's parallel dense‖MoE FFN for layer Ly (enable_moe_block). Unlike the generic
// moeMLPPre/Post (one branch, wacc straight into the residual), Gemma-4 runs TWO branches off the SAME residual h
// and joins them under a shared post-norm, so h is read three times and written only at the end. Mirrors
// decoder/forward_gemma4_moe.go exactly:
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
	r.profMark(&r.profAttn)

	// --- router FIRST (on RAW h): rmsnorm_nw → gemv_f32_f32(folded proj) → moe_route → per-expert-scale
	// fold. It needs only h, and it is the one thing the host waits for (recordRouted); the dense
	// branch below then executes while the routing is read back and the misses are DMA'd. The two
	// share no buffers, so this order is bit-identical to dense-first. ---
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
	if e := r.recordRouted(); e != nil {
		return e
	}
	r.profMark(&r.profRouter)

	// --- dense branch → g4x1 ---
	if r.floatAct {
		if e := r.floatActDense(Ly, x); e != nil {
			return e
		}
		if e := r.normF32(r.g4x1, Ly.g4postFFN1); e != nil {
			return e
		}
		r.profMark(&r.profDense)
		// xe = preFFNNorm2(h) into xF, which the experts read after the gap.
		if e := r.normInto(x, Ly.g4preFFN2); e != nil {
			return e
		}
		r.profMark(&r.profRouter)
		return nil
	}
	if e := r.rms(x, Ly.g4preFFN, r.mq, r.mSc); e != nil { // xd = preFFNNorm(h), int8
		return e
	}
	if e := r.doG(Ly.g, r.mq, r.mSc, nullBias, r.gO, 0); e != nil {
		return e
	}
	if e := r.doG(Ly.u, r.mq, r.mSc, nullBias, r.uO, 0); e != nil {
		return e
	}
	if e := r.launch(r.fSw, r.actCfg(glueQuantThreads, glueQuantThreads*4, r.inter), Arg(r.gO), Arg(r.uO), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(0)),
		gpu.ArgValue(int32(r.inter)), gpu.ArgValue(r.act), Arg(r.dq), Arg(r.dSc), Arg(r.dScr)); e != nil {
		return e
	}
	if e := r.doG(Ly.d, r.dq, r.dSc, nullBias, r.g4x1, 0); e != nil {
		return e
	}
	if e := r.normF32(r.g4x1, Ly.g4postFFN1); e != nil {
		return e
	}
	r.profMark(&r.profDense)

	// xe = preFFNNorm2(h); reuse mq/mSc (dense branch done with them). This is the last static op
	// before the gap: launchToken clears g4x2 (H2D) and, if caching, reads back the routing (D2H).
	if e := r.rms(x, Ly.g4preFFN2, r.mq, r.mSc); e != nil {
		return e
	}
	r.profMark(&r.profRouter)
	return nil
}

// gemma4MoeMLPPost issues the post-readback half: the expert loop accumulating into the (already
// cleared) g4x2, its post-norm, and the join — sum x1+x2, joint post-norm, add the residual h, then
// the per-layer scalar. g4x2 was zeroed and the expert slots filled in the gap before this runs.
func (r *cudaResident) gemma4MoeMLPPost(Ly *cudaLayer, l int, x Buffer) error {
	gu := 2 * r.moeInter
	for j := 0; j < r.topK; j++ {
		if e := r.waitMiss(j); e != nil {
			return e
		}
		if r.floatAct {
			if e := r.floatActExpert(Ly, j); e != nil {
				return e
			}
			continue
		}
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

	// JOIN: sum x1+x2 BEFORE the joint norm; add the residual h AFTER it; then the per-layer scalar. A mis-order is
	// plausible-but-wrong: cosine shows it but will not localize it.
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

// moeState is cudaResident's mixture-of-experts geometry, its per-token scratch and kernels: the sparse router, the routed and
// shared experts, Gemma 4's parallel dense||MoE branch and the expert-major prefill's row kernels. Zero on a dense model.
type moeState struct {
	// Sparse MoE. The router projection stays f32 (gemv_f32_a8) while the experts are int4: its output steers a
	// DISCRETE choice, so a quantization error near a tie does not perturb the result slightly, it runs a DIFFERENT
	// expert and the output is unrelated (the Granite SSM expert-flip class; docs/code-notes/cuda.md#cudaResident.moe).
	// This is the one place in this backend where the cheap precision is not worth it.
	moe                     bool
	nE, topK, moeInter      int
	moeSigmoid, moeNormTopK int32
	moeScale                float32
	nGroup, topkGroup       int
	// Gemma-4 parallel dense||MoE (any layer g4moe: JIT router_f32, alloc g4 scratch, take gemma4MoeMLP). g4cap
	// (GOINFER_G4_CAPTURE) is a DEBUG readback of the four MoE-layer buffers (rn / wgt / x1 / x2) so a whole-forward
	// miss localizes to router vs dense vs expert vs join in ONE run.
	gemma4Moe bool
	// gemma4Dense is true for ANY gemma4 checkpoint (dense or MoE), broader than gemma4Moe (enable_moe_block layers).
	// It gates compiling fScaleVec, the dense per-layer-output-scalar kernel (segB's dense tail), which has nothing
	// router-specific about it.
	gemma4Dense                                             bool
	sharedInter                                             int      // width of the always-on shared expert (0 ⇒ none)
	bMoeFold                                                Pipeline // moe_fold_wacc (moe_fold.cu); zero when it did not load
	bMoeRows, bGluRows                                      Pipeline // gemv_w4a8_moe_rows, glu_quant_rows (moe_rows.cu); zero when they did not load
	moeRowsOff                                              bool     // test seam: run the expert-major path's per-pair launches (the A/B and identity reference)
	fRoute, fRouterGemv, fMoEGemv, fMoEWacc, fSharedCombine Pipeline
	fMoEWaccBias                                            Pipeline // gpt-oss: wacc + per-expert down bias
	fRouterF32, fScaleWgt, fRmsNW, fScaleVec                Pipeline // gemma4 MoE (router_f32 module)
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
	// Shared-expert scratch (allocated only when any layer hasShared). Sized to sharedInter,
	// which is its own width — distinct from both the dense inter and the routed moeInter.
	shGUout, shSc, shScr, shDownOut Buffer
	shGl                            Buffer // [1] the Qwen-MoE shared-gate logit; unused when ungated
	shQ                             Buffer
}

// gptOssState is gpt-oss's own kernels and per-layer tables: the clamped SwiGLU epilogue, the biased router, the attention sinks
// and the expert biases. Zero for every other family.
type gptOssState struct {
	// gpt-oss only (nil elsewhere, and every other family's launches pass ArgNull so the kernels stay bit-identical): the
	// clamped interleaved-SwiGLU expert epilogue, plus the per-layer attention sinks and the per-expert gate‖up bias
	// table it needs.
	gptOssSw                 Pipeline // glu_quant_gptoss (own module — audited glue.ptx/moe.ptx untouched)
	gptOssRoute              Pipeline // route_gptoss — top-k + softmax over the BIASED logits (moe_route's contract is wrong for this family)
	gptOssAlpha, gptOssLimit float32
	gptOssSinks              []Buffer // [layer] → [nH] learned per-head attention sink logits
	gptOssDownBias           []Buffer // [layer] → [nExpert*hidden] per-expert down-projection bias
	gptOssExpBias            []Buffer // [layer] → [nExpert·2·moeInter] gate‖up biases, indexed on-device by the router
}
