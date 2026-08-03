//go:build darwin

package metal

// Mixture-of-experts (MoE) resident decode for the Metal backend — the FeatMoE lever.
// Mirrors the WebGPU implementation (gpu/moe.go): the router runs entirely on-GPU and
// writes idx[k]/wgt[k] device buffers, then the host records a FIXED k dispatches per
// projection, each reading its expert index dynamically (weightRow = idx[slot]*rowsPerExpert
// + n). The command buffer is static every token, so the encode-ahead executor
// (ForwardEmbPipe) still pre-encodes token t+1 while t runs — a host-side top-k readback
// would have broken that pipeline. See docs/task-metal-moe.md.

import (
	"fmt"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

// moeKernels is concatenated onto allKernels (kernels.go) so it shares the one compiled
// library + the #include/using and the UNP8 macro. int4 expert weights = group=32, f16
// group scales, int8 activations — identical quant to the dense W4A8 path.
const moeKernels = `
// Router / shared-gate GEMV: f32 weight [rows,K] x int8 activation (aq,asc). One simdgroup
// (32 lanes) per output row. The router weight stays f32 (loaded full-precision for expert-
// selection stability); reuses the already-quantized post-attn-norm activation mq/mSc.
kernel void gemv_wf32_a8(device const float* wf[[buffer(0)]], device const char* aq[[buffer(1)]],
    device const float* asc[[buffer(2)]], device float* out[[buffer(3)]], constant uint& K[[buffer(4)]],
    uint gid[[threadgroup_position_in_grid]], uint lid[[thread_index_in_threadgroup]]) {
    device const float* wr = wf + (uint)gid*K;
    float acc = 0.0f;
    for (uint k=lid; k<K; k+=32u) acc += wr[k]*float(aq[k]);
    acc = simd_sum(acc);
    if (lid==0) out[gid] = acc*asc[0];
}

// moe_route: on-GPU top-k router. ONE thread does softmax (or per-expert sigmoid) scoring,
// optional +bias selection score, optional DeepSeek group-limiting, top-k, optional
// norm_topk_prob renorm and routed_scale. Writes idx[k] (u32) + wgt[k] (f32 = un-biased
// score). Mirrors decoder.routeExperts / groupLimit exactly (softmax sum is f32 here vs the
// CPU's f64 — a cosmetic near-tie difference, Metal has no double). nE<=256, nGroup<=64.
kernel void moe_route(device const float* logits[[buffer(0)]], device const float* bias[[buffer(1)]],
    device uint* outIdx[[buffer(2)]], device float* outWgt[[buffer(3)]], constant uint& nE[[buffer(4)]],
    constant uint& k[[buffer(5)]], constant uint& sigmoid[[buffer(6)]], constant uint& norm[[buffer(7)]],
    constant float& scale[[buffer(8)]], constant uint& nGroup[[buffer(9)]], constant uint& topkGroup[[buffer(10)]],
    uint tid[[thread_position_in_grid]]) {
    if (tid != 0u) return;
    float score[256];
    float sel[256];
    if (sigmoid != 0u) {
        for (uint i=0u;i<nE;i++) score[i] = 1.0f/(1.0f+exp(-logits[i]));
    } else {
        float mx = logits[0]; for (uint i=1u;i<nE;i++) mx = max(mx, logits[i]);
        float sum = 0.0f; for (uint i=0u;i<nE;i++){ float e = exp(logits[i]-mx); score[i]=e; sum+=e; }
        float inv = 1.0f/sum; for (uint i=0u;i<nE;i++) score[i] *= inv;
    }
    for (uint i=0u;i<nE;i++) sel[i] = score[i] + bias[i]; // bias is zeros when absent
    // DeepSeek group-limiting: partition into nGroup contiguous groups, score each by the
    // sum of its top-2 selection scores, keep the top topkGroup groups, mask the rest to -inf.
    if (nGroup > 1u) {
        uint gsz = nE / nGroup;
        float gscore[64];
        for (uint g=0u; g<nGroup; g++) {
            float t1=-INFINITY, t2=-INFINITY;
            for (uint i=g*gsz; i<(g+1u)*gsz; i++) { float v=sel[i]; if (v>t1){t2=t1;t1=v;} else if (v>t2){t2=v;} }
            gscore[g]=t1+t2;
        }
        bool keep[64];
        for (uint g=0u; g<nGroup; g++) keep[g]=false;
        for (uint j=0u; j<topkGroup; j++) {
            uint bg=0u; float bv=-INFINITY; bool found=false;
            for (uint g=0u; g<nGroup; g++) if (!keep[g] && gscore[g]>bv){ bv=gscore[g]; bg=g; found=true; }
            if (found) keep[bg]=true;
        }
        for (uint g=0u; g<nGroup; g++) if (!keep[g]) for (uint i=g*gsz; i<(g+1u)*gsz; i++) sel[i]=-INFINITY;
    }
    float wsum = 0.0f;
    for (uint j=0u; j<k; j++) {
        uint best=0u; float bv=-INFINITY;
        for (uint i=0u; i<nE; i++) if (sel[i]>bv){ bv=sel[i]; best=i; } // strict > => lowest index on ties
        outIdx[j]=best; outWgt[j]=score[best]; wsum+=score[best]; sel[best]=-INFINITY;
    }
    if (norm != 0u && wsum > 0.0f) for (uint j=0u; j<k; j++) outWgt[j] /= wsum;
    if (scale != 0.0f && scale != 1.0f) for (uint j=0u; j<k; j++) outWgt[j] *= scale;
}

// Indexed Stage-A W4A8 expert GEMV, mode-0 OVERWRITE (fused gate|up). Same body as
// gemv_w4a8_sa but the WEIGHT row is idx[slot]*rowsPerExpert + outRow (the stacked
// all-experts buffer); the OUTPUT row is outRow. rowsPerExpert = 2*inter for gate|up.
kernel void gemv_w4a8_moe(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], device const uint* idx[[buffer(6)]], constant uint& slot[[buffer(7)]],
    constant uint& rowsPerExpert[[buffer(8)]], threadgroup short* As[[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]],
    uint tgs[[threads_per_threadgroup]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]]) {
    for (uint i=tid;i<K;i+=tgs) As[i]=short(aq[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint G = K>>5u;
    uint row = tgid*(tgs>>5u) + sgid;               // output row 0..rowsPerExpert-1
    uint wrow = idx[slot]*rowsPerExpert + row;       // weight row in the stacked buffer
    device const uint4* wr = wq  + (uint)wrow*G;
    device const half*  sr = sct + (uint)wrow*G;
    float acc = 0.0f;
    for (uint g=lane; g<G; g+=32u) {
        uint4 w = wr[g]; threadgroup const short* a = As + g*32u;
        int gi = UNP8(w.x,a) + UNP8(w.y,a+8) + UNP8(w.z,a+16) + UNP8(w.w,a+24);
        acc += float(gi) * float(sr[g]);
    }
    acc = simd_sum(acc);
    if (lane==0) out[row] = acc*asc[0];
}

// Indexed Stage-A W4A8 expert GEMV, mode-1 WEIGHTED-ACCUMULATE (down projection). Weight
// row = idx[slot]*rowsPerExpert + outRow (rowsPerExpert = H); epilogue folds the router
// weight and sums straight into the residual: out[row] += wgt[slot]*acc*asc[0]. This is
// the combine — no separate weighted-sum kernel. Looping the k slots builds the full sum.
kernel void gemv_w4a8_moe_wacc(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], device const uint* idx[[buffer(6)]], device const float* wgt[[buffer(7)]],
    constant uint& slot[[buffer(8)]], constant uint& rowsPerExpert[[buffer(9)]], threadgroup short* As[[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]],
    uint tgs[[threads_per_threadgroup]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]]) {
    for (uint i=tid;i<K;i+=tgs) As[i]=short(aq[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint G = K>>5u;
    uint row = tgid*(tgs>>5u) + sgid;
    uint wrow = idx[slot]*rowsPerExpert + row;
    device const uint4* wr = wq  + (uint)wrow*G;
    device const half*  sr = sct + (uint)wrow*G;
    float acc = 0.0f;
    for (uint g=lane; g<G; g+=32u) {
        uint4 w = wr[g]; threadgroup const short* a = As + g*32u;
        int gi = UNP8(w.x,a) + UNP8(w.y,a+8) + UNP8(w.z,a+16) + UNP8(w.w,a+24);
        acc += float(gi) * float(sr[g]);
    }
    acc = simd_sum(acc);
    if (lane==0) out[row] += wgt[slot]*acc*asc[0];
}

// shared_gate_combine: qwen2_moe gated shared expert — x[i] += sigmoid(gl[0])*src[i].
kernel void shared_gate_combine(device float* x[[buffer(0)]], device const float* src[[buffer(1)]],
    device const float* gl[[buffer(2)]], uint i[[thread_position_in_grid]]) {
    x[i] += (1.0f/(1.0f+exp(-gl[0]))) * src[i];
}
`

// moeLayer holds one MoE layer's device weights: the f32 router (+bias), the stacked all-E
// fused gate|up and down expert buffers (W4A8), and the optional shared expert (dense W4A8
// gate|up + down, plus the f32 sigmoid gate for the qwen2_moe gated variant).
type moeLayer struct {
	routerW    Buffer // f32 [nE, H]
	routerBias Buffer // f32 [nE] (zeros when the arch has no e_score_correction_bias)
	expGuW     Buffer // stacked all-E fused gate|up: rows nE*2*inter (int4 words)
	expGuS     Buffer // ... f16 group scales
	expDW      Buffer // stacked all-E down: rows nE*H
	expDS      Buffer
	// shared expert (sharedInter > 0)
	shGuW, shGuS Buffer // dense fused gate|up: rows 2*sharedInter
	shDW, shDS   Buffer // dense down: rows H
	shGateW      Buffer // f32 [1, H] sigmoid gate (gated variant only; zero if ungated)
}

// moeResident holds the MoE pipelines, config, constant uniforms and scratch shared across
// all MoE layers (config is uniform per model). rIdx/rWgt are the on-GPU router outputs.
type moeResident struct {
	pRouter, pRoute, pGU, pDownWacc, pSharedGate Pipeline

	nE, k, inter, sharedInter     int
	sharedUngated                 bool
	uNE, uK, uSigmoid, uNorm      Buffer // route uniforms
	uScale                        Buffer // f32
	uNGroup, uTopkGroup           Buffer
	uInter, uInter2, uSharedInter Buffer
	uSlot                         []Buffer // k compile-time slot-index constants

	rLogits, rIdx, rWgt Buffer // router scratch: logits[nE], idx[k] (u32), wgt[k]
	shGl, shDown        Buffer // shared-expert scratch: gate logit[1], down out[H]
}

// buildMoE builds the Resident-level MoE state (pipelines, config, uniforms, scratch) from a
// dense-shaped model that actually declares an MoE FFN. Returns nil when the model is not MoE,
// and an error for an MoE variant this path does not implement (so BuildResident declines →
// CPU fallback rather than running wrong).
func buildMoE(d *Device, m *decoder.Model, pipe func(string) Pipeline, H int) (*moeResident, error) {
	nE, k, inter, sharedInter, sigmoid, norm, sharedUngated, scale, nGroup, topkGroup, ok := m.MoEResidentParams()
	if !ok {
		return nil, nil // dense model — no MoE
	}
	// Gemma 4's enable_moe_block (gemma4_text, 26B-A4B) sets arch.MoE, so MoEResidentParams reports
	// ok — but it is NOT the generic Mixtral/Qwen shape this path implements. It is a parallel
	// dense‖MoE FFN with gelu-tanh experts (gemv_w4a8_moe's epilogue is SiLU), a weightless router
	// pre-norm + learned [hidden] scale + per_expert_scale, and seven norms. Building it here would
	// run a wrong (SiLU, plain-router) MoE. Decline until the Metal gemma4MoeMLP path lands (9c
	// Step 5) so gemma4_text falls back to CPU rather than mis-running — same discipline as CUDA's
	// isG4MoE split (cuda/backend.go). Dense Gemma 4 has no MoE, so it returns nil,nil above and
	// admits normally; this only gates the MoE variant.
	if m.HasGemma4MoEResident() {
		// Gemma 4's parallel dense‖MoE is NOT this generic path — it runs encodeGemma4MoEFFN
		// (its own router/experts/join, gemma4_moe.go), routed around here via HasGemma4MoEResident
		// exactly like CUDA's isG4MoE split. Return nil (not MoE for THIS builder) so r.moe stays
		// nil and BuildResident builds r.g4moe instead; the per-layer loop routes on the bundle.
		return nil, nil
	}
	if nE > 256 {
		return nil, fmt.Errorf("metal MoE: nE=%d exceeds router cap 256", nE)
	}
	if nGroup > 64 {
		return nil, fmt.Errorf("metal MoE: nGroup=%d exceeds cap 64", nGroup)
	}
	mo := &moeResident{
		pRouter: pipe("gemv_wf32_a8"), pRoute: pipe("moe_route"),
		pGU: pipe("gemv_w4a8_moe"), pDownWacc: pipe("gemv_w4a8_moe_wacc"),
		pSharedGate:   pipe("shared_gate_combine"),
		nE:            nE,
		k:             k,
		inter:         inter,
		sharedInter:   sharedInter,
		sharedUngated: sharedUngated,
	}
	b1 := func(b bool) uint32 {
		if b {
			return 1
		}
		return 0
	}
	mo.uNE, mo.uK = d.NewBufferU32(uint32(nE)), d.NewBufferU32(uint32(k))
	mo.uSigmoid, mo.uNorm = d.NewBufferU32(b1(sigmoid)), d.NewBufferU32(b1(norm))
	mo.uScale = d.NewBufferFloats([]float32{float32(scale)})
	mo.uNGroup, mo.uTopkGroup = d.NewBufferU32(uint32(nGroup)), d.NewBufferU32(uint32(topkGroup))
	mo.uInter, mo.uInter2 = d.NewBufferU32(uint32(inter)), d.NewBufferU32(uint32(2*inter))
	mo.uSharedInter = d.NewBufferU32(uint32(sharedInter))
	mo.uSlot = make([]Buffer, k)
	for j := range mo.uSlot {
		mo.uSlot[j] = d.NewBufferU32(uint32(j))
	}
	mo.rLogits = d.NewBufferLen(nE)
	mo.rIdx = d.NewBufferUint32s(make([]uint32, k))
	mo.rWgt = d.NewBufferLen(k)
	mo.shGl, mo.shDown = d.NewBufferLen(1), d.NewBufferLen(H)
	return mo, nil
}

// buildMoELayer packs one MoE layer's weights. Experts are row-stacked into ONE buffer per
// projection via the existing int4Concat (all-E gate|up interleaved gate_e,up_e; all-E down),
// so expert e lives at weight row e*2*inter (gate|up) / e*H (down). The router (+bias) stays
// f32; the shared expert is packed like the dense FFN.
func buildMoELayer(d *Device, lw *decoder.LayerWeights, mo *moeResident) *moeLayer {
	ml := &moeLayer{}
	ml.routerW = f32Mat(d, &lw.Router)
	if len(lw.RouterBias) > 0 {
		ml.routerBias = d.NewBufferFloats(lw.RouterBias)
	} else {
		ml.routerBias = d.NewBufferFloats(make([]float32, mo.nE)) // zeros → sel = score
	}
	guMats := make([]*linalg.WeightMat, 0, 2*len(lw.Experts))
	for e := range lw.Experts {
		guMats = append(guMats, &lw.Experts[e].Gate, &lw.Experts[e].Up)
	}
	ml.expGuW, ml.expGuS = int4Concat(d, guMats...)
	downMats := make([]*linalg.WeightMat, 0, len(lw.Experts))
	for e := range lw.Experts {
		downMats = append(downMats, &lw.Experts[e].Down)
	}
	ml.expDW, ml.expDS = int4Concat(d, downMats...)
	if mo.sharedInter > 0 {
		ml.shGuW, ml.shGuS = int4Concat(d, &lw.SharedExpert.Gate, &lw.SharedExpert.Up)
		var e error
		if ml.shDW, ml.shDS, e = int4Buf(d, &lw.SharedExpert.Down); e != nil {
			panic(e)
		}
		if !mo.sharedUngated {
			ml.shGateW = f32Mat(d, &lw.SharedGate)
		}
	}
	return ml
}

// f32Mat uploads a full-precision WeightMat (router / shared-gate) to a Metal f32 buffer.
func f32Mat(d *Device, w *linalg.WeightMat) Buffer {
	f, ok := w.F32()
	if !ok {
		panic(fmt.Sprintf("metal MoE: weight kind %q is not f32 (router/shared-gate must be full precision)", w.Kind()))
	}
	return d.NewBufferFloats(f)
}

// encodeMoEFFN records the MoE FFN for one layer, replacing the dense gate/up/swiglu/down
// dispatches. post-attn norm → router logits → on-GPU top-k → per-selected-expert
// gate|up/swiglu/weighted-down → optional shared expert. Value-independent (idx/wgt read at
// kernel-execution time), so the encode-ahead executor still pre-encodes it.
func (r *Resident) encodeMoEFFN(e *Encoder, L *residLayer) {
	mo := r.moe
	ml := L.moe
	// post-attn RMSNorm → quantized activation mq/mSc (same as the dense FFN entry).
	e.Dispatch(r.pRms, 256, 256, r.x, L.postNorm, r.mq, r.mSc, r.uH, r.uEps, r.uAddOne)
	// router logits: routerW[nE,H] × mq → rLogits[nE]
	e.Dispatch(mo.pRouter, mo.nE*32, 32, ml.routerW, r.mq, r.mSc, mo.rLogits, r.uH)
	// on-GPU top-k → rIdx[k], rWgt[k]
	e.Dispatch(mo.pRoute, 1, 1, mo.rLogits, ml.routerBias, mo.rIdx, mo.rWgt,
		mo.uNE, mo.uK, mo.uSigmoid, mo.uNorm, mo.uScale, mo.uNGroup, mo.uTopkGroup)
	// k selected experts: fused gate|up (overwrite) → swiglu → down (weighted-accumulate into x).
	for j := 0; j < mo.k; j++ {
		e.DispatchTG(mo.pGU, (2*mo.inter)*32, 256, r.H*2, ml.expGuW, ml.expGuS, r.mq, r.mSc, r.gu, r.uH, mo.rIdx, mo.uSlot[j], mo.uInter2)
		e.Dispatch(r.pSw, 256, 256, r.gu, r.gu.At(mo.inter*4), r.dq, r.dSc, mo.uInter, r.uAct)
		e.DispatchTG(mo.pDownWacc, r.H*32, 256, mo.inter*2, ml.expDW, ml.expDS, r.dq, r.dSc, r.x, mo.uInter, mo.rIdx, mo.rWgt, mo.uSlot[j], r.uH)
	}
	// shared always-on expert (Qwen2-MoE gated / GLM ungated).
	if mo.sharedInter > 0 {
		e.DispatchTG(r.pSA, (2*mo.sharedInter)*32, 256, r.H*2, ml.shGuW, ml.shGuS, r.mq, r.mSc, r.gu, r.uH)
		e.Dispatch(r.pSw, 256, 256, r.gu, r.gu.At(mo.sharedInter*4), r.dq, r.dSc, mo.uSharedInter, r.uAct)
		if mo.sharedUngated {
			e.Dispatch(r.pGemvResid, r.H*32, 32, ml.shDW, ml.shDS, r.dq, r.dSc, r.x, mo.uSharedInter) // x += down
		} else {
			e.Dispatch(r.pGemv, r.H*32, 32, ml.shDW, ml.shDS, r.dq, r.dSc, mo.shDown, mo.uSharedInter) // down → scratch
			e.Dispatch(mo.pRouter, 32, 32, ml.shGateW, r.mq, r.mSc, mo.shGl, r.uH)                     // gate logit (1 row)
			e.Dispatch(mo.pSharedGate, r.H, 256, r.x, mo.shDown, mo.shGl)                              // x += sigmoid(gl)*down
		}
	}
}
