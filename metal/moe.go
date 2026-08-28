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
	"os"
	"strconv"

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
// route_gptoss — gpt-oss's MoE router. moe_route's contract (sel=score+bias; top-k on sel;
// weight=score[best]) has the bias steer SELECTION only, which is right for DeepSeek/GLM/
// Qwen-MoE but wrong for gpt-oss: its bias reaches the logits BEFORE top-k, and the weight is
// softmax over the SELECTED biased logits, not the unbiased score. Verified standalone
// (metal/gptoss_kernels_test.go's TestGptOssRoute_metal: a case where the bias changes WHICH
// experts win, not just a valid-looking-but-wrong softmax). Softmax-over-selected-only equals
// softmax-over-all-then-renormalize (the shared normalizer cancels), matching HF's
// GptOssTopKRouter exactly.
kernel void route_gptoss(device const float* logits[[buffer(0)]], device const float* bias[[buffer(1)]],
    device uint* outIdx[[buffer(2)]], device float* outWgt[[buffer(3)]], constant uint& nE[[buffer(4)]],
    constant uint& k[[buffer(5)]],
    uint tid[[thread_position_in_grid]]) {
    if (tid != 0u) return;
    float sc[256];
    for (uint i=0u;i<nE;i++) sc[i] = logits[i] + bias[i];
    bool taken[256];
    for (uint i=0u;i<nE;i++) taken[i]=false;
    float chosen[256];
    for (uint j=0u;j<k;j++) {
        int best=-1; float bv=-INFINITY;
        for (uint i=0u;i<nE;i++) if (!taken[i] && sc[i]>bv) { bv=sc[i]; best=int(i); }
        taken[uint(best)]=true;
        outIdx[j]=uint(best);
        chosen[j]=bv;
    }
    float mx=chosen[0];
    for (uint j=1u;j<k;j++) mx=max(mx,chosen[j]);
    float sum=0.0f;
    for (uint j=0u;j<k;j++) { chosen[j]=exp(chosen[j]-mx); sum+=chosen[j]; }
    float inv=1.0f/sum;
    for (uint j=0u;j<k;j++) outWgt[j]=chosen[j]*inv;
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
        uint4 w = wr[g]; threadgroup const short4* a4 = reinterpret_cast<threadgroup const short4*>(As + g*32u);
        int gi = UNP8V(w.x,a4) + UNP8V(w.y,a4+2) + UNP8V(w.z,a4+4) + UNP8V(w.w,a4+6);
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
        uint4 w = wr[g]; threadgroup const short4* a4 = reinterpret_cast<threadgroup const short4*>(As + g*32u);
        int gi = UNP8V(w.x,a4) + UNP8V(w.y,a4+2) + UNP8V(w.z,a4+4) + UNP8V(w.w,a4+6);
        acc += float(gi) * float(sr[g]);
    }
    acc = simd_sum(acc);
    if (lane==0) out[row] += wgt[slot]*acc*asc[0];
}
// gemv_w4a8_moe_wacc_bias: gpt-oss's down-projection combine. Its bias is added INSIDE the
// expert, before the router weight scales the result (decoder/forward_gptoss.go's
// gptOssExpert: dst = Down·h + downBias; then gptOssMoE combines out += w·dst) — so the bias
// is wgt[slot]*bias[row], NOT a plain += bias[row] the way GPT-2's o-proj/down-proj bias is
// (those have no router weight to fold into). gemv_w4a8_moe_wacc has no bias epilogue at all,
// so this is a genuinely new kernel, not a parameter added to an existing one.
kernel void gemv_w4a8_moe_wacc_bias(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], device const uint* idx[[buffer(6)]], device const float* wgt[[buffer(7)]],
    constant uint& slot[[buffer(8)]], constant uint& rowsPerExpert[[buffer(9)]], device const float* bias[[buffer(10)]],
    threadgroup short* As[[threadgroup(0)]],
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
        uint4 w = wr[g]; threadgroup const short4* a4 = reinterpret_cast<threadgroup const short4*>(As + g*32u);
        int gi = UNP8V(w.x,a4) + UNP8V(w.y,a4+2) + UNP8V(w.z,a4+4) + UNP8V(w.w,a4+6);
        acc += float(gi) * float(sr[g]);
    }
    acc = simd_sum(acc);
    if (lane==0) out[row] += wgt[slot]*(acc*asc[0] + bias[idx[slot]*rowsPerExpert + row]);
}

// shared_gate_combine: qwen2_moe gated shared expert — x[i] += sigmoid(gl[0])*src[i].
kernel void shared_gate_combine(device float* x[[buffer(0)]], device const float* src[[buffer(1)]],
    device const float* gl[[buffer(2)]], uint i[[thread_position_in_grid]]) {
    x[i] += (1.0f/(1.0f+exp(-gl[0]))) * src[i];
}
// gptoss_glu: clamp(gate,max=limit)·sigmoid(alpha·clamp(gate)) · (clamp(up,[-limit,limit])+1) —
// gpt-oss's clamped interleaved-SwiGLU, verified against decoder/forward_gptoss.go's
// gptOssExpert. THE ASYMMETRY IS REAL: gate clamps only from ABOVE, up clamps BOTH sides —
// clamping both symmetrically is the tidy-looking mistake, and it only differs from the
// correct math where the activation saturates, which is exactly where a clamp exists to
// matter (metal/gptoss_kernels_test.go's TestGptOssSwigluQuant_metal drives inputs well past
// ±limit on both branches for that reason).
inline float gptoss_glu(float gx, float ux, float alpha, float limit) {
    gx = min(gx, limit);
    ux = clamp(ux, -limit, limit);
    float glu = gx/(1.0f+exp(-alpha*gx));
    return (ux+1.0f)*glu;
}
// swiglu_quant_gptoss: fused activation+quant for gpt-oss's MoE expert, taking gemv_w4a8_moe's
// RAW (bias-free) gate|up output and adding the per-expert bias itself — biasOff =
// idx[slot]*2*I, matching gate‖up's packing, mirrors how the router decides which expert runs
// while the launch geometry stays static (the whole point of the indexed-stacked-expert
// design). hasBias lets the same kernel serve a bias-free caller (none exist yet, kept for
// symmetry with layernorm_quant's hasBias).
kernel void swiglu_quant_gptoss(device const float* g[[buffer(0)]], device const float* u[[buffer(1)]],
    device char* dq[[buffer(2)]], device float* ds[[buffer(3)]], constant uint& I[[buffer(4)]],
    device const float* biasGU[[buffer(5)]], device const uint* idx[[buffer(6)]], constant uint& slot[[buffer(7)]],
    constant uint& hasBias[[buffer(8)]], constant float& alpha[[buffer(9)]], constant float& limit[[buffer(10)]],
    uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256];
    uint biasOff = hasBias != 0u ? idx[slot]*2u*I : 0u;
    float mx=0;
    for (uint i=tid;i<I;i+=tgs) {
        float gx=g[i], ux=u[i];
        if (hasBias!=0u) { gx += biasGU[biasOff+i]; ux += biasGU[biasOff+I+i]; }
        float dd = gptoss_glu(gx, ux, alpha, limit);
        mx = max(mx, fabs(dd));
    }
    mx = simd_max(mx);
    uint sgid = tid >> 5u, lane = tid & 31u;
    if (lane == 0) red[sgid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nsg = tgs >> 5u;
    if (tid == 0) { float v = red[0]; for (uint k=1; k<nsg; k++) v = max(v, red[k]); red[0] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sc=red[0]/127.0f; if(sc==0)sc=1; if(tid==0)ds[0]=sc; float inv=1/sc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i=tid;i<I;i+=tgs) {
        float gx=g[i], ux=u[i];
        if (hasBias!=0u) { gx += biasGU[biasOff+i]; ux += biasGU[biasOff+I+i]; }
        float dd = gptoss_glu(gx, ux, alpha, limit);
        dq[i]=char(clamp(int(round(dd*inv)),-127,127));
    }
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
	// gpt-oss: per-expert stacked biases, packed to match the weight addressing swiglu_quant_gptoss
	// / gemv_w4a8_moe_wacc_bias read (biasOff = idx[slot]*2*inter / idx[slot]*H). Zero-value Buffer
	// for every other family — the gpt-oss dispatch path in encodeMoEFFN is the only reader.
	expGuBias, expDBias Buffer
	// pool is non-nil when mo.paged: a bounded LRU slot pool + on-demand staging (expertpool.go),
	// generalized from the gemma4-only path (gemma4_moe.go) — same mechanism, no gemma4-specific
	// assumptions in expertPool itself, so this file supplies the stageFn instead of inventing one.
	// expGuW/expGuS/expDW/expDS stay zero-value when paged (the stacked all-E buffer is exactly the
	// 11.96 GB/19.2 GB this exists to avoid).
	pool *expertPool
}

// moeResident holds the MoE pipelines, config, constant uniforms and scratch shared across
// all MoE layers (config is uniform per model). rIdx/rWgt are the on-GPU router outputs.
type moeResident struct {
	pRouter, pRoute, pGU, pDownWacc, pSharedGate Pipeline

	// gpt-oss: its own router/activation/down-combine kernels (route_gptoss,
	// swiglu_quant_gptoss, gemv_w4a8_moe_wacc_bias) replace pRoute/pGU+pSw/pDownWacc in
	// encodeMoEFFN — the bias semantics genuinely differ (see the kernel docs in moeKernels),
	// not just an added parameter. Zero-value Pipelines/Buffers for every other family.
	isGptOss                 bool
	pRouteGptOss, pActGptOss Pipeline
	pDownWaccBias            Pipeline
	uAlpha, uLimit, uHasBias Buffer

	nE, k, inter, sharedInter     int
	sharedUngated                 bool
	uNE, uK, uSigmoid, uNorm      Buffer // route uniforms
	uScale                        Buffer // f32
	uNGroup, uTopkGroup           Buffer
	uInter, uInter2, uSharedInter Buffer
	uSlot                         []Buffer // k compile-time slot-index constants

	rLogits, rIdx, rWgt Buffer // router scratch: logits[nE], idx[k] (u32), wgt[k]
	shGl, shDown        Buffer // shared-expert scratch: gate logit[1], down out[H]

	// Synchronous paging (GOINFER_METAL_MOE_SLOTS=N>0): generalizes gemma4_moe.go's paging to this
	// generic MoE shape (Mixtral/Qwen/GLM/gpt-oss/qwen3_5_moe/qwen3_next) — same env var, same
	// mechanism, same expertPool. N must be >= k (a token's own top-k must fit); N==0/unset ⇒ all
	// experts resident (today's behavior, unchanged). idxZeros lets the paged expert GEMVs read row
	// 0 of a single-expert slot buffer while rWgt is still indexed by the selection slot — the
	// reused gemv_w4a8_moe(_wacc[_bias]) kernels compute byte-identically to the stacked path.
	paged    bool
	slots    int
	idxZeros Buffer

	// giwFile is the re-opened .giw for pread-staging; nil ⇒ the mmap byte-copy path. Shared
	// read-only fd across every layer's pool; closed by resident.Close. Same knob and default as
	// gemma4_moe.go's (GOINFER_MOE_PREAD=0 opts out) — deliberately the same env var, since it is
	// the same mechanism generalized.
	giwFile *os.File
}

// buildMoE builds the resident-level MoE state (pipelines, config, uniforms, scratch) from a
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
	_, _, isGptOss := m.GptOssActResident()
	mo := &moeResident{
		pRouter: pipe("gemv_wf32_a8"), pRoute: pipe("moe_route"),
		pGU: pipe("gemv_w4a8_moe"), pDownWacc: pipe("gemv_w4a8_moe_wacc"),
		pSharedGate:   pipe("shared_gate_combine"),
		isGptOss:      isGptOss,
		nE:            nE,
		k:             k,
		inter:         inter,
		sharedInter:   sharedInter,
		sharedUngated: sharedUngated,
	}
	if isGptOss {
		alpha, limit, _ := m.GptOssActResident()
		mo.pRouteGptOss = pipe("route_gptoss")
		mo.pActGptOss = pipe("swiglu_quant_gptoss")
		mo.pDownWaccBias = pipe("gemv_w4a8_moe_wacc_bias")
		mo.uAlpha, mo.uLimit = NewBufferFloats(d, []float32{alpha}), NewBufferFloats(d, []float32{limit})
		mo.uHasBias = NewBufferU32(d, 1)
	}
	b1 := func(b bool) uint32 {
		if b {
			return 1
		}
		return 0
	}
	mo.uNE, mo.uK = NewBufferU32(d, uint32(nE)), NewBufferU32(d, uint32(k))
	mo.uSigmoid, mo.uNorm = NewBufferU32(d, b1(sigmoid)), NewBufferU32(d, b1(norm))
	mo.uScale = NewBufferFloats(d, []float32{float32(scale)})
	mo.uNGroup, mo.uTopkGroup = NewBufferU32(d, uint32(nGroup)), NewBufferU32(d, uint32(topkGroup))
	mo.uInter, mo.uInter2 = NewBufferU32(d, uint32(inter)), NewBufferU32(d, uint32(2*inter))
	mo.uSharedInter = NewBufferU32(d, uint32(sharedInter))
	mo.uSlot = make([]Buffer, k)
	for j := range mo.uSlot {
		mo.uSlot[j] = NewBufferU32(d, uint32(j))
	}
	mo.rLogits = d.NewBufferLen(nE)
	mo.rIdx = NewBufferUint32s(d, make([]uint32, k))
	mo.rWgt = d.NewBufferLen(k)
	mo.shGl, mo.shDown = d.NewBufferLen(1), d.NewBufferLen(H)
	// Synchronous paging: GOINFER_METAL_MOE_SLOTS=N keeps only N experts/layer resident and stages
	// the routed top-k in per token — same knob and semantics as gemma4_moe.go's (shared env var
	// deliberately: it is the same underlying mechanism, generalized). N==0/unset ⇒ all experts
	// resident (today's behavior).
	if s := os.Getenv("GOINFER_METAL_MOE_SLOTS"); s != "" {
		n, err := strconv.Atoi(s)
		if err != nil || n < k {
			return nil, fmt.Errorf("GOINFER_METAL_MOE_SLOTS=%q invalid (need integer >= topK=%d)", s, k)
		}
		if n < nE { // n>=nE would hold every expert — no paging, just build the stacked path
			mo.paged, mo.slots = true, n
			mo.idxZeros = NewBufferUint32s(d, make([]uint32, k))
		}
	}
	// Stage experts by pread'ing their nibbles straight into the slot buffers instead of byte-copying
	// off the mmap — the refinement gemma4_moe.go measured (cold A/B: 1892→1488 ms/tok, 1.26×; major
	// faults 92.8→0.0/stage, the demand-fault page-in gone) applied to this shape. DEFAULT ON; needs a
	// .giw-mmap'd model, since the offsets are into that file. Re-opened once, shared across layers.
	// GOINFER_MOE_PREAD=0 opts out (the mmap byte-copy baseline, kept for the A/B). If the open fails
	// or the model isn't .giw-backed, buildMoELayer falls back to the byte-copy.
	//
	// GOINFER_MOE_NOCACHE is deliberately NOT wired here: gemma4 measured it and DECLINED (no effect,
	// and its motivating evidence was an ordering confound — see buildGemma4MoEResident). Porting a
	// declined flag would re-open a settled question.
	if mo.paged && os.Getenv("GOINFER_MOE_PREAD") != "0" {
		if p := m.GiwPath(); p != "" {
			if f, err := os.Open(p); err == nil {
				mo.giwFile = f
			} else {
				fmt.Fprintf(os.Stderr, "metal MoE: open(%s) failed (%v) — using mmap byte-copy\n", p, err)
			}
		}
	}
	return mo, nil
}

// buildMoELayer packs one MoE layer's weights. Experts are row-stacked into ONE buffer per
// projection via the existing int4Concat (all-E gate|up interleaved gate_e,up_e; all-E down),
// so expert e lives at weight row e*2*inter (gate|up) / e*H (down). The router (+bias) stays
// f32; the shared expert is packed like the dense FFN.
func buildMoELayer(d *Device, m *decoder.Model, l int, lw *decoder.LayerWeights, mo *moeResident) *moeLayer {
	ml := &moeLayer{}
	ml.routerW = f32Mat(d, &lw.Router)
	if len(lw.RouterBias) > 0 {
		ml.routerBias = NewBufferFloats(d, lw.RouterBias)
	} else {
		ml.routerBias = NewBufferFloats(d, make([]float32, mo.nE)) // zeros → sel = score
	}
	if mo.paged {
		// Paged: don't stack the experts (that is the multi-GB set this exists to avoid). Build a
		// bounded LRU slot pool + a stage fn reading expert e's W4A8 bytes from the layer's own
		// WeightMats on demand. Per-expert buffer sizes come from expert 0. Mirrors
		// buildGemma4MoELayer's paged branch exactly (byte-copy staging; no pread fast path here —
		// that was a later, separately-measured refinement on top of a working paged baseline).
		gw0, gs0, ok1 := int4DirectWords(&lw.Experts[0].Gate)
		uw0, us0, ok1u := int4DirectWords(&lw.Experts[0].Up)
		dw0, ds0, ok2 := int4DirectWords(&lw.Experts[0].Down)
		if !ok1 || !ok1u || !ok2 {
			panic("metal MoE paging: experts are not int4-direct (group-32) — cannot stage without a re-quant")
		}
		nGuW, nGuS := len(gw0)+len(uw0), len(gs0)+len(us0)
		experts := lw.Experts // capture (aliases the model's own weights; kept alive by the Model)
		stage := func(e int) ([]byte, []uint16, []byte, []uint16) {
			gw, gs, _ := int4DirectBytes(&experts[e].Gate)
			uw, us, _ := int4DirectBytes(&experts[e].Up)
			dw, ds, _ := int4DirectBytes(&experts[e].Down)
			// gate|up fused per row-block: the stacked (non-paged) layout is gate rows THEN up rows
			// per expert (int4Concat(gate,up) order), so a slot must reproduce that concatenation —
			// gemv_w4a8_moe reads row r < inter as gate, r >= inter as up, off the SAME buffer.
			guBytes := append(append([]byte(nil), gw...), uw...)
			guScales := append(append([]uint16(nil), gs...), us...)
			return guBytes, guScales, dw, ds
		}
		ml.pool = newExpertPool(d, mo.slots, nGuW, nGuS, len(dw0), len(ds0), stage)
		// pread staging: resolve each expert's nibble file offsets within the .giw mmap (pure pointer
		// arithmetic — MmapByteOffset does NOT touch the pages), then stage by pread'ing straight into
		// the slot's UMA words. Zero mmap faults, one large sequential read per projection.
		//
		// Three reads per expert, not gemma4's two: this shape keeps gate and up as SEPARATE tensors
		// (gemma4's bundle hands over one already-fused gate|up span), so the slot's fused gate|up
		// buffer is filled in two ranges — gate at byte 0, up at byte len(gate) — exactly reproducing
		// the byte-copy path's append(gw, uw...) and hence the stacked layout gemv_w4a8_moe reads.
		// Scales stay f32→f16 from the heap-resident q4s (giwReader.f32 COPIES them, so reading them
		// never faults the mmap).
		//
		// Falls back to the byte-copy path if the fd is absent or ANY expert's nibbles aren't
		// .giw-mmap-backed (e.g. a requantized HF/safetensors load): every offset must resolve for
		// pread to be correct, so this is all-or-nothing per layer, never per expert.
		if mo.giwFile != nil {
			type expertSpan struct {
				gOff, uOff, dOff int64
				gLen, uLen, dLen int
			}
			spans := make([]expertSpan, len(experts))
			resolved := true
			for ei := range experts {
				gq, _, _, okg := experts[ei].Gate.Int4()
				uq, _, _, oku := experts[ei].Up.Int4()
				dq, _, _, okd := experts[ei].Down.Int4()
				if !okg || !oku || !okd {
					resolved = false
					break
				}
				gOff, ok1 := m.MmapByteOffset(gq)
				uOff, ok2 := m.MmapByteOffset(uq)
				dOff, ok3 := m.MmapByteOffset(dq)
				if !ok1 || !ok2 || !ok3 {
					resolved = false
					break
				}
				spans[ei] = expertSpan{gOff, uOff, dOff, len(gq), len(uq), len(dq)}
			}
			if resolved {
				fd := int(mo.giwFile.Fd())
				ml.pool.stagePread = func(ei int, sl expertSlot) {
					sp := spans[ei]
					if err := preadRangeIntoU32Buf(fd, sl.guW, 0, sp.gOff, sp.gLen); err != nil {
						panic(fmt.Sprintf("metal MoE pread gate expert %d: %v", ei, err))
					}
					if err := preadRangeIntoU32Buf(fd, sl.guW, sp.gLen, sp.uOff, sp.uLen); err != nil {
						panic(fmt.Sprintf("metal MoE pread up expert %d: %v", ei, err))
					}
					if err := preadRangeIntoU32Buf(fd, sl.dW, 0, sp.dOff, sp.dLen); err != nil {
						panic(fmt.Sprintf("metal MoE pread down expert %d: %v", ei, err))
					}
					_, gs, _ := int4DirectBytes(&experts[ei].Gate) // f16 scales from heap q4s (no mmap fault)
					_, us, _ := int4DirectBytes(&experts[ei].Up)
					_, ds, _ := int4DirectBytes(&experts[ei].Down)
					guS := sl.guS.U16s()
					copy(guS, gs)
					copy(guS[len(gs):], us)
					copy(sl.dS.U16s(), ds)
				}
			}
		}
	} else {
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
	}
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
	if mo.isGptOss {
		ml.expGuBias = NewBufferFloats(d, m.GptOssExpertBiasResident(l))
		ml.expDBias = NewBufferFloats(d, m.GptOssExpertDownBiasResident(l))
	}
	return ml
}

// f32Mat uploads a full-precision WeightMat (router / shared-gate) to a Metal f32 buffer.
func f32Mat(d *Device, w *linalg.WeightMat) Buffer {
	f, ok := w.F32()
	if !ok {
		panic(fmt.Sprintf("metal MoE: weight kind %q is not f32 (router/shared-gate must be full precision)", w.Kind()))
	}
	return NewBufferFloats(d, f)
}

// encodeMoEFFN records the MoE FFN for one layer, replacing the dense gate/up/swiglu/down
// dispatches. post-attn norm → router logits → on-GPU top-k → per-selected-expert
// gate|up/swiglu/weighted-down → optional shared expert. Value-independent (idx/wgt read at
// kernel-execution time), so the encode-ahead executor still pre-encodes it.
func (r *resident) encodeMoEFFN(e *Encoder, L *residLayer) {
	r.encodeMoERouter(e, L)
	r.encodeMoEExperts(e, L)
}

// encodeMoERouter is the value-INDEPENDENT head of the FFN: post-attn norm → router logits →
// on-GPU top-k, ending with rIdx[k]/rWgt[k] written on-device. In the paged forward this is the
// first command buffer; the host then reads rIdx and stages the routed experts before the second.
func (r *resident) encodeMoERouter(e *Encoder, L *residLayer) {
	mo := r.moe
	ml := L.moe
	// post-attn RMSNorm → quantized activation mq/mSc (same as the dense FFN entry).
	e.Dispatch(r.pRms, 256, 256, r.x, L.postNorm, r.mq, r.mSc, r.uH, r.uEps, r.uAddOne)
	// router logits: routerW[nE,H] × mq → rLogits[nE]
	e.Dispatch(mo.pRouter, mo.nE*32, 32, ml.routerW, r.mq, r.mSc, mo.rLogits, r.uH)
	// on-GPU top-k → rIdx[k], rWgt[k]. gpt-oss's router bias reaches the logits BEFORE top-k and
	// the weight is softmax over the SELECTED biased logits — a genuinely different kernel from
	// the generic DeepSeek/GLM/Qwen-MoE shape (moe_route), not a parameter swap (see route_gptoss).
	if mo.isGptOss {
		e.Dispatch(mo.pRouteGptOss, 1, 1, mo.rLogits, ml.routerBias, mo.rIdx, mo.rWgt, mo.uNE, mo.uK)
	} else {
		e.Dispatch(mo.pRoute, 1, 1, mo.rLogits, ml.routerBias, mo.rIdx, mo.rWgt,
			mo.uNE, mo.uK, mo.uSigmoid, mo.uNorm, mo.uScale, mo.uNGroup, mo.uTopkGroup)
	}
}

// encodeMoEExperts runs the k selected experts out of the STACKED all-E buffer (ml.expGuW/expDW,
// indexed by rIdx at kernel-execution time) plus the optional shared expert. The non-paged path.
func (r *resident) encodeMoEExperts(e *Encoder, L *residLayer) {
	mo := r.moe
	ml := L.moe
	// k selected experts: fused gate|up (overwrite) → clamped-swiglu/swiglu → down (weighted-
	// accumulate into x). gpt-oss fuses its per-expert gate‖up bias into the activation step and
	// its per-expert down bias into the combine step (both INSIDE the router-weight scale — see
	// swiglu_quant_gptoss / gemv_w4a8_moe_wacc_bias's docs), so it swaps both dispatches, not just
	// the router.
	for j := 0; j < mo.k; j++ {
		e.DispatchTG(mo.pGU, (2*mo.inter)*32, 256, r.H*2, ml.expGuW, ml.expGuS, r.mq, r.mSc, r.gu, r.uH, mo.rIdx, mo.uSlot[j], mo.uInter2)
		if mo.isGptOss {
			e.Dispatch(mo.pActGptOss, 256, 256, r.gu, r.gu.At(mo.inter*4), r.dq, r.dSc, mo.uInter,
				ml.expGuBias, mo.rIdx, mo.uSlot[j], mo.uHasBias, mo.uAlpha, mo.uLimit)
			e.DispatchTG(mo.pDownWaccBias, r.H*32, 256, mo.inter*2, ml.expDW, ml.expDS, r.dq, r.dSc, r.x, mo.uInter, mo.rIdx, mo.rWgt, mo.uSlot[j], r.uH, ml.expDBias)
		} else {
			e.Dispatch(r.pSw, 256, 256, r.gu, r.gu.At(mo.inter*4), r.dq, r.dSc, mo.uInter, r.uAct)
			e.DispatchTG(mo.pDownWacc, r.H*32, 256, mo.inter*2, ml.expDW, ml.expDS, r.dq, r.dSc, r.x, mo.uInter, mo.rIdx, mo.rWgt, mo.uSlot[j], r.uH)
		}
	}
	r.encodeMoESharedExpert(e, L)
}

// encodeMoEExpertsPaged is encodeMoEExperts' paged twin: the k selected experts run out of their
// STAGED SLOT buffers (one expert per slot) instead of the stacked all-E buffer. idxZeros makes
// each GEMV read row 0 of its single-expert slot (idxZeros[j]==0) while rWgt is still indexed by
// the selection slot uSlot[j], so the reused gemv_w4a8_moe(_wacc[_bias]) kernels compute
// byte-identically to the stacked path (slot bytes == the stacked buffer's rows for that expert —
// see buildMoELayer's paged stage fn). Mirrors gemma4_moe.go's encodeG4Phase2Paged.
func (r *resident) encodeMoEExpertsPaged(e *Encoder, L *residLayer, slots []expertSlot) {
	mo := r.moe
	ml := L.moe
	for j := 0; j < mo.k; j++ {
		s := slots[j]
		e.DispatchTG(mo.pGU, (2*mo.inter)*32, 256, r.H*2, s.guW, s.guS, r.mq, r.mSc, r.gu, r.uH, mo.idxZeros, mo.uSlot[j], mo.uInter2)
		if mo.isGptOss {
			e.Dispatch(mo.pActGptOss, 256, 256, r.gu, r.gu.At(mo.inter*4), r.dq, r.dSc, mo.uInter,
				ml.expGuBias, mo.idxZeros, mo.uSlot[j], mo.uHasBias, mo.uAlpha, mo.uLimit)
			e.DispatchTG(mo.pDownWaccBias, r.H*32, 256, mo.inter*2, s.dW, s.dS, r.dq, r.dSc, r.x, mo.uInter, mo.idxZeros, mo.rWgt, mo.uSlot[j], r.uH, ml.expDBias)
		} else {
			e.Dispatch(r.pSw, 256, 256, r.gu, r.gu.At(mo.inter*4), r.dq, r.dSc, mo.uInter, r.uAct)
			e.DispatchTG(mo.pDownWacc, r.H*32, 256, mo.inter*2, s.dW, s.dS, r.dq, r.dSc, r.x, mo.uInter, mo.idxZeros, mo.rWgt, mo.uSlot[j], r.uH)
		}
	}
	r.encodeMoESharedExpert(e, L)
}

// encodeMoESharedExpert is the always-on shared expert (Qwen2-MoE gated / GLM ungated) — never
// paged (it is one dense expert, always resident), so identical in both the paged and non-paged
// tail.
func (r *resident) encodeMoESharedExpert(e *Encoder, L *residLayer) {
	mo := r.moe
	ml := L.moe
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

// forwardLogitsMoEPaged is the SYNCHRONOUS expert-paging decode for the generic MoE shape (the
// mechanism gemma4_moe.go's forwardLogitsPaged pioneered, generalized here to also cover a
// DeltaNet mixer layer's FFN half — qwen3_5_moe/qwen3_next pair a recurrent mixer with THIS FFN
// shape, not gemma4's). Per layer: dense/attention-only layers (including a DeltaNet layer whose
// FFN isn't paged, or when GOINFER_METAL_MOE_SLOTS is unset) encode in one command buffer via the
// existing encodeLayer; a paged MoE layer is torn at the router — the value-dependent seam —
// [mixer/attention + router] -> submit+wait -> read rIdx -> stage the routed top-k into the
// layer's LRU slot pool -> [experts-from-slots] -> submit+wait. Assumes the caller filled r.x with
// the embedding and holds the OS thread (ForwardEmb does both).
func (r *resident) forwardLogitsMoEPaged(pos int) (logits []float32) {
	// Same abort discipline as forwardLogitsPaged (audit R-02/C-09): a transient staging error or
	// OOM deep inside expertPool.ensureResident at decode time must not crash the process.
	defer func() {
		if p := recover(); p != nil {
			r.recordExecErr(fmt.Errorf("metal: paged MoE forward aborted: %v", p))
			logits = nil
		}
	}()
	r.uPos.SetU32(uint32(pos))
	r.uNKeys.SetU32(uint32(pos + 1))
	mo := r.moe
	for l := 0; l < r.nL; l++ {
		L := &r.layers[l]
		if L.moe != nil && L.moe.pool != nil {
			e := r.q.Begin() // phase 1: mixer/attention + router -> rIdx/rWgt
			if L.delta != nil {
				r.encodeDeltaNetMixer(e, L)
			} else {
				r.encodeAttention(e, l)
			}
			r.encodeMoERouter(e, L)
			e.End()
			r.recordExecErr(e.Err())
			idx := mo.rIdx.U32s()
			ids := make([]int, mo.k)
			for j := 0; j < mo.k; j++ {
				ids[j] = int(idx[j])
			}
			slots := make([]expertSlot, mo.k)
			for j := 0; j < mo.k; j++ {
				slots[j] = L.moe.pool.ensureResident(ids[j]) // stage on miss, evict LRU
			}
			e2 := r.q.Begin() // phase 2: experts from slots (+ shared expert)
			r.encodeMoEExpertsPaged(e2, L, slots)
			e2.End()
			r.recordExecErr(e2.Err())
			continue
		}
		e := r.q.Begin()
		r.encodeLayer(e, l)
		e.End()
		r.recordExecErr(e.Err())
	}
	e := r.q.Begin()
	e.Dispatch(r.pRms, 256, 256, r.x, r.finalNorm, r.aq, r.aSc, r.uH, r.uEps, r.uAddOne)
	e.Dispatch(r.pGemvW8, (r.V)*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.logits, r.uH)
	e.End()
	r.recordExecErr(e.Err())
	r.finalizeLogits()
	return r.logitsHost
}
