package kernels

// MoeKernels is concatenated onto allKernels (kernels.go) so it shares the one compiled
// library + the #include/using and the UNP8 macro. int4 expert weights = group=32, f16
// group scales, int8 activations — identical quant to the dense W4A8 path.
const MoeKernels = `
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
//
// When no remaining score is above -INFINITY (every one NaN or -INFINITY: a crafted router, or an activation that
// overflowed) a slot takes the lowest expert not yet chosen, with a finite score so the weights stay finite. An index
// outside [0, nE) must never leave this kernel: the indexed expert GEMVs use it as a row offset into the weights.
// route_gptoss_sg and route_gptoss_batch (prefill.go) follow the same rule and write the same bytes.
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
        if (best < 0) {
            best = 0;
            for (uint i=0u;i<nE;i++) if (!taken[i]) { best=int(i); break; }
            bv = -1e30f;
        }
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

// moe_route_sg / route_gptoss_sg (D-P04): moe_route and route_gptoss on one simdgroup, dispatched (32,32), the same
// bytes out. moe_route on one thread measured 68 us per dispatch at 60x4 and 201 us at 128x8 (BenchmarkMoERoute, M1
// Pro), once per MoE layer per token. Lane l owns experts l, l+32, ...; everything elementwise (exp, sigmoid, the
// scale by 1/sum, +bias) runs lane-strided and is the same expression per element. What is order-sensitive stays in
// the serial order: the softmax sum and wsum run over i or j ascending, uniformly in every lane, and lane 0 does the
// output writes, the renorm and the scale exactly as moe_route does. Top-k is exact: a lane's strict-> scan of its
// own experts in ascending order, then simd_max of the value and simd_min of the index among the lanes holding it,
// which is the serial scan's lowest index on ties. When nothing is above -INFINITY, moe_route_sg selects expert 0 as
// moe_route does, and route_gptoss_sg selects the lowest expert not yet chosen as route_gptoss does: a chosen expert's
// score is overwritten with -INFINITY, which a remaining NaN or -INFINITY score cannot be told from, so that kernel
// also keeps the chosen set in a per-lane array that every lane updates alike.
// Group limiting: a group's top-2 sum is order-independent, so each lane scores whole groups; keep is the serial
// selection, run uniformly. Tested against moe_route/route_gptoss in moe_route_sg_test.go.
static inline uint route_sg_pick(threadgroup float* sel, uint nE, uint lane, thread float& m) {
    float bv=-INFINITY; uint bi=0xFFFFFFFFu;
    for (uint i=lane; i<nE; i+=32u) { float v=sel[i]; if (v>bv){ bv=v; bi=i; } }
    m = simd_max(bv);
    return simd_min((bi != 0xFFFFFFFFu && bv == m) ? bi : 0xFFFFFFFFu);
}
kernel void moe_route_sg(device const float* logits[[buffer(0)]], device const float* bias[[buffer(1)]],
    device uint* outIdx[[buffer(2)]], device float* outWgt[[buffer(3)]], constant uint& nE[[buffer(4)]],
    constant uint& k[[buffer(5)]], constant uint& sigmoid[[buffer(6)]], constant uint& norm[[buffer(7)]],
    constant float& scale[[buffer(8)]], constant uint& nGroup[[buffer(9)]], constant uint& topkGroup[[buffer(10)]],
    uint lane[[thread_index_in_threadgroup]]) {
    threadgroup float score[256];
    threadgroup float sel[256];
    threadgroup float gscore[64];
    if (sigmoid != 0u) {
        for (uint i=lane;i<nE;i+=32u) score[i] = 1.0f/(1.0f+exp(-logits[i]));
    } else {
        float mx = -INFINITY; for (uint i=lane;i<nE;i+=32u) mx = max(mx, logits[i]);
        mx = simd_max(mx);
        for (uint i=lane;i<nE;i+=32u) score[i] = exp(logits[i]-mx);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sum = 0.0f; for (uint i=0u;i<nE;i++) sum += score[i];
        float inv = 1.0f/sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i=lane;i<nE;i+=32u) score[i] *= inv;
    }
    for (uint i=lane;i<nE;i+=32u) sel[i] = score[i] + bias[i];
    if (nGroup > 1u) {
        uint gsz = nE / nGroup;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint g=lane; g<nGroup; g+=32u) {
            float t1=-INFINITY, t2=-INFINITY;
            for (uint i=g*gsz; i<(g+1u)*gsz; i++) { float v=sel[i]; if (v>t1){t2=t1;t1=v;} else if (v>t2){t2=v;} }
            gscore[g]=t1+t2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ulong keep = 0ul;
        for (uint j=0u; j<topkGroup; j++) {
            uint bg=0u; float bv=-INFINITY; bool found=false;
            for (uint g=0u; g<nGroup; g++) if (((keep>>g)&1ul)==0ul && gscore[g]>bv){ bv=gscore[g]; bg=g; found=true; }
            if (found) keep |= 1ul<<bg;
        }
        for (uint i=lane;i<nE;i+=32u) { uint g=i/gsz; if (g<nGroup && ((keep>>g)&1ul)==0ul) sel[i]=-INFINITY; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float wsum = 0.0f;
    for (uint j=0u; j<k; j++) {
        float m;
        uint best = route_sg_pick(sel, nE, lane, m);
        if (best == 0xFFFFFFFFu) best = 0u;
        float w = score[best];
        if (lane == 0u) { outIdx[j]=best; outWgt[j]=w; }
        wsum += w;
        if (lane == (best & 31u)) sel[best]=-INFINITY; // the only lane that reads it next round
    }
    if (lane != 0u) return;
    if (norm != 0u && wsum > 0.0f) for (uint j=0u; j<k; j++) outWgt[j] /= wsum;
    if (scale != 0.0f && scale != 1.0f) for (uint j=0u; j<k; j++) outWgt[j] *= scale;
}
kernel void route_gptoss_sg(device const float* logits[[buffer(0)]], device const float* bias[[buffer(1)]],
    device uint* outIdx[[buffer(2)]], device float* outWgt[[buffer(3)]], constant uint& nE[[buffer(4)]],
    constant uint& k[[buffer(5)]],
    uint lane[[thread_index_in_threadgroup]]) {
    threadgroup float sc[256];
    float chosen[256];
    bool taken[256]; // the same in every lane: best is uniform
    for (uint i=0u;i<nE;i++) taken[i]=false;
    for (uint i=lane;i<nE;i+=32u) sc[i] = logits[i] + bias[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j=0u;j<k;j++) {
        float bv; // sc[best]
        uint best = route_sg_pick(sc, nE, lane, bv);
        if (best == 0xFFFFFFFFu) { // nothing above -INFINITY is left: route_gptoss's rule
            best = 0u;
            for (uint i=0u;i<nE;i++) if (!taken[i]) { best=i; break; }
            bv = -1e30f;
        }
        taken[best]=true;
        if (lane == 0u) { outIdx[j]=best; chosen[j]=bv; }
        if (lane == (best & 31u)) sc[best]=-INFINITY;
    }
    if (lane != 0u) return;
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

// gemv_w8a8_moe / gemv_w8a8_moe_wacc (int8 slice 4, docs/tasks/task-metal-int8-2026-10.md): the expert GEMVs above on the
// native int8 path's weights, in their argument order and launch shape. The stacked buffer holds every expert's int8
// rows (K bytes each) with one f32 scale per row; the weight row is idx[slot]*rowsPerExpert + outRow. The body is
// W8A8_SA_BODY's at that row: the activation staged as words, an exact int32 dot, and float(acc)·aScale·wScale through
// simd_broadcast_first. Mode 0 overwrites (fused gate|up); mode 1 adds wgt[slot]·y into the residual (down), the
// combine gemv_w4a8_moe_wacc does.
#define W8A8_MOE_BODY \
    threadgroup uint* As4 = (threadgroup uint*)As; \
    device const uint* a4 = (device const uint*)aq; \
    uint G = K >> 2u; \
    for (uint i = tid; i < G; i += tgs) As4[i] = a4[i]; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    uint row = tgid*(tgs>>5u) + sgid; \
    uint wrow = idx[slot]*rowsPerExpert + row; \
    device const uint* wr = (device const uint*)(wq + (uint)wrow*K); \
    int acc = 0; \
    for (uint g = lane; g < G; g += 32u) acc += DOT4I8(As4[g], wr[g]); \
    acc = simd_sum(acc); \
    float y = simd_broadcast_first(float(acc) * asc[0] * sct[wrow]);
kernel void gemv_w8a8_moe(device const char* wq[[buffer(0)]], device const float* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], device const uint* idx[[buffer(6)]], constant uint& slot[[buffer(7)]],
    constant uint& rowsPerExpert[[buffer(8)]], threadgroup short* As[[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]],
    uint tgs[[threads_per_threadgroup]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]]) {
    W8A8_MOE_BODY
    if (lane==0) out[row] = y;
}
kernel void gemv_w8a8_moe_wacc(device const char* wq[[buffer(0)]], device const float* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], device const uint* idx[[buffer(6)]], device const float* wgt[[buffer(7)]],
    constant uint& slot[[buffer(8)]], constant uint& rowsPerExpert[[buffer(9)]], threadgroup short* As[[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]],
    uint tgs[[threads_per_threadgroup]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]]) {
    W8A8_MOE_BODY
    if (lane==0) out[row] += wgt[slot]*y;
}
// gemv_w8a8_moe_rows / gemv_w8a8_moe_wacc_rows: the two int8 expert GEMVs above with R output rows per simdgroup, one
// read of each staged activation word feeding R exact int32 dots (the integer sum does not depend on the order), then the
// originals' epilogue for each row, the same expression: y = simd_broadcast_first(float(acc)*asc*sct), out = y or
// out += wgt*y. Full threadgroups only (rows % 8R == 0, gemvRowsFor).
#define W8A8_MOE_ROWS_BODY \
    threadgroup uint* As4 = (threadgroup uint*)As; \
    device const uint* a4 = (device const uint*)aq; \
    uint G = K >> 2u; \
    for (uint i = tid; i < G; i += tgs) As4[i] = a4[i]; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    uint row0 = (tgid*(tgs>>5u) + sgid)*R; \
    uint wrow0 = idx[slot]*rowsPerExpert + row0; \
    int acc[R]; \
    SA_ROWS_UNROLL for (uint r=0;r<R;r++) acc[r] = 0; \
    for (uint g = lane; g < G; g += 32u) { \
        uint a = As4[g]; \
        SA_ROWS_UNROLL for (uint r=0;r<R;r++) acc[r] += DOT4I8(a, ((device const uint*)(wq + (uint)(wrow0+r)*K))[g]); \
    } \
    float y[R]; \
    SA_ROWS_UNROLL for (uint r=0;r<R;r++) { int sr = simd_sum(acc[r]); y[r] = simd_broadcast_first(float(sr) * asc[0] * sct[wrow0+r]); }
template <uint R>
kernel void gemv_w8a8_moe_rows(device const char* wq[[buffer(0)]], device const float* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], device const uint* idx[[buffer(6)]], constant uint& slot[[buffer(7)]],
    constant uint& rowsPerExpert[[buffer(8)]], threadgroup short* As[[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]],
    uint tgs[[threads_per_threadgroup]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]]) {
    W8A8_MOE_ROWS_BODY
    if (lane==0) { SA_ROWS_UNROLL for (uint r=0;r<R;r++) out[row0+r] = y[r]; }
}
template <uint R>
kernel void gemv_w8a8_moe_wacc_rows(device const char* wq[[buffer(0)]], device const float* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], device const uint* idx[[buffer(6)]], device const float* wgt[[buffer(7)]],
    constant uint& slot[[buffer(8)]], constant uint& rowsPerExpert[[buffer(9)]], threadgroup short* As[[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]],
    uint tgs[[threads_per_threadgroup]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]]) {
    W8A8_MOE_ROWS_BODY
    if (lane==0) { SA_ROWS_UNROLL for (uint r=0;r<R;r++) out[row0+r] += wgt[slot]*y[r]; }
}
template [[host_name("gemv_w8a8_moe_rows2")]] kernel decltype(gemv_w8a8_moe_rows<2>) gemv_w8a8_moe_rows<2>;
template [[host_name("gemv_w8a8_moe_rows4")]] kernel decltype(gemv_w8a8_moe_rows<4>) gemv_w8a8_moe_rows<4>;
template [[host_name("gemv_w8a8_moe_wacc_rows2")]] kernel decltype(gemv_w8a8_moe_wacc_rows<2>) gemv_w8a8_moe_wacc_rows<2>;
template [[host_name("gemv_w8a8_moe_wacc_rows4")]] kernel decltype(gemv_w8a8_moe_wacc_rows<4>) gemv_w8a8_moe_wacc_rows<4>;

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
// gemv_w4a8_moe_rows / gemv_w4a8_moe_wacc_rows (D-B04's routed-expert half, docs/audit-metal-2026-09-30.md): the two
// expert GEMVs above with R18's R rows per simdgroup. gemv_w4a8_moe is gemv_w4a8_sa's body at the routed expert's rows,
// and sa_rows_acc<R> computes gemv_w4a8_sa's sums for R rows (R18, bit-identical), so these run sa_rows_acc over the
// expert's own weight block (rows idx[slot]*rowsPerExpert onwards) and apply the originals' epilogues: out = acc*asc, and
// the accumulate as the compiler lowers the original's. Full threadgroups only, as sa_rows_acc requires (rows % 8R == 0).
template <uint R>
kernel void gemv_w4a8_moe_rows(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], device const uint* idx[[buffer(6)]], constant uint& slot[[buffer(7)]],
    constant uint& rowsPerExpert[[buffer(8)]], threadgroup half* Ah[[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]],
    uint tgs[[threads_per_threadgroup]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]]) {
    uint G = K>>5u, base = idx[slot]*rowsPerExpert;
    float acc[R]; uint row0;
    sa_rows_acc<R>(wq + (uint)base*G, sct + (uint)base*G, aq, K, Ah, tgid, tid, tgs, sgid, lane, acc, row0);
    if (lane==0) { SA_ROWS_UNROLL for (uint r=0;r<R;r++) out[row0+r] = acc[r]*asc[0]; }
}
template <uint R>
kernel void gemv_w4a8_moe_wacc_rows(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], device const uint* idx[[buffer(6)]], device const float* wgt[[buffer(7)]],
    constant uint& slot[[buffer(8)]], constant uint& rowsPerExpert[[buffer(9)]], threadgroup half* Ah[[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]],
    uint tgs[[threads_per_threadgroup]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]]) {
    uint G = K>>5u, base = idx[slot]*rowsPerExpert;
    float acc[R]; uint row0;
    sa_rows_acc<R>(wq + (uint)base*G, sct + (uint)base*G, aq, K, Ah, tgid, tid, tgs, sgid, lane, acc, row0);
    // fma(wgt*acc, asc, out), written out: it is how the compiler lowers gemv_w4a8_moe_wacc's out += wgt*acc*asc under fast
    // math and under precise (measured, every output of 3 shapes x 3 seeds); left implicit here, fast math reassociated
    // it differently inside the R loop and the outputs differed in the last bit.
    if (lane==0) { SA_ROWS_UNROLL for (uint r=0;r<R;r++) out[row0+r] = fma(wgt[slot]*acc[r], asc[0], out[row0+r]); }
}
template [[host_name("gemv_w4a8_moe_rows2")]] kernel decltype(gemv_w4a8_moe_rows<2>) gemv_w4a8_moe_rows<2>;
template [[host_name("gemv_w4a8_moe_rows4")]] kernel decltype(gemv_w4a8_moe_rows<4>) gemv_w4a8_moe_rows<4>;
template [[host_name("gemv_w4a8_moe_wacc_rows2")]] kernel decltype(gemv_w4a8_moe_wacc_rows<2>) gemv_w4a8_moe_wacc_rows<2>;
template [[host_name("gemv_w4a8_moe_wacc_rows4")]] kernel decltype(gemv_w4a8_moe_wacc_rows<4>) gemv_w4a8_moe_wacc_rows<4>;
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
    device const uint* bidx[[buffer(11)]],
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
    // bidx, NOT idx. On the PAGED path idx is a zero buffer so the weight GEMV reads row 0 of a
    // one-expert slot — correct — but the bias table stays the STACKED all-expert one, so indexing
    // it the same way gave every routed expert expert 0's bias. Finite, plausible, wrong: the class
    // CUDA fixed in d9829ce, here indexed by a constant zero (audit-2026-09-02 C-09). The
    // non-paged path passes the same buffer for both and is unchanged.
    if (lane==0) out[row] += wgt[slot]*(acc*asc[0] + bias[bidx[slot]*rowsPerExpert + row]);
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
