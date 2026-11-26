// router_gemm_f16: batched router projection — normF[M×H](f16) · routerWᵀ[nE×H](f32) → logits[M×nE](f32).
// Grid = M*nE*32 threads (one simdgroup of 32 lanes per (token, expert) pair).
kernel void router_gemm_f16(
    device const half* in[[buffer(0)]],
    device const float* wf[[buffer(1)]],
    device float* logits[[buffer(2)]],
    constant uint& H[[buffer(3)]],
    constant uint& nE[[buffer(4)]],
    constant uint& M[[buffer(5)]],
    uint tid[[thread_position_in_grid]]) {
    uint sg = tid >> 5u;
    uint m = sg / nE;
    uint e = sg % nE;
    if (m >= M || e >= nE) return;
    uint lane = tid & 31u;
    device const float* wr = wf + e * H;
    device const half* ar = in + m * H;
    float acc = 0.0f;
    for (uint h = lane; h < H; h += 32u) {
        acc += wr[h] * float(ar[h]);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        logits[m * nE + e] = acc;
    }
}

// moe_route_batch: batched top-k router selection for all M prompt rows.
// Grid = M*32 threads (simdgroup per token; lane 0 executes route logic).
kernel void moe_route_batch(
    device const float* logits[[buffer(0)]],
    device const float* bias[[buffer(1)]],
    device uint* outIdx[[buffer(2)]],
    device float* outWgt[[buffer(3)]],
    constant uint& nE[[buffer(4)]],
    constant uint& k[[buffer(5)]],
    constant uint& sigmoid[[buffer(6)]],
    constant uint& norm[[buffer(7)]],
    constant float& scale[[buffer(8)]],
    constant uint& nGroup[[buffer(9)]],
    constant uint& topkGroup[[buffer(10)]],
    constant uint& M[[buffer(11)]],
    uint tid[[thread_position_in_grid]]) {
    uint m = tid >> 5u;
    uint lane = tid & 31u;
    if (m >= M || lane != 0u) return;
    device const float* rowLogits = logits + m * nE;
    device uint* rowIdx = outIdx + m * k;
    device float* rowWgt = outWgt + m * k;

    float score[256];
    float sel[256];
    if (sigmoid != 0u) {
        for (uint i=0u;i<nE;i++) score[i] = 1.0f/(1.0f+exp(-rowLogits[i]));
    } else {
        float mx = rowLogits[0]; for (uint i=1u;i<nE;i++) mx = max(mx, rowLogits[i]);
        float sum = 0.0f; for (uint i=0u;i<nE;i++){ float e = exp(rowLogits[i]-mx); score[i]=e; sum+=e; }
        float inv = 1.0f/sum; for (uint i=0u;i<nE;i++) score[i] *= inv;
    }
    for (uint i=0u;i<nE;i++) sel[i] = score[i] + bias[i];

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
        for (uint i=0u; i<nE; i++) if (sel[i]>bv){ bv=sel[i]; best=i; }
        rowIdx[j]=best; rowWgt[j]=score[best]; wsum+=score[best]; sel[best]=-INFINITY;
    }
    if (norm != 0u && wsum > 0.0f) for (uint j=0u; j<k; j++) rowWgt[j] /= wsum;
    if (scale != 0.0f && scale != 1.0f) for (uint j=0u; j<k; j++) rowWgt[j] *= scale;
}

// route_gptoss_batch: batched router for gpt-oss MoE. A slot with no score above -INFINITY takes the lowest expert not
// yet chosen in its row, as route_gptoss does (moe.go), so no index outside [0, nE) is written.
kernel void route_gptoss_batch(device const float* logits[[buffer(0)]], device const float* bias[[buffer(1)]],
    device uint* outIdx[[buffer(2)]], device float* outWgt[[buffer(3)]], constant uint& nE[[buffer(4)]],
    constant uint& k[[buffer(5)]], constant uint& M[[buffer(6)]],
    uint tid[[thread_position_in_grid]]) {
    uint m = tid >> 5u;
    uint lane = tid & 31u;
    if (m >= M || lane != 0u) return;
    device const float* rowLogits = logits + m * nE;
    device uint* rowIdx = outIdx + m * k;
    device float* rowWgt = outWgt + m * k;
    float sc[256];
    for (uint i=0u;i<nE;i++) sc[i] = rowLogits[i] + bias[i];
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
        rowIdx[j]=uint(best);
        chosen[j]=bv;
    }
    float mx=chosen[0];
    for (uint j=1u;j<k;j++) mx=max(mx,chosen[j]);
    float sum=0.0f;
    for (uint j=0u;j<k;j++) { chosen[j]=exp(chosen[j]-mx); sum+=chosen[j]; }
    float inv=1.0f/sum;
    for (uint j=0u;j<k;j++) rowWgt[j]=chosen[j]*inv;
}

// gather_rows_f16: gathers Ce rows from src[M×H] into dst[CePad×H] according to rowIndices[Ce],
// writing 0.0h for any padding rows Ce <= i < CePad.
kernel void gather_rows_f16(
    device const half* src[[buffer(0)]],
    device half* dst[[buffer(1)]],
    device const uint* rowIndices[[buffer(2)]],
    constant uint& H[[buffer(3)]],
    constant uint& Ce[[buffer(4)]],
    constant uint& CePad[[buffer(5)]],
    uint tid[[thread_position_in_grid]]) {
    uint total = CePad * H;
    if (tid >= total) return;
    uint i = tid / H;
    uint h = tid % H;
    if (i < Ce) {
        uint srcRow = rowIndices[i];
        dst[i * H + h] = src[srcRow * H + h];
    } else {
        dst[i * H + h] = 0.0h;
    }
}

// scatter_add_weighted_f16: scatters Ce rows from src[CePad×H] into residual[M×H], multiplying by weights[Ce].
kernel void scatter_add_weighted_f16(
    device const half* src[[buffer(0)]],
    device half* residual[[buffer(1)]],
    device const uint* rowIndices[[buffer(2)]],
    device const float* weights[[buffer(3)]],
    constant uint& H[[buffer(4)]],
    constant uint& Ce[[buffer(5)]],
    constant uint& M[[buffer(6)]],
    uint tid[[thread_position_in_grid]]) {
    uint total = Ce * H;
    if (tid >= total) return;
    uint i = tid / H;
    uint h = tid % H;
    uint dstRow = rowIndices[i];
    if (dstRow >= M) return;
    float w = weights[i];
    float val = float(src[i * H + h]) * w;
    residual[dstRow * H + h] += half(val);
}

