// attention_prefill: one threadgroup per (row m, query head qh). Row m attends CAUSALLY to
// keys[0..startPos+m] over the shared f16 KV cache (GQA via kvh). q read from the fused qkv
// buffer (row stride qStride, q section at offset 0). Reuses the decode attention math (dot →
// threadgroup softmax → value accum), per row with per-row nKeys. O(M²) but that's inherent to
// prefill attention; an MMA/flash version is a later throughput lever.
kernel void attention_prefill(device const half* qkv[[buffer(0)]], device const half* kc[[buffer(1)]],
    device const half* vc[[buffer(2)]], device half* out[[buffer(3)]], constant uint& nH[[buffer(4)]],
    constant uint& nKV[[buffer(5)]], constant uint& hd[[buffer(6)]], constant uint& startPos[[buffer(7)]],
    constant float& scale[[buffer(8)]], constant uint& qStride[[buffer(9)]], constant uint& window[[buffer(10)]],
    device const uint* kt[[buffer(11)]],
    uint gid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]],
    uint tgs[[threads_per_threadgroup]]) {
    uint m = gid / nH, qh = gid % nH;
    uint kvDim = nKV*hd, kvh = qh/(nH/nKV);
    uint nKeys = startPos + m + 1u;
    uint winStart = (window>0u && nKeys>window) ? nKeys-window : 0u;
    uint qDim = nH*hd;
    device const half* qr = qkv + m*qStride + qh*hd;
    device const half* kb = kc + kvh*hd;
    device const half* vb = vc + kvh*hd;
    threadgroup float sc[4096];
    threadgroup float red[128];
    for (uint s=winStart+tid; s<nKeys; s+=tgs) {
        float a=0; device const half* k=kb+kv_row(kt,s)*kvDim; uint d=0;
        // half4 vectorized K-read (coalescing fix, bit-identical; guarded on hd%4==0, scalar tail).
        if ((hd&3u)==0u) for (; d<hd; d+=4u){ half4 k4=*((device const half4*)(k+d)); a+=float(qr[d])*float(k4.x); a+=float(qr[d+1u])*float(k4.y); a+=float(qr[d+2u])*float(k4.z); a+=float(qr[d+3u])*float(k4.w); }
        for (; d<hd; d++) a += float(qr[d])*float(k[d]);
        sc[s]=attn_score(a*scale);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mmax=-INFINITY; for (uint s=winStart+tid;s<nKeys;s+=tgs) mmax=max(mmax,sc[s]);
    mmax = simd_max(mmax);
    uint sgidA = tid >> 5u, laneA = tid & 31u;
    if (laneA == 0) red[sgidA] = mmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nsgA = tgs >> 5u;
    if (tid == 0) { float v = red[0]; for (uint k=1; k<nsgA; k++) v = max(v, red[k]); red[0] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    float ls=0; for (uint s=winStart+tid;s<nKeys;s+=tgs){ float p=exp(sc[s]-mx); sc[s]=p; ls+=p; }
    red[tid]=ls; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st=tgs/2u; st>0u; st>>=1u){ if(tid<st) red[tid]+=red[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float sum=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d=tid; d<hd; d+=tgs){ float a=0; for(uint s=winStart;s<nKeys;s++) a += sc[s]*float(vb[kv_row(kt,s)*kvDim+d]); out[m*qDim + qh*hd + d]=half(a/sum); }
}

// attention_prefill_tiled: attention_prefill for a pass past its 4,096-score buffer (P28). The scores go through sc[] one tile of
// keys at a time, in two passes: the row's max over every key, then per tile the probabilities, each thread's share of
// the softmax sum and the P*V accumulation. Each thread owns the same keys in the same order as attention_prefill's, the sum
// is reduced by the same tree, and P*V walks the keys in order, so the output equals attention_prefill's wherever both run;
// a score is computed twice (once for the max, once per tile) by the same expression. A thread starts each tile at the first
// key it owns there, so any tile size keeps that ownership. Needs hd <= 4 * the threadgroup size (out-dim accumulators in
// registers).
inline float prefill_score(device const half* qr, device const half* k, uint hd, float scale) {
    float a=0; uint d=0;
    if ((hd&3u)==0u) for (; d<hd; d+=4u){ half4 k4=*((device const half4*)(k+d)); a+=float(qr[d])*float(k4.x); a+=float(qr[d+1u])*float(k4.y); a+=float(qr[d+2u])*float(k4.z); a+=float(qr[d+3u])*float(k4.w); }
    for (; d<hd; d++) a += float(qr[d])*float(k[d]);
    return attn_score(a*scale);
}
kernel void attention_prefill_tiled(device const half* qkv[[buffer(0)]], device const half* kc[[buffer(1)]],
    device const half* vc[[buffer(2)]], device half* out[[buffer(3)]], constant uint& nH[[buffer(4)]],
    constant uint& nKV[[buffer(5)]], constant uint& hd[[buffer(6)]], constant uint& startPos[[buffer(7)]],
    constant float& scale[[buffer(8)]], constant uint& qStride[[buffer(9)]], constant uint& window[[buffer(10)]],
    constant uint& tile[[buffer(11)]], device const uint* kt[[buffer(12)]],
    uint gid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]],
    uint tgs[[threads_per_threadgroup]]) {
    uint m = gid / nH, qh = gid % nH;
    uint kvDim = nKV*hd, kvh = qh/(nH/nKV);
    uint nKeys = startPos + m + 1u;
    uint winStart = (window>0u && nKeys>window) ? nKeys-window : 0u;
    uint qDim = nH*hd;
    device const half* qr = qkv + m*qStride + qh*hd;
    device const half* kb = kc + kvh*hd;
    device const half* vb = vc + kvh*hd;
    threadgroup float sc[4096];
    threadgroup float red[128];
    float mmax=-INFINITY;
    for (uint s=winStart+tid; s<nKeys; s+=tgs) mmax=max(mmax, prefill_score(qr, kb+kv_row(kt,s)*kvDim, hd, scale));
    mmax = simd_max(mmax);
    uint sgidA = tid >> 5u, laneA = tid & 31u;
    if (laneA == 0) red[sgidA] = mmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nsgA = tgs >> 5u;
    if (tid == 0) { float v = red[0]; for (uint k=1; k<nsgA; k++) v = max(v, red[k]); red[0] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    float ls=0;
    float acc[4] = {0, 0, 0, 0};
    for (uint t0=winStart; t0<nKeys; t0+=tile) {
        uint t1 = min(t0+tile, nKeys);
        uint own = t0 + (tid + tgs - (t0-winStart)%tgs) % tgs; // this thread's first key in the tile: s - winStart == tid mod tgs
        for (uint s=own; s<t1; s+=tgs){ float p=exp(prefill_score(qr, kb+kv_row(kt,s)*kvDim, hd, scale)-mx); sc[s-t0]=p; ls+=p; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j=0u, d=tid; d<hd; j++, d+=tgs){ float a=acc[j]; for(uint s=t0;s<t1;s++) a += sc[s-t0]*float(vb[kv_row(kt,s)*kvDim+d]); acc[j]=a; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    red[tid]=ls; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st=tgs/2u; st>0u; st>>=1u){ if(tid<st) red[tid]+=red[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float sum=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j=0u, d=tid; d<hd; j++, d+=tgs) out[m*qDim + qh*hd + d]=half(acc[j]/sum);
}

// attention_prefill_img (S17's Metal image prefill): attention_prefill with
// image blocks [b.x, b.y) in absolute positions that attend BIDIRECTIONALLY, as the CPU's KVCache.attendHi and CUDA's
// PrefillImageLast: a query inside a block sees keys up to that block's b.y - 1 (its own block only; several blocks for
// several images, S11), every other query stays causal, and the sliding window's lower bound stays at the causal
// position's (pos + 1), as the CPU's WindowStart. nBlocks == 0 is no block (then this is attention_prefill exactly). A
// separate kernel so every existing dispatch of attention_prefill stays as it was.
kernel void attention_prefill_img(device const half* qkv[[buffer(0)]], device const half* kc[[buffer(1)]],
    device const half* vc[[buffer(2)]], device half* out[[buffer(3)]], constant uint& nH[[buffer(4)]],
    constant uint& nKV[[buffer(5)]], constant uint& hd[[buffer(6)]], constant uint& startPos[[buffer(7)]],
    constant float& scale[[buffer(8)]], constant uint& qStride[[buffer(9)]], constant uint& window[[buffer(10)]],
    device const uint2* blocks[[buffer(11)]], constant uint& nBlocks[[buffer(12)]], device const uint* kt[[buffer(13)]],
    uint gid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]],
    uint tgs[[threads_per_threadgroup]]) {
    uint m = gid / nH, qh = gid % nH;
    uint kvDim = nKV*hd, kvh = qh/(nH/nKV);
    uint pos = startPos + m;
    uint causal = pos + 1u;
    uint winStart = (window>0u && causal>window) ? causal-window : 0u;
    uint nKeys = causal;
    for (uint b=0u; b<nBlocks; b++) { if (pos >= blocks[b].x && pos < blocks[b].y) { nKeys = blocks[b].y; break; } }
    uint qDim = nH*hd;
    device const half* qr = qkv + m*qStride + qh*hd;
    device const half* kb = kc + kvh*hd;
    device const half* vb = vc + kvh*hd;
    threadgroup float sc[4096];
    threadgroup float red[128];
    for (uint s=winStart+tid; s<nKeys; s+=tgs) {
        float a=0; device const half* k=kb+kv_row(kt,s)*kvDim; uint d=0;
        if ((hd&3u)==0u) for (; d<hd; d+=4u){ half4 k4=*((device const half4*)(k+d)); a+=float(qr[d])*float(k4.x); a+=float(qr[d+1u])*float(k4.y); a+=float(qr[d+2u])*float(k4.z); a+=float(qr[d+3u])*float(k4.w); }
        for (; d<hd; d++) a += float(qr[d])*float(k[d]);
        sc[s]=attn_score(a*scale);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mmax=-INFINITY; for (uint s=winStart+tid;s<nKeys;s+=tgs) mmax=max(mmax,sc[s]);
    mmax = simd_max(mmax);
    uint sgidA = tid >> 5u, laneA = tid & 31u;
    if (laneA == 0) red[sgidA] = mmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nsgA = tgs >> 5u;
    if (tid == 0) { float v = red[0]; for (uint k=1; k<nsgA; k++) v = max(v, red[k]); red[0] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    float ls=0; for (uint s=winStart+tid;s<nKeys;s+=tgs){ float p=exp(sc[s]-mx); sc[s]=p; ls+=p; }
    red[tid]=ls; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st=tgs/2u; st>0u; st>>=1u){ if(tid<st) red[tid]+=red[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float sum=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d=tid; d<hd; d+=tgs){ float a=0; for(uint s=winStart;s<nKeys;s++) a += sc[s]*float(vb[kv_row(kt,s)*kvDim+d]); out[m*qDim + qh*hd + d]=half(a/sum); }
}

// attention_prefill_fused: the simdgroup_matrix (MMA) flash-attention twin of attention_prefill.
// One simdgroup per (query head, 8-row query tile);
// QKᵀ and PV are both 8×8-tiled MMA matmuls (hd/8 tiles each), replacing attention_prefill's
// per-key scalar dot products and its fully-serial PV scan. K is read K^T-transposed straight
// from the row-major cache via simdgroup_load's transpose flag (confirmed against
// aikit/gpu/metal_vit.go:539's gemm_f32_sg_big, the only prior art for that flag in this
// codebase) — no explicit transpose step. Online (flash) softmax: running max/sum live in
// per-row threadgroup scratch (stat[sgid]) so no O(nKeys) score buffer is needed (the exact
// kernel above allocates sc[4096] and would silently overrun a longer context); the O
// accumulator is rescaled by alpha=exp(mOld-mNew) each key-tile in scalar code, because a
// simdgroup_matrix has no addressable per-row view — every rescale/mask/reduce step goes
// through the float scratch (sScr) or half scratch (pScr) via simdgroup_store/_load, exactly
// as gemm_w4f16_store's epilogue already does for its own per-element bias/residual step.
// NOT bit-identical (the online-softmax rescale reorders the sum vs. the exact kernel's single
// final normalize) — same P19 category as the CUDA L2 twin. Requires hd%8==0 && hd<=128
// (ATTN_MAXHD); the Go dispatch falls back to attention_prefill outside that range.
// ATTN_SGPT simdgroups/threadgroup must match attnFusedSGPT (backend.go's Go-side grid helper).
//
// ATTN_KTILE (audit-metal-2026-09-12.md M-04): the QKᵀ score computation runs on ATTN_KTILE=32
// keys (4 sub-tiles of 8) before the scalar softmax/rescale/PV phase — which is the barrier-heavy
// part (2 barriers per hd-tile, hdTiles up to 16) — instead of on 8 keys as before, amortising
// that phase 4x (the audit's count: ~34 barriers per 8 keys against 32 MMAs of useful work). O
// itself stays in threadgroup scratch with the same scalar per-lane rescale (oScr, unchanged) —
// this does NOT attempt the audit's further register-accumulator + diagonal-α-MMA rewrite, which
// needs its own f32->f16 round-trip to feed the diagonal as an MMA operand and was not obviously
// fewer barriers once that round-trip is counted; scoped out as a separate, riskier follow-up.
#define ATTN_SGPT 4
#define ATTN_MAXHD 128
#define ATTN_KTILE 32
kernel void attention_prefill_fused(device const half* qkv[[buffer(0)]], device const half* kc[[buffer(1)]],
    device const half* vc[[buffer(2)]], device half* out[[buffer(3)]], constant uint& nH[[buffer(4)]],
    constant uint& nKV[[buffer(5)]], constant uint& hd[[buffer(6)]], constant uint& startPos[[buffer(7)]],
    constant float& scale[[buffer(8)]], constant uint& qStride[[buffer(9)]], constant uint& window[[buffer(10)]],
    constant uint& M[[buffer(11)]], device const uint* kt[[buffer(12)]],
    uint tgid[[threadgroup_position_in_grid]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint sgpt[[simdgroups_per_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    uint numRowTiles = (M + 7u) / 8u;
    uint sg = tgid*sgpt + sgid;
    uint totalTiles = nH * numRowTiles;
    if (sg >= totalTiles) return;
    uint qh = sg / numRowTiles, rt = sg % numRowTiles;
    uint r0 = rt * 8u;
    uint kvDim = nKV*hd, kvh = qh/(nH/nKV);
    uint qDim = nH*hd;
    uint hdTiles = hd/8u;
    device const half* qBase = qkv + r0*qStride + qh*hd;
    device const half* kBase = kc + kvh*hd;
    device const half* vBase = vc + kvh*hd;

    threadgroup float sScr[ATTN_SGPT][8*ATTN_KTILE];
    threadgroup half  pScr[ATTN_SGPT][8*ATTN_KTILE];
    threadgroup float oScr[ATTN_SGPT][8*ATTN_MAXHD];
    threadgroup float stat[ATTN_SGPT][24]; // [0:8)=m  [8:16)=l  [16:24)=alpha (this block)

    for (uint i=lane; i<8u*hd; i+=32u) oScr[sgid][i] = 0.0f;
    if (lane < 8u) { stat[sgid][lane] = -INFINITY; stat[sgid][8u+lane] = 0.0f; }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_half8x8 qTile[ATTN_MAXHD/8];
    for (uint kk=0; kk<hdTiles; kk++) simdgroup_load(qTile[kk], qBase + kk*8u, qStride);

    uint lastRow = min(r0+7u, M-1u);
    uint nKeysMax = startPos + lastRow + 1u;
    uint firstRowKeys = startPos + r0 + 1u;
    uint winStart0 = (window>0u && firstRowKeys>window) ? firstRowKeys-window : 0u;
    uint j0superStart = (winStart0/ATTN_KTILE)*ATTN_KTILE;

    for (uint j0s=j0superStart; j0s<nKeysMax; j0s+=ATTN_KTILE) {
        // nsub: valid 8-key sub-tiles in this group (the last group may be ragged — nKeysMax
        // need not be a multiple of ATTN_KTILE, or even of 8).
        uint nsub = min((nKeysMax - j0s + 7u)/8u, uint(ATTN_KTILE/8));
        for (uint sub=0; sub<nsub; sub++) {
            uint j0 = j0s + sub*8u;
            simdgroup_float8x8 Sacc = make_filled_simdgroup_matrix<float,8,8>(0.0);
            for (uint kk=0; kk<hdTiles; kk++) {
                simdgroup_half8x8 kT;
                simdgroup_load(kT, kBase + kv_row(kt,j0)*kvDim + kk*8u, kvDim, ulong2(0,0), true);
                simdgroup_multiply_accumulate(Sacc, qTile[kk], kT, Sacc);
            }
            simdgroup_store(Sacc, sScr[sgid] + sub*8u, ATTN_KTILE);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        uint ncols = nsub*8u;
        if (lane < 8u) {
            uint row = lane, qi = r0+row;
            if (qi < M) {
                uint qpos = startPos+qi, rowKeys = qpos+1u;
                uint rowWin = (window>0u && rowKeys>window) ? rowKeys-window : 0u;
                float sraw[ATTN_KTILE]; float rowmax = -INFINITY;
                for (uint c=0;c<ncols;c++) {
                    uint j = j0s+c;
                    float v = attn_score(sScr[sgid][row*ATTN_KTILE+c]*scale);
                    if (j>=rowKeys || j<rowWin) v = -INFINITY;
                    sraw[c]=v; rowmax = max(rowmax, v);
                }
                float mOld = stat[sgid][row];
                float mNew = max(mOld, rowmax);
                float alpha = (mOld <= -INFINITY) ? 0.0f : exp(mOld-mNew);
                float blockSum=0.0f;
                for (uint c=0;c<ncols;c++) {
                    float p = (sraw[c] <= -INFINITY) ? 0.0f : exp(sraw[c]-mNew);
                    pScr[sgid][row*ATTN_KTILE+c] = half(p);
                    blockSum += p;
                }
                stat[sgid][row] = mNew;
                stat[sgid][8u+row] = stat[sgid][8u+row]*alpha + blockSum;
                stat[sgid][16u+row] = alpha;
            } else {
                for (uint c=0;c<ncols;c++) pScr[sgid][row*ATTN_KTILE+c] = half(0.0);
                stat[sgid][16u+row] = 1.0f;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (uint cc=0; cc<hdTiles; cc++) {
            simdgroup_float8x8 pvAcc = make_filled_simdgroup_matrix<float,8,8>(0.0);
            for (uint sub=0; sub<nsub; sub++) {
                simdgroup_half8x8 pTile, vTile;
                simdgroup_load(pTile, pScr[sgid] + sub*8u, ATTN_KTILE);
                simdgroup_load(vTile, vBase + kv_row(kt,j0s+sub*8u)*kvDim + cc*8u, kvDim);
                simdgroup_multiply_accumulate(pvAcc, pTile, vTile, pvAcc);
            }
            simdgroup_store(pvAcc, sScr[sgid], 8); // scratch reuse: this group's scores are consumed
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (lane < 8u) {
                uint row = lane; float a = stat[sgid][16u+row];
                for (uint c=0;c<8u;c++) {
                    uint idx = row*hd + cc*8u+c;
                    oScr[sgid][idx] = oScr[sgid][idx]*a + sScr[sgid][row*8u+c];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint e=lane; e<8u*hd; e+=32u) {
        uint row = e/hd, col = e%hd, qi = r0+row;
        if (qi >= M) continue;
        float l = stat[sgid][8u+row];
        out[qi*qDim + qh*hd + col] = half(l > 0.0f ? oScr[sgid][row*hd+col]/l : 0.0f);
    }
}

