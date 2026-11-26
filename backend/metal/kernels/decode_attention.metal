// One THREADGROUP (128 threads) per query head — vs the old 1-thread-per-head (12 threads
// total = 68% of decode time from underutilization). Scores parallel over keys, softmax via
// threadgroup reduction, output parallel over head dims. nKeys ≤ metalCtxCapMax (32768); past 4096 keys the
// scores are tiled (the deep-context path below).
// window>0 (Mistral) restricts the query to the last window keys: keys[winStart..nKeys),
// winStart = max(0, nKeys-window). window==0 is full causal. Derived from nKeys in-kernel, so
// no per-token uniform. (Mistral is all-local; a hypothetical global layer binds window=0.)
// sinks/hasSink: gpt-oss's per-head learned attention sink — an extra logit with NO key and NO
// value, competing for the softmax MAX and joining the DENOMINATOR only (verified standalone,
// metal/gptoss_kernels_test.go's TestGptOssAttnSink_metal, including the sink-DOMINATES case a
// post-hoc denominator patch would get wrong). hasSink==0 is a true no-op for every other family
// (uniform across the threadgroup, so no divergence cost).
kernel void attention(device const float* q[[buffer(0)]], device const half* kc[[buffer(1)]],
    device const half* vc[[buffer(2)]], device float* out[[buffer(3)]], constant uint& nH[[buffer(4)]],
    constant uint& nKV[[buffer(5)]], constant uint& hd[[buffer(6)]], constant uint& nKeys[[buffer(7)]],
    constant float& scale[[buffer(8)]], constant uint& window[[buffer(9)]],
    device const float* sinks[[buffer(10)]], constant uint& hasSink[[buffer(11)]], device const uint* kt[[buffer(12)]],
    uint qh[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    uint kvDim = nKV*hd; uint kvh = qh/(nH/nKV);
    uint winStart = (window>0u && nKeys>window) ? nKeys-window : 0u;
    uint nWin = nKeys - winStart;
    device const float* qr = q + qh*hd;
    device const half*  kb = kc + kvh*hd;   // f16 KV; dot/accum stay in f32 -> parity-neutral
    device const half*  vb = vc + kvh*hd;
    threadgroup float sc[4096];
    threadgroup float red[128];

    // Single-tile fast path (nWin <= 4096): bit-identical reduction tree & float-add order for all
    // historical contexts. Fixes sliding-window indexing: sc[] is indexed relative to winStart so
    // that windowed attention at pos >= 4096 never overflows sc[4096].
    if (nWin <= 4096u) {
        for (uint s=winStart+tid; s<nKeys; s+=tgs) {
            float a=0; device const half* k=kb+kv_row(kt,s)*kvDim; uint d=0;
            // half4 vectorized K-read (1.79x @2048 ctx): the one-thread-per-key access is uncoalesced
            // (adjacent lanes stride kvDim), so 8-byte loads recover sector utilization. SAME sequential
            // accumulation order ⇒ bit-identical; guarded on hd%4==0 for alignment, scalar tail otherwise.
            if ((hd&3u)==0u) for (; d<hd; d+=4u){ half4 k4=*((device const half4*)(k+d)); a+=qr[d]*float(k4.x); a+=qr[d+1u]*float(k4.y); a+=qr[d+2u]*float(k4.z); a+=qr[d+3u]*float(k4.w); }
            for (; d<hd; d++) a += qr[d]*float(k[d]);
            sc[s - winStart]=attn_score(a*scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float m=-INFINITY; for (uint s=tid;s<nWin;s+=tgs) m=max(m,sc[s]);
        red[tid]=m; threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st=tgs/2; st>0; st>>=1){ if(tid<st) red[tid]=max(red[tid],red[tid+st]); threadgroup_barrier(mem_flags::mem_threadgroup); }
        float mx=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
        float sink=0.0f; bool hasS = hasSink != 0u;
        if (hasS) { sink = sinks[qh]; mx = max(mx, sink); }
        float ls=0; for (uint s=tid;s<nWin;s+=tgs){ float p=exp(sc[s]-mx); sc[s]=p; ls+=p; }
        red[tid]=ls; threadgroup_barrier(mem_flags::mem_threadgroup);
        // DENOMINATOR SUM — float-add is non-associative, so this result is coupled to tgs (the reduction
        // WIDTH). tgs is pinned to tgReduceAttn (128, model.go); do NOT parameterize/sweep it, and any
        // alternate attention kernel MUST reduce at the same width or it diverges byte-exactly (past
        // nKeys>width). No existing gate catches this. See ollama-chase §A2-Metal.
        for (uint st=tgs/2; st>0; st>>=1){ if(tid<st) red[tid]+=red[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }
        float sum=red[0];
        if (hasS) sum += exp(sink-mx); // sink joins the denominator only — no value vector, numerator untouched
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // V-READ: order-preserving 8-wide load-batch unroll. Each thread walks nKeys serially at
        // stride kvDim (the same "distinct per-key, latency-exposed" shape the K-read's half4 fix
        // (1.79x @2048, above) already treats) -- issue 8 independent loads ahead, retire the adds in
        // the SAME sequential order as the plain scalar loop, so this is bit-identical by construction
        // (float accumulation order unchanged, only load scheduling). Scalar tail for nKeys not a
        // multiple of 8. Measured (2026-08-21, git-stash A/B, tight-alternated, 2048 ctx): ~26.6%
        // faster than scalar; width-4 also won (~17-18%) but 8 measured further ahead of it.
        for (uint d=tid; d<hd; d+=tgs){
            float a=0; uint s=winStart; uint nMain = winStart + (nWin & ~7u);
            uint vBlock = 0xffffffffu, vBase = 0u; // the table entry of the block in hand
            for (; s<nMain; s+=8u) {
                uint sc_idx = s - winStart;
                float v0, v1, v2, v3, v4, v5, v6, v7;
                if ((s & 63u) <= 56u) { // the tile is inside one block
                    if ((s >> 6u) != vBlock) { vBlock = s >> 6u; vBase = kt[vBlock]*64u; }
                    uint r = (vBase + (s & 63u))*kvDim + d;
                    v0=float(vb[r]);
                    v1=float(vb[r+1u*kvDim]);
                    v2=float(vb[r+2u*kvDim]);
                    v3=float(vb[r+3u*kvDim]);
                    v4=float(vb[r+4u*kvDim]);
                    v5=float(vb[r+5u*kvDim]);
                    v6=float(vb[r+6u*kvDim]);
                    v7=float(vb[r+7u*kvDim]);
                } else {
                    v0=float(vb[kv_row(kt,s+0u)*kvDim+d]);
                    v1=float(vb[kv_row(kt,s+1u)*kvDim+d]);
                    v2=float(vb[kv_row(kt,s+2u)*kvDim+d]);
                    v3=float(vb[kv_row(kt,s+3u)*kvDim+d]);
                    v4=float(vb[kv_row(kt,s+4u)*kvDim+d]);
                    v5=float(vb[kv_row(kt,s+5u)*kvDim+d]);
                    v6=float(vb[kv_row(kt,s+6u)*kvDim+d]);
                    v7=float(vb[kv_row(kt,s+7u)*kvDim+d]);
                }
                a += sc[sc_idx+0u]*v0; a += sc[sc_idx+1u]*v1; a += sc[sc_idx+2u]*v2; a += sc[sc_idx+3u]*v3;
                a += sc[sc_idx+4u]*v4; a += sc[sc_idx+5u]*v5; a += sc[sc_idx+6u]*v6; a += sc[sc_idx+7u]*v7;
            }
            for (; s<nKeys; s++) a += sc[s - winStart]*float(vb[kv_row(kt,s)*kvDim+d]);
            out[qh*hd+d]=a/sum;
        }
        return;
    }

    // Deep-context tiled online softmax path (nWin > 4096):
    // Accumulate across tiles of 4096 keys using online softmax rescaling:
    // m_new = max(m_prev, m_tile), alpha = exp(m_prev - m_new), l_new = l_prev*alpha + sum_tile,
    // acc[slot] = acc[slot]*alpha + sum(p * v).
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    bool hasS = hasSink != 0u;
    if (hasS) {
        m_prev = sinks[qh];
        l_prev = 1.0f;
    }
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint tileBase = winStart; tileBase < nKeys; tileBase += 4096u) {
        uint tileEnd = min(tileBase + 4096u, nKeys);
        uint tileLen = tileEnd - tileBase;

        for (uint s = tileBase + tid; s < tileEnd; s += tgs) {
            float a = 0; device const half* k = kb + kv_row(kt,s)*kvDim; uint d = 0;
            if ((hd&3u) == 0u) for (; d < hd; d += 4u) {
                half4 k4 = *((device const half4*)(k+d));
                a += qr[d]*float(k4.x);
                a += qr[d+1u]*float(k4.y);
                a += qr[d+2u]*float(k4.z);
                a += qr[d+3u]*float(k4.w);
            }
            for (; d < hd; d++) a += qr[d]*float(k[d]);
            sc[s - tileBase] = attn_score(a * scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float m_t = -INFINITY;
        for (uint s = tid; s < tileLen; s += tgs) m_t = max(m_t, sc[s]);
        red[tid] = m_t;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st = tgs/2; st > 0; st >>= 1) {
            if (tid < st) red[tid] = max(red[tid], red[tid+st]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float m_tile = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float m_new = max(m_prev, m_tile);
        float alpha = (m_prev == -INFINITY) ? 0.0f : exp(m_prev - m_new);

        float ls = 0;
        for (uint s = tid; s < tileLen; s += tgs) {
            float p = exp(sc[s] - m_new);
            sc[s] = p;
            ls += p;
        }
        red[tid] = ls;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st = tgs/2; st > 0; st >>= 1) {
            if (tid < st) red[tid] += red[tid+st];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float sum_tile = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        l_prev = l_prev * alpha + sum_tile;
        m_prev = m_new;

        for (uint slot = 0; slot < 4u; slot++) {
            uint d = tid + slot * tgs;
            if (d >= hd) break;
            float a = acc[slot] * alpha;
            uint s = tileBase;
            uint nMain = tileBase + (tileLen & ~7u);
            uint vBlock = 0xffffffffu, vBase = 0u; // the table entry of the block in hand
            for (; s < nMain; s += 8u) {
                uint sc_idx = s - tileBase;
                float v0, v1, v2, v3, v4, v5, v6, v7;
                if ((s & 63u) <= 56u) { // the tile is inside one block
                    if ((s >> 6u) != vBlock) { vBlock = s >> 6u; vBase = kt[vBlock]*64u; }
                    uint r = (vBase + (s & 63u))*kvDim + d;
                    v0=float(vb[r]);
                    v1=float(vb[r+1u*kvDim]);
                    v2=float(vb[r+2u*kvDim]);
                    v3=float(vb[r+3u*kvDim]);
                    v4=float(vb[r+4u*kvDim]);
                    v5=float(vb[r+5u*kvDim]);
                    v6=float(vb[r+6u*kvDim]);
                    v7=float(vb[r+7u*kvDim]);
                } else {
                    v0=float(vb[kv_row(kt,s+0u)*kvDim+d]);
                    v1=float(vb[kv_row(kt,s+1u)*kvDim+d]);
                    v2=float(vb[kv_row(kt,s+2u)*kvDim+d]);
                    v3=float(vb[kv_row(kt,s+3u)*kvDim+d]);
                    v4=float(vb[kv_row(kt,s+4u)*kvDim+d]);
                    v5=float(vb[kv_row(kt,s+5u)*kvDim+d]);
                    v6=float(vb[kv_row(kt,s+6u)*kvDim+d]);
                    v7=float(vb[kv_row(kt,s+7u)*kvDim+d]);
                }
                a += sc[sc_idx+0u]*v0; a += sc[sc_idx+1u]*v1; a += sc[sc_idx+2u]*v2; a += sc[sc_idx+3u]*v3;
                a += sc[sc_idx+4u]*v4; a += sc[sc_idx+5u]*v5; a += sc[sc_idx+6u]*v6; a += sc[sc_idx+7u]*v7;
            }
            for (; s < tileEnd; s++) {
                a += sc[s - tileBase]*float(vb[kv_row(kt,s)*kvDim+d]);
            }
            acc[slot] = a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint slot = 0; slot < 4u; slot++) {
        uint d = tid + slot * tgs;
        if (d >= hd) break;
        out[qh*hd + d] = acc[slot] / l_prev;
    }
}
// attention_fa / attention_fa_combine — R2: decode attention gridded
// by (kvHead, split) instead of by query head, so the G=nH/nKV heads sharing a KV head compute
// their dots/folds from ONE cooperative, coalesced K/V read per key instead of G separate
// per-head reads. hd MUST be 128 (32 lanes x half4 = one coalesced 256B row read per simdgroup) --
// enforced by the Go dispatch site (canUseAttnFA), NOT in-kernel; every other head width keeps the
// shipped attention kernel. NOT bit-identical to it by design (reduction/combine order differs,
// same status as every other non-exact Metal kernel) -- scored by its own tolerance gate.
//
// Two dispatches, always, even at nSplit==1 (one code path; skipping the combine dispatch at
// nSplit==1 is a later, MEASURED optimization, not assumed here):
//
//  1. attention_fa: 128 threads (4 simdgroups) per (kvHead, split) threadgroup. Each simdgroup
//     walks its own key sub-range within the split (sgid-strided, so all 4 make even progress) and
//     runs online softmax (FlashAttention-style: running max m, running sum l, running numerator
//     acc) independently for all G group heads at once, keyed off ONE shared cooperative K/V read
//     (32 lanes x half4 = hd). m/l are simdgroup-uniform after simd_sum; acc is per-lane (each lane
//     owns a distinct 4-wide dim slice, 32*4=hd). The 4 simdgroups' partials are staged to
//     threadgroup memory and combined by the first 32 threads (fixed sg0->sg1->sg2->sg3 online-
//     softmax merge, one lane per dim-slice) into ONE (m, l, acc[hd]) triple per group head,
//     written to partial[kvHead][split][group head].
//  2. attention_fa_combine: nH threadgroups x hd threads, one thread per (query head, dim) --
//     merges the nSplit partials for that head (split-ascending order) and writes the final
//     out[nH*hd]. The (m,l) combine is redundantly recomputed per dim rather than shared across a
//     head's hd threads: O(nSplit) scalar work, negligible next to attention_fa's O(nKeys) pass it
//     follows. Metal serializes dispatches within one encoder in submission order, so the command-
//     buffer boundary between the two is the only sync needed -- no explicit fence.
//
// Split assignment: this threadgroup's key range is [winStart, nKeys) divided into nSplit
// CONTIGUOUS chunks (chunkLen = ceil(nWin/nSplit)), threadgroup index split owns chunk split --
// deterministic, not work-stealing. An empty chunk (chunkStart >= chunkEnd, or a simdgroup with no
// keys in its stride) leaves that split/simdgroup's (m, l) at the online-softmax identity
// (-INFINITY, 0), which the combine's exp(m2-m_new) guard (m2==-INFINITY -> weight 0) treats as a
// true no-op contribution, not a NaN.
#define ATTN_FA_MAXG 8
kernel void attention_fa(
    device const float* q[[buffer(0)]], device const half* kc[[buffer(1)]],
    device const half* vc[[buffer(2)]], device float* partial[[buffer(3)]],
    constant uint& nKV[[buffer(4)]], constant uint& G[[buffer(5)]],
    constant uint& nKeys[[buffer(6)]], constant float& scale[[buffer(7)]],
    constant uint& window[[buffer(8)]], constant uint& nSplit[[buffer(9)]], device const uint* kt[[buffer(10)]],
    threadgroup float* shm[[threadgroup(0)]],   // 128 * (6*G) floats: per-thread (m[G],l[G],acc[G][4])
    uint tgid[[threadgroup_position_in_grid]], // flat kvHead*nSplit + split (1D dispatch API)
    uint tid[[thread_index_in_threadgroup]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]]) {
    const uint hd = 128u;
    const uint kvDim = nKV * hd;
    uint kvh = tgid / nSplit, split = tgid % nSplit;
    uint winStart = (window > 0u && nKeys > window) ? nKeys - window : 0u;
    uint nWin = nKeys - winStart;
    uint chunkLen = (nWin + nSplit - 1u) / nSplit;
    uint chunkStart = winStart + split * chunkLen;
    uint chunkEnd = min(chunkStart + chunkLen, nKeys);

    device const float* qr = q + kvh * G * hd; // this kvHead's G query heads, contiguous
    device const half*  kb = kc + kvh * hd;
    device const half*  vb = vc + kvh * hd;

    float m[ATTN_FA_MAXG]; float l[ATTN_FA_MAXG]; float acc[ATTN_FA_MAXG][4];
    for (uint g = 0; g < G; g++) { m[g] = -INFINITY; l[g] = 0.0f; acc[g][0]=acc[g][1]=acc[g][2]=acc[g][3]=0.0f; }

    float qslice[ATTN_FA_MAXG][4];
    for (uint g = 0; g < G; g++) {
        device const float* qh = qr + g*hd + lane*4u;
        qslice[g][0]=qh[0]; qslice[g][1]=qh[1]; qslice[g][2]=qh[2]; qslice[g][3]=qh[3];
    }

    uint kvBlock = 0xffffffffu, kvBase = 0u; // the table entry of the block in hand: a lane walks its keys in order
    for (uint s = chunkStart + sgid; s < chunkEnd; s += 4u) {
        if ((s >> 6u) != kvBlock) { kvBlock = s >> 6u; kvBase = kt[kvBlock]*64u; }
        uint rowOff = (kvBase + (s & 63u))*kvDim + lane*4u;
        half4 k4 = *((device const half4*)(kb + rowOff));
        half4 v4 = *((device const half4*)(vb + rowOff));
        for (uint g = 0; g < G; g++) {
            float part = qslice[g][0]*float(k4.x) + qslice[g][1]*float(k4.y)
                       + qslice[g][2]*float(k4.z) + qslice[g][3]*float(k4.w);
            float dot = simd_sum(part);
            float score = attn_score(dot * scale);
            float m_new = max(m[g], score);
            float alpha = (m[g] == -INFINITY) ? 0.0f : exp(m[g] - m_new);
            float p = exp(score - m_new);
            l[g] = l[g]*alpha + p;
            acc[g][0] = acc[g][0]*alpha + p*float(v4.x);
            acc[g][1] = acc[g][1]*alpha + p*float(v4.y);
            acc[g][2] = acc[g][2]*alpha + p*float(v4.z);
            acc[g][3] = acc[g][3]*alpha + p*float(v4.w);
            m[g] = m_new;
        }
    }

    uint stride = 6u*G; // m[G] + l[G] + acc[G][4]
    threadgroup float* row = shm + tid*stride;
    for (uint g = 0; g < G; g++) { row[g] = m[g]; row[G+g] = l[g]; }
    for (uint g = 0; g < G; g++) {
        row[2u*G + g*4u+0u]=acc[g][0]; row[2u*G + g*4u+1u]=acc[g][1];
        row[2u*G + g*4u+2u]=acc[g][2]; row[2u*G + g*4u+3u]=acc[g][3];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float fm[ATTN_FA_MAXG]; float fl[ATTN_FA_MAXG]; float facc[ATTN_FA_MAXG][4];
        for (uint g=0; g<G; g++) { fm[g]=-INFINITY; fl[g]=0.0f; facc[g][0]=facc[g][1]=facc[g][2]=facc[g][3]=0.0f; }
        for (uint sg = 0; sg < 4u; sg++) {
            threadgroup float* r = shm + (sg*32u+lane)*stride;
            for (uint g = 0; g < G; g++) {
                float m2 = r[g], l2 = r[G+g];
                float m_new = max(fm[g], m2);
                float a1 = (fm[g]==-INFINITY) ? 0.0f : exp(fm[g]-m_new);
                float a2 = (m2==-INFINITY) ? 0.0f : exp(m2-m_new);
                fl[g] = fl[g]*a1 + l2*a2;
                facc[g][0] = facc[g][0]*a1 + r[2u*G+g*4u+0u]*a2;
                facc[g][1] = facc[g][1]*a1 + r[2u*G+g*4u+1u]*a2;
                facc[g][2] = facc[g][2]*a1 + r[2u*G+g*4u+2u]*a2;
                facc[g][3] = facc[g][3]*a1 + r[2u*G+g*4u+3u]*a2;
                fm[g] = m_new;
            }
        }
        uint pStride = G * (hd + 2u);
        device float* pout = partial + (kvh*nSplit + split) * pStride;
        for (uint g = 0; g < G; g++) {
            device float* og = pout + g*(hd+2u);
            og[0] = fm[g]; og[1] = fl[g];
            og[2u+lane*4u+0u]=facc[g][0]; og[2u+lane*4u+1u]=facc[g][1];
            og[2u+lane*4u+2u]=facc[g][2]; og[2u+lane*4u+3u]=facc[g][3];
        }
    }
}
kernel void attention_fa_combine(
    device const float* partial[[buffer(0)]], device float* out[[buffer(1)]],
    constant uint& G[[buffer(2)]], constant uint& hd[[buffer(3)]], constant uint& nSplit[[buffer(4)]],
    uint qh[[threadgroup_position_in_grid]], uint d[[thread_position_in_threadgroup]]) {
    uint kvh = qh / G, g = qh % G;
    uint pStride = G * (hd + 2u);
    float m = -INFINITY, l = 0.0f, acc = 0.0f;
    for (uint split = 0; split < nSplit; split++) {
        device const float* pg = partial + (kvh*nSplit + split) * pStride + g*(hd+2u);
        float m2 = pg[0], l2 = pg[1], a2 = pg[2u+d];
        float m_new = max(m, m2);
        float a1w = (m==-INFINITY) ? 0.0f : exp(m-m_new);
        float a2w = (m2==-INFINITY) ? 0.0f : exp(m2-m_new);
        l = l*a1w + l2*a2w;
        acc = acc*a1w + a2*a2w;
        m = m_new;
    }
    out[qh*hd + d] = acc / l;
}

