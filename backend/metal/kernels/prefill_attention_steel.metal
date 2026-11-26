// attention_prefill_steel: prefill attention in the shape of MLX 0.32.0's
// steel_attention (bq32 bk16 bd128), for head dim 128. One threadgroup per (query head, 32 query rows); its 4 simdgroups
// own 8 rows each, and each 16-key K/V block is staged ONCE in padded threadgroup memory and read by all 32 rows. O stays
// in registers (16 8x8 fragments per simdgroup, rescaled in place); the online softmax runs on every lane through the
// fragment lane map (mc3_frag, batch.go: this lane holds row fm, columns fn and fn+1; the 4 lanes of a row differ in
// lane bits 0 and 3), in exp2 with log2e folded into the score scale; P is written straight into PV's left-operand
// fragments; blocks past a row's causal or window limit are skipped. Measured 7.20x attention_prefill_fused in sequence
// at K=3900 on the 1.5B (docs/measurements/metal-prefill-attn-2026-09-27.md). NOT bit-identical to the fused kernel
// (accumulation order, exp2): graded by the §3.2 pooled gate on set A. The fused kernel keeps every other head dim.
#define R19_BQ 32
#define R19_BK 16

// KS-D2: the head dim is a template parameter (HD, with LD = HD + 8 the padded row); every other line is R19's. The K/V
// staging walks the block's 16 x HD/4 half4 words, 128 a pass, so at HD = 128 each thread stages the same four words.
template <ushort HD>
kernel void attention_prefill_steel(device const half* qkv[[buffer(0)]], device const half* kc[[buffer(1)]],
    device const half* vc[[buffer(2)]], device half* out[[buffer(3)]], constant uint& nH[[buffer(4)]],
    constant uint& nKV[[buffer(5)]], constant uint& hd[[buffer(6)]], constant uint& startPos[[buffer(7)]],
    constant float& scale[[buffer(8)]], constant uint& qStride[[buffer(9)]], constant uint& window[[buffer(10)]],
    constant uint& M[[buffer(11)]], device const uint* kt[[buffer(12)]],
    uint tgid[[threadgroup_position_in_grid]], ushort sgid[[simdgroup_index_in_threadgroup]],
    ushort lane[[thread_index_in_simdgroup]], ushort tid[[thread_index_in_threadgroup]]) {
    constexpr ushort LD = HD + 8;
    threadgroup half Ks[R19_BK * LD];
    threadgroup half Vs[R19_BK * LD];

    const uint nRB = (M + R19_BQ - 1u) / R19_BQ;
    const uint qh = tgid / nRB, rb = tgid % nRB;
    if (qh >= nH) return;
    const uint kvDim = nKV * HD, kvh = qh / (nH / nKV), qDim = nH * HD;
    const uint tr0 = rb * R19_BQ;                 // the threadgroup's first query row
    const uint r0 = tr0 + uint(sgid) * 8u;        // this simdgroup's first query row
    const ushort qid = lane >> 2;
    const ushort fm = (qid & 4) + ((lane >> 1) & 3);
    const ushort fn = (qid & 2) * 2 + (lane & 1) * 2;
    const uint myRow = r0 + fm;                   // the query row this lane's fragment elements belong to
    const uint myKeys = startPos + min(myRow, M - 1u) + 1u;               // causal: keys [0, myKeys)
    const uint myWin = (window > 0u && myKeys > window) ? myKeys - window : 0u;

    // The threadgroup's key range: its first row's window start to its last row's causal end.
    const uint tLast = min(tr0 + R19_BQ - 1u, M - 1u);
    const uint jEnd = startPos + tLast + 1u;
    const uint firstKeys = startPos + tr0 + 1u;
    const uint jStart = ((window > 0u && firstKeys > window) ? firstKeys - window : 0u) / R19_BK * R19_BK;
    // This simdgroup's own range, to skip blocks it has nothing in.
    const uint sLast = min(r0 + 7u, M - 1u);
    const uint sEnd = startPos + sLast + 1u;
    const uint sFirstKeys = startPos + min(r0, M - 1u) + 1u;
    const uint sStart = (window > 0u && sFirstKeys > window) ? sFirstKeys - window : 0u;

    const float sl2 = scale * 1.4426950408889634f;
    // A simdgroup whose 8 rows are all past M (the last block's tail) loads no Q (the qkv buffer is padded to 8 rows,
    // not 32) and computes nothing; it still helps stage K/V and meets every barrier.
    const bool active = r0 < M;
    simdgroup_half8x8 qT[HD / 8];
    if (active) {
        for (ushort kk = 0; kk < HD / 8; kk++) {
            simdgroup_load(qT[kk], qkv + r0 * qStride + qh * HD + kk * 8u, qStride);
        }
    }
    simdgroup_float8x8 oAcc[HD / 8];
    for (ushort cc = 0; cc < HD / 8; cc++) oAcc[cc] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float mRow = -INFINITY, lRow = 0.0f;

    for (uint j0 = jStart; j0 < jEnd; j0 += R19_BK) {
        // Stage K and V for keys j0 .. j0+15: 16 x HD halves each, in half4 words; keys past jEnd read as zero.
        for (uint idx = uint(tid); idx < R19_BK * HD / 4u; idx += 128u) {
            uint key = idx / (HD / 4u), d4 = (idx % (HD / 4u)) * 4u;
            uint j = j0 + key;
            half4 kv4 = half4(0.0h), vv4 = half4(0.0h);
            if (j < jEnd) {
                uint jr = kv_row(kt, j);
                kv4 = *(device const half4*)(kc + jr * kvDim + kvh * HD + d4);
                vv4 = *(device const half4*)(vc + jr * kvDim + kvh * HD + d4);
            }
            *(threadgroup half4*)(Ks + key * LD + d4) = kv4;
            *(threadgroup half4*)(Vs + key * LD + d4) = vv4;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active && j0 < sEnd && j0 + R19_BK > sStart) {
            // S = Q K^T for this simdgroup's 8 rows and the block's 16 keys (two 8x8 fragments).
            simdgroup_float8x8 S[2];
            for (ushort sub = 0; sub < 2; sub++) {
                S[sub] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                for (ushort kk = 0; kk < HD / 8; kk++) {
                    simdgroup_half8x8 kTile;
                    simdgroup_load(kTile, Ks + (sub * 8u) * LD + kk * 8u, LD, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(S[sub], qT[kk], kTile, S[sub]);
                }
            }
            // This lane's four scores: row myRow, keys j0 + {fn, fn+1, 8+fn, 9+fn}.
            float s[4];
            s[0] = attn_score_log2(S[0].thread_elements()[0] * sl2); s[1] = attn_score_log2(S[0].thread_elements()[1] * sl2);
            s[2] = attn_score_log2(S[1].thread_elements()[0] * sl2); s[3] = attn_score_log2(S[1].thread_elements()[1] * sl2);
            const bool edge = (j0 + R19_BK > startPos + r0 + 1u) || (j0 < myWin + R19_BK) || (myRow >= M);
            if (edge) {
                uint js[4] = {j0 + fn, j0 + fn + 1u, j0 + 8u + fn, j0 + 9u + fn};
                for (ushort e = 0; e < 4; e++) {
                    if (js[e] >= myKeys || js[e] < myWin || myRow >= M) s[e] = -INFINITY;
                }
            }
            float bmax = max(max(s[0], s[1]), max(s[2], s[3]));
            bmax = max(bmax, simd_shuffle_xor(bmax, 1));
            bmax = max(bmax, simd_shuffle_xor(bmax, 8));
            float mNew = max(mRow, bmax);
            float factor = 1.0f;
            float p[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (mNew > -INFINITY) {
                factor = (mRow > -INFINITY) ? exp2(mRow - mNew) : 0.0f;
                for (ushort e = 0; e < 4; e++) p[e] = (s[e] > -INFINITY) ? exp2(s[e] - mNew) : 0.0f;
                mRow = mNew;
            }
            float bsum = (p[0] + p[1]) + (p[2] + p[3]);
            bsum += simd_shuffle_xor(bsum, 1);
            bsum += simd_shuffle_xor(bsum, 8);
            lRow = lRow * factor + bsum;
            simdgroup_half8x8 P[2];
            P[0].thread_elements()[0] = half(p[0]); P[0].thread_elements()[1] = half(p[1]);
            P[1].thread_elements()[0] = half(p[2]); P[1].thread_elements()[1] = half(p[3]);
            for (ushort cc = 0; cc < HD / 8; cc++) {
                oAcc[cc].thread_elements()[0] *= factor;
                oAcc[cc].thread_elements()[1] *= factor;
                for (ushort sub = 0; sub < 2; sub++) {
                    simdgroup_half8x8 vTile;
                    simdgroup_load(vTile, Vs + (sub * 8u) * LD + cc * 8u, LD);
                    simdgroup_multiply_accumulate(oAcc[cc], P[sub], vTile, oAcc[cc]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (myRow < M) {
        const float inv = lRow > 0.0f ? 1.0f / lRow : 0.0f;
        device half* o = out + myRow * qDim + qh * HD + fn;
        for (ushort cc = 0; cc < HD / 8; cc++) {
            o[cc * 8u] = half(oAcc[cc].thread_elements()[0] * inv);
            o[cc * 8u + 1u] = half(oAcc[cc].thread_elements()[1] * inv);
        }
    }
}
template [[host_name("attention_prefill_steel")]] kernel decltype(attention_prefill_steel<128>) attention_prefill_steel<128>;
template [[host_name("attention_prefill_steel_hd64")]] kernel decltype(attention_prefill_steel<64>) attention_prefill_steel<64>;
template [[host_name("attention_prefill_steel_hd80")]] kernel decltype(attention_prefill_steel<80>) attention_prefill_steel<80>;
template [[host_name("attention_prefill_steel_hd96")]] kernel decltype(attention_prefill_steel<96>) attention_prefill_steel<96>;

// attention_prefill_steel_split (KS-D1): attention_prefill_steel for head dims 256 and 512, the head dim split over the
// threadgroup's 8 simdgroups: NS = HD/128 slices of 128 columns by 8/NS row groups of 8 rows (32 query rows a threadgroup at
// 256, 16 at 512), K/V blocks of BK keys (16 at 256, 8 at 512) staged once in padded threadgroup memory. Each simdgroup forms
// its slice's partial Q·Kᵀ, stores it, and sums its row group's NS partials in slice order, so every slice of a row group holds
// the same scores and online-softmax state; it accumulates its own 128 output columns. Key blocks start at multiples of BK and a
// fully masked block leaves the state unchanged, so a row's output does not depend on how the prompt is split into passes.
// Not bit-identical to attention_prefill (half P, exp2, the order): fidelity-gated (KS-D1's gates).
template <ushort HD>
kernel void attention_prefill_steel_split(device const half* qkv[[buffer(0)]], device const half* kc[[buffer(1)]],
    device const half* vc[[buffer(2)]], device half* out[[buffer(3)]], constant uint& nH[[buffer(4)]],
    constant uint& nKV[[buffer(5)]], constant uint& hd[[buffer(6)]], constant uint& startPos[[buffer(7)]],
    constant float& scale[[buffer(8)]], constant uint& qStride[[buffer(9)]], constant uint& window[[buffer(10)]],
    constant uint& M[[buffer(11)]], device const uint* kt[[buffer(12)]],
    uint tgid[[threadgroup_position_in_grid]], ushort sgid[[simdgroup_index_in_threadgroup]],
    ushort lane[[thread_index_in_simdgroup]], ushort tid[[thread_index_in_threadgroup]]) {
    constexpr ushort NS = HD / 128, NR = 8 / NS, BQ = NR * 8;
    constexpr ushort BK = HD == 512 ? 8 : 16, NSUB = BK / 8, LD = HD + 8;
    threadgroup half Ks[BK * LD];
    threadgroup half Vs[BK * LD];
    threadgroup float Px[8 * NSUB * 64];          // each simdgroup's partial scores, [sg][sub][8x8 row-major]

    const uint nRB = (M + BQ - 1u) / BQ;
    const uint qh = tgid / nRB, rb = tgid % nRB;
    if (qh >= nH) return;
    const ushort rg = sgid / NS, sl = sgid % NS;
    const uint kvDim = nKV * HD, kvh = qh / (nH / nKV), qDim = nH * HD;
    const uint c0 = uint(sl) * 128u;              // this simdgroup's 128 columns
    const uint tr0 = rb * BQ;                     // the threadgroup's first query row
    const uint r0 = tr0 + uint(rg) * 8u;          // this row group's first query row
    const ushort qid = lane >> 2;
    const ushort fm = (qid & 4) + ((lane >> 1) & 3);
    const ushort fn = (qid & 2) * 2 + (lane & 1) * 2;
    const uint myRow = r0 + fm;
    const uint myKeys = startPos + min(myRow, M - 1u) + 1u;
    const uint myWin = (window > 0u && myKeys > window) ? myKeys - window : 0u;

    const uint tLast = min(tr0 + BQ - 1u, M - 1u);
    const uint jEnd = startPos + tLast + 1u;
    const uint firstKeys = startPos + tr0 + 1u;
    const uint jStart = ((window > 0u && firstKeys > window) ? firstKeys - window : 0u) / BK * BK;
    const uint sLast = min(r0 + 7u, M - 1u);
    const uint sEnd = startPos + sLast + 1u;
    const uint sFirstKeys = startPos + min(r0, M - 1u) + 1u;
    const uint sStart = (window > 0u && sFirstKeys > window) ? sFirstKeys - window : 0u;

    const float sl2 = scale * 1.4426950408889634f;
    const bool active = r0 < M;
    simdgroup_half8x8 qT[16];
    if (active) {
        for (ushort kk = 0; kk < 16; kk++) {
            simdgroup_load(qT[kk], qkv + r0 * qStride + qh * HD + c0 + kk * 8u, qStride);
        }
    }
    simdgroup_float8x8 oAcc[16];
    for (ushort cc = 0; cc < 16; cc++) oAcc[cc] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float mRow = -INFINITY, lRow = 0.0f;

    for (uint j0 = jStart; j0 < jEnd; j0 += BK) {
        for (uint idx = uint(tid); idx < uint(BK) * HD / 4u; idx += 256u) {
            uint key = idx / (HD / 4u), d4 = (idx % (HD / 4u)) * 4u;
            uint j = j0 + key;
            half4 kv4 = half4(0.0h), vv4 = half4(0.0h);
            if (j < jEnd) {
                uint jr = kv_row(kt, j);
                kv4 = *(device const half4*)(kc + jr * kvDim + kvh * HD + d4);
                vv4 = *(device const half4*)(vc + jr * kvDim + kvh * HD + d4);
            }
            *(threadgroup half4*)(Ks + key * LD + d4) = kv4;
            *(threadgroup half4*)(Vs + key * LD + d4) = vv4;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const bool live = active && j0 < sEnd && j0 + BK > sStart; // the same for every slice of a row group
        if (live) {
            for (ushort sub = 0; sub < NSUB; sub++) {
                simdgroup_float8x8 S = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                for (ushort kk = 0; kk < 16; kk++) {
                    simdgroup_half8x8 kTile;
                    simdgroup_load(kTile, Ks + (sub * 8u) * LD + c0 + kk * 8u, LD, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(S, qT[kk], kTile, S);
                }
                simdgroup_store(S, Px + (uint(sgid) * NSUB + sub) * 64u, 8);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (live) {
            float s[2 * NSUB];
            for (ushort sub = 0; sub < NSUB; sub++) {
                threadgroup const float* p0 = Px + (uint(rg) * NS * NSUB + sub) * 64u + fm * 8u + fn;
                float e0 = p0[0], e1 = p0[1];
                for (ushort t = 1; t < NS; t++) { e0 += p0[t * NSUB * 64u]; e1 += p0[t * NSUB * 64u + 1u]; }
                s[2 * sub] = attn_score_log2(e0 * sl2);
                s[2 * sub + 1] = attn_score_log2(e1 * sl2);
            }
            const bool edge = (j0 + BK > startPos + r0 + 1u) || (j0 < myWin + BK) || (myRow >= M);
            if (edge) {
                for (ushort e = 0; e < 2 * NSUB; e++) {
                    uint js = j0 + (e >> 1) * 8u + fn + (e & 1u);
                    if (js >= myKeys || js < myWin || myRow >= M) s[e] = -INFINITY;
                }
            }
            float bmax = s[0];
            for (ushort e = 1; e < 2 * NSUB; e++) bmax = max(bmax, s[e]);
            bmax = max(bmax, simd_shuffle_xor(bmax, 1));
            bmax = max(bmax, simd_shuffle_xor(bmax, 8));
            float mNew = max(mRow, bmax);
            float factor = 1.0f;
            float p[2 * NSUB];
            for (ushort e = 0; e < 2 * NSUB; e++) p[e] = 0.0f;
            if (mNew > -INFINITY) {
                factor = (mRow > -INFINITY) ? exp2(mRow - mNew) : 0.0f;
                for (ushort e = 0; e < 2 * NSUB; e++) p[e] = (s[e] > -INFINITY) ? exp2(s[e] - mNew) : 0.0f;
                mRow = mNew;
            }
            float bsum = 0.0f;
            for (ushort sub = 0; sub < NSUB; sub++) bsum += p[2 * sub] + p[2 * sub + 1];
            bsum += simd_shuffle_xor(bsum, 1);
            bsum += simd_shuffle_xor(bsum, 8);
            lRow = lRow * factor + bsum;
            simdgroup_half8x8 P[NSUB];
            for (ushort sub = 0; sub < NSUB; sub++) {
                P[sub].thread_elements()[0] = half(p[2 * sub]);
                P[sub].thread_elements()[1] = half(p[2 * sub + 1]);
            }
            for (ushort cc = 0; cc < 16; cc++) {
                oAcc[cc].thread_elements()[0] *= factor;
                oAcc[cc].thread_elements()[1] *= factor;
                for (ushort sub = 0; sub < NSUB; sub++) {
                    simdgroup_half8x8 vTile;
                    simdgroup_load(vTile, Vs + (sub * 8u) * LD + c0 + cc * 8u, LD);
                    simdgroup_multiply_accumulate(oAcc[cc], P[sub], vTile, oAcc[cc]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (myRow < M) {
        const float inv = lRow > 0.0f ? 1.0f / lRow : 0.0f;
        device half* o = out + myRow * qDim + qh * HD + c0 + fn;
        for (ushort cc = 0; cc < 16; cc++) {
            o[cc * 8u] = half(oAcc[cc].thread_elements()[0] * inv);
            o[cc * 8u + 1u] = half(oAcc[cc].thread_elements()[1] * inv);
        }
    }
}
template [[host_name("attention_prefill_steel_hd256")]] kernel decltype(attention_prefill_steel_split<256>) attention_prefill_steel_split<256>;
template [[host_name("attention_prefill_steel_hd512")]] kernel decltype(attention_prefill_steel_split<512>) attention_prefill_steel_split<512>;

