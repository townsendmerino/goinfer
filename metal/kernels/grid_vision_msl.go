package kernels

const GvMSL = `
// x[i] *= s: attention's 1/sqrt(head_dim), on q after RoPE.
kernel void gv_scale(device float* x [[buffer(0)]], constant uint& n [[buffer(1)]], constant float& s [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) x[gid] = x[gid] * s;
}

// tower_attn_hd{64,72,80}: S17 lever A on Metal (docs/tasks/task-multimodal-support-2026-10.md, "The two Metal levers"):
// a fused, non-causal, online-softmax float32 attention over one segment, replacing eg2Ops.attention's per-head chain
// (head-major copies, a GEMM for the scores, a row softmax, a GEMM against V transposed, a scatter). q (already scaled by
// 1/sqrt(head_dim)), k, v and out are [T, nH*HD] with the heads in place. R19's shape (attention_prefill_steel) in f32:
// one threadgroup per (head, 32 query rows), 4 simdgroups of 8 rows; each 16-key K/V block is staged once in padded
// threadgroup memory and read by all 32 rows; O stays in 8x8 float fragments, rescaled in place; the softmax runs in exp2
// on every lane through the fragment lane map (row fm, columns fn and fn+1). The Q tile is staged with bounds through
// the same threadgroup memory, so a segment's last block never reads past its rows. dbg plants a defect for the tests
// (0 in production): 1 skips the rescale when the max moves, 2 lets the key mask run one past the segment, 3 reads V
// one key late.
#define TA_BQ 32u
#define TA_BK 16u
template <uint HD>
inline void tower_attn_body(device const float* q, device const float* k, device const float* v, device float* out,
    uint T, uint nH, uint dbg, threadgroup float* KV, uint tgid, ushort sgid, ushort lane, ushort tid) {
    constexpr uint LD = HD + 8u;
    threadgroup float* Ks = KV;
    threadgroup float* Vs = KV + TA_BK * LD;
    const uint nRB = (T + TA_BQ - 1u) / TA_BQ;
    const uint h = tgid / nRB, rb = tgid % nRB;
    const uint stride = nH * HD;
    const uint tr0 = rb * TA_BQ, r0 = tr0 + uint(sgid) * 8u;
    const ushort qid = lane >> 2;
    const ushort fm = (qid & 4) + ((lane >> 1) & 3);
    const ushort fn = (qid & 2) * 2 + (lane & 1) * 2;
    const uint myRow = r0 + fm;
    device const float* qh = q + h * HD;
    device const float* kh = k + h * HD;
    device const float* vh = v + h * HD;

    for (uint idx = tid; idx < TA_BQ * HD / 4u; idx += 128u) { // the Q tile, 32 rows (zero past T), into KV
        const uint r = idx / (HD / 4u), d4 = (idx % (HD / 4u)) * 4u;
        const uint row = tr0 + r;
        *(threadgroup float4*)(KV + r * LD + d4) = row < T ? *(device const float4*)(qh + row * stride + d4) : float4(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 qT[HD / 8u];
    for (uint kk = 0; kk < HD / 8u; kk++) simdgroup_load(qT[kk], KV + (uint(sgid) * 8u) * LD + kk * 8u, LD);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 oAcc[HD / 8u];
    for (uint cc = 0; cc < HD / 8u; cc++) oAcc[cc] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float mRow = -INFINITY, lRow = 0.0f;
    const float l2e = 1.4426950408889634f;
    const uint kvalid = dbg == 2u ? T + 1u : T;
    const bool active = r0 < T;
    for (uint j0 = 0; j0 < T; j0 += TA_BK) {
        for (uint idx = tid; idx < TA_BK * HD / 4u; idx += 128u) {
            const uint key = idx / (HD / 4u), d4 = (idx % (HD / 4u)) * 4u;
            const uint j = j0 + key, jv = j + (dbg == 3u ? 1u : 0u);
            *(threadgroup float4*)(Ks + key * LD + d4) = j < T ? *(device const float4*)(kh + j * stride + d4) : float4(0.0f);
            *(threadgroup float4*)(Vs + key * LD + d4) = jv < T ? *(device const float4*)(vh + jv * stride + d4) : float4(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            simdgroup_float8x8 S[2];
            for (ushort sub = 0; sub < 2; sub++) {
                S[sub] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                for (uint kk = 0; kk < HD / 8u; kk++) {
                    simdgroup_float8x8 kTile;
                    simdgroup_load(kTile, Ks + (uint(sub) * 8u) * LD + kk * 8u, LD, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(S[sub], qT[kk], kTile, S[sub]);
                }
            }
            float s[4];
            s[0] = S[0].thread_elements()[0] * l2e; s[1] = S[0].thread_elements()[1] * l2e;
            s[2] = S[1].thread_elements()[0] * l2e; s[3] = S[1].thread_elements()[1] * l2e;
            if (j0 + TA_BK > kvalid) {
                const uint js[4] = {j0 + fn, j0 + fn + 1u, j0 + 8u + fn, j0 + 9u + fn};
                for (ushort e = 0; e < 4; e++) if (js[e] >= kvalid) s[e] = -INFINITY;
            }
            float bmax = max(max(s[0], s[1]), max(s[2], s[3]));
            bmax = max(bmax, simd_shuffle_xor(bmax, 1));
            bmax = max(bmax, simd_shuffle_xor(bmax, 8));
            const float mNew = max(mRow, bmax);
            const float factor = (dbg == 1u) ? 1.0f : ((mRow > -INFINITY) ? exp2(mRow - mNew) : 0.0f);
            float p[4];
            for (ushort e = 0; e < 4; e++) p[e] = (s[e] > -INFINITY) ? exp2(s[e] - mNew) : 0.0f;
            mRow = mNew;
            float bsum = (p[0] + p[1]) + (p[2] + p[3]);
            bsum += simd_shuffle_xor(bsum, 1);
            bsum += simd_shuffle_xor(bsum, 8);
            lRow = lRow * factor + bsum;
            simdgroup_float8x8 P[2];
            P[0].thread_elements()[0] = p[0]; P[0].thread_elements()[1] = p[1];
            P[1].thread_elements()[0] = p[2]; P[1].thread_elements()[1] = p[3];
            for (uint cc = 0; cc < HD / 8u; cc++) {
                oAcc[cc].thread_elements()[0] *= factor;
                oAcc[cc].thread_elements()[1] *= factor;
                for (ushort sub = 0; sub < 2; sub++) {
                    simdgroup_float8x8 vTile;
                    simdgroup_load(vTile, Vs + (uint(sub) * 8u) * LD + cc * 8u, LD);
                    simdgroup_multiply_accumulate(oAcc[cc], P[sub], vTile, oAcc[cc]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (myRow < T) {
        const float inv = lRow > 0.0f ? 1.0f / lRow : 0.0f;
        device float* o = out + myRow * stride + h * HD + fn;
        for (uint cc = 0; cc < HD / 8u; cc++) {
            o[cc * 8u] = oAcc[cc].thread_elements()[0] * inv;
            o[cc * 8u + 1u] = oAcc[cc].thread_elements()[1] * inv;
        }
    }
}
#define TOWER_ATTN(HDV) \
kernel void tower_attn_hd##HDV(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]], \
    device const float* v [[buffer(2)]], device float* out [[buffer(3)]], constant uint& T [[buffer(4)]], \
    constant uint& nH [[buffer(5)]], constant uint& dbg [[buffer(6)]], uint tgid [[threadgroup_position_in_grid]], \
    ushort sgid [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]], \
    ushort tid [[thread_index_in_threadgroup]]) { \
    threadgroup float KV[2u * TA_BK * (HDV + 8u)]; \
    tower_attn_body<HDV>(q, k, v, out, T, nH, dbg, KV, tgid, sgid, lane, tid); \
}
TOWER_ATTN(64)
TOWER_ATTN(72)
TOWER_ATTN(80)

// tower_gemm_{w32,w16}: S17 lever B on Metal (docs/tasks/task-multimodal-support-2026-10.md, "S17's Metal lever B"):
// C[M,N] = A[M,K]·Wᵀ (+ bias[N]), A in f32, W [N,K] in f32 (w32) or f16 widened to f32 as it is staged (w16), f32
// accumulation on simdgroup 8x8 tiles. One threadgroup of 16 simdgroups (512 threads) per 64x64 output tile, each
// simdgroup 16x16 (2x2 fragments: 4x4 per simdgroup spilled, 0.08 TFLOPS against aikit's 1.19, 2026-10-08); K steps of 32
// staged as float4 through padded threadgroup memory (one vector of A and one of W per thread per step), bounds-guarded
// element by element only at an edge, so any M, N and K. The epilogue reuses the staging memory to add the bias and guard
// the edges. dbg plants a defect for the tests (0 in production): 1 drops the bias, 2 stages A one K column late, 3 skips
// the last 8-wide quarter of every K step. (A first version that rounded A to f16, the registered lever, failed the
// tower bars at real size; see the doc.)
// tower_w8_scale is tower_gemm_w8's scale for weight row n at column k: one f32 per group of 32 along K
// (ceil(K/32) a row). dbg 4 plants G-S18e's defect: the next group's scale (the row's last group keeps its own).
inline float tower_w8_scale(device const float* WS, uint n, uint k, uint K, uint dbg) {
    const uint nG = (K + 31u) / 32u;
    uint g = k >> 5;
    if (dbg == 4u) g = min(g + 1u, nG - 1u);
    return WS[n * nG + g];
}
#define TG_T 64u
#define TG_BK 32u
#define TG_LD (TG_BK + 4u)
template <typename TW, bool SC>
inline void tower_gemm_body(device const float* A, device const TW* W, device float* C, device const float* bias,
    device const float* WS, uint M, uint N, uint K, uint hasBias, uint dbg, threadgroup float* S, uint2 tg, ushort sgid,
    ushort tid) {
    threadgroup float* As = S;
    threadgroup float* Bs = S + TG_T * TG_LD;
    const uint m0 = tg.y * TG_T, n0 = tg.x * TG_T;
    const uint sr = uint(sgid >> 2) * 16u, sc = uint(sgid & 3) * 16u;
    const uint r = uint(tid) >> 3, c4 = (uint(tid) & 7u) * 4u;
    const uint ga = m0 + r, gw = n0 + r, aShift = dbg == 2u ? 1u : 0u;
    simdgroup_float8x8 acc[2][2];
    for (uint i = 0; i < 2; i++) for (uint j = 0; j < 2; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < K; k0 += TG_BK) {
        const uint gk = k0 + c4;
        float4 av = float4(0.0f), wv = float4(0.0f);
        if (aShift == 0u && ga < M && gk + 4u <= K) {
            av = *(device const float4*)(A + ga * K + gk);
        } else if (ga < M) {
            for (uint e = 0; e < 4u; e++) if (gk + e + aShift < K) av[e] = A[ga * K + gk + e + aShift];
        }
        // An int8 row starts 4-byte aligned only when K is a multiple of 4; a char4 load from an unaligned row reads
        // the wrong bytes (K = 9 and 17 in the kernel gate), so w8 takes the element path there.
        if (gw < N && gk + 4u <= K && (!SC || (K & 3u) == 0u)) {
            wv = float4(*(device const vec<TW, 4>*)(W + gw * K + gk));
            if (SC) wv *= tower_w8_scale(WS, gw, gk, K, dbg); // the 4 share one group: gk is a multiple of 4, a group 32
        } else if (gw < N) {
            for (uint e = 0; e < 4u; e++) if (gk + e < K) wv[e] = float(W[gw * K + gk + e]) * (SC ? tower_w8_scale(WS, gw, gk + e, K, dbg) : 1.0f);
        }
        *(threadgroup float4*)(As + r * TG_LD + c4) = av;
        *(threadgroup float4*)(Bs + r * TG_LD + c4) = wv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < (dbg == 3u ? TG_BK - 8u : TG_BK); kk += 8u) {
            simdgroup_float8x8 a[2], b[2];
            for (uint i = 0; i < 2; i++) simdgroup_load(a[i], As + (sr + i * 8u) * TG_LD + kk, TG_LD);
            for (uint j = 0; j < 2; j++) simdgroup_load(b[j], Bs + (sc + j * 8u) * TG_LD + kk, TG_LD, ulong2(0, 0), true);
            for (uint i = 0; i < 2; i++) for (uint j = 0; j < 2; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 2; i++) for (uint j = 0; j < 2; j++) simdgroup_store(acc[i][j], S + (sr + i * 8u) * TG_T + sc + j * 8u, TG_T);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool addBias = hasBias != 0u && dbg != 1u;
    for (uint idx = tid; idx < TG_T * TG_T; idx += 512u) {
        const uint rr = idx / TG_T, cc = idx % TG_T, gr = m0 + rr, gc = n0 + cc;
        if (gr < M && gc < N) C[gr * N + gc] = S[idx] + (addBias ? bias[gc] : 0.0f);
    }
}
#define TOWER_GEMM(NAME, TW) \
kernel void NAME(device const float* A [[buffer(0)]], device const TW* W [[buffer(1)]], device float* C [[buffer(2)]], \
    device const float* bias [[buffer(3)]], constant uint& M [[buffer(4)]], constant uint& N [[buffer(5)]], \
    constant uint& K [[buffer(6)]], constant uint& hasBias [[buffer(7)]], constant uint& dbg [[buffer(8)]], \
    uint2 tg [[threadgroup_position_in_grid]], ushort sgid [[simdgroup_index_in_threadgroup]], \
    ushort tid [[thread_index_in_threadgroup]]) { \
    threadgroup float S[2u * TG_T * TG_LD]; \
    tower_gemm_body<TW, false>(A, W, C, bias, bias, M, N, K, hasBias, dbg, S, tg, sgid, tid); \
}
TOWER_GEMM(tower_gemm_w32, float)
TOWER_GEMM(tower_gemm_w16, half)
// tower_gemm_w8 (S18): int8 weights in groups of 32 along K, each group's f32 scale applied as the weights widen to f32
// at staging; A and the accumulation f32, as w16. WS is [N, ceil(K/32)].
kernel void tower_gemm_w8(device const float* A [[buffer(0)]], device const char* W [[buffer(1)]], device float* C [[buffer(2)]],
    device const float* bias [[buffer(3)]], constant uint& M [[buffer(4)]], constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]], constant uint& hasBias [[buffer(7)]], constant uint& dbg [[buffer(8)]],
    device const float* WS [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]],
    ushort sgid [[simdgroup_index_in_threadgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float S[2u * TG_T * TG_LD];
    tower_gemm_body<char, true>(A, W, C, bias, WS, M, N, K, hasBias, dbg, S, tg, sgid, tid);
}
`
