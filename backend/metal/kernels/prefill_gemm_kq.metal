// --- the q4k lane's MMA GEMM ---
// gemm_kq_tile is gemm_w4f16_tile over one segment of a q4k-lane projection (q4kSeg: a run of rows of one kind): the same
// tile, the same 32-k slabs staged once into threadgroup memory, the same MMA chain and epilogue; only the weight staging
// differs. Each staged weight is the value the lane's decode kernels compute with, rounded once to half:
//   KIND 0, Q4_K super-blocks (144 bytes): d*sc*q - dmin*m;  KIND 1, Q5_K (176 bytes): the same with q's fifth bit from qh;
//   KIND 2, Q6_K in the lane's planes over the segment (q6kPlanarRow): d*sc*(q - 32);  KIND 3, int8 rows: q*scale (WS, f32 a row).
// d*sc*q and dmin*m are exact in f32 (at most 22 significant bits), so the difference rounds once to f32, then to half:
// TestGemmKQ_stagesDecodeWeights reads every staged weight back through the kernel. The output goes to columns
// [coff, coff+N) of C, rows ldc halves apart, so a fused group's segments write one buffer. K % 256 == 0 for the block kinds.
inline void kq_scale_min(uint j, device const uchar* q, thread float& sc, thread float& m) {
    if (j < 4u) { sc = float(q[j] & 63u); m = float(q[j+4u] & 63u); }
    else { sc = float((q[j+4u] & 15u) | ((q[j-4u] >> 6u) << 4u)); m = float((q[j+4u] >> 4u) | ((q[j] >> 6u) << 4u)); }
}

// kq_stage8 writes the 8 weights k0+8kb .. k0+8kb+7 of row ncol to p[0], p[8], ..., p[56].
template <uint KIND>
inline void kq_stage8(device const uchar* W, device const float* WS, uint ncol, uint N, uint K, uint k0, ushort kb,
    threadgroup half* p) {
#pragma clang fp contract(off)
    const uint NB = K >> 8u, sb = k0 >> 8u, j = (k0 & 255u) >> 5u;
    if constexpr (KIND == 0u || KIND == 1u) {
        const ulong bb = KIND == 0u ? 144ul : 176ul;
        device const uchar* blk = W + ((ulong)ncol*NB + sb)*bb;
        const half2 dd = as_type<half2>(*(device const uint*)blk);
        float sc, m; kq_scale_min(j, blk + 4, sc, m);
        const float dsc = float(dd.x)*sc, dmm = float(dd.y)*m;
        const uint qoff = (KIND == 0u ? 16u : 48u) + (j >> 1u)*32u + uint(kb)*8u;
        const uint2 q = *(device const uint2*)(blk + qoff);
        uint2 h = uint2(0u);
        if constexpr (KIND == 1u) h = *(device const uint2*)(blk + 16u + uint(kb)*8u);
        const uint sh = (j & 1u)*4u;
        for (ushort kl = 0; kl < 8; kl++) {
            const uint by = (kl < 4 ? q.x : q.y) >> (8u*(kl & 3u));
            uint v = (by >> sh) & 15u;
            if constexpr (KIND == 1u) v |= ((((kl < 4 ? h.x : h.y) >> (8u*(kl & 3u))) >> j) & 1u) << 4u;
            p[kl*8] = half(dsc*float(v) - dmm);
        }
    } else if constexpr (KIND == 2u) {
        const uint T = N*NB, g = ncol*NB + sb;
        const uint e = (k0 & 255u) + uint(kb)*8u, n = e >> 7u, qq = (e & 127u) >> 5u, l = e & 31u;
        const uint2 ql = *(device const uint2*)(W + (ulong)g*128ul + n*64u + (qq & 1u)*32u + l);
        const uint2 qh = *(device const uint2*)(W + (ulong)T*128ul + (ulong)g*64ul + n*32u + l);
        const float sc = float(as_type<char>(W[(ulong)T*192ul + (ulong)g*16ul + n*8u + l/16u + 2u*qq]));
        const float d = float(*(device const half*)(W + (ulong)T*208ul + (ulong)g*2ul));
        const float dsc = d*sc;
        const uint shl = (qq >> 1u)*4u, shh = 2u*qq;
        for (ushort kl = 0; kl < 8; kl++) {
            const uint s8 = 8u*(kl & 3u);
            const uint v = (((kl < 4 ? ql.x : ql.y) >> s8 >> shl) & 15u) | ((((kl < 4 ? qh.x : qh.y) >> s8 >> shh) & 3u) << 4u);
            p[kl*8] = half(dsc*float(int(v) - 32));
        }
    } else {
        const float sc = WS[ncol];
        const uint2 wd = *(device const uint2*)(W + (ulong)ncol*K + k0 + uint(kb)*8u);
        const char4 c0 = as_type<char4>(wd.x), c1 = as_type<char4>(wd.y);
        for (ushort kl = 0; kl < 8; kl++) p[kl*8] = half(float(kl < 4 ? c0[kl] : c1[kl - 4]) * sc);
    }
}

template <ushort TM, ushort TN, uint KIND>
kernel void gemm_kq_tile(device const half* A[[buffer(0)]], device const uchar* W[[buffer(1)]],
    device const float* WS[[buffer(2)]], device half* C[[buffer(3)]],
    constant uint& M[[buffer(4)]], constant uint& N[[buffer(5)]], constant uint& K[[buffer(6)]],
    device const float* bias[[buffer(7)]], constant uint& mode[[buffer(8)]],
    constant uint& ldc[[buffer(9)]], constant uint& coff[[buffer(10)]],
    uint2 tgp[[threadgroup_position_in_grid]], ushort tid[[thread_index_in_threadgroup]],
    ushort sg[[simdgroup_index_in_threadgroup]], ushort lane[[thread_index_in_simdgroup]]) {
    constexpr ushort TT = TM/16, TF = TN/16;
    constexpr ushort WT = 128/TN, WPT = 4/WT;
    constexpr ushort AT = TM >= 32 ? 128/TM : 4, BPT = 4/AT;
    threadgroup half sa[(TN/8)*4*72];
    threadgroup half sb[(TM/8)*4*64];
    threadgroup float cs[4*64];

    const uint n0 = tgp.x*uint(TN), m0 = tgp.y*uint(TM);
    const ushort fh = sg & 1, th = sg >> 1;
    simdgroup_float8x8 acc[TT*TF];
    for (ushort i = 0; i < TT*TF; i++) acc[i] = make_filled_simdgroup_matrix<float,8,8>(0.0f);

    const ushort fr = tid / WT, kq = tid % WT;
    const ushort fb = fr >> 3, nl = fr & 7;
    const uint ncol = n0 + fr;
    const bool wok = ncol < N;
    const ushort tr = tid / AT, kq2 = tid % AT;
    const ushort tb = tr >> 3, rl = tr & 7;
    const uint mrow = m0 + tr;
    const bool aok = mrow < M;

    for (uint k0 = 0; k0 < K; k0 += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort w = 0; w < WPT; w++) {
            const ushort kb = kq*WPT + w;
            threadgroup half* p = sa + (fb*4 + kb)*72 + nl;
            if (wok) {
                kq_stage8<KIND>(W, WS, ncol, N, K, k0, kb, p);
            } else {
                for (ushort kl = 0; kl < 8; kl++) p[kl*8] = 0.0h;
            }
        }
        for (ushort j = 0; j < BPT && (TM >= 32 || tr < TM); j++) {
            const ushort kb = kq2*BPT + j;
            threadgroup half4* d4 = (threadgroup half4*)(sb + (tb*4 + kb)*64 + rl*8);
            if (aok) {
                device const half4* s4 = (device const half4*)(A + mrow*K + k0 + kb*8u);
                d4[0] = s4[0]; d4[1] = s4[1];
            } else {
                d4[0] = half4(0.0h); d4[1] = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort ik = 0; ik < 4; ik++) {
            simdgroup_half8x8 a[TT], b[TF];
            for (ushort t = 0; t < TT; t++) simdgroup_load(a[t], sb + ((th*TT + t)*4 + ik)*64, 8);
            for (ushort f = 0; f < TF; f++) simdgroup_load(b[f], sa + ((fh*TF + f)*4 + ik)*72, 8);
            for (ushort t = 0; t < TT; t++)
                for (ushort f = 0; f < TF; f++)
                    simdgroup_multiply_accumulate(acc[t*TF + f], a[t], b[f], acc[t*TF + f]);
        }
    }

    threadgroup float* myc = cs + sg*64;
    for (ushort t = 0; t < TT; t++)
        for (ushort f = 0; f < TF; f++)
            gemm_epilogue_ld(myc, acc[t*TF + f], m0 + (th*TT + t)*8u, n0 + (fh*TF + f)*8u, M, N, ldc, coff, C, bias, mode, lane);
}
#define KQ_TILES(KIND, NAME) \
template [[host_name("gemm_kq_" NAME "_m64n64")]] kernel decltype(gemm_kq_tile<64, 64, KIND>) gemm_kq_tile<64, 64, KIND>; \
template [[host_name("gemm_kq_" NAME "_m32n64")]] kernel decltype(gemm_kq_tile<32, 64, KIND>) gemm_kq_tile<32, 64, KIND>; \
template [[host_name("gemm_kq_" NAME "_m64n32")]] kernel decltype(gemm_kq_tile<64, 32, KIND>) gemm_kq_tile<64, 32, KIND>; \
template [[host_name("gemm_kq_" NAME "_m32n32")]] kernel decltype(gemm_kq_tile<32, 32, KIND>) gemm_kq_tile<32, 32, KIND>; \
template [[host_name("gemm_kq_" NAME "_m16n32")]] kernel decltype(gemm_kq_tile<16, 32, KIND>) gemm_kq_tile<16, 32, KIND>;
KQ_TILES(0u, "q4k")
KQ_TILES(1u, "q5k")
KQ_TILES(2u, "q6k")
KQ_TILES(3u, "int8")

