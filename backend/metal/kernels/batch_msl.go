package kernels

// BatchKernelsHeader is what batchKernels needs from allKernels' preamble: the Metal header and the rows kernels' unroll
// pragma (kernels.go), spelled identically.
const BatchKernelsHeader = `#include <metal_stdlib>
using namespace metal;
#define SA_ROWS_UNROLL _Pragma("clang loop unroll(full)")
`

const BatchKernels = `
inline void mc3_frag(ushort lane, thread ushort& fm, thread ushort& fn) {
    const ushort qid = lane >> 2;
    fm = (qid & 4) + ((lane >> 1) & 3);
    fn = (qid & 2)*2 + (lane & 1)*2;
}

// aT[k][t] = half(aq[t*K + k]) for t < M, 0 above: the MMA kernels' right operand from M rows of int8 activations.
kernel void mc3_pack(device const char* aq[[buffer(0)]], device half* aT[[buffer(1)]], constant uint& M[[buffer(2)]],
    constant uint& K[[buffer(3)]], uint gid[[thread_position_in_grid]]) {
    uint k = gid >> 3u, t = gid & 7u;
    if (k >= K) return;
    aT[gid] = t < M ? half(float(aq[t*K + k])) : 0.0h;
}
// the LM head's permuted layout (see mc3_lm): logical row j = 32s + 8kb + kk holds physical k = 32s + 8(kk/2) + 2kb + kk%2.
kernel void mc3_pack_lm(device const char* aq[[buffer(0)]], device half* aT[[buffer(1)]], constant uint& M[[buffer(2)]],
    constant uint& K[[buffer(3)]], uint gid[[thread_position_in_grid]]) {
    uint j = gid >> 3u, t = gid & 7u;
    if (j >= K) return;
    uint s = j >> 5u, kb = (j >> 3u) & 3u, kk = j & 7u;
    uint k = s*32u + 8u*(kk >> 1u) + 2u*kb + (kk & 1u);
    aT[gid] = t < M ? half(float(aq[t*K + k])) : 0.0h;
}

// mc3_bt: gemv_w4a8_sa_rows (mode 0), _sa_bias_rows (1) and _sa_resid_rows (2) for 8 activation rows. Residue r = g mod
// 32 of the GEMV's lane-strided group loop keeps its own partial (r = 8*sgid + slot); the partials reduce by simd_sum's
// tree — slots ^1/^2/^4 in registers, simdgroups ^8/^16 through Q — so each output is the GEMV's simd_sum exactly.
template <uint FB>
kernel void mc3_bt(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const half* aT[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], constant uint& N[[buffer(6)]], constant uint& M[[buffer(7)]],
    device const float* bias[[buffer(8)]], constant uint& mode[[buffer(9)]],
    threadgroup float2* Q[[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], ushort sgid[[simdgroup_index_in_threadgroup]],
    ushort lane[[thread_index_in_simdgroup]]) {
    const uint G = K >> 5u;
    const uint f0 = tgid * (8u*FB);
    ushort fm, fn; mc3_frag(lane, fm, fn);
    const ushort sh0 = 4u*fn, sh1 = 4u*(fn+1u);
    float2 P[FB][8];
    device const uint4* wrow[FB];
    device const half* srow[FB];
    for (uint b=0;b<FB;b++) {
        uint row = f0 + b*8u + fm; wrow[b] = wq + row*G; srow[b] = sct + row*G;
        for (ushort i=0;i<8;i++) P[b][i] = float2(0.0f);
    }
    const uint nc = (G + 31u) >> 5u;
    for (uint c = 0; c < nc; c++) {
        SA_ROWS_UNROLL for (ushort i = 0; i < 8; i++) {
            const uint g = c*32u + sgid*8u + i;
            if (g < G) {
                simdgroup_half8x8 A[4];
                for (ushort kb=0; kb<4; kb++) simdgroup_load(A[kb], aT + (g*32u + kb*8u)*8u, 8);
                for (uint b=0;b<FB;b++) {
                    uint4 w = wrow[b][g];
                    float s = float(srow[b][g]);
                    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float,8,8>(0.0f);
                    for (ushort kb=0; kb<4; kb++) {
                        uint word = w[kb];
                        simdgroup_half8x8 Wf;
                        Wf.thread_elements()[0] = half(int((word >> sh0) & 0xFu) - 8);
                        Wf.thread_elements()[1] = half(int((word >> sh1) & 0xFu) - 8);
                        simdgroup_multiply_accumulate(acc, Wf, A[kb], acc);
                    }
                    P[b][i].x += acc.thread_elements()[0] * s;
                    P[b][i].y += acc.thread_elements()[1] * s;
                }
            }
        }
    }
    {
#pragma clang fp reassociate(off)
    for (uint b=0;b<FB;b++) {
        float2 q = ((P[b][0] + P[b][1]) + (P[b][2] + P[b][3])) + ((P[b][4] + P[b][5]) + (P[b][6] + P[b][7]));
        Q[(sgid*FB + b)*32u + lane] = q;
    }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid != 0) return;
    {
#pragma clang fp reassociate(off)
    for (uint b=0;b<FB;b++) {
        float2 v = (Q[(0u*FB + b)*32u + lane] + Q[(1u*FB + b)*32u + lane]) + (Q[(2u*FB + b)*32u + lane] + Q[(3u*FB + b)*32u + lane]);
        uint row = f0 + b*8u + fm;
        for (ushort e = 0; e < 2; e++) {
            uint t = fn + e;
            if (t >= M) continue;
            float vv = e == 0 ? v.x : v.y;
            if (mode == 1u) out[t*N + row] = vv*asc[t] + bias[row];
            else if (mode == 2u) out[t*N + row] += vv*asc[t];
            else out[t*N + row] = vv*asc[t];
        }
    }
    }
}
template [[host_name("mc3_bt_fb2")]] kernel decltype(mc3_bt<2>) mc3_bt<2>;

// mc3_btd: gemv_w4a8_resid_staged for 8 rows. That kernel sums per 8-k WORD (lane L: words L, L+32, ...), so each k
// block's MMA runs into its own zero fragment (exactly that word's integer sum) and residues are words.
template <uint FB>
kernel void mc3_btd(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const half* aT[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], constant uint& N[[buffer(6)]], constant uint& M[[buffer(7)]],
    threadgroup float2* Q[[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], ushort sgid[[simdgroup_index_in_threadgroup]],
    ushort lane[[thread_index_in_simdgroup]]) {
    const uint G = K >> 5u;
    const uint f0 = tgid * (8u*FB);
    ushort fm, fn; mc3_frag(lane, fm, fn);
    const ushort sh0 = 4u*fn, sh1 = 4u*(fn+1u);
    float2 P[FB][8];
    device const uint4* wrow[FB];
    device const half* srow[FB];
    for (uint b=0;b<FB;b++) {
        uint row = f0 + b*8u + fm; wrow[b] = wq + row*G; srow[b] = sct + row*G;
        for (ushort i=0;i<8;i++) P[b][i] = float2(0.0f);
    }
    const uint nc = (G + 7u) >> 3u;
    for (uint c = 0; c < nc; c++) {
        SA_ROWS_UNROLL for (ushort j = 0; j < 2; j++) {
            const uint g = c*8u + sgid*2u + j;
            if (g < G) {
                simdgroup_half8x8 A[4];
                for (ushort kb=0; kb<4; kb++) simdgroup_load(A[kb], aT + (g*32u + kb*8u)*8u, 8);
                for (uint b=0;b<FB;b++) {
                    uint4 w = wrow[b][g];
                    float s = float(srow[b][g]);
                    SA_ROWS_UNROLL for (ushort kb=0; kb<4; kb++) {
                        uint word = w[kb];
                        simdgroup_half8x8 Wf;
                        Wf.thread_elements()[0] = half(int((word >> sh0) & 0xFu) - 8);
                        Wf.thread_elements()[1] = half(int((word >> sh1) & 0xFu) - 8);
                        simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float,8,8>(0.0f);
                        simdgroup_multiply_accumulate(acc, Wf, A[kb], acc);
                        P[b][j*4+kb].x += acc.thread_elements()[0] * s;
                        P[b][j*4+kb].y += acc.thread_elements()[1] * s;
                    }
                }
            }
        }
    }
    {
#pragma clang fp reassociate(off)
    for (uint b=0;b<FB;b++) {
        float2 q = ((P[b][0] + P[b][1]) + (P[b][2] + P[b][3])) + ((P[b][4] + P[b][5]) + (P[b][6] + P[b][7]));
        Q[(sgid*FB + b)*32u + lane] = q;
    }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid != 0) return;
    {
#pragma clang fp reassociate(off)
    for (uint b=0;b<FB;b++) {
        float2 v = (Q[(0u*FB + b)*32u + lane] + Q[(1u*FB + b)*32u + lane]) + (Q[(2u*FB + b)*32u + lane] + Q[(3u*FB + b)*32u + lane]);
        uint row = f0 + b*8u + fm;
        if (fn < M) out[fn*N + row] += v.x * asc[fn];
        if (fn+1u < M) out[(fn+1u)*N + row] += v.y * asc[fn+1u];
    }
    }
}
template [[host_name("mc3_btd_fb2")]] kernel decltype(mc3_btd<2>) mc3_btd<2>;

// mc3_lm: gemv_w8a8_coal for 8 rows. A 32-k slab's four MMAs stay below 2^24 (32*128*128), so each lane converts the
// slab's exact integer and sums it in int32 — production's integer, in any order. Inside a slab k is permuted, the same
// way for both operands, so each lane's weights are 8 contiguous bytes (mc3_pack_lm lays the activations out to match).
template <uint FB>
kernel void mc3_lm(device const char* bq[[buffer(0)]], device const float* bsc[[buffer(1)]],
    device const half* aTp[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], constant uint& N[[buffer(6)]], constant uint& M[[buffer(7)]],
    uint tgid[[threadgroup_position_in_grid]], ushort sgid[[simdgroup_index_in_threadgroup]],
    ushort lane[[thread_index_in_simdgroup]]) {
    ushort fm, fn; mc3_frag(lane, fm, fn);
    const uint f0 = (tgid*4u + sgid) * (8u*FB);
    int2 acc[FB];
    device const uint2* wrow[FB];
    for (uint b=0;b<FB;b++) { acc[b] = int2(0); wrow[b] = reinterpret_cast<device const uint2*>(bq + (f0 + b*8u + fm)*K) + (fn >> 1); }
    const uint S = K >> 5u;
    for (uint s = 0; s < S; s++) {
        simdgroup_half8x8 A[4];
        for (ushort kb=0; kb<4; kb++) simdgroup_load(A[kb], aTp + (s*32u + kb*8u)*8u, 8);
        for (uint b=0;b<FB;b++) {
            uint2 w = wrow[b][s*4u];
            char4 cx = as_type<char4>(w.x), cy = as_type<char4>(w.y);
            simdgroup_half8x8 W0, W1, W2, W3;
            W0.thread_elements()[0] = half(cx.x); W0.thread_elements()[1] = half(cx.y);
            W1.thread_elements()[0] = half(cx.z); W1.thread_elements()[1] = half(cx.w);
            W2.thread_elements()[0] = half(cy.x); W2.thread_elements()[1] = half(cy.y);
            W3.thread_elements()[0] = half(cy.z); W3.thread_elements()[1] = half(cy.w);
            simdgroup_float8x8 f = make_filled_simdgroup_matrix<float,8,8>(0.0f);
            simdgroup_multiply_accumulate(f, W0, A[0], f);
            simdgroup_multiply_accumulate(f, W1, A[1], f);
            simdgroup_multiply_accumulate(f, W2, A[2], f);
            simdgroup_multiply_accumulate(f, W3, A[3], f);
            acc[b] += int2(int(f.thread_elements()[0]), int(f.thread_elements()[1]));
        }
    }
    for (uint b=0;b<FB;b++) {
        uint row = f0 + b*8u + fm;
        if (fn < M) out[fn*N + row] = float(acc[b].x) * asc[fn] * bsc[row];
        if (fn+1u < M) out[(fn+1u)*N + row] = float(acc[b].y) * asc[fn+1u] * bsc[row];
    }
}
template [[host_name("mc3_lm_fb4")]] kernel decltype(mc3_lm<4>) mc3_lm<4>;

// mc3_sstree: the build-time check that simd_sum IS the xor tree 1, 2, 4, 8, 16 the kernels above reproduce. Writes, per
// simdgroup, simd_sum(v) and the tree's own value; batchSimdSumTreeOK compares them bit for bit.
kernel void mc3_sstree(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    float v = in[gid];
    float b = v;
    b += simd_shuffle_xor(b, 1); b += simd_shuffle_xor(b, 2); b += simd_shuffle_xor(b, 4);
    b += simd_shuffle_xor(b, 8); b += simd_shuffle_xor(b, 16);
    float s = simd_sum(v);
    if (lane == 0) { uint w = gid >> 5; out[w*2] = s; out[w*2+1] = b; }
}
`
