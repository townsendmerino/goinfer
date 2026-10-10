//go:build darwin

package metal

import (
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"testing"
	"time"
)

// mc3MMAKernels is MC3's matrix-unit prototype (docs/tasks/task-concurrency-2026-09.md), TEST-ONLY: a W4A8 GEMM for
// up to 8 decode sequences on simdgroup_matrix, sized for decode rather than prefill. The prefill GEMM's 64-token tile
// costs the same at M = 1 as at M = 64 (TestMC3S0PrefillGEMMSmallM times it), so this one's token tile
// is a single 8x8 fragment column: C^T[feature][token] = W[feature][k] . A^T[k][token].
//
//   - The left operand (8 features x 8 k) is built IN REGISTERS through thread_elements(): lane (fm, fn) holds features
//     fm, k = fn, fn+1 — two adjacent nibbles of one packed word — so weights never touch threadgroup memory. The
//     mapping is MLX's (steel BaseMMAFrag); TestMC3MMA8 checks it on this device first.
//   - Weights are the raw nibble - 8 and activations the int8 quantized values, both exact in half, so a 32-k group's
//     four MMAs accumulate the INTEGER group sum gi exactly in f32 (|gi| <= 32*8*127). Each lane then scales its two
//     elements by that row's group scale: run += float(gi) * scale, the decode kernels' own per-group product.
//   - What differs from the decode GEMV is only the ORDER of the float sum over groups: here g = 0, 1, 2, ... within a
//     K slice (then slices in order); the GEMV sums lane-strided partials and simd_sums them. So outputs agree to f32
//     summation-order rounding, not bit for bit.
//
// A^T is half [K][8] (token t in column t, tokens past M zero). FB feature blocks of 8 per simdgroup, S simdgroups
// splitting K (contiguous slices) per feature set; 4 simdgroups per threadgroup; grid N / (8*FB*4/S) threadgroups.
const mc3MMAKernels = `
kernel void mc3_te_layout(device float* out [[buffer(0)]], ushort lane [[thread_index_in_simdgroup]]) {
    simdgroup_float8x8 m;
    m.thread_elements()[0] = float(lane*2);
    m.thread_elements()[1] = float(lane*2+1);
    simdgroup_store(m, out, 8);
}

template <uint FB, uint S>
kernel void mc3_mma(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const half* aT[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], constant uint& N[[buffer(6)]], constant uint& M[[buffer(7)]],
    threadgroup float2* red[[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], ushort sgid[[simdgroup_index_in_threadgroup]],
    ushort lane[[thread_index_in_simdgroup]]) {
    const uint G = K >> 5u;
    const ushort fsg = sgid / S, ks = sgid % S;
    const uint f0 = (tgid*(4u/S) + fsg) * (8u*FB);
    const ushort qid = lane >> 2;
    const ushort fm = (qid & 4) + ((lane >> 1) & 3);
    const ushort fn = (qid & 2)*2 + (lane & 1)*2;
    const uint g0 = (G*ks)/S, g1 = (G*(ks+1u))/S;
    float2 run[FB];
    device const uint4* wrow[FB];
    device const half* srow[FB];
    for (uint b=0;b<FB;b++) { uint row = f0 + b*8u + fm; run[b] = float2(0.0f); wrow[b] = wq + row*G; srow[b] = sct + row*G; }
    const ushort sh0 = 4u*fn, sh1 = 4u*(fn+1u);
    for (uint g = g0; g < g1; g++) {
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
            run[b].x += acc.thread_elements()[0] * s;
            run[b].y += acc.thread_elements()[1] * s;
        }
    }
    if (S > 1u) {
        for (uint b=0;b<FB;b++) red[(sgid*FB + b)*32u + lane] = run[b];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ks != 0) return;
        for (ushort j=1;j<S;j++) { for (uint b=0;b<FB;b++) run[b] += red[((sgid+j)*FB + b)*32u + lane]; }
    }
    for (uint b=0;b<FB;b++) {
        uint row = f0 + b*8u + fm;
        if (fn < M) out[fn*N + row] = run[b].x * asc[fn];
        if (fn+1u < M) out[(fn+1u)*N + row] = run[b].y * asc[fn+1u];
    }
}
// mc3_mma_bi: the same fragments and exact integer group sums, accumulated in the decode GEMV's ORDER so the result
// is bit-identical to it. The GEMV's lane L sums groups g = L, L+32, L+64, ... as acc += float(gi)*scale and then
// simd_sums the 32 lanes. Here residue r = g mod 32 keeps its own partial with that same chain (residues interleaved
// over the 4 simdgroups, r = 4i + sgid, so each simdgroup runs 8 of them); at the end every partial goes to
// threadgroup memory as T[residue][output] (row stride 65: bank-conflict padding) and each output is simd_summed with
// residue L on lane L — the GEMV's exact reduction input. All 4 simdgroups share the threadgroup's 8*FB features.
template <uint FB>
kernel void mc3_mma_bi(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const half* aT[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], constant uint& N[[buffer(6)]], constant uint& M[[buffer(7)]],
    threadgroup float* T[[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], ushort sgid[[simdgroup_index_in_threadgroup]],
    ushort lane[[thread_index_in_simdgroup]]) {
    const uint G = K >> 5u;
    const uint f0 = tgid * (8u*FB);
    const ushort qid = lane >> 2;
    const ushort fm = (qid & 4) + ((lane >> 1) & 3);
    const ushort fn = (qid & 2)*2 + (lane & 1)*2;
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
            const uint g = c*32u + i*4u + sgid;
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
    for (uint b=0;b<FB;b++) {
        SA_ROWS_UNROLL for (ushort i=0;i<8;i++) {
            threadgroup float* tr = T + (b*32u + i*4u + sgid)*65u + fm*8u + fn;
            tr[0] = P[b][i].x; tr[1] = P[b][i].y;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint q = sgid; q < 64u*FB; q += 4u) {
        const uint b = q >> 6u, e = q & 63u;
        float v = simd_sum(T[(b*32u + lane)*65u + e]);
        const uint t = e & 7u;
        if (lane == 0 && t < M) out[t*N + f0 + b*8u + (e >> 3u)] = v * asc[t];
    }
}
// mc3_mma_bt: mc3_mma_bi with the reduction done as simd_sum's own tree instead of by simd_sum. TestMC3SimdSumTree
// measured this GPU's simd_sum as exactly the xor butterfly 1, 2, 4, 8, 16 (lane L pairs with L^1, then ^2, ...). So
// residues are mapped r = 8*sgid + i (simdgroup sgid holds 8 CONSECUTIVE residues in slots i): levels ^1, ^2, ^4 pair
// slots within a lane, in registers; levels ^8 and ^16 pair simdgroups (0,1), (2,3), then the two sums, through a
// small threadgroup exchange. Adjacent slots are adjacent groups, so a lane's weight loads walk a row contiguously.
// NOT robust by construction: it is bit-identical only where simd_sum is that tree (a runtime self-check would gate it).
template <uint FB>
kernel void mc3_mma_bt(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const half* aT[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], constant uint& N[[buffer(6)]], constant uint& M[[buffer(7)]],
    threadgroup float2* Q[[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], ushort sgid[[simdgroup_index_in_threadgroup]],
    ushort lane[[thread_index_in_simdgroup]]) {
    const uint G = K >> 5u;
    const uint f0 = tgid * (8u*FB);
    const ushort qid = lane >> 2;
    const ushort fm = (qid & 4) + ((lane >> 1) & 3);
    const ushort fn = (qid & 2)*2 + (lane & 1)*2;
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
        if (fn < M) out[fn*N + row] = v.x * asc[fn];
        if (fn+1u < M) out[(fn+1u)*N + row] = v.y * asc[fn+1u];
    }
    }
}
template [[host_name("mc3_mma_bt_fb1")]] kernel decltype(mc3_mma_bt<1>) mc3_mma_bt<1>;
template [[host_name("mc3_mma_bt_fb2")]] kernel decltype(mc3_mma_bt<2>) mc3_mma_bt<2>;
template [[host_name("mc3_mma_bt_fb4")]] kernel decltype(mc3_mma_bt<4>) mc3_mma_bt<4>;
// mc3_mma_btd: the down projection's version of mc3_mma_bt. Production's down kernel (gemv_w4a8_resid_staged<R>)
// accumulates per WORD (8 k), not per group: lane L sums words w = L, L+32, ... as acc += float(gi_w) * scale(w/4),
// then simd_sums and adds into the residual. So each of a group's four k blocks gets its own zero-initialised
// accumulator (its MMA is exactly that word's integer sum), and residues are words: r = w mod 32 = 8*sgid + i, i.e.
// simdgroup sgid owns groups 8c + 2*sgid + {0, 1}, all four words of each.
template <uint FB>
kernel void mc3_mma_btd(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const half* aT[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], constant uint& N[[buffer(6)]], constant uint& M[[buffer(7)]],
    threadgroup float2* Q[[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], ushort sgid[[simdgroup_index_in_threadgroup]],
    ushort lane[[thread_index_in_simdgroup]]) {
    const uint G = K >> 5u;
    const uint f0 = tgid * (8u*FB);
    const ushort qid = lane >> 2;
    const ushort fm = (qid & 4) + ((lane >> 1) & 3);
    const ushort fn = (qid & 2)*2 + (lane & 1)*2;
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
template [[host_name("mc3_mma_btd_fb1")]] kernel decltype(mc3_mma_btd<1>) mc3_mma_btd<1>;
template [[host_name("mc3_mma_btd_fb2")]] kernel decltype(mc3_mma_btd<2>) mc3_mma_btd<2>;
template [[host_name("mc3_mma_btd_fb4")]] kernel decltype(mc3_mma_btd<4>) mc3_mma_btd<4>;
// mc3_lm_mma: the int8 LM head (production gemv_w8a8_coal: one int32 sum per row, simd_sum of ints, then
// float(acc) * asc * bsc[row]) for 8 sequences. int8 weights and activations are exact in half; a 32-k slab's four
// MMAs sum at most 32*128*128 < 2^24, so the f32 fragment holds that slab's exact integer, which each lane converts
// and adds into an int32 — the same integer as production's, in any order, so the logits are bit-identical by
// construction. The k order INSIDE a slab is permuted (identically for both operands) so each lane's weights are 8
// contiguous bytes: logical (block kb, column kk) = physical k 8*(kk/2) + 2*kb + kk%2. A^T is laid out to match
// (aTp, written by the host here; a batched quant kernel would write it in production).
template <uint FB>
kernel void mc3_lm_mma(device const char* bq[[buffer(0)]], device const float* bsc[[buffer(1)]],
    device const half* aTp[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], constant uint& N[[buffer(6)]], constant uint& M[[buffer(7)]],
    uint tgid[[threadgroup_position_in_grid]], ushort sgid[[simdgroup_index_in_threadgroup]],
    ushort lane[[thread_index_in_simdgroup]]) {
    const ushort qid = lane >> 2;
    const ushort fm = (qid & 4) + ((lane >> 1) & 3);
    const ushort fn = (qid & 2)*2 + (lane & 1)*2;
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
template [[host_name("mc3_lm_mma_fb1")]] kernel decltype(mc3_lm_mma<1>) mc3_lm_mma<1>;
template [[host_name("mc3_lm_mma_fb2")]] kernel decltype(mc3_lm_mma<2>) mc3_lm_mma<2>;
template [[host_name("mc3_lm_mma_fb4")]] kernel decltype(mc3_lm_mma<4>) mc3_lm_mma<4>;
template [[host_name("mc3_mma_bi_fb1")]] kernel decltype(mc3_mma_bi<1>) mc3_mma_bi<1>;
template [[host_name("mc3_mma_bi_fb2")]] kernel decltype(mc3_mma_bi<2>) mc3_mma_bi<2>;
template [[host_name("mc3_mma_fb1_s1")]] kernel decltype(mc3_mma<1,1>) mc3_mma<1,1>;
template [[host_name("mc3_mma_fb2_s1")]] kernel decltype(mc3_mma<2,1>) mc3_mma<2,1>;
template [[host_name("mc3_mma_fb4_s1")]] kernel decltype(mc3_mma<4,1>) mc3_mma<4,1>;
template [[host_name("mc3_mma_fb1_s2")]] kernel decltype(mc3_mma<1,2>) mc3_mma<1,2>;
template [[host_name("mc3_mma_fb2_s2")]] kernel decltype(mc3_mma<2,2>) mc3_mma<2,2>;
template [[host_name("mc3_mma_fb1_s4")]] kernel decltype(mc3_mma<1,4>) mc3_mma<1,4>;
template [[host_name("mc3_mma_fb2_s4")]] kernel decltype(mc3_mma<2,4>) mc3_mma<2,4>;
template [[host_name("mc3_mma_fb4_s4")]] kernel decltype(mc3_mma<4,4>) mc3_mma<4,4>;
`

// TestMC3MMA8: EXPLORATORY S0 for MC3's matrix-unit route. Checks the thread_elements() lane mapping, then for each
// decode shape of the 1.5B and 7B (qkv, o, gate/up: the SA family, with production's rows per simdgroup) times, per
// dispatch with the S0 bench's SLC-defeating weight rotation, production's M = 1 GEMV (gemv_w4a8_sa_rows<R>, R18b)
// against one mc3_mma dispatch carrying 8 sequences (its cost does not depend on M <= 8). Every variant's 8 output rows
// are checked against the production GEMV run on each row: relative error, and how many outputs match bit for bit.
//
//	GOINFER_METAL_MC3=1 go test -count=1 -run '^TestMC3MMA8$' -v ./metal/
func TestMC3MMA8(t *testing.T) {
	if os.Getenv("GOINFER_METAL_MC3") != "1" {
		t.Skip("set GOINFER_METAL_MC3=1 (allocates ~0.5 GB of GPU buffers)")
	}
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer d.ReleaseAll()
	lib, err := d.CompileLibrary(allKernels+"\n"+mc3MMAKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe := func(name string) Pipeline {
		p, err := d.NewComputePipeline(lib, name)
		if err != nil {
			t.Fatalf("pipeline %s: %v", name, err)
		}
		return p
	}
	cq := d.NewCommandQueue()

	// thread_elements() mapping: lane l's elements must sit at (row fm, cols fn, fn+1).
	lay := d.NewBufferLen(64)
	e := cq.Begin()
	e.DispatchTG(pipe("mc3_te_layout"), 32, 32, 0, lay)
	e.End()
	lv := lay.Floats()[:64]
	for l := range 32 {
		q := l / 4
		fm, fn := (q&4)+((l/2)%4), (q&2)*2+(l%2)*2
		for i := range 2 {
			if got := lv[fm*8+fn+i]; got != float32(l*2+i) {
				t.Fatalf("thread_elements mapping: lane %d element %d expected at (%d,%d), found %v there", l, i, fm, fn+i, got)
			}
		}
	}

	t0 := time.Now()
	hb := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[mc3-mma8 %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	seed := uint32(88172645)
	rnd := func() uint32 { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed }
	const reps = 7
	med := func(xs []float64) float64 { s := append([]float64(nil), xs...); sort.Float64s(s); return s[len(s)/2] }
	timeIt := func(per int, enc func(e *Encoder, i int)) float64 {
		var ms []float64
		for r := range reps + 1 {
			e := cq.Begin()
			for i := range per {
				enc(e, i)
			}
			e.End()
			if err := e.Err(); err != nil {
				t.Fatalf("dispatch: %v", err)
			}
			if r > 0 {
				ms = append(ms, (e.GPUEnd()-e.GPUStart())*1e3/float64(per))
			}
		}
		return med(ms)
	}
	type shape struct {
		model, cat string
		N, K, R    int // R: production's rows per simdgroup (gemvRows {qkv:4 o:2 gu:4})
	}
	variants := []struct{ fb, s int }{{1, 1}, {2, 1}, {4, 1}, {1, 2}, {2, 2}, {1, 4}, {2, 4}, {4, 4}}
	for _, s := range []shape{
		{"1.5B", "qkv", 2048, 1536, 4}, {"1.5B", "o", 1536, 1536, 2}, {"1.5B", "gate/up", 17920, 1536, 4},
		{"7B", "qkv", 4608, 3584, 4}, {"7B", "o", 3584, 3584, 2}, {"7B", "gate/up", 37888, 3584, 4},
	} {
		const M = 8
		bytes := s.N*s.K/2 + s.N*(s.K/32)*2
		copies := min(64, max(1, int(math.Ceil(float64(256<<20)/float64(bytes)))))
		aqb := d.NewBufferBytes(M * s.K)
		av := aqb.Int8s()[:M*s.K]
		for i := range av {
			av[i] = int8(rnd()%255) - 127
		}
		ah := make([]uint16, s.K*8) // A^T [K][8]
		for m := range M {
			for k := 0; k < s.K; k++ {
				ah[k*8+m] = f32ToF16(float32(av[m*s.K+k]))
			}
		}
		aT := NewBufferU16s(d, ah)
		ascs := make([]float32, M)
		for i := range ascs {
			ascs[i] = 0.01 + 0.001*float32(i)
		}
		asc := NewBufferFloats(d, ascs)
		uK, uN, uM := NewBufferU32(d, uint32(s.K)), NewBufferU32(d, uint32(s.N)), NewBufferU32(d, M)
		var ws, ss []Buffer
		for range copies {
			nw := s.N * s.K / 8
			wb := d.NewBufferLen(nw)
			v := wb.U32s()[:nw]
			for i := range v {
				v[i] = rnd()
			}
			sc := make([]uint16, s.N*(s.K/32))
			for i := range sc {
				sc[i] = f32ToF16(float32(rnd()%1000+1) * 1e-5)
			}
			ws, ss = append(ws, wb), append(ss, NewBufferU16s(d, sc))
		}
		gemv := pipe(fmt.Sprintf("gemv_w4a8_sa_rows%d", s.R))
		gemvRow := func(e *Encoder, c, m int, out Buffer) {
			e.DispatchTG(gemv, s.N*32/s.R, 256, s.K*2, ws[c], ss[c], aqb.At(m*s.K), asc.At(4*m), out.At(4*m*s.N), uK)
		}
		ref := d.NewBufferLen(M * s.N)
		e := cq.Begin()
		for m := range M {
			gemvRow(e, 0, m, ref)
		}
		e.End()
		refV := append([]float32(nil), ref.Floats()[:M*s.N]...)
		refMax := 0.0
		for _, v := range refV {
			refMax = math.Max(refMax, math.Abs(float64(v)))
		}
		per := max(8, 2*copies)
		t1 := timeIt(per, func(e *Encoder, i int) { gemvRow(e, i%copies, 0, ref) })
		line := fmt.Sprintf("%-4s %-8s N=%-6d K=%-5d production GEMV (R=%d) %.3f ms, x8 = %.3f:", s.model, s.cat, s.N, s.K, s.R, t1, 8*t1)
		for _, vr := range variants {
			fpt := 8 * vr.fb * 4 / vr.s
			if s.N%fpt != 0 {
				continue
			}
			p := pipe(fmt.Sprintf("mc3_mma_fb%d_s%d", vr.fb, vr.s))
			tgs := s.N / fpt
			red := max(16, 4*vr.fb*32*8)
			out := d.NewBufferLen(M * s.N)
			e := cq.Begin()
			e.DispatchTG(p, tgs*128, 128, red, ws[0], ss[0], aT, asc, out, uK, uN, uM)
			e.End()
			if err := e.Err(); err != nil {
				t.Fatalf("mma dispatch: %v", err)
			}
			got := out.Floats()[:M*s.N]
			same, worst := 0, 0.0
			for i := range got {
				if math.Float32bits(got[i]) == math.Float32bits(refV[i]) {
					same++
				}
				worst = math.Max(worst, math.Abs(float64(got[i]-refV[i]))/refMax)
			}
			if worst > 1e-5 {
				t.Errorf("%s %s fb%d s%d: max |mma - GEMV| / max|GEMV| = %.3g (want summation-order level, < 1e-5)", s.model, s.cat, vr.fb, vr.s, worst)
			}
			tm := timeIt(per, func(e *Encoder, i int) {
				e.DispatchTG(p, tgs*128, 128, red, ws[i%copies], ss[i%copies], aT, asc, out, uK, uN, uM)
			})
			line += fmt.Sprintf("  fb%d/s%d %.3f (%.2fx 8 GEMVs, =%.1f GEMVs, relerr %.1e, %d%% bit-equal)", vr.fb, vr.s, tm, 8*t1/tm, tm/t1, worst, 100*same/len(got))
		}
		for _, bv := range []struct {
			kind string
			fb   int
		}{{"bi", 1}, {"bi", 2}, {"bt", 1}, {"bt", 2}, {"bt", 4}} {
			fb := bv.fb
			p := pipe(fmt.Sprintf("mc3_mma_%s_fb%d", bv.kind, fb))
			tgs, tgb := s.N/(8*fb), fb*32*65*4
			if bv.kind == "bt" {
				tgb = 4 * fb * 32 * 8
			}
			out := d.NewBufferLen(M * s.N)
			e := cq.Begin()
			e.DispatchTG(p, tgs*128, 128, tgb, ws[0], ss[0], aT, asc, out, uK, uN, uM)
			e.End()
			if err := e.Err(); err != nil {
				t.Fatalf("mma_bi dispatch: %v", err)
			}
			got := out.Floats()[:M*s.N]
			diff := 0
			for i := range got {
				if math.Float32bits(got[i]) != math.Float32bits(refV[i]) {
					diff++
				}
			}
			if diff != 0 {
				t.Errorf("%s %s mma_%s fb%d: %d of %d outputs differ from the production GEMV", s.model, s.cat, bv.kind, fb, diff, len(got))
			}
			tm := timeIt(per, func(e *Encoder, i int) {
				e.DispatchTG(p, tgs*128, 128, tgb, ws[i%copies], ss[i%copies], aT, asc, out, uK, uN, uM)
			})
			line += fmt.Sprintf("  %s fb%d %.3f (%.2fx 8 GEMVs, =%.1f GEMVs, %d differ)", strings.ToUpper(bv.kind), fb, tm, 8*t1/tm, tm/t1, diff)
		}
		hb("%s", line)
		d.ReleaseAll()
	}
}

// TestMC3MMA8Down: mc3_mma_btd against production's down projection (gemv_w4a8_resid_staged4) on the 1.5B and 7B
// down shapes: bit-identity per activation row (both add into a zeroed residual), then per-dispatch timing with the
// SLC-defeating rotation.
//
//	GOINFER_METAL_MC3=1 go test -count=1 -run '^TestMC3MMA8Down$' -v ./metal/
func TestMC3MMA8Down(t *testing.T) {
	if os.Getenv("GOINFER_METAL_MC3") != "1" {
		t.Skip("set GOINFER_METAL_MC3=1 (allocates ~0.5 GB of GPU buffers)")
	}
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer d.ReleaseAll()
	lib, err := d.CompileLibrary(allKernels+"\n"+mc3MMAKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe := func(name string) Pipeline {
		p, err := d.NewComputePipeline(lib, name)
		if err != nil {
			t.Fatalf("pipeline %s: %v", name, err)
		}
		return p
	}
	cq := d.NewCommandQueue()
	t0 := time.Now()
	seed := uint32(362436069)
	rnd := func() uint32 { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed }
	const reps = 7
	med := func(xs []float64) float64 { s := append([]float64(nil), xs...); sort.Float64s(s); return s[len(s)/2] }
	timeIt := func(per int, enc func(e *Encoder, i int)) float64 {
		var ms []float64
		for r := range reps + 1 {
			e := cq.Begin()
			for i := range per {
				enc(e, i)
			}
			e.End()
			if err := e.Err(); err != nil {
				t.Fatalf("dispatch: %v", err)
			}
			if r > 0 {
				ms = append(ms, (e.GPUEnd()-e.GPUStart())*1e3/float64(per))
			}
		}
		return med(ms)
	}
	for _, s := range []struct {
		model string
		N, K  int
	}{{"1.5B", 1536, 8960}, {"7B", 3584, 18944}} {
		const M = 8
		bytes := s.N*s.K/2 + s.N*(s.K/32)*2
		copies := min(64, max(1, int(math.Ceil(float64(256<<20)/float64(bytes)))))
		aqb := d.NewBufferBytes(M * s.K)
		av := aqb.Int8s()[:M*s.K]
		for i := range av {
			av[i] = int8(rnd()%255) - 127
		}
		ah := make([]uint16, s.K*8)
		for m := range M {
			for k := 0; k < s.K; k++ {
				ah[k*8+m] = f32ToF16(float32(av[m*s.K+k]))
			}
		}
		aT := NewBufferU16s(d, ah)
		ascs := make([]float32, M)
		for i := range ascs {
			ascs[i] = 0.01 + 0.001*float32(i)
		}
		asc := NewBufferFloats(d, ascs)
		uK, uN, uM := NewBufferU32(d, uint32(s.K)), NewBufferU32(d, uint32(s.N)), NewBufferU32(d, M)
		var ws, ss []Buffer
		for range copies {
			nw := s.N * s.K / 8
			wb := d.NewBufferLen(nw)
			v := wb.U32s()[:nw]
			for i := range v {
				v[i] = rnd()
			}
			sc := make([]uint16, s.N*(s.K/32))
			for i := range sc {
				sc[i] = f32ToF16(float32(rnd()%1000+1) * 1e-5)
			}
			ws, ss = append(ws, wb), append(ss, NewBufferU16s(d, sc))
		}
		down := pipe("gemv_w4a8_resid_staged4")
		downRow := func(e *Encoder, c, m int, out Buffer) {
			e.DispatchTG(down, s.N*32/4, 256, s.K, ws[c], ss[c], aqb.At(m*s.K), asc.At(4*m), out.At(4*m*s.N), uK)
		}
		ref := d.NewBufferLen(M * s.N) // zero-filled on allocation
		e := cq.Begin()
		for m := range M {
			downRow(e, 0, m, ref)
		}
		e.End()
		refV := append([]float32(nil), ref.Floats()[:M*s.N]...)
		per := max(8, 2*copies)
		t1 := timeIt(per, func(e *Encoder, i int) { downRow(e, i%copies, 0, ref) })
		line := fmt.Sprintf("%-4s down N=%-5d K=%-5d production staged4 %.3f ms, x8 = %.3f:", s.model, s.N, s.K, t1, 8*t1)
		for _, fb := range []int{1, 2, 4} {
			p := pipe(fmt.Sprintf("mc3_mma_btd_fb%d", fb))
			tgs, tgb := s.N/(8*fb), 4*fb*32*8
			out := d.NewBufferLen(M * s.N)
			e := cq.Begin()
			e.DispatchTG(p, tgs*128, 128, tgb, ws[0], ss[0], aT, asc, out, uK, uN, uM)
			e.End()
			if err := e.Err(); err != nil {
				t.Fatalf("btd dispatch: %v", err)
			}
			got := out.Floats()[:M*s.N]
			diff := 0
			for i := range got {
				if math.Float32bits(got[i]) != math.Float32bits(refV[i]) {
					diff++
				}
			}
			if diff != 0 {
				t.Errorf("%s down btd fb%d: %d of %d outputs differ from production", s.model, fb, diff, len(got))
			}
			tm := timeIt(per, func(e *Encoder, i int) {
				e.DispatchTG(p, tgs*128, 128, tgb, ws[i%copies], ss[i%copies], aT, asc, out, uK, uN, uM)
			})
			line += fmt.Sprintf("  BTD fb%d %.3f (%.2fx 8 GEMVs, =%.1f GEMVs, %d differ)", fb, tm, 8*t1/tm, tm/t1, diff)
		}
		fmt.Fprintf(os.Stderr, "[mc3-mma8d %6.1fs] %s\n", time.Since(t0).Seconds(), line)
		d.ReleaseAll()
	}
}

// TestMC3MMA8LMHead: mc3_lm_mma against production's int8 LM head (gemv_w8a8_coal) at the 1.5B and 7B vocab x hidden
// shapes: bit-identity of all 8 rows' logits, then per-dispatch timing (the head is larger than the SLC, so no
// rotation is needed).
//
//	GOINFER_METAL_MC3=1 go test -count=1 -run '^TestMC3MMA8LMHead$' -v ./metal/
func TestMC3MMA8LMHead(t *testing.T) {
	if os.Getenv("GOINFER_METAL_MC3") != "1" {
		t.Skip("set GOINFER_METAL_MC3=1 (allocates ~1 GB of GPU buffers)")
	}
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer d.ReleaseAll()
	lib, err := d.CompileLibrary(allKernels+"\n"+mc3MMAKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe := func(name string) Pipeline {
		p, err := d.NewComputePipeline(lib, name)
		if err != nil {
			t.Fatalf("pipeline %s: %v", name, err)
		}
		return p
	}
	cq := d.NewCommandQueue()
	t0 := time.Now()
	seed := uint32(521288629)
	rnd := func() uint32 { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed }
	const reps = 7
	med := func(xs []float64) float64 { s := append([]float64(nil), xs...); sort.Float64s(s); return s[len(s)/2] }
	timeIt := func(per int, enc func(e *Encoder, i int)) float64 {
		var ms []float64
		for r := range reps + 1 {
			e := cq.Begin()
			for i := range per {
				enc(e, i)
			}
			e.End()
			if err := e.Err(); err != nil {
				t.Fatalf("dispatch: %v", err)
			}
			if r > 0 {
				ms = append(ms, (e.GPUEnd()-e.GPUStart())*1e3/float64(per))
			}
		}
		return med(ms)
	}
	for _, s := range []struct {
		model string
		V, H  int
	}{{"1.5B", 151936, 1536}, {"7B", 152064, 3584}} {
		const M = 8
		aqb := d.NewBufferBytes(M * s.H)
		av := aqb.Int8s()[:M*s.H]
		for i := range av {
			av[i] = int8(rnd()%255) - 127
		}
		ah := make([]uint16, s.H*8) // permuted A^T, see mc3_lm_mma
		for m := range M {
			for sl := 0; sl < s.H/32; sl++ {
				for kb := range 4 {
					for kk := range 8 {
						k := sl*32 + 8*(kk>>1) + 2*kb + (kk & 1)
						ah[(sl*32+kb*8+kk)*8+m] = f32ToF16(float32(av[m*s.H+k]))
					}
				}
			}
		}
		aTp := NewBufferU16s(d, ah)
		ascs := make([]float32, M)
		for i := range ascs {
			ascs[i] = 0.01 + 0.001*float32(i)
		}
		asc := NewBufferFloats(d, ascs)
		uK, uN, uM := NewBufferU32(d, uint32(s.H)), NewBufferU32(d, uint32(s.V)), NewBufferU32(d, M)
		wb := d.NewBufferBytes(s.V * s.H)
		wv := wb.Int8s()[:s.V*s.H]
		for i := range wv {
			wv[i] = int8(rnd())
		}
		bs := make([]float32, s.V)
		for i := range bs {
			bs[i] = float32(rnd()%1000+1) * 1e-5
		}
		bsc := NewBufferFloats(d, bs)
		w8 := pipe("gemv_w8a8_coal")
		lmRow := func(e *Encoder, m int, out Buffer) {
			e.Dispatch(w8, s.V*32, 32, aqb.At(m*s.H), asc.At(4*m), wb, bsc, out.At(4*m*s.V), uK)
		}
		ref := d.NewBufferLen(M * s.V)
		e := cq.Begin()
		for m := range M {
			lmRow(e, m, ref)
		}
		e.End()
		refV := append([]float32(nil), ref.Floats()[:M*s.V]...)
		t1 := timeIt(4, func(e *Encoder, i int) { lmRow(e, 0, ref) })
		line := fmt.Sprintf("%-4s LM head V=%d H=%d production gemv_w8a8_coal %.3f ms, x8 = %.3f:", s.model, s.V, s.H, t1, 8*t1)
		for _, fb := range []int{1, 2, 4} {
			if s.V%(32*fb) != 0 {
				continue
			}
			p := pipe(fmt.Sprintf("mc3_lm_mma_fb%d", fb))
			tgs := s.V / (32 * fb)
			out := d.NewBufferLen(M * s.V)
			e := cq.Begin()
			e.DispatchTG(p, tgs*128, 128, 0, wb, bsc, aTp, asc, out, uK, uN, uM)
			e.End()
			if err := e.Err(); err != nil {
				t.Fatalf("lm dispatch: %v", err)
			}
			got := out.Floats()[:M*s.V]
			diff := 0
			for i := range got {
				if math.Float32bits(got[i]) != math.Float32bits(refV[i]) {
					diff++
				}
			}
			if diff != 0 {
				t.Errorf("%s LM head fb%d: %d of %d logits differ from production", s.model, fb, diff, len(got))
			}
			tm := timeIt(4, func(e *Encoder, i int) {
				e.DispatchTG(p, tgs*128, 128, 0, wb, bsc, aTp, asc, out, uK, uN, uM)
			})
			line += fmt.Sprintf("  LM fb%d %.3f (%.2fx 8 GEMVs, =%.1f GEMVs, %d differ)", fb, tm, 8*t1/tm, tm/t1, diff)
		}
		fmt.Fprintf(os.Stderr, "[mc3-mma8lm %6.1fs] %s\n", time.Since(t0).Seconds(), line)
		d.ReleaseAll()
	}
}
