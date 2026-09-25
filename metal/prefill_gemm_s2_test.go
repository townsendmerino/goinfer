//go:build darwin && goinfer_testhooks

package metal

import (
	"runtime"
	"testing"
)

// TestS2KernelsCompile compiles R16's prototype kernels and builds every pipeline — seconds, no checkpoint — so an
// MSL error fails here rather than 30 s into a TestMetalPrefillDecomp benchmark run.
func TestS2KernelsCompile(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer d.ReleaseAll()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	pool := NewARPool()
	defer pool.Drain()
	lib, err := d.CompileLibrary(gemmS2Kernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	for _, k := range []string{"gemm_w4f16_tg", "gemm_w4f16_tg2", "gemm_w4f16_tg3", "gemm_w4f16_tg4"} {
		if _, err := d.NewComputePipeline(lib, k); err != nil {
			t.Errorf("%s: %v", k, err)
		}
	}
}

// gemmS2Kernels is R16's first prototype (docs/tasks/red-october.md; design in
// docs/measurements/metal-prefill-gemm-s2-2026-09-25.md): the int4 prefill GEMM restructured in the shape of
// llama.cpp's classic kernel_mul_mm, which S1b measured at >= 2.96 TFLOPS on this machine against the current
// gemm_w4f16_store's ~0.75 under sustained load. TEST-ONLY until R16's band is met — nothing in production
// compiles it.
//
// Same inputs, outputs and epilogue as gemm_w4f16_store (A [M×K] f16 row-major, W [N×K/8] packed nibbles,
// WS [N×K/32] f16 scales, C [M×N] f16; mode 0 plain / 1 +bias / 2 +residual), so the benchmark can swap it in
// with the same buffer list. What changes is the structure:
//
//   - one threadgroup of 128 threads (4 simdgroups) owns a 64-feature × 32-token output tile; simdgroup sg
//     computes features 32*(sg&1).. and tokens 16*(sg>>1).. (8 accumulators);
//   - K advances 32 per slab — exactly one scale group — and BOTH operand tiles are staged into threadgroup
//     memory ONCE per slab and shared by the 4 simdgroups: weights dequantized cooperatively (16 per thread),
//     activations loaded cooperatively (8 halves per thread);
//   - the MMA loop reads threadgroup memory only; no device load waits inside it.
//
// It is meant to be BIT-IDENTICAL to gemm_w4f16_store: each output element still accumulates K in ordered
// 8-wide chunks into an f32 simdgroup_float8x8 via simdgroup_multiply_accumulate(acc, a, b, acc), from the same
// f16 activations and weights dequantized by the same expression, and the epilogue applies bias / residual the
// same way. The benchmark checks that rather than assuming it.
const gemmS2Kernels = `
#include <metal_stdlib>
using namespace metal;

kernel void gemm_w4f16_tg(device const half* A[[buffer(0)]], device const uint* W[[buffer(1)]],
    device const half* WS[[buffer(2)]], device half* C[[buffer(3)]],
    constant uint& M[[buffer(4)]], constant uint& N[[buffer(5)]], constant uint& K[[buffer(6)]],
    device const float* bias[[buffer(7)]], constant uint& mode[[buffer(8)]],
    uint2 tgp[[threadgroup_position_in_grid]], ushort tid[[thread_index_in_threadgroup]],
    ushort sg[[simdgroup_index_in_threadgroup]], ushort lane[[thread_index_in_simdgroup]]) {
    // weights:     [feature block fb 0..7][k block kb 0..3] of 8x8 [k][feature]
    // activations: [token block tb 0..3][k block kb 0..3] of 8x8 [token][k]
    threadgroup half sa[64*32];
    threadgroup half sb[32*32];
    threadgroup float cs[4*64];

    const uint n0 = tgp.x*64u, m0 = tgp.y*32u;
    const uint wpr = K/8u, gpr = K/32u;
    const ushort fh = sg & 1, th = sg >> 1;

    simdgroup_float8x8 acc[8];
    for (ushort i = 0; i < 8; i++) acc[i] = make_filled_simdgroup_matrix<float,8,8>(0.0f);

    // staging assignments
    const ushort fr = tid >> 1, kh = tid & 1;     // weight row 0..63, k half 0..1 (16 values = 2 words)
    const ushort fb = fr >> 3, nl = fr & 7;
    const uint ncol = n0 + fr;
    const ushort tr = tid >> 2, kq = tid & 3;     // token row 0..31, k quarter 0..3 (8 halves)
    const ushort tb = tr >> 3, rl = tr & 7;
    const uint mrow = m0 + tr;

    for (uint k0 = 0; k0 < K; k0 += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ncol < N) {
            float sc = float(WS[ncol*gpr + k0/32u]);
            for (ushort w = 0; w < 2; w++) {
                const ushort kb = kh*2 + w;
                uint word = W[ncol*wpr + k0/8u + kb];
                for (ushort kl = 0; kl < 8; kl++)
                    sa[(fb*4 + kb)*64 + kl*8 + nl] = half(float(int((word >> (4u*kl)) & 0xFu) - 8) * sc);
            }
        } else {
            for (ushort w = 0; w < 2; w++) {
                const ushort kb = kh*2 + w;
                for (ushort kl = 0; kl < 8; kl++) sa[(fb*4 + kb)*64 + kl*8 + nl] = 0.0h;
            }
        }
        threadgroup half* dst = sb + (tb*4 + kq)*64 + rl*8;
        if (mrow < M) {
            device const half* src = A + mrow*K + k0 + kq*8u;
            for (ushort i = 0; i < 8; i++) dst[i] = src[i];
        } else {
            for (ushort i = 0; i < 8; i++) dst[i] = 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort ik = 0; ik < 4; ik++) {
            simdgroup_half8x8 a[2], b[4];
            for (ushort t = 0; t < 2; t++) simdgroup_load(a[t], sb + ((th*2 + t)*4 + ik)*64, 8);
            for (ushort f = 0; f < 4; f++) simdgroup_load(b[f], sa + ((fh*4 + f)*4 + ik)*64, 8);
            for (ushort t = 0; t < 2; t++)
                for (ushort f = 0; f < 4; f++)
                    simdgroup_multiply_accumulate(acc[t*4 + f], a[t], b[f], acc[t*4 + f]);
        }
    }

    // epilogue: exactly gemm_w4f16_store's per-element bias / residual, through per-simdgroup scratch
    threadgroup float* myc = cs + sg*64;
    for (ushort t = 0; t < 2; t++) {
        for (ushort f = 0; f < 4; f++) {
            const uint mb = m0 + (th*2 + t)*8u, nb = n0 + (fh*4 + f)*8u;
            simdgroup_store(acc[t*4 + f], myc, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (ushort e = lane; e < 64; e += 32) {
                const uint m = mb + e/8, n = nb + e%8;
                if (m < M && n < N) {
                    float v = myc[e];
                    if (mode == 1u) v += bias[n];
                    if (mode == 2u) v += float(C[m*N + n]);
                    C[m*N + n] = half(v);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// gemm_w4f16_tg2 is prototype 2: prototype 1 plus the staging fixes adopted from the external review
// (docs/measurements/metal-prefill-gemm-s2-2026-09-25.md): each 8x8 weight block padded to a 72-half stride so one
// store instruction's 32 lanes spread over 16 threadgroup banks instead of 4; activations staged as two half4; the
// two weight words read as one uint2; the dequant store pointer hoisted. Layout, vector width and addressing only —
// operands, accumulation order and epilogue are unchanged, so it must stay bit-identical.
kernel void gemm_w4f16_tg2(device const half* A[[buffer(0)]], device const uint* W[[buffer(1)]],
    device const half* WS[[buffer(2)]], device half* C[[buffer(3)]],
    constant uint& M[[buffer(4)]], constant uint& N[[buffer(5)]], constant uint& K[[buffer(6)]],
    device const float* bias[[buffer(7)]], constant uint& mode[[buffer(8)]],
    uint2 tgp[[threadgroup_position_in_grid]], ushort tid[[thread_index_in_threadgroup]],
    ushort sg[[simdgroup_index_in_threadgroup]], ushort lane[[thread_index_in_simdgroup]]) {
    threadgroup half sa[32*72];
    threadgroup half sb[32*32];
    threadgroup float cs[4*64];

    const uint n0 = tgp.x*64u, m0 = tgp.y*32u;
    const uint wpr = K/8u, gpr = K/32u;
    const ushort fh = sg & 1, th = sg >> 1;

    simdgroup_float8x8 acc[8];
    for (ushort i = 0; i < 8; i++) acc[i] = make_filled_simdgroup_matrix<float,8,8>(0.0f);

    const ushort fr = tid >> 1, kh = tid & 1;
    const ushort fb = fr >> 3, nl = fr & 7;
    const uint ncol = n0 + fr;
    const ushort tr = tid >> 2, kq = tid & 3;
    const ushort tb = tr >> 3, rl = tr & 7;
    const uint mrow = m0 + tr;
    threadgroup half4* dst4 = (threadgroup half4*)(sb + (tb*4 + kq)*64 + rl*8);

    for (uint k0 = 0; k0 < K; k0 += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ncol < N) {
            float sc = float(WS[ncol*gpr + k0/32u]);
            uint2 wv = *(device const uint2*)(W + ncol*wpr + k0/8u + kh*2u);
            for (ushort w = 0; w < 2; w++) {
                uint word = wv[w];
                threadgroup half* p = sa + (fb*4 + kh*2 + w)*72 + nl;
                for (ushort kl = 0; kl < 8; kl++) {
                    *p = half(float(int((word >> (4u*kl)) & 0xFu) - 8) * sc);
                    p += 8;
                }
            }
        } else {
            for (ushort w = 0; w < 2; w++) {
                threadgroup half* p = sa + (fb*4 + kh*2 + w)*72 + nl;
                for (ushort kl = 0; kl < 8; kl++) { *p = 0.0h; p += 8; }
            }
        }
        if (mrow < M) {
            device const half4* s4 = (device const half4*)(A + mrow*K + k0 + kq*8u);
            dst4[0] = s4[0];
            dst4[1] = s4[1];
        } else {
            dst4[0] = half4(0.0h);
            dst4[1] = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort ik = 0; ik < 4; ik++) {
            simdgroup_half8x8 a[2], b[4];
            for (ushort t = 0; t < 2; t++) simdgroup_load(a[t], sb + ((th*2 + t)*4 + ik)*64, 8);
            for (ushort f = 0; f < 4; f++) simdgroup_load(b[f], sa + ((fh*4 + f)*4 + ik)*72, 8);
            for (ushort t = 0; t < 2; t++)
                for (ushort f = 0; f < 4; f++)
                    simdgroup_multiply_accumulate(acc[t*4 + f], a[t], b[f], acc[t*4 + f]);
        }
    }

    threadgroup float* myc = cs + sg*64;
    for (ushort t = 0; t < 2; t++) {
        for (ushort f = 0; f < 4; f++) {
            const uint mb = m0 + (th*2 + t)*8u, nb = n0 + (fh*4 + f)*8u;
            simdgroup_store(acc[t*4 + f], myc, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (ushort e = lane; e < 64; e += 32) {
                const uint m = mb + e/8, n = nb + e%8;
                if (m < M && n < N) {
                    float v = myc[e];
                    if (mode == 1u) v += bias[n];
                    if (mode == 2u) v += float(C[m*N + n]);
                    C[m*N + n] = half(v);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// Shared by prototypes 3 and 4: dequantize one thread's 16 weights (2 words, one scale) into a padded weight tile
// and store its activation half4s — prototype 2's staging, factored so the double-buffered kernel can run it
// AFTER its MMAs on values already fetched into registers.
inline void s2_stage_w(threadgroup half* sa, uint2 wv, float sc, bool wok, ushort fb, ushort kh, ushort nl) {
    for (ushort w = 0; w < 2; w++) {
        uint word = wv[w];
        threadgroup half* p = sa + (fb*4 + kh*2 + w)*72 + nl;
        for (ushort kl = 0; kl < 8; kl++) {
            *p = wok ? half(float(int((word >> (4u*kl)) & 0xFu) - 8) * sc) : 0.0h;
            p += 8;
        }
    }
}

inline void s2_epilogue(threadgroup float* myc, simdgroup_float8x8 acc, uint mb, uint nb, uint M, uint N,
    device half* C, device const float* bias, uint mode, ushort lane) {
    simdgroup_store(acc, myc, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort e = lane; e < 64; e += 32) {
        const uint m = mb + e/8, n = nb + e%8;
        if (m < M && n < N) {
            float v = myc[e];
            if (mode == 1u) v += bias[n];
            if (mode == 2u) v += float(C[m*N + n]);
            C[m*N + n] = half(v);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

// gemm_w4f16_tg3 is prototype 3: prototype 2 with REGISTER-PREFETCH DOUBLE BUFFERING. M1 has no async copy, so each
// iteration issues the NEXT slab's device loads into registers first, runs the current slab's MMAs from threadgroup
// memory while those loads are in flight, and only then dequantizes and stores the prefetched values into the other
// buffer — one threadgroup barrier per slab. (Staging first and computing second would stall on the loaded data
// before any MMA ran.) Same operands and accumulation order: bit-identical.
kernel void gemm_w4f16_tg3(device const half* A[[buffer(0)]], device const uint* W[[buffer(1)]],
    device const half* WS[[buffer(2)]], device half* C[[buffer(3)]],
    constant uint& M[[buffer(4)]], constant uint& N[[buffer(5)]], constant uint& K[[buffer(6)]],
    device const float* bias[[buffer(7)]], constant uint& mode[[buffer(8)]],
    uint2 tgp[[threadgroup_position_in_grid]], ushort tid[[thread_index_in_threadgroup]],
    ushort sg[[simdgroup_index_in_threadgroup]], ushort lane[[thread_index_in_simdgroup]]) {
    threadgroup half sa[2][32*72];
    threadgroup half sb[2][32*32];
    threadgroup float cs[4*64];

    const uint n0 = tgp.x*64u, m0 = tgp.y*32u;
    const uint wpr = K/8u, gpr = K/32u;
    const ushort fh = sg & 1, th = sg >> 1;
    simdgroup_float8x8 acc[8];
    for (ushort i = 0; i < 8; i++) acc[i] = make_filled_simdgroup_matrix<float,8,8>(0.0f);

    const ushort fr = tid >> 1, kh = tid & 1;
    const ushort fb = fr >> 3, nl = fr & 7;
    const uint ncol = n0 + fr;
    const bool wok = ncol < N;
    const ushort tr = tid >> 2, kq = tid & 3;
    const ushort tb = tr >> 3, rl = tr & 7;
    const uint mrow = m0 + tr;
    const bool aok = mrow < M;
    const ushort aoff = (tb*4 + kq)*64 + rl*8;
    device const uint* wrow = W + (wok ? ncol : 0u)*wpr + kh*2u;
    device const half* wsrow = WS + (wok ? ncol : 0u)*gpr;
    device const half* arow = A + (aok ? mrow : 0u)*K + kq*8u;

    uint2 wv = uint2(0u); float sc = 0.0f; half4 x0 = half4(0.0h), x1 = half4(0.0h);
    if (wok) { wv = *(device const uint2*)(wrow); sc = float(wsrow[0]); }
    if (aok) { device const half4* s4 = (device const half4*)(arow); x0 = s4[0]; x1 = s4[1]; }
    s2_stage_w(sa[0], wv, sc, wok, fb, kh, nl);
    { threadgroup half4* d4 = (threadgroup half4*)(sb[0] + aoff); d4[0] = x0; d4[1] = x1; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ushort cur = 0;
    for (uint k0 = 0; k0 < K; k0 += 32u) {
        const bool more = k0 + 32u < K;
        if (more) {  // issue the next slab's loads; nothing below depends on them until after the MMAs
            if (wok) { wv = *(device const uint2*)(wrow + (k0 + 32u)/8u); sc = float(wsrow[(k0 + 32u)/32u]); }
            if (aok) { device const half4* s4 = (device const half4*)(arow + k0 + 32u); x0 = s4[0]; x1 = s4[1]; }
        }
        for (ushort ik = 0; ik < 4; ik++) {
            simdgroup_half8x8 a[2], b[4];
            for (ushort t = 0; t < 2; t++) simdgroup_load(a[t], sb[cur] + ((th*2 + t)*4 + ik)*64, 8);
            for (ushort f = 0; f < 4; f++) simdgroup_load(b[f], sa[cur] + ((fh*4 + f)*4 + ik)*72, 8);
            for (ushort t = 0; t < 2; t++)
                for (ushort f = 0; f < 4; f++)
                    simdgroup_multiply_accumulate(acc[t*4 + f], a[t], b[f], acc[t*4 + f]);
        }
        if (more) {
            s2_stage_w(sa[cur ^ 1], wv, sc, wok, fb, kh, nl);
            threadgroup half4* d4 = (threadgroup half4*)(sb[cur ^ 1] + aoff);
            d4[0] = x0; d4[1] = x1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        cur ^= 1;
    }

    threadgroup float* myc = cs + sg*64;
    for (ushort t = 0; t < 2; t++)
        for (ushort f = 0; f < 4; f++)
            s2_epilogue(myc, acc[t*4 + f], m0 + (th*2 + t)*8u, n0 + (fh*4 + f)*8u, M, N, C, bias, mode, lane);
}

// gemm_w4f16_tg4 is prototype 4: prototype 2 with a LARGER SIMDGROUP TILE — each simdgroup computes 32 tokens × 32
// features (16 accumulators), so a threadgroup covers 64 features × 64 tokens and the MMA loop issues 8 matrix loads
// per 16 MMAs (prototype 2: 6 per 8). Activations: 64 tokens × 32 k per slab, 16 halves per thread. Grid is
// (ceil(N/64), ceil(M/64)). Same operands and accumulation order: bit-identical.
kernel void gemm_w4f16_tg4(device const half* A[[buffer(0)]], device const uint* W[[buffer(1)]],
    device const half* WS[[buffer(2)]], device half* C[[buffer(3)]],
    constant uint& M[[buffer(4)]], constant uint& N[[buffer(5)]], constant uint& K[[buffer(6)]],
    device const float* bias[[buffer(7)]], constant uint& mode[[buffer(8)]],
    uint2 tgp[[threadgroup_position_in_grid]], ushort tid[[thread_index_in_threadgroup]],
    ushort sg[[simdgroup_index_in_threadgroup]], ushort lane[[thread_index_in_simdgroup]]) {
    threadgroup half sa[32*72];
    threadgroup half sb[64*32];   // [token block tb 0..7][k block kb 0..3] of 8x8 [token][k]
    threadgroup float cs[4*64];

    const uint n0 = tgp.x*64u, m0 = tgp.y*64u;
    const uint wpr = K/8u, gpr = K/32u;
    const ushort fh = sg & 1, th = sg >> 1;   // features 32*fh.., tokens 32*th..
    simdgroup_float8x8 acc[16];
    for (ushort i = 0; i < 16; i++) acc[i] = make_filled_simdgroup_matrix<float,8,8>(0.0f);

    const ushort fr = tid >> 1, kh = tid & 1;
    const ushort fb = fr >> 3, nl = fr & 7;
    const uint ncol = n0 + fr;
    const bool wok = ncol < N;
    const ushort tr = tid >> 1, kh2 = tid & 1;  // token row 0..63, k half (two 8-wide k blocks)
    const ushort tb = tr >> 3, rl = tr & 7;
    const uint mrow = m0 + tr;
    const bool aok = mrow < M;

    for (uint k0 = 0; k0 < K; k0 += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint2 wv = uint2(0u); float sc = 0.0f;
        if (wok) { wv = *(device const uint2*)(W + ncol*wpr + k0/8u + kh*2u); sc = float(WS[ncol*gpr + k0/32u]); }
        s2_stage_w(sa, wv, sc, wok, fb, kh, nl);
        for (ushort j = 0; j < 2; j++) {
            const ushort kb = kh2*2 + j;
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
            simdgroup_half8x8 a[4], b[4];
            for (ushort t = 0; t < 4; t++) simdgroup_load(a[t], sb + ((th*4 + t)*4 + ik)*64, 8);
            for (ushort f = 0; f < 4; f++) simdgroup_load(b[f], sa + ((fh*4 + f)*4 + ik)*72, 8);
            for (ushort t = 0; t < 4; t++)
                for (ushort f = 0; f < 4; f++)
                    simdgroup_multiply_accumulate(acc[t*4 + f], a[t], b[f], acc[t*4 + f]);
        }
    }

    threadgroup float* myc = cs + sg*64;
    for (ushort t = 0; t < 4; t++)
        for (ushort f = 0; f < 4; f++)
            s2_epilogue(myc, acc[t*4 + f], m0 + (th*4 + t)*8u, n0 + (fh*4 + f)*8u, M, N, C, bias, mode, lane);
}
`
