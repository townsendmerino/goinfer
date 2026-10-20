// moe_rows.cu — grouped, exact expert GEMVs for the expert-major MoE prefill (cuda/moe_expert_major.go).
//
// The expert-major path runs every (row, rank) pair routed to an expert as its own three launches: the gate||up GEMV,
// glu_quant, the down GEMV. The kernels here take all of one expert's pairs in one launch each, so a prefill chunk issues
// a few hundred launches per layer instead of ~3 per pair, and an expert's weight words are loaded once for a tile of
// pairs rather than once per pair.
//
// Exactness is the contract: every output element is the value gemv_w4a8_moe (moe.cu, the audited moe.ptx) stores for the
// same weight row and activation row, bit for bit. A pair keeps its own accumulator across the whole of K, in the same lane
// order, with the same per-word rounding and the same shuffle tree; only the number of rows sharing one weight load
// changes. moe.ptx's float sequence per 64-word step is mul(s1, p1) -> fma(s0, p0, .) -> add(facc, .), and a single fma
// per word in the ragged tail; both are written here as explicit intrinsics, so no compiler choice remains
// (FMA-linted, kernel_fma_lint_test.go). The gates are in cuda/moe_rows_test.go.

#include <cuda_fp16.h>

#define MR_ROWS 8 // pairs (activation rows) one warp carries per weight row

extern "C" {

// gemv_w4a8_moe_rows: for n pairs p = 0..nPairs-1 and weight rows row = 0..N-1 of the expert in cache slot slotIdx[0]:
//   dst[(dIdx ? dIdx[p] : p) * N + row] = (sum over K of dequant(W[slot*N + row, k]) * A[aIdx ? aIdx[p] : p, k]) * aScale[arow]
// A is int8 packed four per int, 2*Kwords ints per row. One warp per weight row per MR_ROWS-pair tile; grid.y walks the tiles.
__global__ void gemv_w4a8_moe_rows(const unsigned int* __restrict__ W, const int* __restrict__ a,
                                   const __half* __restrict__ gs, const float* __restrict__ aScale,
                                   const unsigned int* __restrict__ slotIdx, int N, int Kwords, int Kgroups,
                                   const unsigned int* __restrict__ aIdx, const unsigned int* __restrict__ dIdx,
                                   int nPairs, float* __restrict__ dst) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (row >= N) return;
    int lane = threadIdx.x & 31;
    int p0 = blockIdx.y * MR_ROWS;
    int np = nPairs - p0;
    if (np > MR_ROWS) np = MR_ROWS;
    long wrow = (long)slotIdx[0] * N + row;
    const unsigned int* wr = W + wrow * Kwords;
    const __half* sr = gs + wrow * Kgroups;
    const int* ar[MR_ROWS];
    float facc[MR_ROWS];
    long arow[MR_ROWS];
#pragma unroll
    for (int j = 0; j < MR_ROWS; j++) {
        arow[j] = 0;
        ar[j] = a;
        facc[j] = 0.f;
        if (j < np) {
            arow[j] = aIdx ? (long)aIdx[p0 + j] : (long)(p0 + j);
            long aoff = (long)arow[j] * 2 * Kwords;
            ar[j] = a + aoff;
        }
    }
    int base = 0;
    for (; base + 64 <= Kwords; base += 64) {
        int wi = base + lane, wj = wi + 32;
        unsigned int w0 = wr[wi], w1 = wr[wj];
        float s0 = __half2float(sr[wi >> 2]), s1 = __half2float(sr[wj >> 2]);
        int l0 = (int)__vsub4(w0 & 0x0F0F0F0Fu, 0x08080808u), h0 = (int)__vsub4((w0 >> 4) & 0x0F0F0F0Fu, 0x08080808u);
        int l1 = (int)__vsub4(w1 & 0x0F0F0F0Fu, 0x08080808u), h1 = (int)__vsub4((w1 >> 4) & 0x0F0F0F0Fu, 0x08080808u);
#pragma unroll
        for (int j = 0; j < MR_ROWS; j++) {
            if (j < np) {
                int q0 = 0, q1 = 0;
                q0 = __dp4a(l0, ar[j][2 * wi], q0);
                q0 = __dp4a(h0, ar[j][2 * wi + 1], q0);
                q1 = __dp4a(l1, ar[j][2 * wj], q1);
                q1 = __dp4a(h1, ar[j][2 * wj + 1], q1);
                facc[j] = __fadd_rn(facc[j], __fmaf_rn(s0, (float)q0, __fmul_rn(s1, (float)q1)));
            }
        }
    }
    for (; base < Kwords; base += 32) {
        int wi = base + lane;
        if (wi < Kwords) { // ragged tail: Kwords need NOT be a multiple of 32
            unsigned int word = wr[wi];
            float s = __half2float(sr[wi >> 2]);
            int l = (int)__vsub4(word & 0x0F0F0F0Fu, 0x08080808u), h = (int)__vsub4((word >> 4) & 0x0F0F0F0Fu, 0x08080808u);
#pragma unroll
            for (int j = 0; j < MR_ROWS; j++) {
                if (j < np) {
                    int q = 0;
                    q = __dp4a(l, ar[j][2 * wi], q);
                    q = __dp4a(h, ar[j][2 * wi + 1], q);
                    facc[j] = __fmaf_rn((float)q, s, facc[j]);
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < MR_ROWS; j++) {
        if (j < np) {
            float f = facc[j];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) f = __fadd_rn(f, __shfl_down_sync(0xffffffffu, f, off));
            if (lane == 0) {
                long drow = dIdx ? (long)dIdx[p0 + j] : (long)(p0 + j);
                dst[drow * N + row] = __fmul_rn(f, aScale[arow[j]]);
            }
        }
    }
}

#define ACT_GELU_TANH 0
#define ACT_SILU      1

// glu_quant_rows: glu_quant (glue.cu) for M rows of a [M][gStride] gate||up buffer (gate at gOff, up at uOff, both I wide),
// one block per row: act(g)*u, then the symmetric int8 quantization. q is [M][I/4] ints, scale [M], dscr [M][I] floats. The
// body is glu_quant's, line for line; only the row offsets are new.
__global__ void glu_quant_rows(const float* __restrict__ gu, int gStride, int gOff, int uOff, int I, int act,
                               int* __restrict__ q, float* __restrict__ scale, float* __restrict__ dscr) {
    int m = blockIdx.x;
    const float* g = gu + (long)m * gStride;
    const float* u = g;
    float* dscratch = dscr + (long)m * I;
    int* qr = q + (long)m * (I / 4);
    extern __shared__ float red[];
    int t = threadIdx.x, nt = blockDim.x;
    float ma = 0.f;
    for (int k = t; k < I; k += nt) {
        float x = g[gOff + k], a;
        if (act == ACT_SILU) {
            a = x / (1.f + __expf(-x));
        } else {
            a = 0.5f * x * (1.f + tanhf(0.7978845608028654f * (__fmaf_rn(0.044715f, __fmul_rn(__fmul_rn(x, x), x), x))));
        }
        float d = a * u[uOff + k]; dscratch[k] = d; ma = fmaxf(ma, fabsf(d));
    }
    for (int o = 16; o > 0; o >>= 1) ma = fmaxf(ma, __shfl_down_sync(0xffffffff, ma, o));
    int lane = t & 31, warp = t >> 5, nWarps = (nt + 31) >> 5;
    if (lane == 0) red[warp] = ma;
    __syncthreads();
    if (warp == 0) {
        float mv = (lane < nWarps) ? red[lane] : 0.f;
        for (int o = 16; o > 0; o >>= 1) mv = fmaxf(mv, __shfl_down_sync(0xffffffff, mv, o));
        if (lane == 0) red[0] = mv;
    }
    __syncthreads();
    float sc = red[0] / 127.f; float inv = sc > 0.f ? 1.f / sc : 0.f;
    if (t == 0) scale[m] = sc;
    for (int j = t; j < I / 4; j += nt) {
        int packed = 0;
        for (int b = 0; b < 4; b++) { int v = __float2int_rn(dscratch[4 * j + b] * inv); v = max(-127, min(127, v)); packed |= (v & 0xff) << (8 * b); }
        qr[j] = packed;
    }
}

} // extern "C"
