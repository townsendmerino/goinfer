// nongated_act.cu — the non-gated MLP's activation for Nemotron-H: relu²(x) = max(x,0)², then the symmetric per-vector int8 quantization the down
// projection reads. relu2_quant is glu_quant (glue.cu) without the gate: the same output contract (packed int8 in q, the scale in *scale, the
// activated values in dscratch), so the down GEMV that follows is the one the gated MLP already uses. A new module with its own PTX: no ordinal is
// added to glu_quant, so no existing PTX moves. Mirrors decoder/rmsnorm.go relu2 and decoder's non-gated MLP.
extern "C" {

__global__ void relu2_quant(const float* __restrict__ u, int uOff, int I,
                            int* __restrict__ q, float* __restrict__ scale, float* __restrict__ dscratch) {
    extern __shared__ float red[];
    int t = threadIdx.x, nt = blockDim.x;
    float ma = 0.f;
    for (int k = t; k < I; k += nt) {
        float x = fmaxf(u[uOff + k], 0.f);
        float d = __fmul_rn(x, x);
        dscratch[k] = d;
        ma = fmaxf(ma, fabsf(d));
    }
    // max is exact and order-independent, so the warp-shuffle form is the same value as any other order (see glu_quant)
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
    if (t == 0) *scale = sc;
    for (int j = t; j < I / 4; j += nt) {
        int packed = 0;
        for (int b = 0; b < 4; b++) { int v = __float2int_rn(__fmul_rn(dscratch[4 * j + b], inv)); v = max(-127, min(127, v)); packed |= (v & 0xff) << (8 * b); }
        q[j] = packed;
    }
}

} // extern "C"
