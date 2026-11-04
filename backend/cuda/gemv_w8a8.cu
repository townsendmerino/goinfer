// W8A8 GEMV: dst[n] = (sum_k W[n,k]*a[k]) * aScale * wScale[n]
// W is [N,K] row-major int8 (K padded to mult-16); a is [K] int8. Turing __dp4a int8 dot.
// One warp per output row: 32 lanes stride K in int4 (4xint8) chunks, warp-reduce.
extern "C" __global__ void gemv_w8a8(
    const int* __restrict__ W,   // [N, K/4] as packed int (4 int8 per int)
    const int* __restrict__ a,   // [K/4] packed int8 activation
    const float* __restrict__ wScale, // [N]
    float aScale, int N, int Kdiv4, float* __restrict__ dst)
{
    int n = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (n >= N) return;
    const int* wrow = W + (long)n * Kdiv4;
    int acc = 0;
    for (int k = lane; k < Kdiv4; k += 32) acc = __dp4a(wrow[k], a[k], acc);
    // warp reduce
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) dst[n] = (float)acc * aScale * wScale[n];
}
