// lora.cu — compute-time LoRA (G3, docs/task-gpu-paths-2026-09.md): the two-GEMV low-rank
// delta y[o] += scale * sum_r B[o,r] * (sum_k A[r,k] * dequant(aq,ascale)[k]), applied AFTER a
// base projection's matmul on the SAME quantized activation the base projection consumed
// (matching decoder/lora.go's applyLoRA CPU reference, and metal/lora.go's / gpu/lora.go's own
// design on the other two backends). Own file/module: cuda/testdata/REGEN.md's rule for a NEW
// kernel is a new .cu/.ptx pair, so this never forces a regen of the audited glue.ptx.
//
// Split into two kernels for their different natural parallelism (same reasoning as the other
// two backends): the down-project reduces over K (potentially thousands of elements) per rank,
// the up-project has no reduction at all (one MAC chain of length R per output row).

extern "C" {

// lora_delta_down: block b reduces ranks b, b+gridDim.x, ... — launched with one block per rank
// (loraDownCfg in lora.go, audit P-11). It used to be ONE block looping over every rank, which
// pulled all R rows of A through a single SM: +11 ms on an 8.9 ms qwen3-4b token. A GridX=1 launch
// is still that old kernel exactly, and each block of a GridX=R launch runs the same per-rank
// arithmetic (same per-thread partial order, same tree reduction), so the two shapes give
// BIT-IDENTICAL output; TestLoRADownGridBitIdenticalCUDA holds that. aq is the SAME packed-int8
// activation (4 int8 per int, low byte first — rmsnorm_quant's own packing, glue.cu) every GEMV
// kernel in this backend already consumes.
__global__ void lora_delta_down(const int* __restrict__ aq, const float* __restrict__ ascale,
                                 const float* __restrict__ amat, float* __restrict__ tout,
                                 int K, int R) {
    extern __shared__ float red[];
    int t = threadIdx.x, nt = blockDim.x;
    float sc = *ascale;
    for (int r = blockIdx.x; r < R; r += gridDim.x) {
        const float* Ar = amat + (long)r * K;
        float part = 0.f;
        for (int k = t; k < K; k += nt) {
            int word = aq[k >> 2];
            int byte = (word >> (8 * (k & 3))) & 0xff;
            signed char sq = (signed char)byte;
            float dq = float(sq) * sc; // dequantize: a plain multiply, not an accumulate
            part = __fmaf_rn(Ar[k], dq, part);
        }
        red[t] = part;
        __syncthreads();
        for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] += red[t + o]; __syncthreads(); }
        if (t == 0) tout[r] = red[0];
        __syncthreads();
    }
}

// lora_delta_up: one thread per output row. Grid may over-provision past Outn (g1cfg rounds up
// to the block size), hence the bounds check — the same convention gemv_w8a8's `if (n >= N)
// return;` uses.
__global__ void lora_delta_up(const float* __restrict__ bmat, const float* __restrict__ tin,
                               float* __restrict__ dst, int R, int Outn, float scale) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= Outn) return;
    const float* Br = bmat + (long)row * R;
    float acc = 0.f;
    for (int r = 0; r < R; r++) acc = __fmaf_rn(Br[r], tin[r], acc);
    dst[row] = __fmaf_rn(scale, acc, dst[row]);
}

}
