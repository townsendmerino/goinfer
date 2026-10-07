// tower_base.cu — the kernels the CUDA vision-tower base adds to aikit's gpu.ViT (S4 of docs/tasks/task-multimodal-support-2026-10.md).
//
// aikit's ViT module already has the f32 GEMMs (with bias / bias+residual epilogues), LayerNorm, RMSNorm, GELU, scaled attention and a NeoX RoPE; what Gemma 4's
// tower and the grid towers need beyond it is below: the patch embedder's two position tables, ClippableLinear's clamps, Gemma 4's axial 2-D RoPE, an elementwise
// multiply (gelu(gate) * up) and a scale (the grid towers' attention scale applied to q). Ported from metal/gemma4_vision.go's and metal/grid_vision.go's kernels.
//
// Every float multiply and add is an explicit intrinsic (__fmul_rn / __fadd_rn / __fsub_rn): the same rule as the repo's other kernels (kernel_fma_lint_test.go),
// so the compiler cannot contract a multiply-add and the rounding is the CPU's (Go does not fuse on amd64).

// x[i, d] += X[pos[i].x, d] + Y[pos[i].y, d]: the patch embedder's two position tables. pos is [n/H, 2] int32; n = rows * H.
extern "C" __global__ void tower_pos_add(float* __restrict__ x, const float* __restrict__ X, const float* __restrict__ Y,
                                         const int* __restrict__ pos, int H, long n)
{
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    long i = g / H;
    int d = (int)(g % H);
    float px = X[(long)pos[2 * i] * H + d];
    float py = Y[(long)pos[2 * i + 1] * H + d];
    x[g] = __fadd_rn(x[g], __fadd_rn(px, py));
}

__device__ __forceinline__ float tower_clampf(float v, float lo, float hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

// A ClippableLinear's output clamp, in place.
extern "C" __global__ void tower_clamp(float* __restrict__ x, long n, float lo, float hi)
{
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g < n) x[g] = tower_clampf(x[g], lo, hi);
}

// Its input clamp, into a copy (q, k and v read the same normed input under different bounds).
extern "C" __global__ void tower_clamp_copy(const float* __restrict__ x, float* __restrict__ out, long n, float lo, float hi)
{
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g < n) out[g] = tower_clampf(x[g], lo, hi);
}

// Axial 2-D RoPE in place over x [T, heads, hd] from the tables cos/sin [T, hd] (aikit's Gemma4RopeTables): each half of a head is rotated rotate-half
// within itself (pairs d, d + hd/4), the first half by the patch's x and the second by its y. n = T * heads * hd/2.
extern "C" __global__ void tower_rope_axial(float* __restrict__ x, const float* __restrict__ cs, const float* __restrict__ sn,
                                            int heads, int hd, long n)
{
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    int half_ = hd / 2, q = hd / 4;
    int p = (int)(g % half_);
    long th = g / half_;
    long t = th / heads;
    int base = (p / q) * half_, d = p % q;
    float* r = x + th * hd + base;
    const float* c = cs + t * hd + base;
    const float* s = sn + t * hd + base;
    float a = r[d], b = r[d + q];
    r[d] = __fsub_rn(__fmul_rn(a, c[d]), __fmul_rn(b, s[d]));
    r[d + q] = __fadd_rn(__fmul_rn(b, c[d + q]), __fmul_rn(a, s[d + q]));
}

// x[i] *= u[i] (gelu(gate) * up, after aikit's gelu_tanh has run in place over the gate).
extern "C" __global__ void tower_mul(float* __restrict__ x, const float* __restrict__ u, long n)
{
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g < n) x[g] = __fmul_rn(x[g], u[g]);
}

// x[i] *= s (the grid towers apply the attention scale to q after RoPE, as the CPU tower does).
extern "C" __global__ void tower_scale(float* __restrict__ x, long n, float s)
{
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g < n) x[g] = __fmul_rn(x[g], s);
}

// NeoX rotate-half RoPE in place over x [T, heads, hd] from the tables cos/sin [T, hd] (aikit's Qwen3 / GLM-OCR RopeTables, rows cat(f, f)): pairs (d, d + hd/2)
// across the whole head, as aikit's rope_qk does on its fused qkv, here on a separate q or k buffer (GLM-OCR norms q and k per head before RoPE, so they cannot
// stay fused). n = T * heads * hd/2.
extern "C" __global__ void tower_rope_half(float* __restrict__ x, const float* __restrict__ cs, const float* __restrict__ sn,
                                           int heads, int hd, long n)
{
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    int half_ = hd / 2;
    int d = (int)(g % half_);
    long th = g / half_;
    long t = th / heads;
    float* r = x + th * hd;
    const float* c = cs + t * hd;
    const float* s = sn + t * hd;
    float a = r[d], b = r[d + half_];
    r[d] = __fsub_rn(__fmul_rn(a, c[d]), __fmul_rn(b, s[d]));
    r[d + half_] = __fadd_rn(__fmul_rn(b, c[d + half_]), __fmul_rn(a, s[d + half_]));
}
