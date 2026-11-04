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

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------
// tower_attn_hd{64,72,80}: S17 lever A (docs/tasks/task-multimodal-support-2026-10.md, "S17's CUDA lever A"): a fused, query-tiled, online-softmax float32 attention for the towers,
// replacing aikit's AttentionTiled (which the step 0 profile measured at 0.12-0.14 TFLOPS, 92-95% of every tower's time). Non-causal, one segment per call (the caller offsets the
// pointers), q/k/v/out all [T, nH*HD] with the heads in place, softmax(scale * q k^T) v.
//
// One block = one head x 64 queries, 256 threads as 16 (ty) x 16 (tx). The S tile (64 queries x 32 keys) is register-blocked 4 rows x 2 keys per thread; the output tile is 4 rows x
// NC columns (column c = tx + 16 j) per thread. K tiles are staged transposed ([d][key], so the S loop reads a float2 of keys) and V tiles as they lie ([key][d]), both in one shared
// buffer (K, then V); P goes through a padded [query][key] shared tile. The softmax is online (running max m, running sum l, the output rescaled when the max moves) and works in base 2:
// sl2 = scale * log2(e) is applied to the raw score and exp2f used. Every multiply-add is an explicit intrinsic (the file's rule). `dbg` plants kernel defects for the tests: 1 skips the
// rescale when the max moves, 2 lets the key mask run one past the segment (the extra key scores 0 and carries V = 0, so it only skews the denominator, and is invisible over thousands of keys),
// 3 reads each V tile one key late (the K/V pairing off by one).
#define TA_BQ 64
#define TA_BK 32
#define TA_NT 256
#define TA_NEG_INF __int_as_float(0xff800000)

template <int HD>
__device__ __forceinline__ void tower_attn_body(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v, float* __restrict__ out,
                                                int T, int nH, float sl2, int dbg)
{
    constexpr int NC = (HD + 15) / 16;
    __shared__ float Qs[HD * TA_BQ];
    __shared__ float KV[HD * TA_BK];
    __shared__ float Ps[TA_BQ * (TA_BK + 1)];
    const long stride = (long)nH * HD;
    const int h = blockIdx.y, q0 = blockIdx.x * TA_BQ;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const float* qh = q + (long)h * HD;
    const float* kh = k + (long)h * HD;
    const float* vh = v + (long)h * HD;
    float* oh = out + (long)h * HD;

    // Q tile, transposed to [d][query] (lanes run along the query so the shared write is conflict-free; the global read is strided but its lines are reused across d from L1)
    for (int idx = tid; idx < TA_BQ * HD; idx += TA_NT) {
        const int r = idx % TA_BQ, d = idx / TA_BQ;
        const int row = q0 + r;
        Qs[d * TA_BQ + r] = row < T ? qh[(long)row * stride + d] : 0.f;
    }
    float m[4], l[4], acc[4][NC];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        m[r] = TA_NEG_INF;
        l[r] = 0.f;
#pragma unroll
        for (int j = 0; j < NC; ++j) acc[r][j] = 0.f;
    }
    const int nTiles = (T + TA_BK - 1) / TA_BK;
    const int kvalid = dbg == 2 ? T + 1 : T;
    for (int kt = 0; kt < nTiles; ++kt) {
        const int k0 = kt * TA_BK;
        for (int idx = tid; idx < TA_BK * HD; idx += TA_NT) {   // K tile -> KV[d][key]
            const int kk = idx % TA_BK, d = idx / TA_BK;
            const int key = k0 + kk;
            KV[d * TA_BK + kk] = key < T ? kh[(long)key * stride + d] : 0.f;
        }
        __syncthreads();
        float s[4][2];
#pragma unroll
        for (int r = 0; r < 4; ++r) { s[r][0] = 0.f; s[r][1] = 0.f; }
#pragma unroll
        for (int d = 0; d < HD; ++d) {
            const float4 qv = *reinterpret_cast<const float4*>(&Qs[d * TA_BQ + ty * 4]);
            const float2 kv = *reinterpret_cast<const float2*>(&KV[d * TA_BK + tx * 2]);
            s[0][0] = __fmaf_rn(qv.x, kv.x, s[0][0]); s[0][1] = __fmaf_rn(qv.x, kv.y, s[0][1]);
            s[1][0] = __fmaf_rn(qv.y, kv.x, s[1][0]); s[1][1] = __fmaf_rn(qv.y, kv.y, s[1][1]);
            s[2][0] = __fmaf_rn(qv.z, kv.x, s[2][0]); s[2][1] = __fmaf_rn(qv.z, kv.y, s[2][1]);
            s[3][0] = __fmaf_rn(qv.w, kv.x, s[3][0]); s[3][1] = __fmaf_rn(qv.w, kv.y, s[3][1]);
        }
        __syncthreads();
        for (int idx = tid; idx < TA_BK * HD; idx += TA_NT) {   // V tile -> KV[key][d] (its load overlaps the softmax below)
            const int d = idx % HD, kk = idx / HD;
            const int key = k0 + kk + (dbg == 3 ? 1 : 0);   // dbg 3 plants a V tile read one key late
            KV[kk * HD + d] = key < T ? vh[(long)key * stride + d] : 0.f;
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const float a0 = (k0 + tx * 2 < kvalid) ? __fmul_rn(s[r][0], sl2) : TA_NEG_INF;
            const float a1 = (k0 + tx * 2 + 1 < kvalid) ? __fmul_rn(s[r][1], sl2) : TA_NEG_INF;
            float mt = fmaxf(a0, a1);
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, o));
            const float mn = fmaxf(m[r], mt);
            const float alpha = dbg == 1 ? 1.f : exp2f(m[r] - mn);
            const float p0 = exp2f(a0 - mn), p1 = exp2f(a1 - mn);
            float ps = __fadd_rn(p0, p1);
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) ps = __fadd_rn(ps, __shfl_xor_sync(0xffffffffu, ps, o));
            l[r] = __fmaf_rn(l[r], alpha, ps);
            m[r] = mn;
#pragma unroll
            for (int j = 0; j < NC; ++j) acc[r][j] = __fmul_rn(acc[r][j], alpha);
            Ps[(ty * 4 + r) * (TA_BK + 1) + tx * 2] = p0;
            Ps[(ty * 4 + r) * (TA_BK + 1) + tx * 2 + 1] = p1;
        }
        __syncthreads();
#pragma unroll 4
        for (int kk = 0; kk < TA_BK; ++kk) {
            const float p0 = Ps[(ty * 4 + 0) * (TA_BK + 1) + kk];
            const float p1 = Ps[(ty * 4 + 1) * (TA_BK + 1) + kk];
            const float p2 = Ps[(ty * 4 + 2) * (TA_BK + 1) + kk];
            const float p3 = Ps[(ty * 4 + 3) * (TA_BK + 1) + kk];
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                const int c = tx + 16 * j;
                const float vv = (HD % 16 == 0 || c < HD) ? KV[kk * HD + c] : 0.f;
                acc[0][j] = __fmaf_rn(p0, vv, acc[0][j]);
                acc[1][j] = __fmaf_rn(p1, vv, acc[1][j]);
                acc[2][j] = __fmaf_rn(p2, vv, acc[2][j]);
                acc[3][j] = __fmaf_rn(p3, vv, acc[3][j]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = q0 + ty * 4 + r;
        if (row >= T) continue;
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const int c = tx + 16 * j;
            if (HD % 16 == 0 || c < HD) oh[(long)row * stride + c] = __fdiv_rn(acc[r][j], l[r]);
        }
    }
}

extern "C" __global__ void __launch_bounds__(TA_NT, 2) tower_attn_hd64(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v, float* __restrict__ out,
                                                                    int T, int nH, float sl2, int dbg) { tower_attn_body<64>(q, k, v, out, T, nH, sl2, dbg); }
extern "C" __global__ void __launch_bounds__(TA_NT, 2) tower_attn_hd72(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v, float* __restrict__ out,
                                                                    int T, int nH, float sl2, int dbg) { tower_attn_body<72>(q, k, v, out, T, nH, sl2, dbg); }
extern "C" __global__ void __launch_bounds__(TA_NT, 2) tower_attn_hd80(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v, float* __restrict__ out,
                                                                    int T, int nH, float sl2, int dbg) { tower_attn_body<80>(q, k, v, out, T, nH, sl2, dbg); }
