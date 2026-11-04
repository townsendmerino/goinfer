// vit.cu — the kernel set for a native, cgo-free ViT forward (docs/task-native-gpu.md,
// Phase 3). These are aikit's own compute, built on the cuda.go device layer, and they
// exist to run vision.ResidentEncoder on NVIDIA the way goinfer's WebGPU backend runs it
// on any GPU — but cgo-free.
//
// BIT-IDENTITY-EXEMPT: no bit-identity contract exists for this tower, so the bare float
// MACs below are free to contract. The gate is gpu/visioncuda/encoder_test.go, which
// requires cosine >= 1-1e-6 against the pure-Go CPU tower — a TOLERANCE, deliberately:
// that tower accumulates LayerNorm/softmax/GELU in float64 while these kernels work in
// f32/int32, so bit-equality is impossible BY CONSTRUCTION, not merely unachieved. The
// int8 dots are exact, so the residual divergence is float reassociation only, which is
// why the bar is tight rather than loose. Nothing downstream consumes these outputs
// bit-for-bit. If that ever changes — if some path must match this kernel exactly — flip
// this to BIT-IDENTITY-CONTRACT and expect real work: the reductions here are unprotected.
//
// Related, and NOT the same claim: gpu/metal_vit.go pins its threadgroup width
// (ViTBlock/LNBLOCK) as a bit-identity dependency. That is about the Metal kernel matching
// ITSELF across configurations — a perf sweep of the width silently moves its bits and the
// tolerance gate would not notice — not about matching the CPU tower. gpu/cuda_vit.go's
// ViTBlock carries the same structural coupling but documents it only as an array size.
//
// PARITY IS THE POINT, SO THE MATH MIRRORS THE CPU TOWER EXACTLY
// --------------------------------------------------------------
// Every kernel here is written against vision/encoder.go's arithmetic, not against
// "standard" formulations, because the gate is cosine ≈ 1.0 against that CPU forward:
//   - layernorm accumulates mean and variance in DOUBLE and applies
//     (x-mean)*rsqrt(var+eps)*w + b, matching layerNormInto. eps is added to the
//     VARIANCE, not to the deviation.
//   - gelu is the TANH approximation with c = sqrt(2/pi), evaluated in double —
//     geluTanh. Not erf-gelu; they differ enough to move a cosine.
//   - softmax subtracts the row max, exponentiates and sums in DOUBLE — softmaxRow.
//   - attention is BIDIRECTIONAL (no causal mask) and scales scores by 1/sqrt(hd)
//     BEFORE the softmax.
// Turing runs fp64 at 1/32 rate, so the double accumulations are deliberately confined
// to the per-row reductions (a vanishing fraction of a tower's FLOPs) where they buy
// parity; every bulk matmul stays in int32/f32.
//
// Conventions from gpu/cuda.go: scalars by value, each kernel bounds-checks its own
// global index (a CUDA launch rounds up to whole blocks), and weight matrices are
// ROW-MAJOR [N,K] — i.e. the B-transposed layout linalg.MatmulBT uses, so a GEMM here
// is C[m,n] = dot(A[m,:], B[n,:]).

#define LNBLOCK 256

// quant_rows: per-row symmetric int8 quantization, byte-identical to linalg's
// quantizeRowInt8 — scale = maxAbs/127, q = round(x/scale) clamped to +-127, and an
// all-zero row yields scale 0. One block per row. roundf is round-half-away-from-zero,
// which is what Go's math.Round does; rintf (half-to-even) would NOT match.
extern "C" __global__ void quant_rows(
    const float* __restrict__ x, signed char* __restrict__ q, float* __restrict__ s,
    int rows, int dim)
{
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* xr = x + (long)r * dim;
    __shared__ float sm[LNBLOCK];
    float m = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float a = fabsf(xr[i]);
        if (a > m) m = a;
    }
    sm[threadIdx.x] = m;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off && sm[threadIdx.x + off] > sm[threadIdx.x]) sm[threadIdx.x] = sm[threadIdx.x + off];
        __syncthreads();
    }
    float maxAbs = sm[0];
    float scale = maxAbs / 127.f;
    if (threadIdx.x == 0) s[r] = (maxAbs == 0.f) ? 0.f : scale;
    signed char* qr = q + (long)r * dim;
    if (maxAbs == 0.f) {
        for (int i = threadIdx.x; i < dim; i += blockDim.x) qr[i] = 0;
        return;
    }
    float inv = 1.f / scale;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = roundf(xr[i] * inv);
        if (v > 127.f) v = 127.f;
        else if (v < -127.f) v = -127.f;
        qr[i] = (signed char)v;
    }
}

// gemm_w8a8: C[M,N] = (A_q[M,K] . B_q[N,K]) * aScale[M] * bScale[N]. The int8 dot is
// EXACT integer arithmetic, so this contributes no error of its own — the only rounding
// is the final rescale. One thread per output element; correctness-first, not tiled.
extern "C" __global__ void gemm_w8a8(
    const signed char* __restrict__ A, const float* __restrict__ aScale,
    const signed char* __restrict__ B, const float* __restrict__ bScale,
    float* __restrict__ C, int M, int N, int K)
{
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long)M * N) return;
    int m = (int)(g / N), n = (int)(g % N);
    const signed char* ar = A + (long)m * K;
    const signed char* br = B + (long)n * K;
    int acc = 0;
    for (int k = 0; k < K; k++) acc += (int)ar[k] * (int)br[k];
    C[g] = (float)acc * aScale[m] * bScale[n];
}

// gemm_f32: C[M,N] = A[M,K] . B[N,K] in f32 (B row-major = B-transposed), for the
// patch-embed projection, whose weights are f32 rather than quantized.
extern "C" __global__ void gemm_f32(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K)
{
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long)M * N) return;
    int m = (int)(g / N), n = (int)(g % N);
    const float* ar = A + (long)m * K;
    const float* br = B + (long)n * K;
    float acc = 0.f;
    for (int k = 0; k < K; k++) acc += ar[k] * br[k];
    C[g] = acc;
}

// add_bias: x[r,d] += bias[d] — a per-row broadcast (projection biases).
extern "C" __global__ void add_bias(
    float* __restrict__ x, const float* __restrict__ bias, int rows, int dim)
{
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long)rows * dim) return;
    x[g] += bias[g % dim];
}

// add_vec: x[i] += v[i] — elementwise, for the positional embedding and the two
// residual adds per layer.
extern "C" __global__ void add_vec(
    float* __restrict__ x, const float* __restrict__ v, int n)
{
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    x[g] += v[g];
}

// layernorm: standard mean/variance LayerNorm (NOT RMS), one block per row, double
// accumulation to match layerNormInto. Two passes over the row: mean, then variance.
extern "C" __global__ void layernorm(
    const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
    float* __restrict__ out, int rows, int dim, float eps)
{
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* xr = x + (long)r * dim;
    float* dst = out + (long)r * dim;
    __shared__ double sm[LNBLOCK];

    double acc = 0.0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) acc += (double)xr[i];
    sm[threadIdx.x] = acc;
    __syncthreads();
    // Cross-thread SUM reduction: the order is fixed by blockDim.x (== host ViTBlock
    // == LNBLOCK), and f64 addition is not associative either, so this width is a
    // BIT-IDENTITY dependency, not a tuning knob. See cuda_vit.go's ViTBlock.
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) sm[threadIdx.x] += sm[threadIdx.x + off];
        __syncthreads();
    }
    double mean = sm[0] / (double)dim;
    __syncthreads();

    acc = 0.0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        double d = (double)xr[i] - mean;
        acc += d * d;
    }
    sm[threadIdx.x] = acc;
    __syncthreads();
    // Cross-thread SUM reduction: the order is fixed by blockDim.x (== host ViTBlock
    // == LNBLOCK), and f64 addition is not associative either, so this width is a
    // BIT-IDENTITY dependency, not a tuning knob. See cuda_vit.go's ViTBlock.
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) sm[threadIdx.x] += sm[threadIdx.x + off];
        __syncthreads();
    }
    double var = sm[0] / (double)dim;
    double inv = 1.0 / sqrt(var + (double)eps);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        dst[i] = (float)(((double)xr[i] - mean) * inv) * w[i] + b[i];
    }
}

// gelu_tanh: the tanh approximation, in double, matching geluTanh exactly.
extern "C" __global__ void gelu_tanh(float* __restrict__ x, int n)
{
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    const double c = 0.7978845608028654; // sqrt(2/pi)
    double v = (double)x[g];
    x[g] = (float)(0.5 * v * (1.0 + tanh(c * (v + 0.044715 * v * v * v))));
}

// ATTN_KTILE / ATTN_MAXHD size the attention kernel's shared K stage (audit
// M-14). 16 rows x 128 dims x 4 B = 8 KB, on top of the np*4 B dynamic score
// row and the reduction scratch — comfortably inside a 48 KB block budget even
// at np=4096. ATTN_MAXHD is a ceiling, not an assumption: a larger head dim
// takes the unstaged path.
#define ATTN_KTILE 16
#define ATTN_MAXHD 128

// ATTN_STAGE_SHARED_MAX caps the shared memory a block may use before staging
// is declined. The score row is DYNAMIC shared of np*4 bytes, so at large np the
// extra 8 KB tile costs a block of occupancy per SM and the staging stops
// paying. Measured on an RTX 2070 SUPER, staged vs unstaged:
//
//     np=729   -31.4%     np=2048  -30.0%
//     np=1024  -33.8%     np=3072   -9.0%
//     np=1024  -36.6% (qwen, hd=80) np=4096  +11.1%   <- inverts
//
// 20 KB admits everything up to np=3072 (12 KB + 8 KB) and excludes np=4096
// (16 KB + 8 KB). The threshold is in BYTES rather than patches because the
// mechanism is the shared budget, not the patch count — a different hd or a
// different score dtype moves the crossover and this moves with it.
#define ATTN_STAGE_SHARED_MAX 20480

// attention: bidirectional multi-head self-attention over np patches, ONE BLOCK per
// (head, query). The block stages that query's np scores in dynamic shared memory,
// softmaxes them (max-subtract, double-accumulated sum — softmaxRow), then each thread
// accumulates one output dimension. No causal mask: every query attends to every patch.
//
// q/k/v are [np, nH*hd] with head h at column offset h*hd, so no gather/transpose pass
// is needed — the kernel indexes the heads in place.
//
// Shared memory required: np * sizeof(float), sized by the caller.
extern "C" __global__ void attention(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ out, int np, int nH, int hd, float scale)
{
    int blk = blockIdx.x;
    if (blk >= nH * np) return;
    int h = blk / np, i = blk % np;
    int hidden = nH * hd, off = h * hd;
    extern __shared__ float sc[];

    const float* qi = q + (long)i * hidden + off;

    // scores[j] = dot(q_i, k_j) * scale, with K STAGED THROUGH SHARED MEMORY
    // (audit M-14).
    //
    // The direct form — thread j walking k + j*hidden + off — has adjacent
    // threads reading addresses `hidden` floats apart, so every lane of a warp
    // touches a DIFFERENT cache line and uses hd of the 32 floats it pulls in.
    // Staging a KTILE-row tile cooperatively makes the global reads contiguous
    // (consecutive t -> consecutive d within a row) and the dot then reads from
    // shared. The query vector is staged too: every thread in the block reads
    // the same qi[d], which is a broadcast from shared instead of np/blockDim
    // redundant global loads.
    //
    // BIT-IDENTICAL: each dot still accumulates d ASCENDING over the same
    // values, and no reduction order changes. Only where the operands are read
    // from moves. The softmax max/sum trees below are untouched — their width
    // is a bit-identity dependency, as this file notes, not a tuning knob.
    //
    // hd > ATTN_MAXHD, or a shared budget that staging would push past, falls
    // back to the direct form rather than silently truncating or regressing; no
    // shipped tower is near the hd ceiling (so400m 72, Qwen 80).
    __shared__ float qs[ATTN_MAXHD];
    __shared__ float ks[ATTN_KTILE][ATTN_MAXHD];
    if (hd <= ATTN_MAXHD && (long)np * 4 + ATTN_KTILE * ATTN_MAXHD * 4 <= ATTN_STAGE_SHARED_MAX) {
        for (int d = threadIdx.x; d < hd; d += blockDim.x) qs[d] = qi[d];
        __syncthreads();
        for (int j0 = 0; j0 < np; j0 += ATTN_KTILE) {
            int cnt = min(ATTN_KTILE, np - j0);
            for (int t = threadIdx.x; t < cnt * hd; t += blockDim.x) {
                int jj = t / hd, dd = t - jj * hd;
                ks[jj][dd] = k[(long)(j0 + jj) * hidden + off + dd];
            }
            __syncthreads();
            for (int jj = threadIdx.x; jj < cnt; jj += blockDim.x) {
                float acc = 0.f;
                for (int d = 0; d < hd; d++) acc += qs[d] * ks[jj][d];
                sc[j0 + jj] = acc * scale;
            }
            __syncthreads();
        }
    } else {
        for (int j = threadIdx.x; j < np; j += blockDim.x) {
            const float* kj = k + (long)j * hidden + off;
            float acc = 0.f;
            for (int d = 0; d < hd; d++) acc += qi[d] * kj[d];
            sc[j] = acc * scale;
        }
        __syncthreads();
    }

    // row max, then exp/sum in double — both reduced through shared memory.
    __shared__ float smax[LNBLOCK];
    __shared__ double ssum[LNBLOCK];
    float m = -3.402823466e+38f;
    for (int j = threadIdx.x; j < np; j += blockDim.x) if (sc[j] > m) m = sc[j];
    smax[threadIdx.x] = m;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o && smax[threadIdx.x + o] > smax[threadIdx.x]) smax[threadIdx.x] = smax[threadIdx.x + o];
        __syncthreads();
    }
    float mx = smax[0];
    __syncthreads();

    double su = 0.0;
    for (int j = threadIdx.x; j < np; j += blockDim.x) {
        double e = exp((double)sc[j] - (double)mx);
        sc[j] = (float)e;
        su += e;
    }
    ssum[threadIdx.x] = su;
    __syncthreads();
    // Cross-thread SUM reduction: the order is fixed by blockDim.x (== host ViTBlock
    // == LNBLOCK), and f64 addition is not associative either, so this width is a
    // BIT-IDENTITY dependency, not a tuning knob. See cuda_vit.go's ViTBlock.
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) ssum[threadIdx.x] += ssum[threadIdx.x + o];
        __syncthreads();
    }
    double inv = 1.0 / ssum[0];
    __syncthreads();
    for (int j = threadIdx.x; j < np; j += blockDim.x) sc[j] = (float)((double)sc[j] * inv);
    __syncthreads();

    // out[i, off+d] = sum_j softmax[j] * v[j, off+d]
    float* oi = out + (long)i * hidden + off;
    if (hd <= ATTN_MAXHD && (long)np * 4 + ATTN_KTILE * ATTN_MAXHD * 4 <= ATTN_STAGE_SHARED_MAX) {
        float acc = 0.f;
        for (int j0 = 0; j0 < np; j0 += ATTN_KTILE) {
            int cnt = min(ATTN_KTILE, np - j0);
            for (int t = threadIdx.x; t < cnt * hd; t += blockDim.x) {
                int jj = t / hd, dd = t - jj * hd;
                ks[jj][dd] = v[(long)(j0 + jj) * hidden + off + dd];
            }
            __syncthreads();
            if (threadIdx.x < hd) {
                for (int jj = 0; jj < cnt; jj++) {
                    acc += sc[j0 + jj] * ks[jj][threadIdx.x];
                }
            }
            __syncthreads();
        }
        if (threadIdx.x < hd) {
            oi[threadIdx.x] = acc;
        }
    } else {
        for (int d = threadIdx.x; d < hd; d += blockDim.x) {
            float acc = 0.f;
            for (int j = 0; j < np; j++) acc += sc[j] * v[(long)j * hidden + off + d];
            oi[d] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// attention_tiled — query-tiled, online-softmax (FlashAttention-style)
// bidirectional multi-head self-attention.
//
// Dynamic shared memory: 0 bytes. Static shared memory: 16 KB (Ks and Vs).
// Staging K and V in shared tiles across AT_QTILE=32 queries amortizes global
// memory reads by up to 32x.
// ---------------------------------------------------------------------------
#define AT_MAXHD 128
#define AT_KTILE 16
#define AT_QTILE 32

extern "C" __global__ void attention_tiled(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ out,
    int np, int nH, int hd, float scale)
{
    __shared__ float Ks[AT_KTILE][AT_MAXHD];
    __shared__ float Vs[AT_KTILE][AT_MAXHD];

    int tilesPerHead = (np + AT_QTILE - 1) / AT_QTILE;
    int blk = blockIdx.x;
    int h = blk / tilesPerHead;
    int tileIdx = blk % tilesPerHead;
    int qStart = tileIdx * AT_QTILE;
    int tid = threadIdx.x;
    int tgsz = blockDim.x; // AT_QTILE = 32
    int i = qStart + tid;
    int hidden = nH * hd, off = h * hd;
    bool active = (i < np);

    float m = -3.402823466e+38f;
    float l = 0.0f;
    float acc[AT_MAXHD];
    #pragma unroll
    for (int d = 0; d < AT_MAXHD; d++) acc[d] = 0.0f;

    for (int k0 = 0; k0 < np; k0 += AT_KTILE) {
        int chunk = min(AT_KTILE, np - k0);
        for (int idx = tid; idx < chunk * hd; idx += tgsz) {
            int kk = idx / hd, d = idx % hd;
            Ks[kk][d] = k[(long)(k0 + kk) * hidden + off + d];
            Vs[kk][d] = v[(long)(k0 + kk) * hidden + off + d];
        }
        __syncthreads();

        if (active) {
            const float* qi = q + (long)i * hidden + off;
            float s[AT_KTILE];
            float blockMax = -3.402823466e+38f;
            #pragma unroll
            for (int kk = 0; kk < AT_KTILE; kk++) {
                if (kk < chunk) {
                    float dot = 0.0f;
                    for (int d = 0; d < hd; d++) dot += qi[d] * Ks[kk][d];
                    float sv = dot * scale;
                    s[kk] = sv;
                    if (sv > blockMax) blockMax = sv;
                }
            }
            float newMax = fmaxf(m, blockMax);
            float corr = expf(m - newMax);
            l *= corr;
            for (int d = 0; d < hd; d++) acc[d] *= corr;
            float blockSum = 0.0f;
            for (int kk = 0; kk < chunk; kk++) {
                float e = expf(s[kk] - newMax);
                blockSum += e;
                for (int d = 0; d < hd; d++) acc[d] += e * Vs[kk][d];
            }
            l += blockSum;
            m = newMax;
        }
        __syncthreads();
    }

    if (active) {
        float* oi = out + (long)i * hidden + off;
        float inv = 1.0f / l;
        for (int d = 0; d < hd; d++) oi[d] = acc[d] * inv;
    }
}

// ---------------------------------------------------------------------------
// Qwen2.5-VL ViT additions (Phase 3). The SigLIP set above covers the shared ops;
// these are the five Qwen needs on top, and they are written against
// vision/qwen_encoder.go's arithmetic for the same parity reason.
//
// Note gelu_erf below is a DIFFERENT function from gelu_tanh above. Qwen's patch
// merger uses nn.GELU()'s exact erf form while its SigLIP counterpart uses the tanh
// approximation; they differ by ~5e-4, which is far more than the parity bar. Both
// are shipped, and callers must pick deliberately.
// ---------------------------------------------------------------------------

// rmsnorm: weight-only RMS normalization (no mean subtraction, no bias) —
// x * rsqrt(mean(x^2) + eps) * w, double-accumulated to match rmsNorm. One block
// per row.
extern "C" __global__ void rmsnorm(
    const float* __restrict__ x, const float* __restrict__ w,
    float* __restrict__ out, int rows, int dim, float eps)
{
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* xr = x + (long)r * dim;
    float* dst = out + (long)r * dim;
    __shared__ double sm[LNBLOCK];
    double acc = 0.0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        double v = (double)xr[i];
        acc += v * v;
    }
    sm[threadIdx.x] = acc;
    __syncthreads();
    // Cross-thread SUM reduction: the order is fixed by blockDim.x (== host ViTBlock
    // == LNBLOCK), and f64 addition is not associative either, so this width is a
    // BIT-IDENTITY dependency, not a tuning knob. See cuda_vit.go's ViTBlock.
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) sm[threadIdx.x] += sm[threadIdx.x + o];
        __syncthreads();
    }
    double inv = 1.0 / sqrt(sm[0] / (double)dim + (double)eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        dst[i] = (float)((double)xr[i] * inv) * w[i];
    }
}

// rope_qk: NeoX rotate_half 2D rotary applied IN PLACE to the q and k thirds of a
// fused qkv buffer [seq, 3*hidden] — row layout [3, nH, hd], so q is at offset 0 and
// k at offset hidden. v is untouched. Splitting qkv into three buffers first would be
// a pure copy; indexing the thirds here avoids it.
//
// Each thread owns one (patch, head, d) pair with d < hd/2 and rotates BOTH q and k,
// reading x and y before writing either — the pairs are disjoint, so this is
// race-free and bit-identical to the CPU's in-place pairwise form.
extern "C" __global__ void rope_qk(
    float* __restrict__ qkv, const float* __restrict__ cos, const float* __restrict__ sin,
    int seq, int nH, int hd)
{
    int half = hd / 2;
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)seq * nH * half;
    if (g >= total) return;
    int d = (int)(g % half);
    long rest = g / half;
    int head = (int)(rest % nH);
    int i = (int)(rest / nH);
    int hidden = nH * hd;

    const float* co = cos + (long)i * hd;
    const float* si = sin + (long)i * hd;
    long qoff = (long)i * 3 * hidden + (long)head * hd;
    long koff = qoff + hidden;

    float x = qkv[qoff + d], y = qkv[qoff + d + half];
    qkv[qoff + d] = x * co[d] - y * si[d];
    qkv[qoff + d + half] = y * co[d + half] + x * si[d + half];

    x = qkv[koff + d]; y = qkv[koff + d + half];
    qkv[koff + d] = x * co[d] - y * si[d];
    qkv[koff + d + half] = y * co[d + half] + x * si[d + half];
}

// attention_seg: bidirectional MHA restricted to each patch's cu_seqlens SEGMENT —
// a window for most blocks, a whole image for the fullatt blocks. The caller passes
// per-patch segment bounds (segStart/segEnd) rather than the cu_seqlens prefix array,
// so the kernel needs no search.
//
// Reads q/k/v from the FUSED qkv buffer [seq, 3*hidden] at offsets 0/hidden/2*hidden.
// One block per (head, query); the segment's scores stage in dynamic shared memory,
// so the caller sizes it to maxSegment*4 bytes.
extern "C" __global__ void attention_seg(
    const float* __restrict__ qkv, float* __restrict__ out,
    const int* __restrict__ segStart, const int* __restrict__ segEnd,
    int seq, int nH, int hd, float scale)
{
    int blk = blockIdx.x;
    if (blk >= nH * seq) return;
    int h = blk / seq, i = blk % seq;
    int hidden = nH * hd, off = h * hd;
    int s0 = segStart[i], s1 = segEnd[i], n = s1 - s0;
    extern __shared__ float sc[];

    const float* qi = qkv + (long)i * 3 * hidden + off;
    __shared__ float qs[ATTN_MAXHD];
    if (hd <= ATTN_MAXHD) {
        for (int d = threadIdx.x; d < hd; d += blockDim.x) qs[d] = qi[d];
        __syncthreads();
        for (int t = threadIdx.x; t < n; t += blockDim.x) {
            const float* kj = qkv + (long)(s0 + t) * 3 * hidden + hidden + off;
            float acc = 0.f;
            for (int d = 0; d < hd; d++) acc += qs[d] * kj[d];
            sc[t] = acc * scale;
        }
    } else {
        for (int t = threadIdx.x; t < n; t += blockDim.x) {
            const float* kj = qkv + (long)(s0 + t) * 3 * hidden + hidden + off;
            float acc = 0.f;
            for (int d = 0; d < hd; d++) acc += qi[d] * kj[d];
            sc[t] = acc * scale;
        }
    }
    __syncthreads();

    __shared__ float smax[LNBLOCK];
    __shared__ double ssum[LNBLOCK];
    float m = -3.402823466e+38f;
    for (int t = threadIdx.x; t < n; t += blockDim.x) if (sc[t] > m) m = sc[t];
    smax[threadIdx.x] = m;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o && smax[threadIdx.x + o] > smax[threadIdx.x]) smax[threadIdx.x] = smax[threadIdx.x + o];
        __syncthreads();
    }
    float mx = smax[0];
    __syncthreads();

    double su = 0.0;
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
        double e = exp((double)sc[t] - (double)mx);
        sc[t] = (float)e;
        su += e;
    }
    ssum[threadIdx.x] = su;
    __syncthreads();
    // Cross-thread SUM reduction: the order is fixed by blockDim.x (== host ViTBlock
    // == LNBLOCK), and f64 addition is not associative either, so this width is a
    // BIT-IDENTITY dependency, not a tuning knob. See cuda_vit.go's ViTBlock.
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) ssum[threadIdx.x] += ssum[threadIdx.x + o];
        __syncthreads();
    }
    double inv = 1.0 / ssum[0];
    __syncthreads();
    for (int t = threadIdx.x; t < n; t += blockDim.x) sc[t] = (float)((double)sc[t] * inv);
    __syncthreads();

    float* oi = out + (long)i * hidden + off;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t < n; t++) acc += sc[t] * qkv[(long)(s0 + t) * 3 * hidden + 2 * hidden + off + d];
        oi[d] = acc;
    }
}

// attention_seg_tiled: segment-aware query-tiled online-softmax bidirectional MHA.
//
// Unlike attention_seg which stages an entire segment score row in dynamic shared
// memory (which exceeds the 48 KB hardware limit on large images), this uses a
// constant 16 KB static shared memory allocation (Ks and Vs tiles of AT_KTILE=16),
// lifting the shared memory limit completely and amortizing K/V loads across
// AT_QTILE=32 queries by up to 32x.
extern "C" __global__ void attention_seg_tiled(
    const float* __restrict__ qkv, float* __restrict__ out,
    const int* __restrict__ segStart, const int* __restrict__ segEnd,
    int seq, int nH, int hd, float scale)
{
    __shared__ float Ks[AT_KTILE][AT_MAXHD];
    __shared__ float Vs[AT_KTILE][AT_MAXHD];

    int tilesPerHead = (seq + AT_QTILE - 1) / AT_QTILE;
    int blk = blockIdx.x;
    int h = blk / tilesPerHead;
    int tileIdx = blk % tilesPerHead;
    int qStart = tileIdx * AT_QTILE;
    int tid = threadIdx.x;
    int tgsz = blockDim.x; // AT_QTILE = 32
    int i = qStart + tid;
    int hidden = nH * hd, off = h * hd;
    bool active = (i < seq);

    int s0 = active ? segStart[i] : 0;
    int s1 = active ? segEnd[i] : 0;

    float m = -3.402823466e+38f;
    float l = 0.0f;
    float acc[AT_MAXHD];
    #pragma unroll
    for (int d = 0; d < AT_MAXHD; d++) acc[d] = 0.0f;

    for (int k0 = 0; k0 < seq; k0 += AT_KTILE) {
        int chunk = min(AT_KTILE, seq - k0);
        for (int idx = tid; idx < chunk * hd; idx += tgsz) {
            int kk = idx / hd, d = idx % hd;
            int kj_idx = k0 + kk;
            Ks[kk][d] = qkv[(long)kj_idx * 3 * hidden + hidden + off + d];
            Vs[kk][d] = qkv[(long)kj_idx * 3 * hidden + 2 * hidden + off + d];
        }
        __syncthreads();

        if (active) {
            int chunkStart = k0;
            int chunkEnd = k0 + chunk;
            int validStart = max(s0, chunkStart);
            int validEnd = min(s1, chunkEnd);

            if (validStart < validEnd) {
                const float* qi = qkv + (long)i * 3 * hidden + off;
                float s[AT_KTILE];
                float blockMax = -3.402823466e+38f;
                #pragma unroll
                for (int kk = 0; kk < AT_KTILE; kk++) {
                    if (kk < chunk) {
                        int kIdx = k0 + kk;
                        if (kIdx >= s0 && kIdx < s1) {
                            float dot = 0.0f;
                            for (int d = 0; d < hd; d++) dot += qi[d] * Ks[kk][d];
                            float sv = dot * scale;
                            s[kk] = sv;
                            if (sv > blockMax) blockMax = sv;
                        }
                    }
                }
                if (blockMax > -3.402823466e+38f) {
                    float newMax = fmaxf(m, blockMax);
                    float corr = expf(m - newMax);
                    l *= corr;
                    for (int d = 0; d < hd; d++) acc[d] *= corr;
                    float blockSum = 0.0f;
                    for (int kk = 0; kk < chunk; kk++) {
                        int kIdx = k0 + kk;
                        if (kIdx >= s0 && kIdx < s1) {
                            float e = expf(s[kk] - newMax);
                            blockSum += e;
                            for (int d = 0; d < hd; d++) acc[d] += e * Vs[kk][d];
                        }
                    }
                    l += blockSum;
                    m = newMax;
                }
            }
        }
        __syncthreads();
    }

    if (active) {
        float* oi = out + (long)i * hidden + off;
        float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
        for (int d = 0; d < hd; d++) oi[d] = acc[d] * inv;
    }
}

// silu_mul: gate = silu(gate) * up, the gated-MLP activation. silu in double to match
// the CPU's float64 exp.
extern "C" __global__ void silu_mul(
    float* __restrict__ gate, const float* __restrict__ up, int n)
{
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    double v = (double)gate[g];
    gate[g] = (float)(v / (1.0 + exp(-v))) * up[g];
}

// gelu_erf: the EXACT (erf) GELU — nn.GELU()'s default, used by Qwen's patch merger.
// Distinct from gelu_tanh above; see the header note.
extern "C" __global__ void gelu_erf(float* __restrict__ x, int n)
{
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    double v = (double)x[g];
    x[g] = (float)(0.5 * v * (1.0 + erf(v / 1.4142135623730951)));
}

// ---------------------------------------------------------------------------
// TILED GEMMs (throughput). The gemm_w8a8 / gemm_f32 above are correctness-first:
// one thread per output element, each re-reading a whole row of A and of B from
// global memory. At ViT shapes that is the dominant cost — every A row is re-read N
// times and every B row M times.
//
// These tile through shared memory instead: a TILE×TILE block cooperatively stages
// one A-tile and one B-tile per K-chunk, so each element is read from global memory
// once per tile rather than once per output. Same arithmetic, same operand order
// within a thread's accumulation.
//
// The W8A8 variant is expected to be BIT-IDENTICAL to the untiled one, and its test
// asserts exactly that: the accumulator is int32 and integer addition is associative,
// so re-chunking the K loop cannot change the result. That is a much sharper gate than
// a tolerance, and it is available here precisely because the dot is integer.
// gemm_f32_tiled cannot make that claim — f32 addition reassociates — so its gate is a
// tight relative bound instead.
//
// Launch with a 2-D grid: grid = (ceil(N/TILE), ceil(M/TILE)), block = (TILE, TILE).
// ---------------------------------------------------------------------------

#define TILE 16

extern "C" __global__ void gemm_w8a8_tiled(
    const signed char* __restrict__ A, const float* __restrict__ aScale,
    const signed char* __restrict__ B, const float* __restrict__ bScale,
    float* __restrict__ C, int M, int N, int K)
{
    __shared__ signed char As[TILE][TILE];
    __shared__ signed char Bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int m = blockIdx.y * TILE + ty;
    int n = blockIdx.x * TILE + tx;
    int bRow = blockIdx.x * TILE + ty; // the B row this thread STAGES (not the one it uses)
    int acc = 0;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        int k = k0 + tx;
        As[ty][tx] = (m < M && k < K) ? A[(long)m * K + k] : (signed char)0;
        Bs[ty][tx] = (bRow < N && k < K) ? B[(long)bRow * K + k] : (signed char)0;
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < TILE; kk++) acc += (int)As[ty][kk] * (int)Bs[tx][kk];
        __syncthreads();
    }
    if (m < M && n < N) C[(long)m * N + n] = (float)acc * aScale[m] * bScale[n];
}

// ---------------------------------------------------------------------------
// gemm_w8a8_reg — the int8 twin of gemm_f32_reg (audit M-12).
//
// gemm_w8a8_tiled above computes ONE output per thread with byte-granular
// shared staging: its PTX is 32 ld.shared.u8 per 4 dp4a. That is LSU-bound, not
// MAC-bound, and the roofline campaign measured the same shape at 360 GMAC/s
// against this card's 4876 GMAC/s dp4a roof — 7% — then RETIRED it from the ANN
// path (anncuda/gemv_w8a8.cu). The retirement never reached the ViT path, where
// every int8 projection still ran it: ~300 GMAC per so400m image, ~0.83 s.
//
// Two changes, both from gemm_f32_reg's playbook:
//
//   PACKED-WORD STAGING. Shared memory holds int (four int8), not signed char,
//   so one ld.shared.u32 feeds a whole dp4a instead of four ld.shared.u8 feeding
//   a quarter of one. A is reinterpreted as int* with K/4 words per row, which
//   is why K%4 is required; device allocations are 256-byte aligned so the row
//   starts are too.
//
//   4x4 REGISTER BLOCKING. Each thread owns a 4x4 micro-tile, so the 8 shared
//   words it loads per k-word feed 16 dp4a rather than 1. Per 16-element k-step
//   that is 32 ld.shared.u32 for 64 dp4a — 0.5 loads per dp4a against the tiled
//   kernel's 8, a 16x better ratio, which is the whole point since the kernel is
//   load-bound.
//
// K steps 16 elements (IBK=4 words) rather than gemm_f32_reg's 16 floats, so the
// alignment requirement stays K%16==0 — SigLIP-so400m's inter of 4304 is a
// multiple of 16 but NOT of 64, and a K%64 kernel would have missed the MLP
// projections that are most of the work.
//
// The epilogue is byte-for-byte gemm_w8a8_tiled's: an exact int32 accumulator
// scaled by aScale[m]*bScale[n] in the same order, so the two kernels agree
// exactly and GEMMW8A8Plan can route between them freely.
//
// Requires only K%16==0 (IBK words of four int8). M and N are free: edge tiles
// stage zeros and skip their stores, so the inner loop is still unguarded.
// GEMMW8A8Plan routes a K%16!=0 shape to gemm_w8a8_tiled.
// ---------------------------------------------------------------------------

#define IBM 64
#define IBN 64
#define IBK 4  // words of four int8 = 16 K-elements per step
#define IBTM 4 // per-thread micro-tile rows (mirrors gemm_f32_reg's RTM)
#define IBTN 4 // per-thread micro-tile cols (mirrors gemm_f32_reg's RTN)

extern "C" __global__ void gemm_w8a8_reg(
    const signed char* __restrict__ A, const float* __restrict__ aScale,
    const signed char* __restrict__ B, const float* __restrict__ bScale,
    float* __restrict__ C, int M, int N, int K)
{
    __shared__ int As[IBK][IBM + 1];
    __shared__ int Bs[IBK][IBN + 1];
    const int* __restrict__ Aw = (const int*)A;
    const int* __restrict__ Bw = (const int*)B;
    int kw = K >> 2; // words per row

    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0..255
    int m0 = blockIdx.y * IBM, n0 = blockIdx.x * IBN;
    int lr = tid >> 2, lc = tid & 3; // 64 rows x 4 k-words per pass

    int acc[IBTM][IBTN];
    #pragma unroll
    for (int i = 0; i < IBTM; i++)
        #pragma unroll
        for (int j = 0; j < IBTN; j++) acc[i][j] = 0;

    for (int k0 = 0; k0 < kw; k0 += IBK) {
        // Edge tiles are handled by STAGING ZEROS, not by bounds-checking the
        // inner loop. An out-of-range row contributes zero to its accumulator
        // and its output is simply not stored, so the 64 dp4a per k-step stay
        // unguarded. Two predicated global loads per thread per k-step is
        // nothing against that, and it is what lets this kernel take N=4304 —
        // SigLIP-so400m's intermediate width, a multiple of 16 but not of 64,
        // and the shape the MLP projections use, which is most of a ViT's work.
        As[lc][lr] = (m0 + lr < M) ? Aw[(long)(m0 + lr) * kw + k0 + lc] : 0;
        Bs[lc][lr] = (n0 + lr < N) ? Bw[(long)(n0 + lr) * kw + k0 + lc] : 0;
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < IBK; kk++) {
            int a[IBTM], b[IBTN];
            #pragma unroll
            for (int i = 0; i < IBTM; i++) a[i] = As[kk][threadIdx.y * IBTM + i];
            #pragma unroll
            for (int j = 0; j < IBTN; j++) b[j] = Bs[kk][threadIdx.x * IBTN + j];
            #pragma unroll
            for (int i = 0; i < IBTM; i++)
                #pragma unroll
                for (int j = 0; j < IBTN; j++) acc[i][j] = __dp4a(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < IBTM; i++) {
        int m = m0 + threadIdx.y * IBTM + i;
        if (m >= M) continue;
        #pragma unroll
        for (int j = 0; j < IBTN; j++) {
            int n = n0 + threadIdx.x * IBTN + j;
            if (n < N) C[(long)m * N + n] = (float)acc[i][j] * aScale[m] * bScale[n];
        }
    }
}

// gemm_w8a8_bias — fused GEMM + bias addition.
// Writes C[m, n] = (acc * aScale * bScale) + bias[n], eliminating separate add_bias kernel.
extern "C" __global__ void gemm_w8a8_bias(
    const signed char* __restrict__ A, const float* __restrict__ aScale,
    const signed char* __restrict__ B, const float* __restrict__ bScale,
    const float* __restrict__ bias,
    float* __restrict__ C, int M, int N, int K)
{
    __shared__ int As[IBK][IBM + 1];
    __shared__ int Bs[IBK][IBN + 1];
    const int* __restrict__ Aw = (const int*)A;
    const int* __restrict__ Bw = (const int*)B;
    int kw = K >> 2;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int m0 = blockIdx.y * IBM, n0 = blockIdx.x * IBN;
    int lr = tid >> 2, lc = tid & 3;

    int acc[IBTM][IBTN];
    #pragma unroll
    for (int i = 0; i < IBTM; i++)
        #pragma unroll
        for (int j = 0; j < IBTN; j++) acc[i][j] = 0;

    for (int k0 = 0; k0 < kw; k0 += IBK) {
        As[lc][lr] = (m0 + lr < M) ? Aw[(long)(m0 + lr) * kw + k0 + lc] : 0;
        Bs[lc][lr] = (n0 + lr < N) ? Bw[(long)(n0 + lr) * kw + k0 + lc] : 0;
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < IBK; kk++) {
            int a[IBTM], b[IBTN];
            #pragma unroll
            for (int i = 0; i < IBTM; i++) a[i] = As[kk][threadIdx.y * IBTM + i];
            #pragma unroll
            for (int j = 0; j < IBTN; j++) b[j] = Bs[kk][threadIdx.x * IBTN + j];
            #pragma unroll
            for (int i = 0; i < IBTM; i++)
                #pragma unroll
                for (int j = 0; j < IBTN; j++) acc[i][j] = __dp4a(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < IBTM; i++) {
        int m = m0 + threadIdx.y * IBTM + i;
        if (m >= M) continue;
        #pragma unroll
        for (int j = 0; j < IBTN; j++) {
            int n = n0 + threadIdx.x * IBTN + j;
            if (n < N) C[(long)m * N + n] = (float)acc[i][j] * aScale[m] * bScale[n] + bias[n];
        }
    }
}

// gemm_w8a8_bias_add — fused GEMM + bias + residual addition.
// Writes residual[m, n] += (acc * aScale * bScale) + bias[n], eliminating both add_bias and add_vec.
extern "C" __global__ void gemm_w8a8_bias_add(
    const signed char* __restrict__ A, const float* __restrict__ aScale,
    const signed char* __restrict__ B, const float* __restrict__ bScale,
    const float* __restrict__ bias,
    float* __restrict__ residual, int M, int N, int K)
{
    __shared__ int As[IBK][IBM + 1];
    __shared__ int Bs[IBK][IBN + 1];
    const int* __restrict__ Aw = (const int*)A;
    const int* __restrict__ Bw = (const int*)B;
    int kw = K >> 2;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int m0 = blockIdx.y * IBM, n0 = blockIdx.x * IBN;
    int lr = tid >> 2, lc = tid & 3;

    int acc[IBTM][IBTN];
    #pragma unroll
    for (int i = 0; i < IBTM; i++)
        #pragma unroll
        for (int j = 0; j < IBTN; j++) acc[i][j] = 0;

    for (int k0 = 0; k0 < kw; k0 += IBK) {
        As[lc][lr] = (m0 + lr < M) ? Aw[(long)(m0 + lr) * kw + k0 + lc] : 0;
        Bs[lc][lr] = (n0 + lr < N) ? Bw[(long)(n0 + lr) * kw + k0 + lc] : 0;
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < IBK; kk++) {
            int a[IBTM], b[IBTN];
            #pragma unroll
            for (int i = 0; i < IBTM; i++) a[i] = As[kk][threadIdx.y * IBTM + i];
            #pragma unroll
            for (int j = 0; j < IBTN; j++) b[j] = Bs[kk][threadIdx.x * IBTN + j];
            #pragma unroll
            for (int i = 0; i < IBTM; i++)
                #pragma unroll
                for (int j = 0; j < IBTN; j++) acc[i][j] = __dp4a(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < IBTM; i++) {
        int m = m0 + threadIdx.y * IBTM + i;
        if (m >= M) continue;
        #pragma unroll
        for (int j = 0; j < IBTN; j++) {
            int n = n0 + threadIdx.x * IBTN + j;
            if (n < N) {
                long idx = (long)m * N + n;
                residual[idx] += (float)acc[i][j] * aScale[m] * bScale[n] + bias[n];
            }
        }
    }
}

extern "C" __global__ void gemm_f32_tiled(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int m = blockIdx.y * TILE + ty;
    int n = blockIdx.x * TILE + tx;
    int bRow = blockIdx.x * TILE + ty;
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        int k = k0 + tx;
        As[ty][tx] = (m < M && k < K) ? A[(long)m * K + k] : 0.f;
        Bs[ty][tx] = (bRow < N && k < K) ? B[(long)bRow * K + k] : 0.f;
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < TILE; kk++) acc += As[ty][kk] * Bs[tx][kk];
        __syncthreads();
    }
    if (m < M && n < N) C[(long)m * N + n] = acc;
}

// ---------------------------------------------------------------------------
// gemm_f32_reg — the register-blocked f32 GEMM (Phase 4 throughput).
//
// WHY THIS IS NOT A PORT OF METAL'S gemm_f32_sg_big. Metal's win came from dropping
// shared staging entirely and loading A/B fragments straight from device memory. That
// does not transfer, and the reason is the operand layout rather than the platform:
// A is [M,K] and B is [N,K], both row-major, so BOTH operands are contiguous along K.
// A thread owning output column n reads B[n*K + k] — consecutive threads are then K
// floats apart, so a staging-free load is fully uncoalesced. The existing tiled kernel
// is coalesced precisely BECAUSE it stages (consecutive threads read consecutive k).
//
// So this keeps shared staging to buy coalescing, and takes its speed from REGISTER
// BLOCKING instead: each thread computes a 4x4 micro-tile, so one shared load feeds 16
// FMAs instead of 1. Same idea as Metal's ("do more per thread"), opposite conclusion
// about staging, for a layout reason — measured, not assumed.
//
// Geometry: 64x64 output tile per block, K stepped 16 at a time, 256 threads (16x16),
// 4x4 outputs each. ALIGNED ONLY (M%64==0, N%64==0, K%16==0) so the inner loops carry
// no bounds checks; GEMMF32Plan routes everything else to gemm_f32_tiled.
//
// The shared arrays are padded by one float. Without the pad, As is [16][64] and a
// half-warp writing As[lc][lr] hits stride 64 — 64 % 32 == 0, so all 16 lanes land in
// ONE bank (a 16-way conflict). Padding to 65 makes the stride odd and the writes
// conflict-free. This is invisible in correctness and worth ~a third of the throughput.
// ---------------------------------------------------------------------------

#define RBM 64
#define RBN 64
#define RBK 16
#define RTM 4
#define RTN 4

extern "C" __global__ void gemm_f32_reg(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K)
{
    __shared__ float As[RBK][RBM + 1];
    __shared__ float Bs[RBK][RBN + 1];
    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0..255
    int m0 = blockIdx.y * RBM, n0 = blockIdx.x * RBN;
    int lr = tid / RBK, lc = tid % RBK; // 16 rows x 16 k-cols per pass

    float acc[RTM][RTN];
    #pragma unroll
    for (int i = 0; i < RTM; i++)
        #pragma unroll
        for (int j = 0; j < RTN; j++) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < K; k0 += RBK) {
        // Coalesced global reads: consecutive tid → consecutive k.
        #pragma unroll
        for (int i = 0; i < RBM; i += 16) As[lc][lr + i] = A[(long)(m0 + lr + i) * K + k0 + lc];
        #pragma unroll
        for (int i = 0; i < RBN; i += 16) Bs[lc][lr + i] = B[(long)(n0 + lr + i) * K + k0 + lc];
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < RBK; kk++) {
            float a[RTM], b[RTN];
            #pragma unroll
            for (int i = 0; i < RTM; i++) a[i] = As[kk][threadIdx.y * RTM + i];
            #pragma unroll
            for (int j = 0; j < RTN; j++) b[j] = Bs[kk][threadIdx.x * RTN + j];
            #pragma unroll
            for (int i = 0; i < RTM; i++)
                #pragma unroll
                for (int j = 0; j < RTN; j++) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < RTM; i++) {
        int m = m0 + threadIdx.y * RTM + i;
        #pragma unroll
        for (int j = 0; j < RTN; j++) C[(long)m * N + n0 + threadIdx.x * RTN + j] = acc[i][j];
    }
}

// gemm_f32_bias — register-blocked f32 GEMM with fused bias addition.
// Writes C[m, n] = acc + bias[n]. Accepts arbitrary M and N when K is a multiple of 16.
extern "C" __global__ void gemm_f32_bias(
    const float* __restrict__ A, const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C, int M, int N, int K)
{
    __shared__ float As[RBK][RBM + 1];
    __shared__ float Bs[RBK][RBN + 1];
    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0..255
    int m0 = blockIdx.y * RBM, n0 = blockIdx.x * RBN;
    int lr = tid / RBK, lc = tid % RBK; // 16 rows x 16 k-cols per pass

    float acc[RTM][RTN];
    #pragma unroll
    for (int i = 0; i < RTM; i++)
        #pragma unroll
        for (int j = 0; j < RTN; j++) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < K; k0 += RBK) {
        #pragma unroll
        for (int i = 0; i < RBM; i += 16) {
            int r = m0 + lr + i;
            int c = k0 + lc;
            As[lc][lr + i] = (r < M && c < K) ? A[(long)r * K + c] : 0.f;
        }
        #pragma unroll
        for (int i = 0; i < RBN; i += 16) {
            int r = n0 + lr + i;
            int c = k0 + lc;
            Bs[lc][lr + i] = (r < N && c < K) ? B[(long)r * K + c] : 0.f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < RBK; kk++) {
            float a[RTM], b[RTN];
            #pragma unroll
            for (int i = 0; i < RTM; i++) a[i] = As[kk][threadIdx.y * RTM + i];
            #pragma unroll
            for (int j = 0; j < RTN; j++) b[j] = Bs[kk][threadIdx.x * RTN + j];
            #pragma unroll
            for (int i = 0; i < RTM; i++)
                #pragma unroll
                for (int j = 0; j < RTN; j++) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < RTM; i++) {
        int m = m0 + threadIdx.y * RTM + i;
        if (m >= M) continue;
        #pragma unroll
        for (int j = 0; j < RTN; j++) {
            int n = n0 + threadIdx.x * RTN + j;
            if (n < N) C[(long)m * N + n] = acc[i][j] + bias[n];
        }
    }
}

// gemm_f32_bias_add — register-blocked f32 GEMM with fused bias and residual addition.
// Writes residual[m, n] += acc + bias[n], eliminating both add_bias and add_vec.
extern "C" __global__ void gemm_f32_bias_add(
    const float* __restrict__ A, const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ residual, int M, int N, int K)
{
    __shared__ float As[RBK][RBM + 1];
    __shared__ float Bs[RBK][RBN + 1];
    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0..255
    int m0 = blockIdx.y * RBM, n0 = blockIdx.x * RBN;
    int lr = tid / RBK, lc = tid % RBK; // 16 rows x 16 k-cols per pass

    float acc[RTM][RTN];
    #pragma unroll
    for (int i = 0; i < RTM; i++)
        #pragma unroll
        for (int j = 0; j < RTN; j++) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < K; k0 += RBK) {
        #pragma unroll
        for (int i = 0; i < RBM; i += 16) {
            int r = m0 + lr + i;
            int c = k0 + lc;
            As[lc][lr + i] = (r < M && c < K) ? A[(long)r * K + c] : 0.f;
        }
        #pragma unroll
        for (int i = 0; i < RBN; i += 16) {
            int r = n0 + lr + i;
            int c = k0 + lc;
            Bs[lc][lr + i] = (r < N && c < K) ? B[(long)r * K + c] : 0.f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < RBK; kk++) {
            float a[RTM], b[RTN];
            #pragma unroll
            for (int i = 0; i < RTM; i++) a[i] = As[kk][threadIdx.y * RTM + i];
            #pragma unroll
            for (int j = 0; j < RTN; j++) b[j] = Bs[kk][threadIdx.x * RTN + j];
            #pragma unroll
            for (int i = 0; i < RTM; i++)
                #pragma unroll
                for (int j = 0; j < RTN; j++) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < RTM; i++) {
        int m = m0 + threadIdx.y * RTM + i;
        if (m >= M) continue;
        #pragma unroll
        for (int j = 0; j < RTN; j++) {
            int n = n0 + threadIdx.x * RTN + j;
            if (n < N) residual[(long)m * N + n] += acc[i][j] + bias[n];
        }
    }
}

