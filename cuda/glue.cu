// glue.cu — the per-token elementwise/attention kernels the GEMV-only projection omitted,
// so the cgo-free CUDA decode measurement is end-to-end (docs/prompts/cuda-measure-e2e-decode.md).
// Dense Qwen2/Llama decode. Compiled to PTX offline (NVRTC compute_75), go:embed'd, driver-JIT'd.
// Each is cosine-validated vs a CPU reference in the harness; math matches goinfer's CPU decode.

extern "C" {

// rmsnorm_quant: y = x * w * rsqrt(mean(x^2)+eps); then per-vector symmetric int8 quant
// (scale = maxabs(y)/127), packed 4 int8 per int (the GEMV activation layout). One block.
//
// addOne selects the scale, mirroring decoder/rmsnorm.go: Gemma stores norm weights as
// deviations from 1.0 and scales by (1+w); Llama/Qwen scale by w directly. Getting this
// wrong silently shifts every layer — it is a per-family knob (Architecture.RMSAddOne).
__global__ void rmsnorm_quant(const float* __restrict__ x, const float* __restrict__ w,
                              int H, float eps, int addOne, int* __restrict__ aq, float* __restrict__ aScale) {
    extern __shared__ float sh[];        // [H] normed values + reduction scratch
    float* normed = sh;
    float* red = sh + H;
    int t = threadIdx.x, nt = blockDim.x;
    // pass 1: sum of squares
    float ss = 0.f;
    for (int k = t; k < H; k += nt) ss += x[k] * x[k];
    red[t] = ss; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] += red[t + o]; __syncthreads(); }
    float rnorm = rsqrtf(red[0] / H + eps); __syncthreads();
    // pass 2: normed + maxabs
    float ma = 0.f;
    for (int k = t; k < H; k += nt) { float g = addOne ? (1.f + w[k]) : w[k]; float v = x[k] * g * rnorm; normed[k] = v; ma = fmaxf(ma, fabsf(v)); }
    red[t] = ma; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] = fmaxf(red[t], red[t + o]); __syncthreads(); }
    float sc = red[0] / 127.f; float inv = sc > 0.f ? 1.f / sc : 0.f;
    if (t == 0) *aScale = sc;
    // pass 3: quant + pack 4 int8 per int
    for (int j = t; j < H / 4; j += nt) {
        int packed = 0;
        for (int b = 0; b < 4; b++) {
            int q = __float2int_rn(normed[4 * j + b] * inv);
            q = max(-127, min(127, q));
            packed |= (q & 0xff) << (8 * b);
        }
        aq[j] = packed;
    }
}

// rmsnorm_f32: plain in-place RMSNorm of a [H] vector — no quantization fused in.
//
// This is Gemma's SANDWICH norm (Architecture.NormPlacement == NormSandwich4). Unlike every
// other norm in this file it normalizes a SUBLAYER OUTPUT rather than a GEMV input, so it
// must NOT quantize: the result is added straight into the f32 residual stream
// (decoder/model.go runLayersFromEmbed — `if sandwich { normalize(scr.sub, PostAttnNorm) }`
// then `h += scr.sub`). One block; addOne as in rmsnorm_quant.
__global__ void rmsnorm_f32(float* __restrict__ x, const float* __restrict__ w,
                            int H, float eps, int addOne) {
    extern __shared__ float red[];
    int t = threadIdx.x, nt = blockDim.x;
    float ss = 0.f;
    for (int k = t; k < H; k += nt) ss += x[k] * x[k];
    red[t] = ss; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] += red[t + o]; __syncthreads(); }
    float rnorm = rsqrtf(red[0] / H + eps); __syncthreads();
    for (int k = t; k < H; k += nt) { float g = addOne ? (1.f + w[k]) : w[k]; x[k] = x[k] * rnorm * g; }
}

// quant_vec: symmetric int8 quant of a float vector (attention ctx / SwiGLU output), packed.
__global__ void quant_vec(const float* __restrict__ x, int N, int* __restrict__ q, float* __restrict__ scale) {
    extern __shared__ float red[];
    int t = threadIdx.x, nt = blockDim.x;
    float ma = 0.f;
    for (int k = t; k < N; k += nt) ma = fmaxf(ma, fabsf(x[k]));
    red[t] = ma; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] = fmaxf(red[t], red[t + o]); __syncthreads(); }
    float sc = red[0] / 127.f; float inv = sc > 0.f ? 1.f / sc : 0.f;
    if (t == 0) *scale = sc;
    for (int j = t; j < N / 4; j += nt) {
        int packed = 0;
        for (int b = 0; b < 4; b++) {
            int v = __float2int_rn(x[4 * j + b] * inv); v = max(-127, min(127, v));
            packed |= (v & 0xff) << (8 * b);
        }
        q[j] = packed;
    }
}

// rope: rotate pairs (d, d+half) of each head vector by theta = pos * invFreq[d]. In place.
// vec is [nHeads * hd]; rotaryDim == hd (full RoPE, Qwen2/Llama). One thread per (head,d<half).
__global__ void rope(float* __restrict__ vec, const float* __restrict__ invFreq,
                     int nHeads, int hd, int pos) {
    int half = hd / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nHeads * half) return;
    int h = idx / half, d = idx % half;
    float ang = pos * invFreq[d];
    float c = cosf(ang), s = sinf(ang);
    float* base = vec + h * hd;
    float a = base[d], b = base[d + half];
    base[d] = a * c - b * s;
    base[d + half] = a * s + b * c;
}

// attention: decode single-query GQA. q[nH*hd], kc/vc are [nKeys*kvDim] (kvDim=nKV*hd),
// layout [key*kvDim + head*hd + d], k post-RoPE. One block per q-head. ctx[nH*hd].
// window: 0 = full causal (a global layer, or an arch with no window). Otherwise attend only
// over the last `window` keys, matching decoder/kvcache.go WindowStart:
//   start = max(pos - window + 1, 0)  and with nKeys = pos+1  ⇒  max(nKeys - window, 0).
// Scores are indexed from winStart, so shared memory is bounded by min(nKeys, window) rather
// than nKeys — which also lifts the ~12k-key ceiling the old (nKeys+blockDim)*4 request hit
// against the 48 KB per-block limit.
__global__ void attention(const float* __restrict__ q, const float* __restrict__ kc,
                          const float* __restrict__ vc, int nH, int nKV, int hd, int nKeys,
                          float scale, int window, float* __restrict__ ctx) {
    extern __shared__ float sm[];       // [nWin] scores/weights + [blockDim] reduce
    int winStart = (window > 0 && nKeys > window) ? nKeys - window : 0;
    int nWin = nKeys - winStart;
    float* sc = sm;
    float* red = sm + nWin;
    int h = blockIdx.x; if (h >= nH) return;
    int kvDim = nKV * hd, group = nH / nKV, kvh = h / group;
    const float* qh = q + h * hd;
    int t = threadIdx.x, nt = blockDim.x;
    // scores + local max
    float lm = -1e30f;
    for (int s = winStart + t; s < nKeys; s += nt) {
        const float* ks = kc + s * kvDim + kvh * hd;
        float dot = 0.f;
        for (int d = 0; d < hd; d++) dot += qh[d] * ks[d];
        dot *= scale; sc[s - winStart] = dot; lm = fmaxf(lm, dot);
    }
    red[t] = lm; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] = fmaxf(red[t], red[t + o]); __syncthreads(); }
    float mx = red[0]; __syncthreads();
    // exp + sum
    float ls = 0.f;
    for (int s = winStart + t; s < nKeys; s += nt) { float e = __expf(sc[s - winStart] - mx); sc[s - winStart] = e; ls += e; }
    red[t] = ls; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] += red[t + o]; __syncthreads(); }
    float inv = 1.f / red[0]; __syncthreads();
    // ctx[d] = sum_s (sc[s]*inv) * V[s,d]  over the window only
    for (int d = t; d < hd; d += nt) {
        float acc = 0.f;
        for (int s = winStart; s < nKeys; s++) acc += sc[s - winStart] * vc[s * kvDim + kvh * hd + d];
        ctx[h * hd + d] = acc * inv;
    }
}

// glu_quant: d = act(g) * u; then symmetric int8 quant (packed). Mirrors decoder/mlp.go
// gatedMLP: `gate[i] = act(gate[i]) * up[i]` for both activations.
//
// act selects the gated activation (Architecture.Act):
//   ACT_SILU (1) — SwiGLU: silu(g) = g/(1+e^-g).                 Llama / Mistral / Qwen
//   ACT_GELU_TANH (0) — GeGLU: 0.5g(1+tanh(√(2/π)(g+0.044715g³))). Gemma
// The numbering matches decoder's ActKind iota so the Go side passes int32(arch.Act)
// straight through; the constants below make the coupling explicit rather than magic.
//
// Both compute in f32 (__expf fast intrinsic / tanhf) where the CPU reference computes in
// f64 — the same trade the SwiGLU path already shipped with, and it clears the 3% near-tie
// parity bar. Do not "fix" this to f64 without re-measuring: it is elementwise over I and
// f64 is 1/32 rate on this card.
#define ACT_GELU_TANH 0
#define ACT_SILU      1
// gOff/uOff index into g/u. The dense path keeps two separate buffers and passes 0/0; MoE's
// stacked-expert GEMV produces gate and up CONCATENATED in one buffer (one launch over
// N=2*inter), so it passes the same pointer twice with uOff=inter. gocudrv exposes no buffer
// view, so the offset has to be an argument rather than pointer arithmetic on the Go side.
__global__ void glu_quant(const float* __restrict__ g, const float* __restrict__ u,
                          int gOff, int uOff, int I, int act,
                          int* __restrict__ q, float* __restrict__ scale, float* __restrict__ dscratch) {
    extern __shared__ float red[];
    int t = threadIdx.x, nt = blockDim.x;
    float ma = 0.f;
    for (int k = t; k < I; k += nt) {
        float x = g[gOff + k], a;
        if (act == ACT_SILU) {
            a = x / (1.f + __expf(-x));
        } else { // ACT_GELU_TANH — 0.7978845608028654 = sqrt(2/pi), matching decoder/rmsnorm.go geluTanh
            a = 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
        }
        float d = a * u[uOff + k]; dscratch[k] = d; ma = fmaxf(ma, fabsf(d));
    }
    red[t] = ma; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] = fmaxf(red[t], red[t + o]); __syncthreads(); }
    float sc = red[0] / 127.f; float inv = sc > 0.f ? 1.f / sc : 0.f;
    if (t == 0) *scale = sc;
    for (int j = t; j < I / 4; j += nt) {
        int packed = 0;
        for (int b = 0; b < 4; b++) { int v = __float2int_rn(dscratch[4 * j + b] * inv); v = max(-127, min(127, v)); packed |= (v & 0xff) << (8 * b); }
        q[j] = packed;
    }
}

// residual: x += y (elementwise), plus optional bias on y's source handled by GEMV.
__global__ void residual(float* __restrict__ x, const float* __restrict__ y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] += y[i];
}

// argmax: greedy token = argmax(logits[V]). Two-level: block-local argmax → atomic to global.
__global__ void argmax_reduce(const float* __restrict__ logits, int V, int* __restrict__ outIdx, float* __restrict__ outVal) {
    extern __shared__ float sv[];
    int* si = (int*)(sv + blockDim.x);
    int t = threadIdx.x, nt = blockDim.x;
    float bv = -1e30f; int bi = -1;
    for (int k = t; k < V; k += nt) if (logits[k] > bv) { bv = logits[k]; bi = k; }
    sv[t] = bv; si[t] = bi; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o && sv[t + o] > sv[t]) { sv[t] = sv[t + o]; si[t] = si[t + o]; } __syncthreads(); }
    if (t == 0) { *outIdx = si[0]; *outVal = sv[0]; }
}

} // extern "C"
