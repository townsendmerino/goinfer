// ---------------------------------------------------------------------------
// Qwen2.5-VL ViT additions (Phase 3) — the Metal mirror of vit.cu's Qwen section,
// the five kernels Qwen needs on top of the shared SigLIP set. Written against
// vision/qwen_encoder.go's arithmetic, same as the CUDA side.
//
// gelu_erf below is a DIFFERENT function from gelu_tanh above: Qwen's patch merger
// uses nn.GELU()'s exact erf form while SigLIP uses the tanh approximation. They
// differ by ~5e-4, far above any parity bar — shipping one for both is a silent
// numeric bug, so both are present and callers pick deliberately.
// ---------------------------------------------------------------------------

// rmsnorm: weight-only RMS normalization (no mean subtraction, no bias) —
// x * rsqrt(mean(x^2) + eps) * w, one threadgroup per row. The mean-square reduces
// in f32 (the double vit.cu uses does not port); the divide/rsqrt stay exact because
// the library is compiled fast-math OFF (CompileLibraryPrecise) — the same guard that
// kept the quant scale from drifting a ULP on the SigLIP set.
kernel void rmsnorm(
    device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int& rows [[buffer(3)]], constant int& dim [[buffer(4)]], constant float& eps [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]], uint r [[threadgroup_position_in_grid]],
    uint tgsz [[threads_per_threadgroup]])
{
    device const float* xr = x + (uint)r * (uint)dim;
    device float* dst = out + (uint)r * (uint)dim;
    threadgroup float sm[LNBLOCK];
    float acc = 0.0f;
    for (int i = tid; i < dim; i += tgsz) { float vv = xr[i]; acc += vv * vv; }
    sm[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // f32 cross-thread SUM — the reduction order is fixed by tgsz (== ViTBlock/LNBLOCK).
    // f32 addition is not associative, so this width is a bit-identity dependency, not a
    // perf knob: sweeping it moves the bits and needs a parity-gate re-baseline (see ViTBlock).
    for (uint o = tgsz >> 1; o > 0; o >>= 1) { if (tid < o) sm[tid] += sm[tid + o]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv = rsqrt(sm[0] / (float)dim + eps);
    for (int i = tid; i < dim; i += tgsz) dst[i] = (xr[i] * inv) * w[i];
}

// rope_qk: NeoX rotate_half 2D rotary applied IN PLACE to the q and k thirds of a fused
// qkv buffer [seq, 3*hidden] (row layout [3, nH, hd]); v is untouched. One thread per
// (patch, head, d<hd/2), rotating both q and k. dispatchThreads launches exactly the
// seq*nH*(hd/2) threads, so no bounds check.
kernel void rope_qk(
    device float* qkv [[buffer(0)]], device const float* cos [[buffer(1)]], device const float* sin [[buffer(2)]],
    constant int& seq [[buffer(3)]], constant int& nH [[buffer(4)]], constant int& hd [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int hf = hd / 2; // named hf, not half: half is MSL's f16 type keyword.
    int g = (int)gid;
    int d = g % hf;
    int rest = g / hf;
    int head = rest % nH;
    int i = rest / nH;
    int hidden = nH * hd;
    device const float* co = cos + (uint)i * (uint)hd;
    device const float* si = sin + (uint)i * (uint)hd;
    uint qoff = (uint)i * 3u * (uint)hidden + (uint)head * (uint)hd;
    uint koff = qoff + (uint)hidden;
    float x = qkv[qoff + d], y = qkv[qoff + d + hf];
    qkv[qoff + d] = x * co[d] - y * si[d];
    qkv[qoff + d + hf] = y * co[d + hf] + x * si[d + hf];
    x = qkv[koff + d]; y = qkv[koff + d + hf];
    qkv[koff + d] = x * co[d] - y * si[d];
    qkv[koff + d + hf] = y * co[d + hf] + x * si[d + hf];
}

// attention_seg: bidirectional MHA restricted to each patch's segment (a window for
// most blocks, a whole image for the fullatt blocks). Reads q/k/v from the FUSED qkv
// buffer [seq, 3*hidden] at offsets 0/hidden/2*hidden; the per-query score row stages
// in dynamic threadgroup memory (sc, maxSeg floats, bound at index 0 by the caller).
// segStart/segEnd are PER-PATCH bounds, not cu_seqlens. One threadgroup per (head,
// query). Softmax sum is f32 (the vit.cu double does not port).
kernel void attention_seg(
    device const float* qkv [[buffer(0)]], device float* out [[buffer(1)]],
    device const int* segStart [[buffer(2)]], device const int* segEnd [[buffer(3)]],
    constant int& seq [[buffer(4)]], constant int& nH [[buffer(5)]], constant int& hd [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    threadgroup float* sc [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]], uint blk [[threadgroup_position_in_grid]],
    uint tgsz [[threads_per_threadgroup]])
{
    int h = (int)blk / seq, i = (int)blk % seq;
    int hidden = nH * hd, off = h * hd;
    int s0 = segStart[i], s1 = segEnd[i], n = s1 - s0;
    device const float* qi = qkv + (uint)i * 3u * (uint)hidden + off;
    threadgroup float qs[128];
    if (hd <= 128) {
        for (int d = tid; d < hd; d += tgsz) qs[d] = qi[d];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int t = tid; t < n; t += tgsz) {
            device const float* kj = qkv + (uint)(s0 + t) * 3u * (uint)hidden + (uint)hidden + off;
            float acc = 0.0f;
            for (int d = 0; d < hd; d++) acc += qs[d] * kj[d];
            sc[t] = acc * scale;
        }
    } else {
        for (int t = tid; t < n; t += tgsz) {
            device const float* kj = qkv + (uint)(s0 + t) * 3u * (uint)hidden + (uint)hidden + off;
            float acc = 0.0f;
            for (int d = 0; d < hd; d++) acc += qi[d] * kj[d];
            sc[t] = acc * scale;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float smax[LNBLOCK];
    threadgroup float ssum[LNBLOCK];
    float m = -3.402823466e+38f;
    for (int t = tid; t < n; t += tgsz) if (sc[t] > m) m = sc[t];
    smax[tid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = tgsz >> 1; o > 0; o >>= 1) { if (tid < o && smax[tid + o] > smax[tid]) smax[tid] = smax[tid + o]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float mx = smax[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float su = 0.0f;
    for (int t = tid; t < n; t += tgsz) { float e = exp(sc[t] - mx); sc[t] = e; su += e; }
    ssum[tid] = su;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // f32 cross-thread SUM — the reduction order is fixed by tgsz (== ViTBlock/LNBLOCK).
    // f32 addition is not associative, so this width is a bit-identity dependency, not a
    // perf knob: sweeping it moves the bits and needs a parity-gate re-baseline (see ViTBlock).
    for (uint o = tgsz >> 1; o > 0; o >>= 1) { if (tid < o) ssum[tid] += ssum[tid + o]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv = 1.0f / ssum[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int t = tid; t < n; t += tgsz) sc[t] = sc[t] * inv;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* oi = out + (uint)i * (uint)hidden + off;
    for (int d = tid; d < hd; d += tgsz) {
        float acc = 0.0f;
        for (int t = 0; t < n; t++) acc += sc[t] * qkv[(uint)(s0 + t) * 3u * (uint)hidden + 2u * (uint)hidden + off + d];
        oi[d] = acc;
    }
}

// attention_seg_tiled — query-tiled online-softmax attention restricted to per-patch
// cu_seqlens segment bounds [segStart[i], segEnd[i]].
//
// Unlike attention_seg which stages an entire segment score row in dynamic threadgroup
// memory (which exceeds Apple Silicon's 32 KiB threadgroup ceiling at maxSeg > 7000),
// this uses a constant 16 KiB threadgroup allocation (Ks and Vs tiles of AT_KTILE=16),
// lifting the threadgroup memory limit completely and allowing arbitrarily large images.
kernel void attention_seg_tiled(
    device const float* qkv [[buffer(0)]], device float* out [[buffer(1)]],
    device const int* segStart [[buffer(2)]], device const int* segEnd [[buffer(3)]],
    constant int& seq [[buffer(4)]], constant int& nH [[buffer(5)]], constant int& hd [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]], uint blk [[threadgroup_position_in_grid]],
    uint tgsz [[threads_per_threadgroup]])
{
    threadgroup float Ks[AT_KTILE][AT_MAXHD];
    threadgroup float Vs[AT_KTILE][AT_MAXHD];

    int tilesPerHead = (seq + AT_QTILE - 1) / AT_QTILE;
    int h = (int)blk / tilesPerHead;
    int tileIdx = (int)blk % tilesPerHead;
    int qStart = tileIdx * AT_QTILE;
    int i = qStart + (int)tid;
    int hidden = nH * hd, off = h * hd;
    bool active = (i < seq);

    int s0 = active ? segStart[i] : 0;
    int s1 = active ? segEnd[i] : 0;

    float m = -3.402823466e+38f;
    float l = 0.0f;
    float acc[AT_MAXHD];
    for (int d = 0; d < hd; d++) acc[d] = 0.0f;

    for (int k0 = 0; k0 < seq; k0 += AT_KTILE) {
        int chunk = min(AT_KTILE, seq - k0);
        for (int idx = (int)tid; idx < chunk * hd; idx += (int)tgsz) {
            int kk = idx / hd, d = idx % hd;
            int kj_idx = k0 + kk;
            Ks[kk][d] = qkv[(uint)kj_idx * 3u * (uint)hidden + (uint)hidden + (uint)off + (uint)d];
            Vs[kk][d] = qkv[(uint)kj_idx * 3u * (uint)hidden + 2u * (uint)hidden + (uint)off + (uint)d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            int chunkStart = k0;
            int chunkEnd = k0 + chunk;
            int validStart = max(s0, chunkStart);
            int validEnd = min(s1, chunkEnd);

            if (validStart < validEnd) {
                device const float* qi = qkv + (uint)i * 3u * (uint)hidden + off;
                float s[AT_KTILE];
                float blockMax = -3.402823466e+38f;
                for (int kk = 0; kk < chunk; kk++) {
                    int kIdx = k0 + kk;
                    if (kIdx >= s0 && kIdx < s1) {
                        float dot = 0.0f;
                        for (int d = 0; d < hd; d++) dot += qi[d] * Ks[kk][d];
                        float sv = dot * scale;
                        s[kk] = sv;
                        if (sv > blockMax) blockMax = sv;
                    }
                }
                if (blockMax > -3.402823466e+38f) {
                    float newMax = max(m, blockMax);
                    float corr = exp(m - newMax);
                    l *= corr;
                    for (int d = 0; d < hd; d++) acc[d] *= corr;
                    float blockSum = 0.0f;
                    for (int kk = 0; kk < chunk; kk++) {
                        int kIdx = k0 + kk;
                        if (kIdx >= s0 && kIdx < s1) {
                            float e = exp(s[kk] - newMax);
                            blockSum += e;
                            for (int d = 0; d < hd; d++) acc[d] += e * Vs[kk][d];
                        }
                    }
                    l += blockSum;
                    m = newMax;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        device float* oi = out + (uint)i * (uint)hidden + off;
        float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
        for (int d = 0; d < hd; d++) oi[d] = acc[d] * inv;
    }
}

// silu_mul: gate = silu(gate) * up, the gated-MLP activation. silu in f32 (vit.cu's
// double does not port), which is why the per-kernel bar is looser than CUDA's 1e-6.
kernel void silu_mul(
    device float* gate [[buffer(0)]], device const float* up [[buffer(1)]], constant int& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    float vv = gate[gid];
    gate[gid] = (vv / (1.0f + exp(-vv))) * up[gid];
}

// erf_approx: Abramowitz & Stegun 7.1.26, max abs error ~1.5e-7 — at the f32 floor.
// MSL's standard library has NO erf (unlike CUDA), so the EXACT GELU below supplies its
// own. This is what makes gelu_erf a genuine f32 approximation of the float64 reference,
// hence its looser (but stated) per-kernel bar; the discrimination from gelu_tanh
// (~5e-4) is unaffected and remains the proof of correct FORM.
inline float erf_approx(float x) {
    float s = sign(x);
    float ax = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-ax * ax);
    return s * y;
}

// gelu_erf: the EXACT (erf) GELU — nn.GELU()'s default, used by Qwen's patch merger.
// Distinct from gelu_tanh; see the header note. The divide matches vit.cu (exact under
// fast-math-off); erf is erf_approx above because MSL lacks a stdlib erf.
kernel void gelu_erf(device float* x [[buffer(0)]], constant int& n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    float vv = x[gid];
    x[gid] = 0.5f * vv * (1.0f + erf_approx(vv / 1.4142135623730951f));
}

