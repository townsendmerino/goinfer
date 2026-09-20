//go:build darwin

package metal

// gumbelMSLKernels — device-side temperature-only sampling by Gumbel-max, the Mac half of R7b
// (docs/tasks/red-october.md). Port of cuda/gumbel.cu and gpu/gumbel.go (WGSL) — read either
// first; this mirrors their structure kernel-for-kernel over MSL's own constraints, checked
// directly against the Metal Shading Language Specification (2026-06-04 rev), not assumed:
//
//   - MSL HAS NO log1p. It does not appear anywhere in Table 8.1 or 8.2's function lists (the
//     complete standard math function set), unlike CUDA's log1pf. So the small-w branch below
//     uses the SAME 12-term Taylor polynomial gpu/gumbel.go's WGSL kernel already uses (w < 0.25:
//     -w*p(w), computed directly from w, never forming 1-w) — for the same reason WGSL needed it:
//     computing 1.0-w in f32 for w below ~6e-8 rounds EXACTLY to 1.0 (below f32's representable
//     ULP spacing near 1), losing the input entirely regardless of which log variant runs on the
//     result afterward. That is a cancellation problem in the subtraction, not a log-accuracy
//     problem, so a more-accurate log alone cannot fix it — only avoiding the subtraction does.
//   - Unlike WGSL, MSL natively supports 64-bit unsigned integers (ulong, Metal 2.4+; this
//     library already targets MSL3_1), so Philox's mulhi is a direct 64-bit multiply + shift —
//     the same computation decoder.philox4x32 does in Go — not WGSL's 16-bit-limb reconstruction.
//   - precise::log is used for the w >= 0.25 branch and the h >= 2^31 branch. Table 8.1 (accuracy
//     with fast math off, i.e. what metal::precise selects) gives log <= 4 ulp for any x > 0;
//     Table 8.2 (fast math, the library-wide default here — metal/model.go's preciseMathCompile
//     defaults to false / CompileLibrary, not CompileLibraryPrecise) gives only "absolute error <=
//     2^-21 near x=1, else <= 3 ulp" — the same weak near-1 guarantee WGSL's own comment already
//     flags. Verified against the spec's actual tables, not assumed to match WGSL's.
//   - NO FMA IN THE KEY. #pragma METAL fp contract(off) brackets both kernels below. The spec
//     states the library's default (fast) contraction fuses a*b+c ACROSS STATEMENTS, not only
//     within one expression (a scale-then-add pair of separate statements is exactly the case
//     "fast" is documented to still fuse) — so writing the multiply and add as two statements,
//     the way decoder.gumbelKey's explicit float32(...) casts do on the host, is NOT on its own
//     sufficient here. The pragma is the documented mechanism to forbid contraction at exactly
//     this kernel's scope without recompiling the whole allKernels library under
//     CompileLibraryPrecise (metal/model.go), which would also change every other kernel's bits
//     and timing — not what a two-kernel addition should do. Restored to `fast` immediately after
//     the second kernel so nothing else in allKernels is affected.
//
// TWO KERNELS, matching CUDA/WGSL: gumbel_stage1, one thread per group of 4 vocabulary entries
// (one Philox call, four keys), threadgroup-level argmax reduction, one (key, index) pair per
// threadgroup. gumbel_stage2, one threadgroup reduces those into the final id. Ties go to the
// LOWEST index (strict >), as on the host and every other backend.
//
// THREADGROUP MEMORY LAYOUT. DispatchTG takes one byte count for one [[threadgroup(0)]] pointer,
// so a (key, index) pair cannot use two separately-typed threadgroup arrays the way a plain CUDA
// __shared__ pair can. Each slot t packs its key at shm[2t] and its index bit-reinterpreted via
// as_type<float> at shm[2t+1] — the same bit-reinterpretation WGSL's own kernel already relies on
// (bitcast<f32>(prm[1]) for invT), not a raw pointer cast between element types.
const gumbelMSLKernels = `
#pragma METAL fp contract(off)
#define GB_THREADS 256u

inline uint gb_mulhi(uint a, uint b) {
    return uint((ulong(a) * ulong(b)) >> 32u);
}

// Philox4x32-10. Must match decoder.philox4x32 (Random123 known-answer vectors, decoder/philox_test.go).
inline void gb_philox(uint c0, uint c1, uint c2, uint c3, uint k0, uint k1, thread uint* out) {
    for (uint r = 0u; r < 10u; r++) {
        if (r > 0u) { k0 += 0x9E3779B9u; k1 += 0xBB67AE85u; }
        uint hi0 = gb_mulhi(0xD2511F53u, c0), lo0 = 0xD2511F53u * c0;
        uint hi1 = gb_mulhi(0xCD9E8D57u, c2), lo1 = 0xCD9E8D57u * c2;
        uint n0 = hi1 ^ c1 ^ k0, n2 = hi0 ^ c3 ^ k1;
        c0 = n0; c1 = lo1; c2 = n2; c3 = lo0;
    }
    out[0] = c0; out[1] = c1; out[2] = c2; out[3] = c3;
}

// log(1-w) for w in [0, 0.25), computed directly from w (never forming 1-w): the same 12-term
// series gpu/gumbel.go's log1p_neg uses (truncation error 0.25^12/13 ~= 5e-9 relative). Callers
// wanting e = -log1p(-w) = -log(1-w) (gb_gumbel below) must negate this function's result — it is
// named and shaped after WGSL's own log1p_neg, which the caller there negates the same way.
inline float gb_log1p_neg(float w) {
    float p = 1.0f / 12.0f;
    p = p * w + 1.0f / 11.0f;
    p = p * w + 1.0f / 10.0f;
    p = p * w + 1.0f / 9.0f;
    p = p * w + 1.0f / 8.0f;
    p = p * w + 1.0f / 7.0f;
    p = p * w + 1.0f / 6.0f;
    p = p * w + 1.0f / 5.0f;
    p = p * w + 1.0f / 4.0f;
    p = p * w + 1.0f / 3.0f;
    p = p * w + 0.5f;
    p = p * w + 1.0f;
    return -w * p;
}

inline float gb_gumbel(uint h) {
    float e;
    if (h < 0x80000000u) {
        float w = (float(h) + 0.5f) * 2.3283064365386963e-10f;
        e = (w < 0.25f) ? -gb_log1p_neg(w) : -precise::log(1.0f - w);
    } else {
        float v = (float(~h) + 0.5f) * 2.3283064365386963e-10f; // = 1 - w, exactly
        e = -precise::log(v);
    }
    return -precise::log(e);
}

// a beats b: larger key first; on an equal key the smaller index wins (idx < 0 means "none").
inline bool gb_better(float ak, int ai, float bk, int bi) {
    if (ai < 0) return false;
    if (bi < 0) return true;
    return ak > bk || (ak == bk && ai < bi);
}

kernel void gumbel_stage1(
    device const float* logits [[buffer(0)]], constant uint& V [[buffer(1)]],
    constant float& invT [[buffer(2)]], constant uint& k0 [[buffer(3)]], constant uint& k1 [[buffer(4)]],
    constant uint& d0 [[buffer(5)]], constant uint& d1 [[buffer(6)]],
    device float* outKey [[buffer(7)]], device int* outIdx [[buffer(8)]],
    threadgroup float* shm [[threadgroup(0)]], // GB_THREADS pairs: shm[2t]=key, shm[2t+1]=as_type<float>(idx)
    uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    uint b = tgid * GB_THREADS + tid; // group of 4 consecutive vocabulary entries
    float best = -3.402823466e38f;
    int bi = -1;
    if (b * 4u < V) {
        uint r[4];
        gb_philox(b, d0, d1, 0u, k0, k1, r);
        for (uint lane = 0u; lane < 4u; lane++) {
            uint i = b * 4u + lane;
            if (i < V) {
                float scaled = logits[i] * invT;
                float k = scaled + gb_gumbel(r[lane]);
                if (k > best) { best = k; bi = int(i); }
            }
        }
    }
    shm[tid * 2u] = best; shm[tid * 2u + 1u] = as_type<float>(bi);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = GB_THREADS >> 1u; o > 0u; o >>= 1u) {
        if (tid < o) {
            float ak = shm[tid * 2u], bk = shm[(tid + o) * 2u];
            int ai = as_type<int>(shm[tid * 2u + 1u]), bidx = as_type<int>(shm[(tid + o) * 2u + 1u]);
            if (gb_better(bk, bidx, ak, ai)) { shm[tid * 2u] = bk; shm[tid * 2u + 1u] = as_type<float>(bidx); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) { outKey[tgid] = shm[0]; outIdx[tgid] = as_type<int>(shm[1]); }
}

kernel void gumbel_stage2(
    device const float* inKey [[buffer(0)]], device const int* inIdx [[buffer(1)]], constant uint& n [[buffer(2)]],
    device int* out [[buffer(3)]],
    threadgroup float* shm [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]]) {
    float best = -3.402823466e38f;
    int bi = -1;
    for (uint i = tid; i < n; i += GB_THREADS) {
        if (gb_better(inKey[i], inIdx[i], best, bi)) { best = inKey[i]; bi = inIdx[i]; }
    }
    shm[tid * 2u] = best; shm[tid * 2u + 1u] = as_type<float>(bi);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = GB_THREADS >> 1u; o > 0u; o >>= 1u) {
        if (tid < o) {
            float ak = shm[tid * 2u], bk = shm[(tid + o) * 2u];
            int ai = as_type<int>(shm[tid * 2u + 1u]), bidx = as_type<int>(shm[(tid + o) * 2u + 1u]);
            if (gb_better(bk, bidx, ak, ai)) { shm[tid * 2u] = bk; shm[tid * 2u + 1u] = as_type<float>(bidx); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) out[0] = as_type<int>(shm[1]); // -1 when nothing was comparable: the caller falls back to argmax
}
#pragma METAL fp contract(fast)
`
