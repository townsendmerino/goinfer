// topk.cu — device-side bounded top-K selection for sampled decode (R7, docs/tasks/red-october.md).
//
// A SEPARATE .cu on purpose, for the same reason argmax.cu and router_f32.cu are: adding a kernel to
// glue.cu would regenerate every audited glue kernel's codegen. This file adds only topk.ptx.
//
// WHAT IT RETURNS. The K highest logits of one V-wide row, ordered (logit DESCENDING, token id
// ASCENDING), plus, when asked, the full-vocabulary softmax denominator Z = Σ exp((l − max)·invT). The
// host sampler runs its existing exact filter over these K instead of over V, and falls back to a
// full-row readback on the rare token where K is not enough (decoder/sampler_topk.go). Output layout
// (ONE buffer, one D2H): out[0..K) ids, out[K..2K) logit bits, out[2K..2K+2) Z hi/lo bits.
//
// THE TIE ORDER IS A CONTRACT, not a preference. decoder.topKByLogit keeps the k best by
// (logit desc, id asc) — "ties evict larger ids" — and that order feeds the cumulative-CDF draw.
// -0.0 and +0.0 compare EQUAL on the host (!=), and here every comparison is a float compare or a
// canonicalised key, so they tie as they do there.
//
// TWO PATHS, ONE ANSWER. Both return exactly the first K of "sort the row by (logit desc, id asc)".
//
//   FAST PATH (the normal case). Only ~K of ~150k entries matter, so do not histogram the row. Sample
//   it at a fixed stride, sort the samples, and take the R-th largest as a threshold t0 chosen so the
//   expected number of row entries >= t0 is a few times K. One pass compacts every entry >= t0 (a
//   plain atomic append: a few hundred to a few thousand entries), a bitonic sort orders them, and the
//   first K are the answer. That is exact whenever K <= count <= CAP, because every entry >= t0 was
//   collected and the K-th largest is >= t0. The threshold is only an ESTIMATE; correctness never
//   depends on it — a count outside [K, CAP] hands off to the slow path.
//
//   SLOW PATH (exact, estimate-free). A 3-pass radix select (11+11+10 bits over an order-preserving
//   float->uint key, -0 canonicalised to +0) finds the K-th largest key exactly; a gather takes every
//   entry above it plus the first `need` equal to it by ASCENDING id when more tie than are needed;
//   a bitonic sort orders the K survivors. Used for short vocabularies, K close to V, and any row
//   whose sampled estimate missed (heavy ties, adversarial layouts).
//
// WHY NOT ONE HISTOGRAM PASS: measured on the RTX 2070 SUPER, the radix path alone took ~220 us for
// V=152k on normal logits. Logit keys concentrate into a few dozen bins, so shared-memory atomics
// serialise, and __match_any_sync (the warp-aggregation fix) is microcoded on Turing as a BREV/FLO loop
// over distinct values and was slower still (ncu source-level stall sampling, 2026-09-20).
//
// The selection path is compares and integer ops; the only floating-point arithmetic is the Z sum.
// TestKernelFMALint applies to this file like any other.

#define TK_THREADS 1024
#define TK_KMAX 1024
#define TK_BINS 2048
#define TK_U 8          // loads in flight per thread per scan step
#define TK_CAP 4096     // fast-path candidate capacity
#define TK_SAMPLES 2048 // strided samples used to estimate the threshold

extern "C" {

__device__ __forceinline__ unsigned int tk_key(float x) {
    if (x == 0.0f) x = 0.0f; // -0 -> +0: the host's != treats them as one value
    unsigned int u = __float_as_uint(x);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// a sorts before b: larger logit first, then smaller id.
__device__ __forceinline__ bool tk_before(float av, int ai, float bv, int bi) {
    return av > bv || (av == bv && ai < bi);
}

// Bitonic sort of P (a power of two) (val, id) pairs in shared memory into (val desc, id asc) order.
// Every thread of the block must call it (it contains barriers).
__device__ void tk_sort(float* sv, int* si, int P) {
    const int t = threadIdx.x;
    for (int kk = 2; kk <= P; kk <<= 1) {
        for (int j = kk >> 1; j > 0; j >>= 1) {
            for (int i = t; i < P; i += TK_THREADS) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const bool firstBlock = ((i & kk) == 0);
                    const bool ib = tk_before(sv[i], si[i], sv[ixj], si[ixj]);
                    if (firstBlock ? !ib : ib) {
                        float tv = sv[i]; sv[i] = sv[ixj]; sv[ixj] = tv;
                        int ti = si[i]; si[i] = si[ixj]; si[ixj] = ti;
                    }
                }
            }
            __syncthreads();
        }
    }
}

__global__ void __launch_bounds__(TK_THREADS)
topk_select(const float* __restrict__ logits, int V, int K, float invT, int wantZ,
            int* __restrict__ out) {
    // One raw scratch region, carved differently per path (the paths never overlap in time):
    //   fast:  samples float[2048] @0, then cv float[4096] @0 and ci int[4096] @16384
    //   slow:  hist int[2048] @0, sval float[1024] @8192, sidx int[1024] @12288, red double[1024] @16384
    __shared__ __align__(16) unsigned char raw[32768];
    __shared__ int segSum[TK_BINS / 64];
    __shared__ int wg[32], we[32], wgo[32], weo[32];
    __shared__ unsigned int s_prefix;
    __shared__ int s_need, s_greater, s_outGt, s_eqTaken, s_totGt, s_totEq, s_eqCnt;
    __shared__ int s_cnt, s_ok;
    __shared__ float s_t0, s_mx;

    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    if (K > V) K = V;
    if (K > TK_KMAX) K = TK_KMAX;

    float* samples = (float*)raw;
    float* cv = (float*)raw;
    int* ci = (int*)(raw + 16384);
    int* hist = (int*)raw;
    float* sval = (float*)(raw + 8192);
    int* sidx = (int*)(raw + 12288);
    double* red = (double*)(raw + 16384);

    // ================= FAST PATH =================
    if (t == 0) { s_ok = 0; s_cnt = 0; }
    __syncthreads();
    const bool tryFast = (V >= 8192) && (K * 8 <= V);
    if (tryFast) {
        // Threshold estimate: the R-th largest of TK_SAMPLES evenly-strided entries, R chosen so the
        // expected number of row entries >= it is ~5K (capped so it stays well inside TK_CAP).
        for (int i = t; i < TK_SAMPLES; i += TK_THREADS) {
            const long long idx = ((long long)i * (long long)V) / TK_SAMPLES;
            samples[i] = logits[idx];
        }
        __syncthreads();
        // Bitonic sort of the samples, descending, values only.
        for (int kk = 2; kk <= TK_SAMPLES; kk <<= 1) {
            for (int j = kk >> 1; j > 0; j >>= 1) {
                for (int i = t; i < TK_SAMPLES; i += TK_THREADS) {
                    const int ixj = i ^ j;
                    if (ixj > i) {
                        const bool firstBlock = ((i & kk) == 0);
                        const float a = samples[i], b = samples[ixj];
                        // firstBlock keeps the larger at the lower index; otherwise the smaller.
                        if (firstBlock ? (a < b) : (a > b)) { samples[i] = b; samples[ixj] = a; }
                    }
                }
                __syncthreads();
            }
        }
        if (t == 0) {
            float target = 5.0f * (float)K;
            if (target > 2400.0f) target = 2400.0f;
            int R = (int)(target * (float)TK_SAMPLES / (float)V + 0.5f);
            if (R < 8) R = 8;
            if (R > 64) R = 64;
            s_t0 = samples[R - 1];
        }
        __syncthreads();
        const float t0 = s_t0;
        // Compact every entry >= t0. cv/ci alias the (now consumed) samples.
        for (int base = 0; base < V; base += TK_THREADS * TK_U) {
            float xv[TK_U];
#pragma unroll
            for (int u = 0; u < TK_U; u++) {
                const int i = base + u * TK_THREADS + t;
                xv[u] = (i < V) ? logits[i] : -3.402823466e38f;
            }
#pragma unroll
            for (int u = 0; u < TK_U; u++) {
                const int i = base + u * TK_THREADS + t;
                if (i < V && xv[u] >= t0) {
                    const int slot = atomicAdd(&s_cnt, 1);
                    if (slot < TK_CAP) { cv[slot] = xv[u]; ci[slot] = i; }
                }
            }
        }
        __syncthreads();
        const int cnt = s_cnt;
        if (cnt >= K && cnt <= TK_CAP) {
            int P = 1;
            while (P < cnt) P <<= 1;
            for (int i = cnt + t; i < P; i += TK_THREADS) { cv[i] = -3.402823466e38f; ci[i] = 0x7fffffff; }
            __syncthreads();
            tk_sort(cv, ci, P);
            for (int i = t; i < K; i += TK_THREADS) { out[i] = ci[i]; out[K + i] = __float_as_int(cv[i]); }
            if (t == 0) { s_mx = cv[0]; s_ok = 1; }
        }
        __syncthreads();
    }

    // ================= SLOW PATH (exact, estimate-free) =================
    if (!s_ok) {
        unsigned int thr = 0;
        int need = K, greater = 0, eqCnt = 0;

        if (K < V) {
            if (t == 0) { s_prefix = 0; s_need = K; s_greater = 0; s_eqCnt = 0; }
            __syncthreads();
            // 11+11+10 bits, MSB first. Computed, not tabulated: a 3-element local array puts 24 B/thread
            // of per-thread local memory in the module, which the local-memory census (rightly) refuses
            // to leave unmeasured.
            for (int pass = 0; pass < 3; pass++) {
                const int shift = (pass == 0) ? 21 : ((pass == 1) ? 10 : 0);
                const int width = (pass == 2) ? 10 : 11;
                const int nb = 1 << width;
                for (int i = t; i < TK_BINS; i += TK_THREADS) hist[i] = 0;
                __syncthreads();
                const unsigned int prefix = s_prefix;
                for (int i = t; i < V; i += TK_THREADS) {
                    const unsigned int k = tk_key(logits[i]);
                    if (pass == 0 || (k >> (shift + width)) == prefix)
                        atomicAdd(&hist[(k >> shift) & (nb - 1)], 1);
                }
                __syncthreads();
                const int nseg = nb >> 6;
                if (t < nseg) {
                    int s = 0;
                    for (int j = 0; j < 64; j++) s += hist[nb - 1 - (t * 64 + j)];
                    segSum[t] = s;
                }
                __syncthreads();
                if (t == 0) {
                    int nd = s_need, cum = 0, sg = 0;
                    for (; sg < nseg; sg++) {
                        if (cum + segSum[sg] >= nd) break;
                        cum += segSum[sg];
                    }
                    int chosen = 0;
                    for (int j = 0; j < 64; j++) {
                        const int b = nb - 1 - (sg * 64 + j);
                        const int c = hist[b];
                        if (cum + c >= nd) { chosen = b; break; }
                        cum += c;
                    }
                    s_need = nd - cum;
                    s_greater += cum;
                    if (pass == 2) s_eqCnt = hist[chosen]; // the final 10 bits fix the key: #(key == thr)
                    s_prefix = (prefix << width) | (unsigned int)chosen;
                }
                __syncthreads();
            }
            thr = s_prefix;
            need = s_need;
            greater = s_greater;
            eqCnt = s_eqCnt;
        } else {
            thr = 0; need = 0; greater = V; eqCnt = 0; // K == V: every entry is selected
        }

        if (t == 0) { s_outGt = 0; s_eqTaken = 0; }
        __syncthreads();
        const bool allTied = (K == V) || (eqCnt == need); // every entry equal to thr is wanted
        if (allTied) {
            // The selected set is exactly {key >= thr}; arrival order is fixed by the sort below.
            for (int i = t; i < V; i += TK_THREADS) {
                const float x = logits[i];
                if (K == V || tk_key(x) >= thr) {
                    const int slot = atomicAdd(&s_outGt, 1);
                    sval[slot] = x; sidx[slot] = i;
                }
            }
            __syncthreads();
        }
        // More entries tie at thr than are needed: WHICH survive is decided by ascending id (the host's
        // tie-break): every key > thr, then the first `need` keys == thr in id order.
        for (int base = 0; !allTied && base < V; base += TK_THREADS) {
            const int i = base + t;
            unsigned int k = 0; float x = 0.0f;
            const bool inb = i < V;
            if (inb) { x = logits[i]; k = tk_key(x); }
            const bool gt = inb && k > thr;
            const bool eq = inb && k == thr;
            const unsigned int gtm = __ballot_sync(0xffffffffu, gt);
            const unsigned int eqm = __ballot_sync(0xffffffffu, eq);
            if (lane == 0) { wg[warp] = __popc(gtm); we[warp] = __popc(eqm); }
            __syncthreads();
            if (t == 0) {
                int ag = 0, ae = 0;
                for (int w = 0; w < 32; w++) { wgo[w] = ag; weo[w] = ae; ag += wg[w]; ae += we[w]; }
                s_totGt = ag; s_totEq = ae;
            }
            __syncthreads();
            const unsigned int lm = (1u << lane) - 1u;
            if (gt) {
                const int slot = s_outGt + wgo[warp] + __popc(gtm & lm);
                sval[slot] = x; sidx[slot] = i;
            } else if (eq) {
                const int rank = s_eqTaken + weo[warp] + __popc(eqm & lm);
                if (rank < need) { const int slot = greater + rank; sval[slot] = x; sidx[slot] = i; }
            }
            __syncthreads();
            if (t == 0) { s_outGt += s_totGt; s_eqTaken += s_totEq; }
            __syncthreads();
            if (s_outGt >= greater && s_eqTaken >= need) break; // block-uniform: everything selected
        }

        int P = 1;
        while (P < K) P <<= 1;
        for (int i = K + t; i < P; i += TK_THREADS) { sval[i] = -3.402823466e38f; sidx[i] = 0x7fffffff; }
        __syncthreads();
        tk_sort(sval, sidx, P);
        for (int i = t; i < K; i += TK_THREADS) { out[i] = sidx[i]; out[K + i] = __float_as_int(sval[i]); }
        if (t == 0) s_mx = sval[0];
        __syncthreads();
    }

    // ================= Z = Σ exp((l - max) * invT), for top-p's nucleus cut =================
    // Per-thread float partial sums, block sum in double.
    if (wantZ) {
        const float mx = s_mx;
        float acc = 0.0f;
        for (int base = 0; base < V; base += TK_THREADS * TK_U) {
            float xv[TK_U];
#pragma unroll
            for (int u = 0; u < TK_U; u++) {
                const int i = base + u * TK_THREADS + t;
                xv[u] = (i < V) ? logits[i] : mx;
            }
#pragma unroll
            for (int u = 0; u < TK_U; u++) {
                const int i = base + u * TK_THREADS + t;
                if (i < V) {
                    const float z = __fmul_rn(xv[u] - mx, invT); // explicit: no contraction into the sum
                    acc = __fadd_rn(acc, expf(z));
                }
            }
        }
        red[t] = (double)acc;
        __syncthreads();
        for (int o = TK_THREADS >> 1; o > 0; o >>= 1) {
            if (t < o) red[t] += red[t + o];
            __syncthreads();
        }
        if (t == 0) {
            const double z = red[0];
            const float hi = (float)z;
            out[2 * K] = __float_as_int(hi);
            out[2 * K + 1] = __float_as_int((float)(z - (double)hi)); // hi+lo reconstructs z to ~1e-15 relative
        }
    }
}

} // extern "C"
