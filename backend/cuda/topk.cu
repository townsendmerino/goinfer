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
// THREE LAUNCHES, in stream order (cuda/topk.go:cudaResident.topkReadback): topk_compact over the grid (fast
// path only), topk_finish on one block, and topk_z over the grid when Z is wanted. The global scratch they
// share (the candidates, their count, the max, Z's partials and a done counter) is the resident's.
//
// THE TIE ORDER IS A CONTRACT, not a preference. decoder.topKByLogit keeps the k best by
// (logit desc, id asc) — "ties evict larger ids" — and that order feeds the cumulative-CDF draw.
// -0.0 and +0.0 compare EQUAL on the host (!=), and here every comparison is a float compare or a
// canonicalised key, so they tie as they do there.
//
// TWO PATHS, ONE ANSWER. Both return exactly the first K of "sort the row by (logit desc, id asc)".
//
//   FAST PATH (the normal case). Only ~K of ~150k entries matter, so do not histogram the row. Sample
//   it at a fixed stride and take the R-th largest sample (an exact selection) as a threshold t0 chosen
//   so the expected number of row entries >= t0 is a few times K. One pass over the grid compacts every
//   entry >= t0 (a plain atomic append: a few hundred to a few thousand entries); one block selects the
//   first K of them and a bitonic sort orders those (the whole set when more tie at the K boundary than
//   are needed). That is exact whenever K <= count <= CAP, because every entry >= t0 was collected and
//   the K-th largest is >= t0. The threshold is only an ESTIMATE; correctness never depends on it — a
//   count outside [K, CAP] hands off to the slow path.
//
//   SLOW PATH (exact, estimate-free). A 3-pass radix select (11+11+10 bits over an order-preserving
//   float->uint key, -0 canonicalised to +0) finds the K-th largest key exactly; a gather takes every
//   entry above it plus the first `need` equal to it by ASCENDING id when more tie than are needed;
//   a bitonic sort orders the K survivors. Used for short vocabularies, K close to V, and any row
//   whose sampled estimate missed (heavy ties, adversarial layouts).
//
// WHY NOT A HISTOGRAM-ONLY SELECT: the one clean ncu reading of a radix-select kernel here (warp-aggregated
// histogram, V=152k, normal logits, RTX 2070 SUPER, 2026-09-20) was ~222 us, with the hottest instructions on
// BREV — how __match_any_sync is microcoded on Turing. That was too slow for the budget, so the fast path
// above avoids histogramming the whole row; the radix path remains as the exact fallback.
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

// The float a canonical key stands for (tk_key's inverse; -0 comes back as +0).
__device__ __forceinline__ float tk_val(unsigned int k) {
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}

// tk_kth finds the kth largest canonical key (1-based) of v[0..n): on return *s_prefix is that key, *s_greater the count
// of keys above it and *s_eqCnt the count equal to it, so kth - *s_greater of the equal ones are needed. The same
// 11+11+10-bit radix passes and scan as the slow path's, over an array instead of the row. Every thread of the block must
// call it (it contains barriers).
__device__ void tk_kth(const float* v, int n, int kth, int* hist, int* segSum, unsigned int* s_prefix, int* s_need,
                       int* s_greater, int* s_eqCnt) {
    const int t = threadIdx.x;
    if (t == 0) { *s_prefix = 0; *s_need = kth; *s_greater = 0; *s_eqCnt = 0; }
    __syncthreads();
    for (int pass = 0; pass < 3; pass++) {
        const int shift = (pass == 0) ? 21 : ((pass == 1) ? 10 : 0);
        const int width = (pass == 2) ? 10 : 11;
        const int nb = 1 << width;
        for (int i = t; i < TK_BINS; i += TK_THREADS) hist[i] = 0;
        __syncthreads();
        const unsigned int prefix = *s_prefix;
        for (int i = t; i < n; i += TK_THREADS) {
            const unsigned int k = tk_key(v[i]);
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
            int nd = *s_need, cum = 0, sg = 0;
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
            *s_need = nd - cum;
            *s_greater += cum;
            if (pass == 2) *s_eqCnt = hist[chosen];
            *s_prefix = (prefix << width) | (unsigned int)chosen;
        }
        __syncthreads();
    }
}

// tk_fast reports whether a (V, K) row takes the sampled fast path; the host launches topk_compact only then.
__device__ __forceinline__ bool tk_fast(int V, int K) { return (V >= 8192) && (K * 8 <= V); }

// topk_compact is the fast path's first half, spread over the grid: every block finds the same threshold t0 (the R-th
// largest of TK_SAMPLES strided samples, by exact selection rather than a sort; the value is the same) and appends every
// entry >= t0 of its slice of the row to gcv/gci, counting all of them in *gcnt (zero on entry; topk_finish resets it).
// Launched only when tk_fast(V, K).
__global__ void __launch_bounds__(TK_THREADS)
topk_compact(const float* __restrict__ logits, int V, int K, float* __restrict__ gcv, int* __restrict__ gci,
             int* __restrict__ gcnt) {
    __shared__ float samples[TK_SAMPLES];
    __shared__ int hist[TK_BINS];
    __shared__ int segSum[TK_BINS / 64];
    __shared__ unsigned int s_prefix;
    __shared__ int s_need, s_greater, s_eqCnt;
    const int t = threadIdx.x;
    if (K > V) K = V;
    if (K > TK_KMAX) K = TK_KMAX;
    for (int i = t; i < TK_SAMPLES; i += TK_THREADS) {
        const long long idx = ((long long)i * (long long)V) / TK_SAMPLES;
        samples[i] = logits[idx];
    }
    __syncthreads();
    float target = 5.0f * (float)K;
    if (target > 2400.0f) target = 2400.0f;
    int R = (int)(target * (float)TK_SAMPLES / (float)V + 0.5f);
    if (R < 8) R = 8;
    if (R > 64) R = 64;
    tk_kth(samples, TK_SAMPLES, R, hist, segSum, &s_prefix, &s_need, &s_greater, &s_eqCnt);
    const float t0 = tk_val(s_prefix);
    const int span = (V + gridDim.x - 1) / gridDim.x;
    const int lo = blockIdx.x * span, hi = (lo + span < V) ? lo + span : V;
    for (int base = lo; base < hi; base += TK_THREADS * TK_U) {
        float xv[TK_U];
#pragma unroll
        for (int u = 0; u < TK_U; u++) {
            const int i = base + u * TK_THREADS + t;
            xv[u] = (i < hi) ? logits[i] : -3.402823466e38f;
        }
#pragma unroll
        for (int u = 0; u < TK_U; u++) {
            const int i = base + u * TK_THREADS + t;
            if (i < hi && xv[u] >= t0) {
                const int slot = atomicAdd(gcnt, 1);
                if (slot < TK_CAP) { gcv[slot] = xv[u]; gci[slot] = i; }
            }
        }
    }
}

// topk_finish orders the selection and writes ids and logit bits to out, and the largest logit to *mxOut for topk_z.
// When tk_fast(V, K) holds, topk_compact has already collected the candidates; a count inside [K, TK_CAP] is ordered
// here, and any other takes the exact slow path over the row, as before. It resets *gcnt for the next row.
__global__ void __launch_bounds__(TK_THREADS)
topk_finish(const float* __restrict__ logits, int V, int K, const float* __restrict__ gcv, const int* __restrict__ gci,
            int* __restrict__ gcnt, int* __restrict__ out, float* __restrict__ mxOut) {
    // One raw scratch region, carved differently per path (the paths never overlap in time):
    //   fast:  cv float[4096] @0 and ci int[4096] @16384, with chist (its own array) for the selection
    //   slow:  hist int[2048] @0, sval float[1024] @8192, sidx int[1024] @12288
    __shared__ __align__(16) unsigned char raw[32768];
    __shared__ int chist[TK_BINS];
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

    // ================= FAST PATH =================
    if (t == 0) { s_ok = 0; s_cnt = 0; }
    __syncthreads();
    const bool tryFast = tk_fast(V, K);
    if (tryFast) {
        const int cnt = *gcnt;
        if (cnt >= K && cnt <= TK_CAP) {
            for (int i = t; i < cnt; i += TK_THREADS) { cv[i] = gcv[i]; ci[i] = gci[i]; }
            __syncthreads();
            // Only the first K of (logit desc, id asc) are written, so select them before sorting: every candidate with a
            // key above the K-th largest, plus the equal ones when all of them are needed. When more tie at the boundary
            // than are needed, which survive is decided by id, and the whole candidate set is sorted instead.
            tk_kth(cv, cnt, K, chist, segSum, &s_prefix, &s_need, &s_greater, &s_eqCnt);
            const unsigned int thr = s_prefix;
            int P = 1;
            if (s_eqCnt == s_need) {
                constexpr int PER = TK_CAP / TK_THREADS;
                float rv[PER];
                int ri[PER];
                bool keep[PER];
#pragma unroll
                for (int u = 0; u < PER; u++) {
                    const int i = u * TK_THREADS + t;
                    keep[u] = (i < cnt) && tk_key(cv[i]) >= thr;
                    rv[u] = keep[u] ? cv[i] : 0.0f;
                    ri[u] = keep[u] ? ci[i] : 0;
                }
                __syncthreads();
#pragma unroll
                for (int u = 0; u < PER; u++) {
                    if (keep[u]) {
                        const int slot = atomicAdd(&s_cnt, 1);
                        cv[slot] = rv[u]; ci[slot] = ri[u];
                    }
                }
                __syncthreads();
                while (P < K) P <<= 1;
                for (int i = K + t; i < P; i += TK_THREADS) { cv[i] = -3.402823466e38f; ci[i] = 0x7fffffff; }
            } else {
                while (P < cnt) P <<= 1;
                for (int i = cnt + t; i < P; i += TK_THREADS) { cv[i] = -3.402823466e38f; ci[i] = 0x7fffffff; }
            }
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

    if (t == 0) { *mxOut = s_mx; *gcnt = 0; }
}

// topk_z writes Z = Σ exp((l - max) * invT), for top-p's nucleus cut, to out[2K..2K+2) as hi/lo float bits. It is the
// single block's sum spread over the grid without changing a rounding: virtual thread t = blockIdx.x * blockDim.x +
// threadIdx.x of TK_THREADS sums row entries t, t + TK_THREADS, ... in ascending order in f32, exactly the partial the
// one-block form gave thread t, and the last block to finish runs the same double tree over the TK_THREADS partials.
// Launch with gridDim.x * blockDim.x == TK_THREADS and blockDim.x a multiple of 32. *done is zero on entry and is reset.
__global__ void topk_z(const float* __restrict__ logits, int V, int K, float invT, const float* __restrict__ mxIn,
                       float* __restrict__ partial, int* __restrict__ done, int* __restrict__ out) {
    __shared__ double red[TK_THREADS];
    __shared__ int s_last;
    if (K > V) K = V;
    if (K > TK_KMAX) K = TK_KMAX;
    const int vt = blockIdx.x * blockDim.x + threadIdx.x;
    const float mx = *mxIn;
    float acc = 0.0f;
    for (int base = vt; base < V; base += TK_THREADS * TK_U) {
        float xv[TK_U];
#pragma unroll
        for (int u = 0; u < TK_U; u++) {
            const int i = base + u * TK_THREADS;
            xv[u] = (i < V) ? logits[i] : mx;
        }
#pragma unroll
        for (int u = 0; u < TK_U; u++) {
            const int i = base + u * TK_THREADS;
            if (i < V) {
                const float z = __fmul_rn(xv[u] - mx, invT); // explicit: no contraction into the sum
                acc = __fadd_rn(acc, expf(z));
            }
        }
    }
    partial[vt] = acc;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) s_last = (atomicAdd(done, 1) == (int)gridDim.x - 1);
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    for (int i = threadIdx.x; i < TK_THREADS; i += blockDim.x) red[i] = (double)__ldcg(&partial[i]);
    __syncthreads();
    for (int o = TK_THREADS >> 1; o > 0; o >>= 1) {
        for (int i = threadIdx.x; i < o; i += blockDim.x) red[i] += red[i + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const double z = red[0];
        const float hi = (float)z;
        out[2 * K] = __float_as_int(hi);
        out[2 * K + 1] = __float_as_int((float)(z - (double)hi)); // hi+lo reconstructs z to ~1e-15 relative
        *done = 0;
    }
}

} // extern "C"
