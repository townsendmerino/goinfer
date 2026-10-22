// kvpage.cuh — paged KV addressing (docs/tasks/task-kv-architecture-2026-10.md §5.3).
//
// A resident layer's K and V live in a pool of 64-row blocks, [numBlocks][64][kvDim]. A sequence's table names, for each 64 positions
// of it, the pool block that holds them; entry 0 is the zero-filled null block, so a released or unused entry reads zeros, never
// another sequence's rows. Every kernel that reads or writes the cache takes the table beside its cache pointers and addresses key k
// through KV_ROW, per key: tiles, partitions and thread-to-key maps stay in absolute key units, so the arithmetic is the same whatever
// the table holds, and a table that maps block i to block i reproduces the contiguous layout byte for byte.
#ifndef GOINFER_KVPAGE_CUH
#define GOINFER_KVPAGE_CUH
// KV_ROW(t, k): the physical row of logical position k (k >= 0) under table t, in 64-bit arithmetic so a large pool cannot overflow.
#define KV_ROW(t, k) ((long)(t)[(k) >> 6] * 64 + (long)((k) & 63))
// KV_RUN_END(k, hi): the end of the run of keys [k, hi) that k's block holds. A loop that walks keys in ascending order looks a block up
// once per run and steps a pointer inside it, so the table costs nothing in its inner loop and the arithmetic keeps its order.
#define KV_RUN_END(k, hi) (((k) | 63) + 1 < (hi) ? ((k) | 63) + 1 : (hi))
// KV_SUM_RUNS(base, t, stride, lo, hi, acc, w): acc = fma(w, KV_LD(row of s), acc) for s in [lo, hi), ascending, where base is the
// element of row 0 to read and w an expression of s. One lookup per 64-key run, whose next run's row is read before the run starts,
// and the run fully unrolled: a latency-bound fold (one thread per output element walking every key) then issues its loads as far
// ahead as the flat loop over a contiguous cache did, which a two-level loop without the unroll does not.
#define KV_SUM_RUNS(base, t, stride, lo, hi, acc, w)                                                                          \
    {                                                                                                                        \
        const KV_T* kvsb_ = (base);                                                                                          \
        const KV_T* kvsp_ = kvsb_ + KV_ROW(t, lo) * (stride);                                                               \
        for (int s = (lo); s < (hi);) {                                                                                      \
            const int e_ = KV_RUN_END(s, hi);                                                                                \
            const KV_T* kvsn_ = (e_ < (hi)) ? kvsb_ + (long)(t)[e_ >> 6] * 64 * (stride) : kvsp_;                           \
            _Pragma("unroll 64") for (; s < e_; s++, kvsp_ += (stride)) acc = __fmaf_rn(w, KV_LD(kvsp_), acc);                \
            kvsp_ = kvsn_;                                                                                                   \
        }                                                                                                                    \
    }
#endif
