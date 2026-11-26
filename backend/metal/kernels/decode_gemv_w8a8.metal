kernel void gemv_w8a8(device const char* aq[[buffer(0)]], device const float* asc[[buffer(1)]],
    device const char* bq[[buffer(2)]], device const float* bsc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], uint n[[thread_position_in_grid]]) {
    int acc=0; device const char* brow=bq+(uint)n*K;
    for(uint k=0;k<K;k++) acc+=int(aq[k])*int(brow[k]);
    out[n]=float(acc)*asc[0]*bsc[n];
}
// Coalesced W8A8 GEMV: ONE simdgroup (32 lanes) per output row. Adjacent lanes read
// adjacent weight bytes (brow[lid], brow[lid+32], …) — coalesced, the memory-access fix
// the CUDA arc's 43%→80% tuning was all about. simd_sum reduces the 32 partials.
// Launch total = N*32 threads, threadgroup = 32.
kernel void gemv_w8a8_coal(device const char* aq[[buffer(0)]], device const float* asc[[buffer(1)]],
    device const char* bq[[buffer(2)]], device const float* bsc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], uint gid[[threadgroup_position_in_grid]], uint lid[[thread_index_in_threadgroup]]) {
    device const char* brow = bq + (uint)gid*K;
    // CANDIDATE: vectorized 32-bit (4-packed-byte) loads instead of scalar per-byte loads. This is
    // a DIFFERENT lever than round 2's ILP-unroll (which found no win, confirming the kernel is
    // bandwidth-bound, not ILP-bound) -- reducing the LOAD INSTRUCTION COUNT 4x (same total bytes,
    // fewer/wider transactions) is the lever a bandwidth-bound kernel should actually respond to.
    // Same coalescing shape as before, just 4 bytes/lane/iteration instead of 1 (lane l reads word
    // l, l+32, ... -- adjacent lanes still hit adjacent memory). Exact integer math either way, so
    // still bit-identical regardless of grouping.
    device const uint* aq4 = (device const uint*)aq;
    device const uint* brow4 = (device const uint*)brow;
    uint G = K >> 2u;
    int acc = 0;
    for (uint g = lid; g < G; g += 32u) {
        uint aw = aq4[g], bw = brow4[g];
        acc += int(char(aw & 0xFFu))         * int(char(bw & 0xFFu));
        acc += int(char((aw >> 8) & 0xFFu))  * int(char((bw >> 8) & 0xFFu));
        acc += int(char((aw >> 16) & 0xFFu)) * int(char((bw >> 16) & 0xFFu));
        acc += int(char((aw >> 24) & 0xFFu)) * int(char((bw >> 24) & 0xFFu));
    }
    acc = simd_sum(acc);
    if (lid == 0) out[gid] = float(acc) * asc[0] * bsc[gid];
}
// W8A8 decode GEMVs for the dense body: int8 weights row-major
// [N][K] with one f32 scale per row (int8Buf), against the same int8 activations and single f32 scale (aq/asc) the
// W4A8 kernels read. Each takes its W4A8 twin's arguments in the same order and runs at the same launch shape, so
// buildResident swaps pipelines for an int8 model and no dispatch site changes. The _body kernels are the coal
// shape (one row per 32-thread threadgroup); the _sa kernels stage the activation in threadgroup memory and run 8
// rows per 256-thread threadgroup, as gemv_w4a8_sa does. The sum is exact int32 (K·127² stays far below 2³¹ at
// every decode shape), so neither shape nor lane order changes a bit. The epilogue is the CPU's,
// float(acc)·aScale·wScale (linalg's w8a8Span), rounded, then the bias, then the residual. The rounding is forced:
// fast math otherwise fuses the product into the following add as one FMA, which F1 measured off the CPU by many
// ulps wherever the residual cancels most of the sum (TestGemvW8A8Body_matchesCPU). y is uniform across the
// simdgroup after simd_sum, so simd_broadcast_first returns it unchanged, and as an opaque cross-lane op it cannot
// be fused through. K must be a multiple of 4: each lane reads one 32-bit word of four codes.
#define DOT4I8(aw, bw) ( \
    int(char((aw) & 0xFFu))         * int(char((bw) & 0xFFu)) \
  + int(char(((aw) >> 8) & 0xFFu))  * int(char(((bw) >> 8) & 0xFFu)) \
  + int(char(((aw) >> 16) & 0xFFu)) * int(char(((bw) >> 16) & 0xFFu)) \
  + int(char(((aw) >> 24) & 0xFFu)) * int(char(((bw) >> 24) & 0xFFu)) )
#define W8A8_COAL_BODY \
    device const uint* brow = (device const uint*)(bq + (uint)gid*K); \
    device const uint* a4 = (device const uint*)aq; \
    uint G = K >> 2u; \
    int acc = 0; \
    for (uint g = lid; g < G; g += 32u) acc += DOT4I8(a4[g], brow[g]); \
    acc = simd_sum(acc); \
    float y = simd_broadcast_first(float(acc) * asc[0] * bsc[gid]);
kernel void gemv_w8a8_body(device const char* bq[[buffer(0)]], device const float* bsc[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], uint gid[[threadgroup_position_in_grid]], uint lid[[thread_index_in_threadgroup]]) {
    W8A8_COAL_BODY
    if (lid == 0) out[gid] = y;
}
kernel void gemv_w8a8_body_resid(device const char* bq[[buffer(0)]], device const float* bsc[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], uint gid[[threadgroup_position_in_grid]], uint lid[[thread_index_in_threadgroup]]) {
    W8A8_COAL_BODY
    if (lid == 0) out[gid] += y;
}
kernel void gemv_w8a8_body_resid_bias(device const char* bq[[buffer(0)]], device const float* bsc[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    device const float* bias[[buffer(5)]], constant uint& K[[buffer(6)]],
    uint gid[[threadgroup_position_in_grid]], uint lid[[thread_index_in_threadgroup]]) {
    W8A8_COAL_BODY
    if (lid == 0) out[gid] += y + bias[gid];
}
#define W8A8_SA_BODY \
    threadgroup uint* As4 = (threadgroup uint*)As; \
    device const uint* a4 = (device const uint*)aq; \
    uint G = K >> 2u; \
    for (uint i = tid; i < G; i += tgs) As4[i] = a4[i]; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    uint row = tgid*(tgs>>5u) + sgid; \
    device const uint* wr = (device const uint*)(wq + (uint)row*K); \
    int acc = 0; \
    for (uint g = lane; g < G; g += 32u) acc += DOT4I8(As4[g], wr[g]); \
    acc = simd_sum(acc); \
    float y = simd_broadcast_first(float(acc) * asc[0] * sct[row]);
kernel void gemv_w8a8_sa(device const char* wq[[buffer(0)]], device const float* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], threadgroup short* As [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    W8A8_SA_BODY
    if (lane==0) out[row] = y;
}
kernel void gemv_w8a8_sa_bias(device const char* wq[[buffer(0)]], device const float* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    device const float* bias[[buffer(5)]], constant uint& K[[buffer(6)]], threadgroup short* As [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    W8A8_SA_BODY
    if (lane==0) out[row] = y + bias[row];
}
kernel void gemv_w8a8_sa_resid(device const char* wq[[buffer(0)]], device const float* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], threadgroup short* As [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    W8A8_SA_BODY
    if (lane==0) out[row] += y;
}
kernel void gemv_w8a8_sa_bias_resid(device const char* wq[[buffer(0)]], device const float* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    device const float* bias[[buffer(5)]], constant uint& K[[buffer(6)]], threadgroup short* As [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    W8A8_SA_BODY
    if (lane==0) out[row] += y + bias[row];
}
