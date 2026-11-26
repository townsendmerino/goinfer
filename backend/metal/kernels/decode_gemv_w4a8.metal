// COALESCED W4A8 GEMV core (shared by _coal/_bias/_resid). ONE simdgroup (32 lanes) per
// output row; lane l reads word l, l+32, l+64… so adjacent lanes hit adjacent memory (vs the
// old stride-4 group-per-lane pattern). Per-word int8·nibble sum × the word's group scale
// (4 words/group → scale index = word>>2; the group scale distributes over its words, so
// per-word is exact). 8-nibble inner unroll = ILP; simd_sum reduces the 32 lane partials.
#define W4A8_BODY \
    uint wpr = K/8u; \
    device const uint*  brow = bq  + (uint)gid*wpr; \
    device const half*  srow = bsc + (uint)gid*(K/32u); \
    float acc = 0.0f; \
    for (uint wi = lid; wi < wpr; wi += 32u) { \
        uint x = brow[wi]; device const char* a = aq + wi*8u; \
        int gi = (int((x)&0xF)-8)*int(a[0]) + (int((x>>4)&0xF)-8)*int(a[1]) \
               + (int((x>>8)&0xF)-8)*int(a[2]) + (int((x>>12)&0xF)-8)*int(a[3]) \
               + (int((x>>16)&0xF)-8)*int(a[4]) + (int((x>>20)&0xF)-8)*int(a[5]) \
               + (int((x>>24)&0xF)-8)*int(a[6]) + (int((x>>28)&0xF)-8)*int(a[7]); \
        acc += float(gi) * float(srow[wi>>2]); \
    } \
    acc = simd_sum(acc);

// Launch total = N*32, tg = 32. int4 weights = half the bytes of int8 (the target-quant
// bandwidth win). _coal is the plain projection; _bias/_resid fuse an epilogue.
kernel void gemv_w4a8_coal(device const uint* bq[[buffer(0)]], device const half* bsc[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], uint gid[[threadgroup_position_in_grid]], uint lid[[thread_index_in_threadgroup]]) {
    W4A8_BODY
    if (lid == 0) out[gid] = acc * asc[0];
}
kernel void gemv_w4a8_bias(device const uint* bq[[buffer(0)]], device const half* bsc[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    device const float* bias[[buffer(5)]], constant uint& K[[buffer(6)]],
    uint gid[[threadgroup_position_in_grid]], uint lid[[thread_index_in_threadgroup]]) {
    W4A8_BODY
    if (lid == 0) out[gid] = acc*asc[0] + bias[gid];
}
// gemv_w4a8_resid_bias: the coal-family counterpart to gemv_w4a8_sa_bias_resid — needed for any
// bias+residual projection whose K exceeds the SA family's 1536 cap (GPT-2's FFN down-proj:
// K=intermediate=4*hidden, e.g. 3072 for GPT-2 small — gemv_w4a8_resid has no bias epilogue,
// gemv_w4a8_bias overwrites instead of accumulating; neither is down-proj's shape for a family
// with a down-proj bias). Not currently dispatched anywhere — gated standalone until wired.
kernel void gemv_w4a8_resid_bias(device const uint* bq[[buffer(0)]], device const half* bsc[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    device const float* bias[[buffer(5)]], constant uint& K[[buffer(6)]],
    uint gid[[threadgroup_position_in_grid]], uint lid[[thread_index_in_threadgroup]]) {
    W4A8_BODY
    if (lid == 0) out[gid] += acc*asc[0] + bias[gid];
}
kernel void gemv_w4a8_resid(device const uint* bq[[buffer(0)]], device const half* bsc[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]],
    uint gid[[threadgroup_position_in_grid]], uint lid[[thread_index_in_threadgroup]]) {
    W4A8_BODY
    if (lid == 0) out[gid] += acc*asc[0];
}

// Stage A (Fable): simdgroup-per-row like _coal, but three ALU/LSU wins, NO repack:
//  (1) uint4 loads — one 128-bit load = one full 32-element scale group (4 words), so the
//      load width IS the parity structure (1 int group-sum, 1 f32 MAC per group).
//  (2) int8 activation staged once into threadgroup short (pre-widened) — replaces the
//      per-row device byte-gather (17920× re-reads) that dominates LSU issue.
//  (3) 8 simdgroups/threadgroup (tg=256) so all cores stay fed; each simdgroup = one row.
// UNP8 = 8 (nibble-8)*int8 terms, bit-identical to _coal's per-word math. As is host-sized to K
// shorts per dispatch; K is bounded by the M-11 threadgroup-memory guard at buildResident time
// (2*K bytes <= d.MaxThreadgroupMemoryLength(), ~32 KiB on Apple GPUs — K<=1536 here is stale,
// left from before that guard existed and roughly 10x too conservative against today's actual cap).
#define UNP8(x, a) ( \
    (int((x)&0xF)-8)*int((a)[0]) + (int(((x)>>4)&0xF)-8)*int((a)[1]) \
  + (int(((x)>>8)&0xF)-8)*int((a)[2]) + (int(((x)>>12)&0xF)-8)*int((a)[3]) \
  + (int(((x)>>16)&0xF)-8)*int((a)[4]) + (int(((x)>>20)&0xF)-8)*int((a)[5]) \
  + (int(((x)>>24)&0xF)-8)*int((a)[6]) + (int(((x)>>28)&0xF)-8)*int((a)[7]) )
#define UNP8V(xw, a4) ( \
    (int((xw)&0xF)-8)*int((a4)[0].x) + (int(((xw)>>4)&0xF)-8)*int((a4)[0].y) \
  + (int(((xw)>>8)&0xF)-8)*int((a4)[0].z) + (int(((xw)>>12)&0xF)-8)*int((a4)[0].w) \
  + (int(((xw)>>16)&0xF)-8)*int((a4)[1].x) + (int(((xw)>>20)&0xF)-8)*int((a4)[1].y) \
  + (int(((xw)>>24)&0xF)-8)*int((a4)[1].z) + (int(((xw)>>28)&0xF)-8)*int((a4)[1].w) )
#define SA_BODY \
    for (uint i=tid;i<K;i+=tgs) As[i]=short(aq[i]); \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    uint G = K>>5u; \
    uint row = tgid*(tgs>>5u) + sgid; \
    device const uint4* wr = wq + (uint)row*G; \
    device const half*  sr = sct + (uint)row*G; \
    float acc = 0.0f; \
    for (uint g=lane; g<G; g+=32u) { \
        uint4 w = wr[g]; threadgroup const short4* a4 = reinterpret_cast<threadgroup const short4*>(As + g*32u); \
        int gi = UNP8V(w.x,a4) + UNP8V(w.y,a4+2) + UNP8V(w.z,a4+4) + UNP8V(w.w,a4+6); \
        acc += float(gi) * float(sr[g]); \
    } \
    acc = simd_sum(acc);
kernel void gemv_w4a8_sa(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], threadgroup short* As [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    SA_BODY
    if (lane==0) out[row] = acc*asc[0];
}
// gemv_w4a8_sa_qv: quant_vec fused into gemv_w4a8_sa — item #5 of the 9-finding audit.
// MEASURED (metal/sa_qv_fusion_test.go, interleaved A/B timing at real dims K=N=1536):
// roughly NEUTRAL, leaning slightly negative (~0.97x — a few percent SLOWER, not faster). NOT
// wired into any production dispatch site — the dispatch-count argument that motivated this
// doesn't survive contact with a wall-clock measurement, so it stays a correctness-proven,
// kept-for-the-record experiment, not a live kernel. Takes the RAW f32 context x directly
// instead of pre-quantized aq/asc, and does the amax-reduction + quantize step that quant_vec
// normally does as its own dispatch, INLINE, per-threadgroup, writing straight into As instead
// of reading pre-quantized aq. Why it doesn't win: quant_vec's amax reduction is a single O(K)
// pass done ONCE; every threadgroup gemv_w4a8_sa launches (~N/8 of them, N = output rows) redoes
// that SAME O(K) reduction independently here, since Metal has no cheap way for one threadgroup
// to hand a computed scale to another within one dispatch — and that redundant cost roughly
// cancels the removed dispatch launch + the aq/asc device-memory round-trip it saves.
kernel void gemv_w4a8_sa_qv(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const float* x[[buffer(2)]], device float* out[[buffer(3)]],
    constant uint& K[[buffer(4)]], threadgroup short* As [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    threadgroup float red[256];
    float mx = 0.0f;
    for (uint i=tid; i<K; i+=tgs) mx = max(mx, fabs(x[i]));
    red[tid] = mx; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s=tgs/2; s>0; s>>=1) { if (tid<s) red[tid]=max(red[tid],red[tid+s]); threadgroup_barrier(mem_flags::mem_threadgroup); }
    float sc = red[0]/127.0f; if (sc==0) sc=1; float inv=1/sc;
    for (uint i=tid;i<K;i+=tgs) As[i]=short(clamp(int(round(x[i]*inv)),-127,127));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint G = K>>5u;
    uint row = tgid*(tgs>>5u) + sgid;
    device const uint4* wr = wq  + (uint)row*G;
    device const half*  sr = sct + (uint)row*G;
    float acc = 0.0f;
    for (uint g=lane; g<G; g+=32u) {
        uint4 w = wr[g]; threadgroup const short* a = As + g*32u;
        int gi = UNP8(w.x,a) + UNP8(w.y,a+8) + UNP8(w.z,a+16) + UNP8(w.w,a+24);
        acc += float(gi) * float(sr[g]);
    }
    acc = simd_sum(acc);
    if (lane==0) out[row] = acc*sc;
}

// BATCH-K W4A8 GEMM (the speculation lever): one weight matrix × KK token activations in ONE
// pass. Each simdgroup owns one output row; it unpacks each weight group's 32 nibbles ONCE and
// MACs them against ALL kk staged activation vectors (kk accumulators). The issue-bound nibble
// unpack (extract+widen, no DP4A on Apple) is thus amortized across kk tokens — the mechanism
// that converts the unused bandwidth into throughput when a speculator drafts kk candidates.
// Activations staged [kk][K] int8→short in threadgroup memory; out is row-major [kk][N].
#define KK_MAX 10
kernel void gemv_w4a8_sa_bk(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], constant uint& N[[buffer(6)]], constant uint& kk[[buffer(7)]],
    threadgroup short* As [[threadgroup(0)]],        // kk*K shorts, host-sized (occupancy: only what k needs)
    uint tgid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]],
    uint tgs[[threads_per_threadgroup]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]]) {
    for (uint i=tid; i<kk*K; i+=tgs) As[i] = short(aq[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint G = K>>5u;
    uint row = tgid*(tgs>>5u) + sgid;
    if (row >= N) return;
    device const uint4* wr = wq  + (uint)row*G;
    device const half*  sr = sct + (uint)row*G;
    float acc[KK_MAX];
    for (uint j=0;j<kk;j++) acc[j]=0.0f;
    for (uint g=lane; g<G; g+=32u) {
        uint4 w = wr[g];
        // Unpack each word's 8 nibbles ONCE into scalars (compiler keeps them in registers),
        // then reuse across all kk activations via the unrolled 8-term dot (UNP8 form).
        int x0=int(w.x&0xF)-8, x1=int((w.x>>4)&0xF)-8, x2=int((w.x>>8)&0xF)-8, x3=int((w.x>>12)&0xF)-8,
            x4=int((w.x>>16)&0xF)-8, x5=int((w.x>>20)&0xF)-8, x6=int((w.x>>24)&0xF)-8, x7=int((w.x>>28)&0xF)-8;
        int y0=int(w.y&0xF)-8, y1=int((w.y>>4)&0xF)-8, y2=int((w.y>>8)&0xF)-8, y3=int((w.y>>12)&0xF)-8,
            y4=int((w.y>>16)&0xF)-8, y5=int((w.y>>20)&0xF)-8, y6=int((w.y>>24)&0xF)-8, y7=int((w.y>>28)&0xF)-8;
        int z0=int(w.z&0xF)-8, z1=int((w.z>>4)&0xF)-8, z2=int((w.z>>8)&0xF)-8, z3=int((w.z>>12)&0xF)-8,
            z4=int((w.z>>16)&0xF)-8, z5=int((w.z>>20)&0xF)-8, z6=int((w.z>>24)&0xF)-8, z7=int((w.z>>28)&0xF)-8;
        int u0=int(w.w&0xF)-8, u1=int((w.w>>4)&0xF)-8, u2=int((w.w>>8)&0xF)-8, u3=int((w.w>>12)&0xF)-8,
            u4=int((w.w>>16)&0xF)-8, u5=int((w.w>>20)&0xF)-8, u6=int((w.w>>24)&0xF)-8, u7=int((w.w>>28)&0xF)-8;
        float sc = float(sr[g]);
        for (uint j=0;j<kk;j++) {
            threadgroup const short* a = As + j*K + g*32u;
            int gi = x0*int(a[0])+x1*int(a[1])+x2*int(a[2])+x3*int(a[3])+x4*int(a[4])+x5*int(a[5])+x6*int(a[6])+x7*int(a[7])
                   + y0*int(a[8])+y1*int(a[9])+y2*int(a[10])+y3*int(a[11])+y4*int(a[12])+y5*int(a[13])+y6*int(a[14])+y7*int(a[15])
                   + z0*int(a[16])+z1*int(a[17])+z2*int(a[18])+z3*int(a[19])+z4*int(a[20])+z5*int(a[21])+z6*int(a[22])+z7*int(a[23])
                   + u0*int(a[24])+u1*int(a[25])+u2*int(a[26])+u3*int(a[27])+u4*int(a[28])+u5*int(a[29])+u6*int(a[30])+u7*int(a[31]);
            acc[j] = fma(float(gi), sc, acc[j]);
        }
    }
    for (uint j=0;j<kk;j++) {
        float s = simd_sum(acc[j]);
        if (lane==0) out[j*N + row] = s * asc[j];
    }
}
kernel void gemv_w4a8_sa_bias(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    device const float* bias[[buffer(5)]], constant uint& K[[buffer(6)]], threadgroup short* As [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    SA_BODY
    if (lane==0) out[row] = acc*asc[0] + bias[row];
}
kernel void gemv_w4a8_sa_resid(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], threadgroup short* As [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    SA_BODY
    if (lane==0) out[row] += acc*asc[0];
}

