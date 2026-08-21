//go:build darwin

package metal

// allKernels is the full dense-decode-layer MSL kernel set in one library (W8A8 path —
// W4A8 is validated separately; this proves ASSEMBLY, not the int4 packing again).
const allKernels = `
#include <metal_stdlib>
using namespace metal;

// addOne selects Gemma's (1+w) RMS offset vs plain w — mirrors decoder/rmsnorm.go, which
// applies the weight AFTER the normalize: (v*inv) * (1+w[i]).
kernel void rmsnorm_quant(device const float* x[[buffer(0)]], device const float* w[[buffer(1)]],
    device char* aq[[buffer(2)]], device float* asc[[buffer(3)]], constant uint& H[[buffer(4)]],
    constant float& eps[[buffer(5)]], constant uint& addOne[[buffer(6)]],
    uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256]; float ss=0;
    for(uint i=tid;i<H;i+=tgs) ss+=x[i]*x[i];
    red[tid]=ss; threadgroup_barrier(mem_flags::mem_threadgroup);
    // SUM-OF-SQUARES reduction — width-coupled (non-associative float add). tgs pinned to tgReduceNorm
    // (256, model.go); not a tuning knob. Representative of the norm-class kernels (rmsnorm_f32,
    // rmsnorm_*_f16, qk_norm*) — all share this contract. See ollama-chase §A2-Metal.
    for(uint s=tgs/2;s>0;s>>=1){ if(tid<s) red[tid]+=red[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup);}
    float rms=rsqrt(red[0]/float(H)+eps); threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx=0; for(uint i=tid;i<H;i+=tgs){ float g=addOne!=0u?(1.0f+w[i]):w[i]; mx=max(mx,fabs(x[i]*rms*g)); }
    red[tid]=mx; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=tgs/2;s>0;s>>=1){ if(tid<s) red[tid]=max(red[tid],red[tid+s]); threadgroup_barrier(mem_flags::mem_threadgroup);}
    float sc=red[0]/127.0f; if(sc==0)sc=1; if(tid==0)asc[0]=sc; float inv=1/sc;
    for(uint i=tid;i<H;i+=tgs){ float g=addOne!=0u?(1.0f+w[i]):w[i]; aq[i]=char(clamp(int(round(x[i]*rms*g*inv)),-127,127)); }
}
// rmsnorm_f32: plain IN-PLACE RMSNorm of a [H] vector — no fused quant, because it norms a
// SUBLAYER OUTPUT into the f32 residual stream rather than a GEMV input. This is Gemma's
// sandwich norm (NormSandwich4): y = proj(...); y = rms(y)*(1+w_post); x += y — which is why
// the fused _resid GEMV epilogue can't be used on that path.
kernel void rmsnorm_f32(device float* x[[buffer(0)]], device const float* w[[buffer(1)]],
    constant uint& H[[buffer(2)]], constant float& eps[[buffer(3)]], constant uint& addOne[[buffer(4)]],
    uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256]; float ss=0;
    for(uint i=tid;i<H;i+=tgs) ss+=x[i]*x[i];
    red[tid]=ss; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=tgs/2;s>0;s>>=1){ if(tid<s) red[tid]+=red[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup);}
    float rms=rsqrt(red[0]/float(H)+eps); threadgroup_barrier(mem_flags::mem_threadgroup);
    // Vectorized float4 in-place read/write -- no reduction at all, each x[i] is computed fully
    // independently, so batching 4 elements per iteration is mathematically identical regardless
    // of grouping (same argument verified safe for quant_vec/layernorm_quant this session).
    uint H4 = H >> 2u;
    device float4* x4 = (device float4*)x;
    device const float4* w4 = (device const float4*)w;
    for (uint i4=tid; i4<H4; i4+=tgs) {
        float4 xv=x4[i4], wv=w4[i4];
        float4 gv = addOne!=0u ? (float4(1.0f)+wv) : wv;
        x4[i4] = xv*rms*gv;
    }
    for (uint i=(H4<<2u)+tid; i<H; i+=tgs) { float g=addOne!=0u?(1.0f+w[i]):w[i]; x[i]=x[i]*rms*g; }
}
// layernorm_quant: FeatLayerNorm's fused norm+quant, mirroring rmsnorm_quant's contract but
// mean-centered (decoder/rmsnorm.go's layerNorm): y = (x-mean)/sqrt(var+eps)*w + b, then
// quantized. hasBias selects GPT-2's weight+bias LayerNorm vs Cohere's bias-free variant (both
// declare Norm==NormLayer; the CPU reference branches on a nil bias the same way). Three
// reduction passes (mean, variance, maxabs) vs rmsnorm_quant's two — LayerNorm needs the mean
// subtracted before anything else can be computed, which RMSNorm has no equivalent of.
kernel void layernorm_quant(device const float* x[[buffer(0)]], device const float* w[[buffer(1)]],
    device const float* b[[buffer(2)]], device char* aq[[buffer(3)]], device float* asc[[buffer(4)]],
    constant uint& H[[buffer(5)]], constant float& eps[[buffer(6)]], constant uint& hasBias[[buffer(7)]],
    uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256];
    float s=0; for(uint i=tid;i<H;i+=tgs) s+=x[i];
    red[tid]=s; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint st=tgs/2;st>0;st>>=1){ if(tid<st) red[tid]+=red[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup);}
    float mean=red[0]/float(H); threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss=0; for(uint i=tid;i<H;i+=tgs){ float d=x[i]-mean; ss+=d*d; }
    red[tid]=ss; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint st=tgs/2;st>0;st>>=1){ if(tid<st) red[tid]+=red[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup);}
    float inv=rsqrt(red[0]/float(H)+eps); threadgroup_barrier(mem_flags::mem_threadgroup);
    // Vectorized float4/char4 load/store for the maxabs-scan and quantize-write loops ONLY --
    // the mean/variance reductions above are order-sensitive float sums and stay untouched. max
    // is exact/order-independent and the quantize-write has no reduction at all, same argument
    // verified safe for quant_vec (this session) but NOT for rmsnorm_quant's sibling loops
    // (see scripts/autoresearch_rmsnorm_results.tsv) -- verified here against
    // TestGPT2ResidentParityMetal, not just the isolated unit test.
    uint H4 = H >> 2u;
    device const float4* x4 = (device const float4*)x;
    device const float4* w4 = (device const float4*)w;
    device const float4* b4 = (device const float4*)b;
    float mx=0;
    for (uint i4=tid; i4<H4; i4+=tgs) {
        float4 xv=x4[i4], wv=w4[i4];
        float4 yv=(xv-mean)*inv*wv;
        if (hasBias!=0u) yv += b4[i4];
        float4 av=fabs(yv);
        mx=max(mx, max(max(av.x,av.y), max(av.z,av.w)));
    }
    for (uint i=(H4<<2u)+tid; i<H; i+=tgs) { float y=(x[i]-mean)*inv*w[i]; if(hasBias!=0u) y+=b[i]; mx=max(mx,fabs(y)); }
    // 2-level SIMD-shuffle reduction instead of the 8-step threadgroup-barrier tree -- max is
    // exact/order-independent regardless of reduction structure (unlike the mean/variance sums
    // above, which stay untouched). Cuts barrier count from 8 to 2 (verified safe on quant_vec).
    mx = simd_max(mx);
    uint sgid = tid >> 5u, lane = tid & 31u;
    if (lane == 0) red[sgid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nsg = tgs >> 5u;
    if (tid == 0) { float v = red[0]; for (uint k=1; k<nsg; k++) v = max(v, red[k]); red[0] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sc=red[0]/127.0f; if(sc==0)sc=1; if(tid==0)asc[0]=sc; float invsc=1/sc;
    device char4* aq4 = (device char4*)aq;
    for (uint i4=tid; i4<H4; i4+=tgs) {
        float4 xv=x4[i4], wv=w4[i4];
        float4 yv=(xv-mean)*inv*wv;
        if (hasBias!=0u) yv += b4[i4];
        int4 qv=int4(round(yv*invsc));
        qv=clamp(qv,-127,127);
        aq4[i4]=char4(qv);
    }
    for (uint i=(H4<<2u)+tid; i<H; i+=tgs) { float y=(x[i]-mean)*inv*w[i]; if(hasBias!=0u) y+=b[i]; aq[i]=char(clamp(int(round(y*invsc)),-127,127)); }
}
kernel void quant_vec(device const float* x[[buffer(0)]], device char* aq[[buffer(1)]],
    device float* asc[[buffer(2)]], constant uint& H[[buffer(3)]],
    uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256];
    // Vectorized float4/char4 load/store for both loops -- max is exact/order-independent and
    // the quantize-write has no reduction at all, so this is mathematically identical regardless
    // of grouping. Verified against both the isolated unit test AND the whole-model snapshot
    // golden (unlike rmsnorm_quant's sibling loops, which drift at the whole-model level despite
    // an identical argument -- see scripts/autoresearch_rmsnorm_results.tsv for that history).
    uint H4 = H >> 2u;
    device const float4* x4 = (device const float4*)x;
    float mx=0;
    for (uint i4=tid; i4<H4; i4+=tgs) {
        float4 xv=x4[i4];
        float4 av=fabs(xv);
        mx=max(mx, max(max(av.x,av.y), max(av.z,av.w)));
    }
    for (uint i=(H4<<2u)+tid; i<H; i+=tgs) mx=max(mx,fabs(x[i]));
    // 2-level SIMD-shuffle reduction instead of the 8-step threadgroup-barrier tree -- max is
    // exact/order-independent regardless of HOW the reduction is structured (unlike a sum, which
    // is why this technique is scoped to maxabs reductions only, never the sum-of-squares class).
    // simd_max reduces within each 32-lane simdgroup with no barrier at all; only the nsg=tgs/32
    // simdgroup leaders write to threadgroup memory, cutting barrier count from 8 to 2.
    mx = simd_max(mx);
    uint sgid = tid >> 5u, lane = tid & 31u;
    if (lane == 0) red[sgid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nsg = tgs >> 5u;
    if (tid == 0) { float v = red[0]; for (uint k=1; k<nsg; k++) v = max(v, red[k]); red[0] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sc=red[0]/127.0f; if(sc==0)sc=1; if(tid==0)asc[0]=sc; float inv=1/sc;
    device char4* aq4 = (device char4*)aq;
    for (uint i4=tid; i4<H4; i4+=tgs) {
        float4 xv=x4[i4];
        int4 qv=int4(round(xv*inv));
        qv=clamp(qv,-127,127);
        aq4[i4]=char4(qv);
    }
    for (uint i=(H4<<2u)+tid; i<H; i+=tgs) aq[i]=char(clamp(int(round(x[i]*inv)),-127,127));
}
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
// UNP8 = 8 (nibble-8)*int8 terms, bit-identical to _coal's per-word math. K<=1536 (As sized).
#define UNP8(x, a) ( \
    (int((x)&0xF)-8)*int((a)[0]) + (int(((x)>>4)&0xF)-8)*int((a)[1]) \
  + (int(((x)>>8)&0xF)-8)*int((a)[2]) + (int(((x)>>12)&0xF)-8)*int((a)[3]) \
  + (int(((x)>>16)&0xF)-8)*int((a)[4]) + (int(((x)>>20)&0xF)-8)*int((a)[5]) \
  + (int(((x)>>24)&0xF)-8)*int((a)[6]) + (int(((x)>>28)&0xF)-8)*int((a)[7]) )
#define SA_BODY \
    for (uint i=tid;i<K;i+=tgs) As[i]=short(aq[i]); \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    uint G = K>>5u; \
    uint row = tgid*(tgs>>5u) + sgid; \
    device const uint4* wr = wq + (uint)row*G; \
    device const half*  sr = sct + (uint)row*G; \
    float acc = 0.0f; \
    for (uint g=lane; g<G; g+=32u) { \
        uint4 w = wr[g]; threadgroup const short* a = As + g*32u; \
        int gi = UNP8(w.x,a) + UNP8(w.y,a+8) + UNP8(w.z,a+16) + UNP8(w.w,a+24); \
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
// gemv_w4a8_sa_bias_resid: FeatOutBias's kernel — the o-proj GEMV needs BOTH an additive
// per-row bias AND direct residual accumulation (gemv_w4a8_sa_resid has no bias epilogue,
// gemv_w4a8_sa_bias overwrites instead of accumulating; neither alone is o-proj's shape for a
// family with OutBias, e.g. GPT-2/gpt-oss). Not currently dispatched anywhere — no family
// resident on Metal declares FeatOutBias yet — gated standalone (gptoss_kernels_test.go-style)
// against a hand-computed reference until a real family wires it in.
kernel void gemv_w4a8_sa_bias_resid(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    device const float* bias[[buffer(5)]], constant uint& K[[buffer(6)]], threadgroup short* As [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    SA_BODY
    if (lane==0) out[row] += acc*asc[0] + bias[row];
}

// Fused block-argmax lm-head (Fable): computes the SAME logits as gemv_w4a8_sa but never
// materializes them — each threadgroup emits (maxLogit, rowIndex) over its 8 rows; a tiny
// second pass (argmax_finish) reduces the tiles to one token. Kills the 608KB logit readback
// + CPU scan. Merge key (v, -idx) is a commutative monoid → order-independent, tie-broken
// identically to a CPU first-max-wins scan (strict >, lower index wins).
struct AmaxPart { float v; uint i; };
// NOTE (N-09): NOT currently dispatched — no pipeline is created for it (see model.go). Unlike the
// batch-K variant it takes no N and has no row>=N guard, so before wiring it, add N and mask any
// out-of-range row's logit to -INFINITY (it's a reduction: all simdgroups must reach the barrier, so
// do NOT early-return like the store variants). SA_BODY reads weight row=row, so guard the read too.
kernel void gemv_w4a8_sa_amax(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device AmaxPart* part[[buffer(4)]],
    constant uint& K[[buffer(5)]], threadgroup short* As [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    SA_BODY                                       // acc = this row's dot; row = output index
    threadgroup float tv[8]; threadgroup uint ti[8];
    if (lane==0) { tv[sgid] = acc*asc[0]; ti[sgid] = row; }   // the logit, exactly as the store variant
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid==0) {
        uint nsg = tgs>>5u; float bv = tv[0]; uint bi = ti[0];
        for (uint s=1u; s<nsg; s++) if (tv[s]>bv || (tv[s]==bv && ti[s]<bi)) { bv=tv[s]; bi=ti[s]; }
        part[tgid].v = bv; part[tgid].i = bi;
    }
}
// int8 twin of gemv_w4a8_sa_amax — the LM head is logit-critical and pinned at int8, so the
// fused block-argmax must read it as int8 too or the greedy fast path disagrees with
// argmax(full logits). Same launch shape (8 simdgroups/threadgroup, one AmaxPart per group over
// its 8 rows), so the tile count and argmax_finish are unchanged.
kernel void gemv_w8a8_amax(device const char* aq[[buffer(0)]], device const float* asc[[buffer(1)]],
    device const char* bq[[buffer(2)]], device const float* bsc[[buffer(3)]], device AmaxPart* part[[buffer(4)]],
    constant uint& K[[buffer(5)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    uint row = tgid*(tgs>>5u) + sgid;
    device const char* brow = bq + (uint)row*K;
    // Vectorized 32-bit (4-packed-byte) loads instead of scalar per-byte loads — the same
    // technique that won ~11% on gemv_w8a8_coal, applied here since this kernel shares its
    // exact bandwidth-bound inner-loop shape. Exact integer math either way.
    device const uint* aq4 = (device const uint*)aq;
    device const uint* brow4 = (device const uint*)brow;
    uint G = K >> 2u;
    int acc = 0;
    for (uint g = lane; g < G; g += 32u) {
        uint aw = aq4[g], bw = brow4[g];
        acc += int(char(aw & 0xFFu))         * int(char(bw & 0xFFu));
        acc += int(char((aw >> 8) & 0xFFu))  * int(char((bw >> 8) & 0xFFu));
        acc += int(char((aw >> 16) & 0xFFu)) * int(char((bw >> 16) & 0xFFu));
        acc += int(char((aw >> 24) & 0xFFu)) * int(char((bw >> 24) & 0xFFu));
    }
    acc = simd_sum(acc);
    threadgroup float tv[8]; threadgroup uint ti[8];
    if (lane==0) { tv[sgid] = float(acc)*asc[0]*bsc[row]; ti[sgid] = row; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid==0) {
        uint nsg = tgs>>5u; float bv = tv[0]; uint bi = ti[0];
        for (uint s=1u; s<nsg; s++) if (tv[s]>bv || (tv[s]==bv && ti[s]<bi)) { bv=tv[s]; bi=ti[s]; }
        part[tgid].v = bv; part[tgid].i = bi;
    }
}
kernel void argmax_finish(device const AmaxPart* part[[buffer(0)]], device uint* tok[[buffer(1)]],
    constant uint& P[[buffer(2)]], uint tid[[thread_index_in_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    float v=-INFINITY; uint idx=0xFFFFFFFFu;
    for (uint p=tid; p<P; p+=256u) { float cv=part[p].v; uint ci=part[p].i; if (cv>v||(cv==v&&ci<idx)){v=cv;idx=ci;} }
    for (uint off=16u; off>0u; off>>=1u) { float ov=simd_shuffle_down(v,off); uint oi=simd_shuffle_down(idx,off); if(ov>v||(ov==v&&oi<idx)){v=ov;idx=oi;} }
    threadgroup float tv[8]; threadgroup uint ti[8];
    if (lane==0u){tv[sgid]=v;ti[sgid]=idx;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid==0u){ float bv=tv[0];uint bi=ti[0]; for(uint s=1u;s<8u;s++) if(tv[s]>bv||(tv[s]==bv&&ti[s]<bi)){bv=tv[s];bi=ti[s];} tok[0]=bi; }
}
// rope: NeoX half-split. Rotates pairs (d, half+d) for d in [0,half) within each head (stride
// hd), where half = rotaryDim/2 = len(invf). half<hd/2 is PARTIAL rotary (Phi): dims
// [2*half, hd) pass through unrotated. total = nHeads*half (the rotate-pair count).
// scale is YaRN's mscale (attention_factor), applied to cos/sin exactly like
// decoder/rope.go's applyRoPE (c := cos(theta)*scale; s := sin(theta)*scale) — NOT to the
// rotated output afterward, which is a different (and wrong) place to apply it. 1.0 for
// every family without YaRN, so this is a no-op multiply, not a new branch. FeatRopeMscale
// (decoder/features.go) is not yet declared for Metal — nothing resident here exercises
// scale != 1.0 end-to-end — but the kernel needs to accept it before wiring can proceed, and
// a no-op multiply on every existing dispatch is the lowest-risk way to add the parameter.
kernel void rope(device float* x[[buffer(0)]], device const float* invf[[buffer(1)]],
    constant uint& hd[[buffer(2)]], constant uint& pos[[buffer(3)]], constant uint& total[[buffer(4)]],
    constant uint& rhalf[[buffer(5)]], constant float& scale[[buffer(6)]], uint gid[[thread_position_in_grid]]) {
    if(gid>=total) return; uint head=gid/rhalf; uint dd=gid%rhalf; uint base=head*hd;
    float th=float(pos)*invf[dd]; float c=cos(th)*scale,s=sin(th)*scale;
    float x0=x[base+dd],x1=x[base+rhalf+dd]; x[base+dd]=x0*c-x1*s; x[base+rhalf+dd]=x0*s+x1*c;
}
// rope2: merges the Q and K rope dispatches for one layer into ONE dispatch — every production
// call site launches rope twice per layer back-to-back, once for Q (buffer offset 0 into the
// fused qkv buffer) and once for K (bound at a Metal buffer-offset kOff via aikit/gpu's
// Buffer.At). Bind-time buffer offsets can't express "two ranges of the SAME dispatch", so this
// takes the UNOFFSET base buffer and does the offset itself: gid in [0,qTotal) addresses Q at
// x[dd..], gid in [qTotal,qTotal+kTotal) addresses K at x[kOff+dd..]. kOff is in ELEMENTS
// (floats), matching how the kernel indexes x — reuse geom's existing uNHhd (= nH*hd) buffer
// unchanged rather than adding a new one, since it is already exactly that value. Same math as
// rope otherwise (verified bit-identical to running rope twice — TestRope2_matchesTwoRope,
// metal/rope2_test.go); rope itself is untouched and kept for its own standalone kernel tests.
kernel void rope2(device float* x[[buffer(0)]], device const float* invf[[buffer(1)]],
    constant uint& hd[[buffer(2)]], constant uint& pos[[buffer(3)]], constant uint& qTotal[[buffer(4)]],
    constant uint& kTotal[[buffer(5)]], constant uint& rhalf[[buffer(6)]], constant float& scale[[buffer(7)]],
    constant uint& kOff[[buffer(8)]], uint gid[[thread_position_in_grid]]) {
    uint total = qTotal + kTotal;
    if (gid >= total) return;
    uint g = gid; uint off = 0u;
    if (gid >= qTotal) { g = gid - qTotal; off = kOff; }
    uint head = g/rhalf; uint dd = g%rhalf; uint base = off + head*hd;
    float th=float(pos)*invf[dd]; float c=cos(th)*scale,s=sin(th)*scale;
    float x0=x[base+dd],x1=x[base+rhalf+dd]; x[base+dd]=x0*c-x1*s; x[base+rhalf+dd]=x0*s+x1*c;
}
kernel void kv_store(device const float* k[[buffer(0)]], device const float* v[[buffer(1)]],
    device half* kc[[buffer(2)]], device half* vc[[buffer(3)]], constant uint& kvDim[[buffer(4)]],
    constant uint& pos[[buffer(5)]], uint i[[thread_position_in_grid]]) {
    kc[pos*kvDim+i]=half(k[i]); vc[pos*kvDim+i]=half(v[i]); // f16 KV: half the cache bytes + read BW
}
// kv_store_f32 / attention_f32 — the FULL-PRECISION KV twins, used on Gemma's sandwich path only
// (resident.kvF32). Gemma's low-magnitude attention contexts amplify f16-KV rounding into a
// catastrophic per-layer context error (0.64 vs f32's 0.92 cosine; matched-input confirmer
// isolated it to the KV cache). Qwen is insensitive and keeps the f16 path (half the cache BW).
kernel void kv_store_f32(device const float* k[[buffer(0)]], device const float* v[[buffer(1)]],
    device float* kc[[buffer(2)]], device float* vc[[buffer(3)]], constant uint& kvDim[[buffer(4)]],
    constant uint& pos[[buffer(5)]], uint i[[thread_position_in_grid]]) {
    kc[pos*kvDim+i]=k[i]; vc[pos*kvDim+i]=v[i];
}
// One THREADGROUP (128 threads) per query head — vs the old 1-thread-per-head (12 threads
// total = 68% of decode time from underutilization). Scores parallel over keys, softmax via
// threadgroup reduction, output parallel over head dims. nKeys ≤ metalCtxCap (4096).
// window>0 (Mistral) restricts the query to the last window keys: keys[winStart..nKeys),
// winStart = max(0, nKeys-window). window==0 is full causal. Derived from nKeys in-kernel, so
// no per-token uniform. (Mistral is all-local; a hypothetical global layer binds window=0.)
// sinks/hasSink: gpt-oss's per-head learned attention sink — an extra logit with NO key and NO
// value, competing for the softmax MAX and joining the DENOMINATOR only (verified standalone,
// metal/gptoss_kernels_test.go's TestGptOssAttnSink_metal, including the sink-DOMINATES case a
// post-hoc denominator patch would get wrong). hasSink==0 is a true no-op for every other family
// (uniform across the threadgroup, so no divergence cost).
kernel void attention(device const float* q[[buffer(0)]], device const half* kc[[buffer(1)]],
    device const half* vc[[buffer(2)]], device float* out[[buffer(3)]], constant uint& nH[[buffer(4)]],
    constant uint& nKV[[buffer(5)]], constant uint& hd[[buffer(6)]], constant uint& nKeys[[buffer(7)]],
    constant float& scale[[buffer(8)]], constant uint& window[[buffer(9)]],
    device const float* sinks[[buffer(10)]], constant uint& hasSink[[buffer(11)]],
    uint qh[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    uint kvDim = nKV*hd; uint kvh = qh/(nH/nKV);
    uint winStart = (window>0u && nKeys>window) ? nKeys-window : 0u;
    device const float* qr = q + qh*hd;
    device const half*  kb = kc + kvh*hd;   // f16 KV; dot/accum stay in f32 -> parity-neutral
    device const half*  vb = vc + kvh*hd;
    threadgroup float sc[4096];
    threadgroup float red[128];
    for (uint s=winStart+tid; s<nKeys; s+=tgs) {
        float a=0; device const half* k=kb+s*kvDim; uint d=0;
        // half4 vectorized K-read (1.79x @2048 ctx): the one-thread-per-key access is uncoalesced
        // (adjacent lanes stride kvDim), so 8-byte loads recover sector utilization. SAME sequential
        // accumulation order ⇒ bit-identical; guarded on hd%4==0 for alignment, scalar tail otherwise.
        if ((hd&3u)==0u) for (; d<hd; d+=4u){ half4 k4=*((device const half4*)(k+d)); a+=qr[d]*float(k4.x); a+=qr[d+1u]*float(k4.y); a+=qr[d+2u]*float(k4.z); a+=qr[d+3u]*float(k4.w); }
        for (; d<hd; d++) a += qr[d]*float(k[d]);
        sc[s]=a*scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m=-INFINITY; for (uint s=winStart+tid;s<nKeys;s+=tgs) m=max(m,sc[s]);
    red[tid]=m; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st=tgs/2; st>0; st>>=1){ if(tid<st) red[tid]=max(red[tid],red[tid+st]); threadgroup_barrier(mem_flags::mem_threadgroup); }
    float mx=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    float sink=0.0f; bool hasS = hasSink != 0u;
    if (hasS) { sink = sinks[qh]; mx = max(mx, sink); }
    float ls=0; for (uint s=winStart+tid;s<nKeys;s+=tgs){ float p=exp(sc[s]-mx); sc[s]=p; ls+=p; }
    red[tid]=ls; threadgroup_barrier(mem_flags::mem_threadgroup);
    // DENOMINATOR SUM — float-add is non-associative, so this result is coupled to tgs (the reduction
    // WIDTH). tgs is pinned to tgReduceAttn (128, model.go); do NOT parameterize/sweep it, and any
    // alternate attention kernel MUST reduce at the same width or it diverges byte-exactly (past
    // nKeys>width). No existing gate catches this. See ollama-chase §A2-Metal.
    for (uint st=tgs/2; st>0; st>>=1){ if(tid<st) red[tid]+=red[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float sum=red[0];
    if (hasS) sum += exp(sink-mx); // sink joins the denominator only — no value vector, numerator untouched
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d=tid; d<hd; d+=tgs){ float a=0; for(uint s=winStart;s<nKeys;s++) a += sc[s]*float(vb[s*kvDim+d]); out[qh*hd+d]=a/sum; }
}
// attention_f32 — identical to attention but reads an f32 KV cache (Gemma sandwich path). Same
// math (the f16 version already accumulated in f32); only the cache element type changes.
kernel void attention_f32(device const float* q[[buffer(0)]], device const float* kc[[buffer(1)]],
    device const float* vc[[buffer(2)]], device float* out[[buffer(3)]], constant uint& nH[[buffer(4)]],
    constant uint& nKV[[buffer(5)]], constant uint& hd[[buffer(6)]], constant uint& nKeys[[buffer(7)]],
    constant float& scale[[buffer(8)]], constant uint& window[[buffer(9)]],
    uint qh[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    uint kvDim = nKV*hd; uint kvh = qh/(nH/nKV);
    uint winStart = (window>0u && nKeys>window) ? nKeys-window : 0u;
    device const float* qr = q + qh*hd;
    device const float* kb = kc + kvh*hd;
    device const float* vb = vc + kvh*hd;
    threadgroup float sc[4096];
    threadgroup float red[128];
    for (uint s=winStart+tid; s<nKeys; s+=tgs) {
        float a=0; device const float* k=kb+s*kvDim; uint d=0;
        // float4 vectorized K-read (same coalescing fix as attention, f32 KV). Bit-identical.
        if ((hd&3u)==0u) for (; d<hd; d+=4u){ float4 k4=*((device const float4*)(k+d)); a+=qr[d]*k4.x; a+=qr[d+1u]*k4.y; a+=qr[d+2u]*k4.z; a+=qr[d+3u]*k4.w; }
        for (; d<hd; d++) a += qr[d]*k[d];
        sc[s]=a*scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m=-INFINITY; for (uint s=winStart+tid;s<nKeys;s+=tgs) m=max(m,sc[s]);
    red[tid]=m; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st=tgs/2; st>0; st>>=1){ if(tid<st) red[tid]=max(red[tid],red[tid+st]); threadgroup_barrier(mem_flags::mem_threadgroup); }
    float mx=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    float ls=0; for (uint s=winStart+tid;s<nKeys;s+=tgs){ float p=exp(sc[s]-mx); sc[s]=p; ls+=p; }
    red[tid]=ls; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st=tgs/2; st>0; st>>=1){ if(tid<st) red[tid]+=red[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float sum=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d=tid; d<hd; d+=tgs){ float a=0; for(uint s=winStart;s<nKeys;s++) a += sc[s]*vb[s*kvDim+d]; out[qh*hd+d]=a/sum; }
}
// glu_act selects the gated MLP's activation. The ordinals are deliberately decoder.ActKind's
// iota (ActGeluTanh=0, ActSiLU=1) so the host passes int(m.GatedActResident()) straight through.
// GELU-tanh matches decoder/rmsnorm.go geluTanh; both compute in f32 where the CPU reference
// uses f64 — the same trade the SwiGLU path already shipped, and it clears the 3% near-tie bar.
#define ACT_GELU_TANH 0u
#define ACT_SILU      1u
inline float glu_act(float x, uint act) {
    if (act == ACT_SILU) return x/(1.0f+exp(-x));
    // GELU-tanh. CLAMP the tanh argument: for a massive-activation gate (Gemma's <bos> hits
    // x~12 → arg~73), MSL's tanh overflows its internal exp() to NaN, which quantizes to 0 and
    // silently drops the channel that BUILDS the massive activation (the entire dormant-Gemma
    // residual traced to exactly this). tanh saturates to ±1 by |arg|~9, so clamping to ±15 is
    // numerically exact for every real input and merely defuses the overflow. SiLU is unaffected
    // (no tanh), which is why SwiGLU models never hit this.
    float a = 0.7978845608028654f*(x+0.044715f*x*x*x); // sqrt(2/pi)
    return 0.5f*x*(1.0f+tanh(clamp(a, -15.0f, 15.0f)));
}
kernel void swiglu_quant(device const float* g[[buffer(0)]], device const float* u[[buffer(1)]],
    device char* dq[[buffer(2)]], device float* ds[[buffer(3)]], constant uint& I[[buffer(4)]],
    constant uint& act[[buffer(5)]],
    uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256]; float mx=0;
    for(uint i=tid;i<I;i+=tgs){ float s=glu_act(g[i],act)*u[i]; mx=max(mx,fabs(s)); }
    red[tid]=mx; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=tgs/2;s>0;s>>=1){ if(tid<s) red[tid]=max(red[tid],red[tid+s]); threadgroup_barrier(mem_flags::mem_threadgroup);}
    float sc=red[0]/127.0f; if(sc==0)sc=1; if(tid==0)ds[0]=sc; float inv=1/sc;
    for(uint i=tid;i<I;i+=tgs){ float s=glu_act(g[i],act)*u[i]; dq[i]=char(clamp(int(round(s*inv)),-127,127)); }
}
// act_quant: FeatNonGatedMLP's fused activation+quant — up->act->down (GPT-2, Nemotron relu²),
// no gate multiply, unlike swiglu_quant. Reuses glu_act (same ordinals), so it inherits the
// GELU-tanh clamp fix for free. Only ACT_GELU_TANH/ACT_SILU are wired (glu_act's only branches);
// exact-erf GELU (decoder's ActGelu, HF's plain "gelu") is NOT implemented — GPT-2's real
// released checkpoints use "gelu_new" (ActGeluTanh), which this covers.
kernel void act_quant(device const float* u[[buffer(0)]], device char* dq[[buffer(1)]],
    device float* ds[[buffer(2)]], constant uint& I[[buffer(3)]], constant uint& act[[buffer(4)]],
    uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256]; float mx=0;
    // Vectorized float4/char4 load/store for both loops -- glu_act stays a scalar call per lane
    // (extracted from the loaded float4), so this is bit-identical to the scalar loop, only
    // batching the memory traffic. max is exact/order-independent and the quantize-write has no
    // reduction at all (same argument verified safe for quant_vec/layernorm_quant this session).
    uint I4 = I >> 2u;
    device const float4* u4 = (device const float4*)u;
    for (uint i4=tid; i4<I4; i4+=tgs) {
        float4 uv=u4[i4];
        float s0=glu_act(uv.x,act), s1=glu_act(uv.y,act), s2=glu_act(uv.z,act), s3=glu_act(uv.w,act);
        mx=max(mx,fabs(s0)); mx=max(mx,fabs(s1)); mx=max(mx,fabs(s2)); mx=max(mx,fabs(s3));
    }
    for (uint i=(I4<<2u)+tid; i<I; i+=tgs){ float s=glu_act(u[i],act); mx=max(mx,fabs(s)); }
    red[tid]=mx; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=tgs/2;s>0;s>>=1){ if(tid<s) red[tid]=max(red[tid],red[tid+s]); threadgroup_barrier(mem_flags::mem_threadgroup);}
    float sc=red[0]/127.0f; if(sc==0)sc=1; if(tid==0)ds[0]=sc; float inv=1/sc;
    device char4* dq4 = (device char4*)dq;
    for (uint i4=tid; i4<I4; i4+=tgs) {
        float4 uv=u4[i4];
        float s0=glu_act(uv.x,act), s1=glu_act(uv.y,act), s2=glu_act(uv.z,act), s3=glu_act(uv.w,act);
        char q0=char(clamp(int(round(s0*inv)),-127,127));
        char q1=char(clamp(int(round(s1*inv)),-127,127));
        char q2=char(clamp(int(round(s2*inv)),-127,127));
        char q3=char(clamp(int(round(s3*inv)),-127,127));
        dq4[i4]=char4(q0,q1,q2,q3);
    }
    for (uint i=(I4<<2u)+tid; i<I; i+=tgs){ float s=glu_act(u[i],act); dq[i]=char(clamp(int(round(s*inv)),-127,127)); }
}
kernel void residual(device float* x[[buffer(0)]], device const float* y[[buffer(1)]], uint i[[thread_position_in_grid]]) { x[i]+=y[i]; }

// qk_norm: per-head RMSNorm on Q and K in the fused qkv buffer, applied BEFORE RoPE (Qwen3,
// Gemma3). ONE threadgroup per head: head < nH is a Q head (weight qn), else a K head (weight
// kn, index head-nH). Norm over the head dim (hd, a power of 2). addOne selects Gemma's (1+w)
// vs plain w. Matches decoder/rmsnorm.go rmsNorm(q/k, QNorm/KNorm, ..., RMSAddOne).
kernel void qk_norm(device float* qkv[[buffer(0)]], device const float* qn[[buffer(1)]],
    device const float* kn[[buffer(2)]], constant uint& nH[[buffer(3)]], constant uint& nKV[[buffer(4)]],
    constant uint& hd[[buffer(5)]], constant uint& nHhd[[buffer(6)]], constant float& eps[[buffer(7)]],
    constant uint& addOne[[buffer(8)]], uint head[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[128];
    bool isQ = head < nH;
    device const float* w = isQ ? qn : kn;
    uint base = isQ ? head*hd : nHhd + (head-nH)*hd;
    device float* x = qkv + base;
    float ss=0; for(uint i=tid;i<hd;i+=tgs){ float v=x[i]; ss+=v*v; }
    red[tid]=ss; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=tgs/2u;s>0u;s>>=1u){ if(tid<s) red[tid]+=red[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float rms=rsqrt(red[0]/float(hd)+eps);
    for(uint i=tid;i<hd;i+=tgs){ float wt = addOne!=0u ? (1.0f+w[i]) : w[i]; x[i]=x[i]*rms*wt; }
}
` + moeKernels + gemma4MoeKernels + deltaNetKernels
