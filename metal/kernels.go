//go:build darwin

package metal

// allKernels is the full dense-decode-layer MSL kernel set in one library (W8A8 path —
// W4A8 is validated separately; this proves ASSEMBLY, not the int4 packing again).
//
// N-17 (audit-metal-2026-09-12.md): six of these kernels have no PRODUCTION pipeline —
// gemv_w4a8_bias, gemv_w4a8_sa_amax, gemv_w4a8_sa_bk, gemv_w4a8_sa_qv, gemv_w8a8, rope2_kv — each
// backs a dedicated micro-benchmark or recorded-negative regression test instead (profile_test.go,
// batchk_test.go, sa_qv_fusion_test.go, gemv_test.go, rope2_kv_test.go respectively;
// gemv_w4a8_sa_amax has no reference anywhere and is the one genuinely dead survivor of this
// list — kept rather than deleted alongside it so its own history stays visible next to the
// others, not because anything still needs it). None of these is "safe to delete because nothing
// production calls it" — deleting one breaks the test that keeps its measurement/negative result
// honest. A seventh, gemm_w4f16 (metal/prefill.go, a genuinely dead duplicate of
// gemm_w4f16_store with no reference anywhere, test included), was deleted outright.
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
    float rms=precise::rsqrt(red[0]/float(H)+eps); threadgroup_barrier(mem_flags::mem_threadgroup);
    // Vectorized LOAD (batch 4 scalar x[i]/w[i] reads into one float4 read). rms is pinned via
    // precise:: (above) so this is stable regardless of surrounding code shape -- confirmed via
    // GOINFER_PRECISE_MATH A/B plus a scalar-vs-vectorized cross-check under default fast-math with
    // rms pinned (both variants byte-identical to each other; argmax unchanged from today's shipped
    // output at every checkpoint, only deep-mantissa sha bits move -- see the round's own commit).
    float mx=0;
    uint H4v = H >> 2u;
    device const float4* xv4 = (device const float4*)x;
    device const float4* wv4 = (device const float4*)w;
    for(uint i4=tid;i4<H4v;i4+=tgs){
        float4 xv=xv4[i4], wv=wv4[i4];
        float g0=addOne!=0u?(1.0f+wv.x):wv.x;
        float g1=addOne!=0u?(1.0f+wv.y):wv.y;
        float g2=addOne!=0u?(1.0f+wv.z):wv.z;
        float g3=addOne!=0u?(1.0f+wv.w):wv.w;
        float p0=xv.x*rms*g0, p1=xv.y*rms*g1, p2=xv.z*rms*g2, p3=xv.w*rms*g3;
        mx=max(mx,fabs(p0)); mx=max(mx,fabs(p1)); mx=max(mx,fabs(p2)); mx=max(mx,fabs(p3));
    }
    for(uint i=(H4v<<2u)+tid;i<H;i+=tgs){ float g=addOne!=0u?(1.0f+w[i]):w[i]; mx=max(mx,fabs(x[i]*rms*g)); }
    mx = simd_max(mx);
    uint sgid = tid >> 5u, lane = tid & 31u;
    if (lane == 0) red[sgid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nsg = tgs >> 5u;
    if (tid == 0) { float v = red[0]; for (uint k=1; k<nsg; k++) v = max(v, red[k]); red[0] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sc=red[0]/127.0f; if(sc==0)sc=1; if(tid==0)asc[0]=sc; float inv=1/sc;
    for(uint i=tid;i<H;i+=tgs){ float g=addOne!=0u?(1.0f+w[i]):w[i]; aq[i]=char(clamp(int(round(x[i]*rms*g*inv)),-127,127)); }
}
// rmsnorm_f16_act: rmsnorm_quant's f32-in shape with NO quantization at all — writes the true
// normed value straight to half. The input-precision half of R1's W4F16 decode lane
// (docs/tasks/red-october.md; gemv_w4f16_sa* above is the weight-stream half). Same
// reduction/precise::rsqrt pinning as rmsnorm_quant for consistency; a scalar loop, not
// vectorized — this kernel is not the bottleneck rmsnorm_quant's amax-scan vectorization was
// written for (no second pass here at all), so the extra complexity isn't earning its keep yet.
kernel void rmsnorm_f16_act(device const float* x[[buffer(0)]], device const float* w[[buffer(1)]],
    device half* out[[buffer(2)]], constant uint& H[[buffer(3)]],
    constant float& eps[[buffer(4)]], constant uint& addOne[[buffer(5)]],
    uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256]; float ss=0;
    for(uint i=tid;i<H;i+=tgs) ss+=x[i]*x[i];
    red[tid]=ss; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=tgs/2;s>0;s>>=1){ if(tid<s) red[tid]+=red[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup);}
    float rms=precise::rsqrt(red[0]/float(H)+eps);
    for(uint i=tid;i<H;i+=tgs){ float g=addOne!=0u?(1.0f+w[i]):w[i]; out[i]=half(x[i]*rms*g); }
}
// f32_to_f16: plain element-wise convert, grid = N. R1's o-proj-input half of the W4F16 lane —
// the attention context (r.ctx) has no norm/weight before o-proj (quant_vec is a bare
// quantizer, not rmsnorm_quant), so the f16 lane needs a bare convert here, not a norm kernel.
kernel void f32_to_f16(device const float* x[[buffer(0)]], device half* out[[buffer(1)]],
    uint i[[thread_position_in_grid]]) { out[i] = half(x[i]); }
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
// gemv_w4f16_* — R1 (docs/tasks/red-october.md): a decode GEMV in the f16-FMA regime MLX and
// llama.cpp use, instead of gemv_w4a8_sa's int8-activation W4A8 regime. NOT bit-identical to the
// shipped path (no int8 activation quantization at all — arguably HIGHER fidelity, not lower;
// still a fidelity-gated lane per the 2026-09-18 owner decision, §4). First slice only: the plain
// dense (non-MoE, non-paged, non-sandwich, non-DeltaNet) QKV/o-proj/gate-up path — the down-proj
// (gemv_w4a8_resid, the "coal" family, a different kernel shape entirely) stays W4A8 for now.
//
// nib2half: the exponent-bias dequant trick named in the brief. A raw 4-bit nibble packed into a
// half's mantissa bits [9:6] with a FIXED exponent field (biased 15 = 2^0) gives the half value
// 1.0 + nibble/16 (nibble<<6 never overflows the 10-bit mantissa: max 15<<6=960<1024) — no
// int-to-float convert instruction. Subtracting 1.5h centers it ((nibble-8)/16) and *16 recovers
// nibble-8 exactly (small integers are exact in half). Verified against a scalar reference for
// all 16 nibble values before this kernel is dispatched anywhere — see gemv_w4f16_test.go.
inline half nib2half(uint nibble) {
    half v = as_type<half>(ushort(0x3C00u | (nibble << 6)));
    return (v - half(1.5h)) * half(16.0h);
}
// UNP8HV: the half-arithmetic analogue of UNP8V above. Each nibble's dequant is promoted to
// float and the 8-term dot for one uint word is accumulated in float — not left in half — so a
// 32-element group's within-group sum does not inherit half's ~3-decimal-digit precision; only
// the per-group SCALE multiply (sr[g], already half in the existing buffer layout, unchanged)
// stays at its existing precision, matching gemv_w4a8_sa's own scale handling exactly.
#define UNP8HV(xw, a4) ( \
    float(nib2half((xw)&0xFu))      *float((a4)[0].x) + float(nib2half(((xw)>>4)&0xFu)) *float((a4)[0].y) \
  + float(nib2half(((xw)>>8)&0xFu)) *float((a4)[0].z) + float(nib2half(((xw)>>12)&0xFu))*float((a4)[0].w) \
  + float(nib2half(((xw)>>16)&0xFu))*float((a4)[1].x) + float(nib2half(((xw)>>20)&0xFu))*float((a4)[1].y) \
  + float(nib2half(((xw)>>24)&0xFu))*float((a4)[1].z) + float(nib2half(((xw)>>28)&0xFu))*float((a4)[1].w) )
// SA_F16_BODY: gemv_w4a8_sa's SA_BODY with the activation staged as half (from a norm producer
// with NO quantization step, e.g. rmsnorm_f16_act) instead of int8, and no separate activation
// scale — the f16 activation values are already the true values; only the per-group WEIGHT
// scale (sr[g]) still applies, exactly as it does in the int8 kernels.
#define SA_F16_BODY \
    for (uint i=tid;i<K;i+=tgs) As[i]=ax[i]; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    uint G = K>>5u; \
    uint row = tgid*(tgs>>5u) + sgid; \
    device const uint4* wr = wq + (uint)row*G; \
    device const half*  sr = sct + (uint)row*G; \
    float acc = 0.0f; \
    for (uint g=lane; g<G; g+=32u) { \
        uint4 w = wr[g]; threadgroup const half4* a4 = reinterpret_cast<threadgroup const half4*>(As + g*32u); \
        float gi = UNP8HV(w.x,a4) + UNP8HV(w.y,a4+2) + UNP8HV(w.z,a4+4) + UNP8HV(w.w,a4+6); \
        acc += gi * float(sr[g]); \
    } \
    acc = simd_sum(acc);
// gemv_w4f16_sa: base, overwrite-out epilogue (gate|up, no bias — Qwen2.5's shape).
kernel void gemv_w4f16_sa(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const half* ax[[buffer(2)]], device float* out[[buffer(3)]],
    constant uint& K[[buffer(4)]], threadgroup half* As [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    SA_F16_BODY
    if (lane==0) out[row] = acc;
}
// gemv_w4f16_sa_bias: additive per-row bias, overwrite-out (QKV's shape).
kernel void gemv_w4f16_sa_bias(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const half* ax[[buffer(2)]], device const float* bias[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], threadgroup half* As [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    SA_F16_BODY
    if (lane==0) out[row] = acc + bias[row];
}
// gemv_w4f16_sa_resid: accumulate into out (o-proj + residual's shape — out IS x at the call site).
kernel void gemv_w4f16_sa_resid(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const half* ax[[buffer(2)]], device float* out[[buffer(3)]],
    constant uint& K[[buffer(4)]], threadgroup half* As [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    SA_F16_BODY
    if (lane==0) out[row] += acc;
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
// qTempScale (buffer 9, Ministral 3 FeatAttnTemp): a post-rotation multiplier on Q ONLY, applied
// after the pair rotation below — decoder/attention.go's sequential path does the SAME thing in
// the SAME order (rope first, attn-temp on top), never touching K. 1.0 (exact no-op) for every
// family without this feature — Model.AttnTempScale's own comment has the formula.
kernel void rope2(device float* x[[buffer(0)]], device const float* invf[[buffer(1)]],
    constant uint& hd[[buffer(2)]], constant uint& pos[[buffer(3)]], constant uint& qTotal[[buffer(4)]],
    constant uint& kTotal[[buffer(5)]], constant uint& rhalf[[buffer(6)]], constant float& scale[[buffer(7)]],
    constant uint& kOff[[buffer(8)]], constant float& qTempScale[[buffer(9)]], uint gid[[thread_position_in_grid]]) {
    uint total = qTotal + kTotal;
    if (gid >= total) return;
    bool isQ = gid < qTotal;
    uint g = gid; uint off = 0u;
    if (!isQ) { g = gid - qTotal; off = kOff; }
    uint head = g/rhalf; uint dd = g%rhalf; uint base = off + head*hd;
    float th=float(pos)*invf[dd]; float c=cos(th)*scale,s=sin(th)*scale;
    float x0=x[base+dd],x1=x[base+rhalf+dd];
    float r0=x0*c-x1*s, r1=x0*s+x1*c;
    if (isQ) { r0 *= qTempScale; r1 *= qTempScale; }
    x[base+dd]=r0; x[base+rhalf+dd]=r1;
}
kernel void kv_store(device const float* k[[buffer(0)]], device const float* v[[buffer(1)]],
    device half* kc[[buffer(2)]], device half* vc[[buffer(3)]], constant uint& kvDim[[buffer(4)]],
    constant uint& pos[[buffer(5)]], uint i[[thread_position_in_grid]]) {
    kc[pos*kvDim+i]=half(k[i]); vc[pos*kvDim+i]=half(v[i]); // f16 KV: half the cache bytes + read BW
}
// kv_store_i8 — stores K and V as symmetric int8 per-head with f32 scales (ks, vs).
// Each thread handles one KV head: reduces absmax → scale (maxabs/127) → stores quantized int8.
kernel void kv_store_i8(device const float* k[[buffer(0)]], device const float* v[[buffer(1)]],
    device char* kc[[buffer(2)]], device char* vc[[buffer(3)]],
    device float* ks[[buffer(4)]], device float* vs[[buffer(5)]],
    constant uint& nKV[[buffer(6)]], constant uint& hd[[buffer(7)]],
    constant uint& pos[[buffer(8)]], uint kvh[[thread_position_in_grid]]) {
    if (kvh >= nKV) return;
    uint kvDim = nKV * hd;
    uint base = kvh * hd;
    float amax_k = 0.0f, amax_v = 0.0f;
    for (uint d = 0; d < hd; d++) {
        amax_k = max(amax_k, abs(k[base + d]));
        amax_v = max(amax_v, abs(v[base + d]));
    }
    float sc_k = amax_k / 127.0f; if (sc_k == 0.0f) sc_k = 1.0f;
    float sc_v = amax_v / 127.0f; if (sc_v == 0.0f) sc_v = 1.0f;
    ks[pos * nKV + kvh] = sc_k;
    vs[pos * nKV + kvh] = sc_v;
    float inv_k = 1.0f / sc_k;
    float inv_v = 1.0f / sc_v;
    for (uint d = 0; d < hd; d++) {
        kc[pos * kvDim + base + d] = char(clamp(round(k[base + d] * inv_k), -127.0f, 127.0f));
        vc[pos * kvDim + base + d] = char(clamp(round(v[base + d] * inv_v), -127.0f, 127.0f));
    }
}
// rope2_kv: fuses rope2 (merged Q+K RoPE) with kv_store (K/V cache write) into ONE dispatch --
// every production call site launches these back-to-back, RoPE-then-store, and K's cache write
// needs exactly the ROTATED value RoPE just computed, so storing it inline removes a full
// device round-trip (rope2 writes rotated K into the qkv buffer, kv_store immediately re-reads
// those same bytes). Grid: [0,qTotal) rotates Q (no cache write -- Q isn't cached, same as
// rope2); [qTotal,qTotal+kTotal) rotates K AND stores it into kc; [qTotal+kTotal,qTotal+2*kTotal)
// copies V (untouched by RoPE) into vc. V shares K's thread count and pairing since kvDim
// (=nKV*hd) is always exactly 2*kTotal (=2*nKV*half). Same math as running rope2 then kv_store --
// verified bit-identical on real weights (TestRope2Kv_matchesRope2ThenKv; TestMetalSnapshotGolden
// and TestGPT2ResidentParityMetal wired against this call site too, also byte-identical).
// MEASURED (TestQwen35ResidentDecodeRateMetal, 20 samples each way): eliminating one dispatch/
// layer is a null result at the whole-decode level (~0.6%, within noise) -- consistent with
// TestLayerA_bindingTax's own finding that per-dispatch marginal cost (~2.2us) is tiny against
// everything else in a token's critical path. NOT wired into any production dispatch site --
// correctness-proven, kept for the record, not a live kernel (same status as gemv_w4a8_sa_qv).
kernel void rope2_kv(device float* x[[buffer(0)]], device const float* invf[[buffer(1)]],
    constant uint& hd[[buffer(2)]], constant uint& pos[[buffer(3)]], constant uint& qTotal[[buffer(4)]],
    constant uint& kTotal[[buffer(5)]], constant uint& rhalf[[buffer(6)]], constant float& scale[[buffer(7)]],
    constant uint& kOff[[buffer(8)]], constant uint& vOff[[buffer(9)]], device half* kc[[buffer(10)]],
    device half* vc[[buffer(11)]], constant uint& kvDim[[buffer(12)]], uint gid[[thread_position_in_grid]]) {
    uint total = qTotal + 2u*kTotal;
    if (gid >= total) return;
    if (gid < qTotal) {
        uint head = gid/rhalf; uint dd = gid%rhalf; uint base = head*hd;
        float th=float(pos)*invf[dd]; float c=cos(th)*scale,s=sin(th)*scale;
        float x0=x[base+dd],x1=x[base+rhalf+dd];
        x[base+dd]=x0*c-x1*s; x[base+rhalf+dd]=x0*s+x1*c;
    } else if (gid < qTotal+kTotal) {
        uint g = gid - qTotal; uint head = g/rhalf; uint dd = g%rhalf; uint base = kOff + head*hd;
        float th=float(pos)*invf[dd]; float c=cos(th)*scale,s=sin(th)*scale;
        float x0=x[base+dd],x1=x[base+rhalf+dd];
        float k0=x0*c-x1*s, k1=x0*s+x1*c;
        x[base+dd]=k0; x[base+rhalf+dd]=k1;
        uint kvi = head*hd;
        kc[pos*kvDim+kvi+dd]=half(k0); kc[pos*kvDim+kvi+rhalf+dd]=half(k1);
    } else {
        uint g = gid - qTotal - kTotal; uint head = g/rhalf; uint dd = g%rhalf; uint base = vOff + head*hd;
        uint kvi = head*hd;
        vc[pos*kvDim+kvi+dd]=half(x[base+dd]); vc[pos*kvDim+kvi+rhalf+dd]=half(x[base+rhalf+dd]);
    }
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
// threadgroup reduction, output parallel over head dims. nKeys ≤ metalCtxCapMax (4096).
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
    uint nWin = nKeys - winStart;
    device const float* qr = q + qh*hd;
    device const half*  kb = kc + kvh*hd;   // f16 KV; dot/accum stay in f32 -> parity-neutral
    device const half*  vb = vc + kvh*hd;
    threadgroup float sc[4096];
    threadgroup float red[128];

    // Single-tile fast path (nWin <= 4096): bit-identical reduction tree & float-add order for all
    // historical contexts. Fixes sliding-window indexing: sc[] is indexed relative to winStart so
    // that windowed attention at pos >= 4096 never overflows sc[4096].
    if (nWin <= 4096u) {
        for (uint s=winStart+tid; s<nKeys; s+=tgs) {
            float a=0; device const half* k=kb+s*kvDim; uint d=0;
            // half4 vectorized K-read (1.79x @2048 ctx): the one-thread-per-key access is uncoalesced
            // (adjacent lanes stride kvDim), so 8-byte loads recover sector utilization. SAME sequential
            // accumulation order ⇒ bit-identical; guarded on hd%4==0 for alignment, scalar tail otherwise.
            if ((hd&3u)==0u) for (; d<hd; d+=4u){ half4 k4=*((device const half4*)(k+d)); a+=qr[d]*float(k4.x); a+=qr[d+1u]*float(k4.y); a+=qr[d+2u]*float(k4.z); a+=qr[d+3u]*float(k4.w); }
            for (; d<hd; d++) a += qr[d]*float(k[d]);
            sc[s - winStart]=a*scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float m=-INFINITY; for (uint s=tid;s<nWin;s+=tgs) m=max(m,sc[s]);
        red[tid]=m; threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st=tgs/2; st>0; st>>=1){ if(tid<st) red[tid]=max(red[tid],red[tid+st]); threadgroup_barrier(mem_flags::mem_threadgroup); }
        float mx=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
        float sink=0.0f; bool hasS = hasSink != 0u;
        if (hasS) { sink = sinks[qh]; mx = max(mx, sink); }
        float ls=0; for (uint s=tid;s<nWin;s+=tgs){ float p=exp(sc[s]-mx); sc[s]=p; ls+=p; }
        red[tid]=ls; threadgroup_barrier(mem_flags::mem_threadgroup);
        // DENOMINATOR SUM — float-add is non-associative, so this result is coupled to tgs (the reduction
        // WIDTH). tgs is pinned to tgReduceAttn (128, model.go); do NOT parameterize/sweep it, and any
        // alternate attention kernel MUST reduce at the same width or it diverges byte-exactly (past
        // nKeys>width). No existing gate catches this. See ollama-chase §A2-Metal.
        for (uint st=tgs/2; st>0; st>>=1){ if(tid<st) red[tid]+=red[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }
        float sum=red[0];
        if (hasS) sum += exp(sink-mx); // sink joins the denominator only — no value vector, numerator untouched
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // V-READ: order-preserving 8-wide load-batch unroll. Each thread walks nKeys serially at
        // stride kvDim (the same "distinct per-key, latency-exposed" shape the K-read's half4 fix
        // (1.79x @2048, above) already treats) -- issue 8 independent loads ahead, retire the adds in
        // the SAME sequential order as the plain scalar loop, so this is bit-identical by construction
        // (float accumulation order unchanged, only load scheduling). Scalar tail for nKeys not a
        // multiple of 8. Measured (2026-08-21, git-stash A/B, tight-alternated, 2048 ctx): ~26.6%
        // faster than scalar; width-4 also won (~17-18%) but 8 measured further ahead of it.
        for (uint d=tid; d<hd; d+=tgs){
            float a=0; uint s=winStart; uint nMain = winStart + (nWin & ~7u);
            for (; s<nMain; s+=8u) {
                uint sc_idx = s - winStart;
                float v0=float(vb[(s+0u)*kvDim+d]);
                float v1=float(vb[(s+1u)*kvDim+d]);
                float v2=float(vb[(s+2u)*kvDim+d]);
                float v3=float(vb[(s+3u)*kvDim+d]);
                float v4=float(vb[(s+4u)*kvDim+d]);
                float v5=float(vb[(s+5u)*kvDim+d]);
                float v6=float(vb[(s+6u)*kvDim+d]);
                float v7=float(vb[(s+7u)*kvDim+d]);
                a += sc[sc_idx+0u]*v0; a += sc[sc_idx+1u]*v1; a += sc[sc_idx+2u]*v2; a += sc[sc_idx+3u]*v3;
                a += sc[sc_idx+4u]*v4; a += sc[sc_idx+5u]*v5; a += sc[sc_idx+6u]*v6; a += sc[sc_idx+7u]*v7;
            }
            for (; s<nKeys; s++) a += sc[s - winStart]*float(vb[s*kvDim+d]);
            out[qh*hd+d]=a/sum;
        }
        return;
    }

    // Deep-context tiled online softmax path (nWin > 4096):
    // Accumulate across tiles of 4096 keys using online softmax rescaling:
    // m_new = max(m_prev, m_tile), alpha = exp(m_prev - m_new), l_new = l_prev*alpha + sum_tile,
    // acc[slot] = acc[slot]*alpha + sum(p * v).
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    bool hasS = hasSink != 0u;
    if (hasS) {
        m_prev = sinks[qh];
        l_prev = 1.0f;
    }
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint tileBase = winStart; tileBase < nKeys; tileBase += 4096u) {
        uint tileEnd = min(tileBase + 4096u, nKeys);
        uint tileLen = tileEnd - tileBase;

        for (uint s = tileBase + tid; s < tileEnd; s += tgs) {
            float a = 0; device const half* k = kb + s*kvDim; uint d = 0;
            if ((hd&3u) == 0u) for (; d < hd; d += 4u) {
                half4 k4 = *((device const half4*)(k+d));
                a += qr[d]*float(k4.x);
                a += qr[d+1u]*float(k4.y);
                a += qr[d+2u]*float(k4.z);
                a += qr[d+3u]*float(k4.w);
            }
            for (; d < hd; d++) a += qr[d]*float(k[d]);
            sc[s - tileBase] = a * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float m_t = -INFINITY;
        for (uint s = tid; s < tileLen; s += tgs) m_t = max(m_t, sc[s]);
        red[tid] = m_t;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st = tgs/2; st > 0; st >>= 1) {
            if (tid < st) red[tid] = max(red[tid], red[tid+st]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float m_tile = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float m_new = max(m_prev, m_tile);
        float alpha = (m_prev == -INFINITY) ? 0.0f : exp(m_prev - m_new);

        float ls = 0;
        for (uint s = tid; s < tileLen; s += tgs) {
            float p = exp(sc[s] - m_new);
            sc[s] = p;
            ls += p;
        }
        red[tid] = ls;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st = tgs/2; st > 0; st >>= 1) {
            if (tid < st) red[tid] += red[tid+st];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float sum_tile = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        l_prev = l_prev * alpha + sum_tile;
        m_prev = m_new;

        for (uint slot = 0; slot < 4u; slot++) {
            uint d = tid + slot * tgs;
            if (d >= hd) break;
            float a = acc[slot] * alpha;
            uint s = tileBase;
            uint nMain = tileBase + (tileLen & ~7u);
            for (; s < nMain; s += 8u) {
                uint sc_idx = s - tileBase;
                float v0 = float(vb[(s+0u)*kvDim+d]);
                float v1 = float(vb[(s+1u)*kvDim+d]);
                float v2 = float(vb[(s+2u)*kvDim+d]);
                float v3 = float(vb[(s+3u)*kvDim+d]);
                float v4 = float(vb[(s+4u)*kvDim+d]);
                float v5 = float(vb[(s+5u)*kvDim+d]);
                float v6 = float(vb[(s+6u)*kvDim+d]);
                float v7 = float(vb[(s+7u)*kvDim+d]);
                a += sc[sc_idx+0u]*v0; a += sc[sc_idx+1u]*v1; a += sc[sc_idx+2u]*v2; a += sc[sc_idx+3u]*v3;
                a += sc[sc_idx+4u]*v4; a += sc[sc_idx+5u]*v5; a += sc[sc_idx+6u]*v6; a += sc[sc_idx+7u]*v7;
            }
            for (; s < tileEnd; s++) {
                a += sc[s - tileBase]*float(vb[s*kvDim+d]);
            }
            acc[slot] = a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint slot = 0; slot < 4u; slot++) {
        uint d = tid + slot * tgs;
        if (d >= hd) break;
        out[qh*hd + d] = acc[slot] / l_prev;
    }
}
// attention_fa / attention_fa_combine — R2 (docs/tasks/red-october.md): decode attention gridded
// by (kvHead, split) instead of by query head, so the G=nH/nKV heads sharing a KV head compute
// their dots/folds from ONE cooperative, coalesced K/V read per key instead of G separate
// per-head reads. hd MUST be 128 (32 lanes x half4 = one coalesced 256B row read per simdgroup) --
// enforced by the Go dispatch site (canUseAttnFA), NOT in-kernel; every other head width keeps the
// shipped attention kernel. NOT bit-identical to it by design (reduction/combine order differs,
// same status as every other non-exact Metal kernel) -- scored by its own tolerance gate.
//
// Two dispatches, always, even at nSplit==1 (one code path; skipping the combine dispatch at
// nSplit==1 is a later, MEASURED optimization, not assumed here):
//
//  1. attention_fa: 128 threads (4 simdgroups) per (kvHead, split) threadgroup. Each simdgroup
//     walks its own key sub-range within the split (sgid-strided, so all 4 make even progress) and
//     runs online softmax (FlashAttention-style: running max m, running sum l, running numerator
//     acc) independently for all G group heads at once, keyed off ONE shared cooperative K/V read
//     (32 lanes x half4 = hd). m/l are simdgroup-uniform after simd_sum; acc is per-lane (each lane
//     owns a distinct 4-wide dim slice, 32*4=hd). The 4 simdgroups' partials are staged to
//     threadgroup memory and combined by the first 32 threads (fixed sg0->sg1->sg2->sg3 online-
//     softmax merge, one lane per dim-slice) into ONE (m, l, acc[hd]) triple per group head,
//     written to partial[kvHead][split][group head].
//  2. attention_fa_combine: nH threadgroups x hd threads, one thread per (query head, dim) --
//     merges the nSplit partials for that head (split-ascending order) and writes the final
//     out[nH*hd]. The (m,l) combine is redundantly recomputed per dim rather than shared across a
//     head's hd threads: O(nSplit) scalar work, negligible next to attention_fa's O(nKeys) pass it
//     follows. Metal serializes dispatches within one encoder in submission order, so the command-
//     buffer boundary between the two is the only sync needed -- no explicit fence.
//
// Split assignment: this threadgroup's key range is [winStart, nKeys) divided into nSplit
// CONTIGUOUS chunks (chunkLen = ceil(nWin/nSplit)), threadgroup index split owns chunk split --
// deterministic, not work-stealing. An empty chunk (chunkStart >= chunkEnd, or a simdgroup with no
// keys in its stride) leaves that split/simdgroup's (m, l) at the online-softmax identity
// (-INFINITY, 0), which the combine's exp(m2-m_new) guard (m2==-INFINITY -> weight 0) treats as a
// true no-op contribution, not a NaN.
#define ATTN_FA_MAXG 8
kernel void attention_fa(
    device const float* q[[buffer(0)]], device const half* kc[[buffer(1)]],
    device const half* vc[[buffer(2)]], device float* partial[[buffer(3)]],
    constant uint& nKV[[buffer(4)]], constant uint& G[[buffer(5)]],
    constant uint& nKeys[[buffer(6)]], constant float& scale[[buffer(7)]],
    constant uint& window[[buffer(8)]], constant uint& nSplit[[buffer(9)]],
    threadgroup float* shm[[threadgroup(0)]],   // 128 * (6*G) floats: per-thread (m[G],l[G],acc[G][4])
    uint tgid[[threadgroup_position_in_grid]], // flat kvHead*nSplit + split (1D dispatch API)
    uint tid[[thread_index_in_threadgroup]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]]) {
    const uint hd = 128u;
    const uint kvDim = nKV * hd;
    uint kvh = tgid / nSplit, split = tgid % nSplit;
    uint winStart = (window > 0u && nKeys > window) ? nKeys - window : 0u;
    uint nWin = nKeys - winStart;
    uint chunkLen = (nWin + nSplit - 1u) / nSplit;
    uint chunkStart = winStart + split * chunkLen;
    uint chunkEnd = min(chunkStart + chunkLen, nKeys);

    device const float* qr = q + kvh * G * hd; // this kvHead's G query heads, contiguous
    device const half*  kb = kc + kvh * hd;
    device const half*  vb = vc + kvh * hd;

    float m[ATTN_FA_MAXG]; float l[ATTN_FA_MAXG]; float acc[ATTN_FA_MAXG][4];
    for (uint g = 0; g < G; g++) { m[g] = -INFINITY; l[g] = 0.0f; acc[g][0]=acc[g][1]=acc[g][2]=acc[g][3]=0.0f; }

    float qslice[ATTN_FA_MAXG][4];
    for (uint g = 0; g < G; g++) {
        device const float* qh = qr + g*hd + lane*4u;
        qslice[g][0]=qh[0]; qslice[g][1]=qh[1]; qslice[g][2]=qh[2]; qslice[g][3]=qh[3];
    }

    for (uint s = chunkStart + sgid; s < chunkEnd; s += 4u) {
        half4 k4 = *((device const half4*)(kb + s*kvDim + lane*4u));
        half4 v4 = *((device const half4*)(vb + s*kvDim + lane*4u));
        for (uint g = 0; g < G; g++) {
            float part = qslice[g][0]*float(k4.x) + qslice[g][1]*float(k4.y)
                       + qslice[g][2]*float(k4.z) + qslice[g][3]*float(k4.w);
            float dot = simd_sum(part);
            float score = dot * scale;
            float m_new = max(m[g], score);
            float alpha = (m[g] == -INFINITY) ? 0.0f : exp(m[g] - m_new);
            float p = exp(score - m_new);
            l[g] = l[g]*alpha + p;
            acc[g][0] = acc[g][0]*alpha + p*float(v4.x);
            acc[g][1] = acc[g][1]*alpha + p*float(v4.y);
            acc[g][2] = acc[g][2]*alpha + p*float(v4.z);
            acc[g][3] = acc[g][3]*alpha + p*float(v4.w);
            m[g] = m_new;
        }
    }

    uint stride = 6u*G; // m[G] + l[G] + acc[G][4]
    threadgroup float* row = shm + tid*stride;
    for (uint g = 0; g < G; g++) { row[g] = m[g]; row[G+g] = l[g]; }
    for (uint g = 0; g < G; g++) {
        row[2u*G + g*4u+0u]=acc[g][0]; row[2u*G + g*4u+1u]=acc[g][1];
        row[2u*G + g*4u+2u]=acc[g][2]; row[2u*G + g*4u+3u]=acc[g][3];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float fm[ATTN_FA_MAXG]; float fl[ATTN_FA_MAXG]; float facc[ATTN_FA_MAXG][4];
        for (uint g=0; g<G; g++) { fm[g]=-INFINITY; fl[g]=0.0f; facc[g][0]=facc[g][1]=facc[g][2]=facc[g][3]=0.0f; }
        for (uint sg = 0; sg < 4u; sg++) {
            threadgroup float* r = shm + (sg*32u+lane)*stride;
            for (uint g = 0; g < G; g++) {
                float m2 = r[g], l2 = r[G+g];
                float m_new = max(fm[g], m2);
                float a1 = (fm[g]==-INFINITY) ? 0.0f : exp(fm[g]-m_new);
                float a2 = (m2==-INFINITY) ? 0.0f : exp(m2-m_new);
                fl[g] = fl[g]*a1 + l2*a2;
                facc[g][0] = facc[g][0]*a1 + r[2u*G+g*4u+0u]*a2;
                facc[g][1] = facc[g][1]*a1 + r[2u*G+g*4u+1u]*a2;
                facc[g][2] = facc[g][2]*a1 + r[2u*G+g*4u+2u]*a2;
                facc[g][3] = facc[g][3]*a1 + r[2u*G+g*4u+3u]*a2;
                fm[g] = m_new;
            }
        }
        uint pStride = G * (hd + 2u);
        device float* pout = partial + (kvh*nSplit + split) * pStride;
        for (uint g = 0; g < G; g++) {
            device float* og = pout + g*(hd+2u);
            og[0] = fm[g]; og[1] = fl[g];
            og[2u+lane*4u+0u]=facc[g][0]; og[2u+lane*4u+1u]=facc[g][1];
            og[2u+lane*4u+2u]=facc[g][2]; og[2u+lane*4u+3u]=facc[g][3];
        }
    }
}
kernel void attention_fa_combine(
    device const float* partial[[buffer(0)]], device float* out[[buffer(1)]],
    constant uint& G[[buffer(2)]], constant uint& hd[[buffer(3)]], constant uint& nSplit[[buffer(4)]],
    uint qh[[threadgroup_position_in_grid]], uint d[[thread_position_in_threadgroup]]) {
    uint kvh = qh / G, g = qh % G;
    uint pStride = G * (hd + 2u);
    float m = -INFINITY, l = 0.0f, acc = 0.0f;
    for (uint split = 0; split < nSplit; split++) {
        device const float* pg = partial + (kvh*nSplit + split) * pStride + g*(hd+2u);
        float m2 = pg[0], l2 = pg[1], a2 = pg[2u+d];
        float m_new = max(m, m2);
        float a1w = (m==-INFINITY) ? 0.0f : exp(m-m_new);
        float a2w = (m2==-INFINITY) ? 0.0f : exp(m2-m_new);
        l = l*a1w + l2*a2w;
        acc = acc*a1w + a2*a2w;
        m = m_new;
    }
    out[qh*hd + d] = acc / l;
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
    uint nWin = nKeys - winStart;
    device const float* qr = q + qh*hd;
    device const float* kb = kc + kvh*hd;
    device const float* vb = vc + kvh*hd;
    threadgroup float sc[4096];
    threadgroup float red[128];

    // Single-tile fast path (nWin <= 4096)
    if (nWin <= 4096u) {
        for (uint s=winStart+tid; s<nKeys; s+=tgs) {
            float a=0; device const float* k=kb+s*kvDim; uint d=0;
            // float4 vectorized K-read (same coalescing fix as attention, f32 KV). Bit-identical.
            if ((hd&3u)==0u) for (; d<hd; d+=4u){ float4 k4=*((device const float4*)(k+d)); a+=qr[d]*k4.x; a+=qr[d+1u]*k4.y; a+=qr[d+2u]*k4.z; a+=qr[d+3u]*k4.w; }
            for (; d<hd; d++) a += qr[d]*k[d];
            sc[s - winStart]=a*scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float m=-INFINITY; for (uint s=tid;s<nWin;s+=tgs) m=max(m,sc[s]);
        red[tid]=m; threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st=tgs/2; st>0; st>>=1){ if(tid<st) red[tid]=max(red[tid],red[tid+st]); threadgroup_barrier(mem_flags::mem_threadgroup); }
        float mx=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
        float ls=0; for (uint s=tid;s<nWin;s+=tgs){ float p=exp(sc[s]-mx); sc[s]=p; ls+=p; }
        red[tid]=ls; threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st=tgs/2; st>0; st>>=1){ if(tid<st) red[tid]+=red[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }
        float sum=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d=tid; d<hd; d+=tgs){ float a=0; for(uint s=winStart;s<nKeys;s++) a += sc[s - winStart]*vb[s*kvDim+d]; out[qh*hd+d]=a/sum; }
        return;
    }

    // Deep-context tiled online softmax path (nWin > 4096)
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint tileBase = winStart; tileBase < nKeys; tileBase += 4096u) {
        uint tileEnd = min(tileBase + 4096u, nKeys);
        uint tileLen = tileEnd - tileBase;

        for (uint s = tileBase + tid; s < tileEnd; s += tgs) {
            float a = 0; device const float* k = kb + s*kvDim; uint d = 0;
            if ((hd&3u) == 0u) for (; d < hd; d += 4u) {
                float4 k4 = *((device const float4*)(k+d));
                a += qr[d]*k4.x;
                a += qr[d+1u]*k4.y;
                a += qr[d+2u]*k4.z;
                a += qr[d+3u]*k4.w;
            }
            for (; d < hd; d++) a += qr[d]*k[d];
            sc[s - tileBase] = a * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float m_t = -INFINITY;
        for (uint s = tid; s < tileLen; s += tgs) m_t = max(m_t, sc[s]);
        red[tid] = m_t;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st = tgs/2; st > 0; st >>= 1) {
            if (tid < st) red[tid] = max(red[tid], red[tid+st]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float m_tile = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float m_new = max(m_prev, m_tile);
        float alpha = (m_prev == -INFINITY) ? 0.0f : exp(m_prev - m_new);

        float ls = 0;
        for (uint s = tid; s < tileLen; s += tgs) {
            float p = exp(sc[s] - m_new);
            sc[s] = p;
            ls += p;
        }
        red[tid] = ls;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st = tgs/2; st > 0; st >>= 1) {
            if (tid < st) red[tid] += red[tid+st];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float sum_tile = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        l_prev = l_prev * alpha + sum_tile;
        m_prev = m_new;

        for (uint slot = 0; slot < 4u; slot++) {
            uint d = tid + slot * tgs;
            if (d >= hd) break;
            float a = acc[slot] * alpha;
            for (uint s = tileBase; s < tileEnd; s++) {
                a += sc[s - tileBase]*vb[s*kvDim+d];
            }
            acc[slot] = a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint slot = 0; slot < 4u; slot++) {
        uint d = tid + slot * tgs;
        if (d >= hd) break;
        out[qh*hd + d] = acc[slot] / l_prev;
    }
}
// attention_i8: reads int8 KV cache with per-(position,KV-head) f32 scales (ks, vs).
// Halves KV memory bandwidth and cache memory footprint vs f16 KV.
kernel void attention_i8(device const float* q[[buffer(0)]], device const char* kc[[buffer(1)]],
    device const char* vc[[buffer(2)]], device const float* ks[[buffer(3)]],
    device const float* vs[[buffer(4)]], device float* out[[buffer(5)]],
    constant uint& nH[[buffer(6)]], constant uint& nKV[[buffer(7)]], constant uint& hd[[buffer(8)]],
    constant uint& nKeys[[buffer(9)]], constant float& scale[[buffer(10)]],
    constant uint& window[[buffer(11)]], device const float* sinks[[buffer(12)]],
    constant uint& hasSink[[buffer(13)]],
    uint qh[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    uint kvDim = nKV*hd; uint kvh = qh/(nH/nKV);
    uint winStart = (window>0u && nKeys>window) ? nKeys-window : 0u;
    uint nWin = nKeys - winStart;
    device const float* qr = q + qh*hd;
    device const char*  kb = kc + kvh*hd;
    device const char*  vb = vc + kvh*hd;
    threadgroup float sc[4096];
    threadgroup float red[128];

    // Single-tile fast path (nWin <= 4096)
    if (nWin <= 4096u) {
        for (uint s=winStart+tid; s<nKeys; s+=tgs) {
            float a=0; device const char* k=kb+s*kvDim; uint d=0;
            float k_scale = ks[s*nKV + kvh];
            if ((hd&3u)==0u) for (; d<hd; d+=4u){ char4 k4=*((device const char4*)(k+d)); a+=qr[d]*float(k4.x); a+=qr[d+1u]*float(k4.y); a+=qr[d+2u]*float(k4.z); a+=qr[d+3u]*float(k4.w); }
            for (; d<hd; d++) a += qr[d]*float(k[d]);
            sc[s - winStart]=(a * k_scale)*scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float m=-INFINITY; for (uint s=tid;s<nWin;s+=tgs) m=max(m,sc[s]);
        red[tid]=m; threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st=tgs/2; st>0; st>>=1){ if(tid<st) red[tid]=max(red[tid],red[tid+st]); threadgroup_barrier(mem_flags::mem_threadgroup); }
        float mx=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
        float sink=0.0f; bool hasS = hasSink != 0u;
        if (hasS) { sink = sinks[qh]; mx = max(mx, sink); }
        float ls=0; for (uint s=tid;s<nWin;s+=tgs){ float p=exp(sc[s]-mx); sc[s]=p; ls+=p; }
        red[tid]=ls; threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st=tgs/2; st>0; st>>=1){ if(tid<st) red[tid]+=red[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }
        float sum=red[0];
        if (hasS) sum += exp(sink-mx);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d=tid; d<hd; d+=tgs){
            float a=0; uint s=winStart; uint nMain = winStart + (nWin & ~7u);
            for (; s<nMain; s+=8u) {
                uint sc_idx = s - winStart;
                float vs0=vs[(s+0u)*nKV+kvh], vs1=vs[(s+1u)*nKV+kvh], vs2=vs[(s+2u)*nKV+kvh], vs3=vs[(s+3u)*nKV+kvh];
                float vs4=vs[(s+4u)*nKV+kvh], vs5=vs[(s+5u)*nKV+kvh], vs6=vs[(s+6u)*nKV+kvh], vs7=vs[(s+7u)*nKV+kvh];
                float v0=float(vb[(s+0u)*kvDim+d])*vs0;
                float v1=float(vb[(s+1u)*kvDim+d])*vs1;
                float v2=float(vb[(s+2u)*kvDim+d])*vs2;
                float v3=float(vb[(s+3u)*kvDim+d])*vs3;
                float v4=float(vb[(s+4u)*kvDim+d])*vs4;
                float v5=float(vb[(s+5u)*kvDim+d])*vs5;
                float v6=float(vb[(s+6u)*kvDim+d])*vs6;
                float v7=float(vb[(s+7u)*kvDim+d])*vs7;
                a += sc[sc_idx+0u]*v0; a += sc[sc_idx+1u]*v1; a += sc[sc_idx+2u]*v2; a += sc[sc_idx+3u]*v3;
                a += sc[sc_idx+4u]*v4; a += sc[sc_idx+5u]*v5; a += sc[sc_idx+6u]*v6; a += sc[sc_idx+7u]*v7;
            }
            for (; s<nKeys; s++) a += sc[s - winStart]*(float(vb[s*kvDim+d])*vs[s*nKV+kvh]);
            out[qh*hd+d]=a/sum;
        }
        return;
    }

    // Deep-context tiled online softmax path (nWin > 4096)
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    bool hasS = hasSink != 0u;
    if (hasS) {
        m_prev = sinks[qh];
        l_prev = 1.0f;
    }
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint tileBase = winStart; tileBase < nKeys; tileBase += 4096u) {
        uint tileEnd = min(tileBase + 4096u, nKeys);
        uint tileLen = tileEnd - tileBase;

        for (uint s = tileBase + tid; s < tileEnd; s += tgs) {
            float a = 0; device const char* k = kb + s*kvDim; uint d = 0;
            float k_scale = ks[s*nKV + kvh];
            if ((hd&3u) == 0u) for (; d < hd; d += 4u) {
                char4 k4 = *((device const char4*)(k+d));
                a += qr[d]*float(k4.x);
                a += qr[d+1u]*float(k4.y);
                a += qr[d+2u]*float(k4.z);
                a += qr[d+3u]*float(k4.w);
            }
            for (; d < hd; d++) a += qr[d]*float(k[d]);
            sc[s - tileBase] = (a * k_scale) * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float m_t = -INFINITY;
        for (uint s = tid; s < tileLen; s += tgs) m_t = max(m_t, sc[s]);
        red[tid] = m_t;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st = tgs/2; st > 0; st >>= 1) {
            if (tid < st) red[tid] = max(red[tid], red[tid+st]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float m_tile = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float m_new = max(m_prev, m_tile);
        float alpha = (m_prev == -INFINITY) ? 0.0f : exp(m_prev - m_new);

        float ls = 0;
        for (uint s = tid; s < tileLen; s += tgs) {
            float p = exp(sc[s] - m_new);
            sc[s] = p;
            ls += p;
        }
        red[tid] = ls;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st = tgs/2; st > 0; st >>= 1) {
            if (tid < st) red[tid] += red[tid+st];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float sum_tile = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        l_prev = l_prev * alpha + sum_tile;
        m_prev = m_new;

        for (uint slot = 0; slot < 4u; slot++) {
            uint d = tid + slot * tgs;
            if (d >= hd) break;
            float a = acc[slot] * alpha;
            uint s = tileBase;
            uint nMain = tileBase + (tileLen & ~7u);
            for (; s < nMain; s += 8u) {
                uint sc_idx = s - tileBase;
                float vs0=vs[(s+0u)*nKV+kvh], vs1=vs[(s+1u)*nKV+kvh], vs2=vs[(s+2u)*nKV+kvh], vs3=vs[(s+3u)*nKV+kvh];
                float vs4=vs[(s+4u)*nKV+kvh], vs5=vs[(s+5u)*nKV+kvh], vs6=vs[(s+6u)*nKV+kvh], vs7=vs[(s+7u)*nKV+kvh];
                float v0 = float(vb[(s+0u)*kvDim+d]) * vs0;
                float v1 = float(vb[(s+1u)*kvDim+d]) * vs1;
                float v2 = float(vb[(s+2u)*kvDim+d]) * vs2;
                float v3 = float(vb[(s+3u)*kvDim+d]) * vs3;
                float v4 = float(vb[(s+4u)*kvDim+d]) * vs4;
                float v5 = float(vb[(s+5u)*kvDim+d]) * vs5;
                float v6 = float(vb[(s+6u)*kvDim+d]) * vs6;
                float v7 = float(vb[(s+7u)*kvDim+d]) * vs7;
                a += sc[sc_idx+0u]*v0; a += sc[sc_idx+1u]*v1; a += sc[sc_idx+2u]*v2; a += sc[sc_idx+3u]*v3;
                a += sc[sc_idx+4u]*v4; a += sc[sc_idx+5u]*v5; a += sc[sc_idx+6u]*v6; a += sc[sc_idx+7u]*v7;
            }
            for (; s < tileEnd; s++) {
                a += sc[s - tileBase]*(float(vb[s*kvDim+d])*vs[s*nKV+kvh]);
            }
            acc[slot] = a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint slot = 0; slot < 4u; slot++) {
        uint d = tid + slot * tgs;
        if (d >= hd) break;
        out[qh*hd + d] = acc[slot] / l_prev;
    }
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
// glu_act_pinned: swiglu_quant-only variant, precise::tanh instead of tanh -- PROBE testing
// whether pinning the transcendental removes swiglu_quant's cross-code-shape drift, the same
// mechanism confirmed for rmsnorm_quant's rsqrt (see that round's commit). Not used by act_quant.
inline float glu_act_pinned(float x, uint act) {
    if (act == ACT_SILU) return x/(1.0f+exp(-x));
    float a = 0.7978845608028654f*(x+0.044715f*x*x*x);
    return 0.5f*x*(1.0f+precise::tanh(clamp(a, -15.0f, 15.0f)));
}
kernel void swiglu_quant(device const float* g[[buffer(0)]], device const float* u[[buffer(1)]],
    device char* dq[[buffer(2)]], device float* ds[[buffer(3)]], constant uint& I[[buffer(4)]],
    constant uint& act[[buffer(5)]],
    uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256]; float mx=0;
    // PROBE: vectorized float4 load of g/u (batch 4 scalar reads), glu_act_pinned kept scalar-per-
    // lane -- tanh is pinned via precise:: above, so this is stable regardless of load shape.
    {
        uint I4v = I >> 2u;
        device const float4* g4v = (device const float4*)g;
        device const float4* u4v = (device const float4*)u;
        for(uint i4=tid;i4<I4v;i4+=tgs){
            float4 gv=g4v[i4], uv=u4v[i4];
            float4 sv = float4(glu_act_pinned(gv.x,act),glu_act_pinned(gv.y,act),glu_act_pinned(gv.z,act),glu_act_pinned(gv.w,act)) * uv;
            float4 av = fabs(sv);
            mx=max(mx,av.x); mx=max(mx,av.y); mx=max(mx,av.z); mx=max(mx,av.w);
        }
        for(uint i=(I4v<<2u)+tid;i<I;i+=tgs){ float s=glu_act_pinned(g[i],act)*u[i]; mx=max(mx,fabs(s)); }
    }
    mx = simd_max(mx);
    uint sgid = tid >> 5u, lane = tid & 31u;
    if (lane == 0) red[sgid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nsg = tgs >> 5u;
    if (tid == 0) { float v = red[0]; for (uint k=1; k<nsg; k++) v = max(v, red[k]); red[0] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sc=red[0]/127.0f; if(sc==0)sc=1; if(tid==0)ds[0]=sc; float inv=1/sc;
    {
        uint I4w = I >> 2u;
        device const float4* g4w = (device const float4*)g;
        device const float4* u4w = (device const float4*)u;
        device char4* dq4w = (device char4*)dq;
        for(uint i4=tid;i4<I4w;i4+=tgs){
            float4 gv=g4w[i4], uv=u4w[i4];
            float4 sv = float4(glu_act_pinned(gv.x,act),glu_act_pinned(gv.y,act),glu_act_pinned(gv.z,act),glu_act_pinned(gv.w,act)) * uv;
            dq4w[i4] = char4(char(clamp(int(round(sv.x*inv)),-127,127)), char(clamp(int(round(sv.y*inv)),-127,127)),
                              char(clamp(int(round(sv.z*inv)),-127,127)), char(clamp(int(round(sv.w*inv)),-127,127)));
        }
        for(uint i=(I4w<<2u)+tid;i<I;i+=tgs){ float s=glu_act_pinned(g[i],act)*u[i]; dq[i]=char(clamp(int(round(s*inv)),-127,127)); }
    }
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
    // 2-level SIMD-shuffle reduction instead of the 8-step threadgroup-barrier tree -- max is
    // exact/order-independent regardless of reduction structure. Cuts barrier count from 8 to 2
    // (verified safe on quant_vec/layernorm_quant this session). Does not touch glu_act's own
    // computation at all -- only how the already-computed per-lane mx values combine.
    mx = simd_max(mx);
    uint sgid = tid >> 5u, lane = tid & 31u;
    if (lane == 0) red[sgid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nsg = tgs >> 5u;
    if (tid == 0) { float v = red[0]; for (uint k=1; k<nsg; k++) v = max(v, red[k]); red[0] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
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

// lora_delta: compute-time LoRA (G3, docs/tasks/task-gpu-paths-2026-09.md), the two-GEMV low-rank
// delta y[o] += scale·Σ_r B[o,r]·(A·x)[r], fused into ONE dispatch (P-11, audit-2026-09-10 —
// mirrors CUDA's own P-11 finding, "14 extra launches per layer per token... before the
// serialized reduction": up to 7 targeted projections/layer each paid TWO dispatches — down then
// up — for pure per-dispatch launch overhead no different in kind from CUDA's). x is the SAME
// quantized activation (aq/asc) the base projection this delta rides alongside already consumed
// — matching applyLoRA's CPU reference, which takes the identical input the base matmul does
// (decoder/lora.go).
//
// M-08 (audit-metal-2026-09-12.md): P-11's fused-into-one-threadgroup shape traded a whole
// dispatch for making the up stage SERIAL over Out — every one of 256 threads striding across
// however many output rows there are, up to thousands. Now launched over ceil(Out/256)
// threadgroups (tgid selects a FIXED, disjoint 256-row block each owns outright — no stride, no
// race, still exactly one thread per row across the whole dispatch), each independently
// RECOMPUTING the down-stage's t[R] rather than reading it from a device-memory scratch buffer a
// separate kernel wrote (CUDA's own P-11 fix, cuda/lora.cu's lora_delta_down/_up split): still
// ONE dispatch (no second launch's overhead, the more expensive line item on Metal's launch/sync
// ceiling), and the redundant R reductions over K this costs are cheap relative to a second
// dispatch — A stays L2-resident across the ceil(Out/256) threadgroups reading it, and R·K MACs
// is small next to Out·R's up-stage work at any Out worth splitting over more than one
// threadgroup. A/B are read as half (PEFT ships them f32; SetAdapter converts once at bind time,
// metal/lora.go) — the delta feeds an int8-quantised activation, so f32 A/B precision was never
// load-bearing, and halving their bytes matters most here: A is re-read whole by every
// threadgroup in the group, not just once per dispatch.
kernel void lora_delta(device const char* aq[[buffer(0)]], device const float* asc[[buffer(1)]],
    device const half* A[[buffer(2)]], device const half* B[[buffer(3)]],
    device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], constant uint& R[[buffer(6)]], constant uint& Out[[buffer(7)]],
    constant float& scale[[buffer(8)]],
    uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    // Fixed-size STATIC threadgroup arrays (matches rmsnorm_quant's own reduction, kernels.go
    // top) — tgs is pinned to tgReduceNorm (256, model.go) by the dispatch site, same as every
    // other norm-class reduction kernel in this file. Deliberately NOT a [[threadgroup(0)]]
    // dynamic parameter: that form requires the caller to use DispatchTG (which sets the length
    // explicitly) rather than plain Dispatch — using a static array here avoids that footgun
    // entirely, the same way every other single-threadgroup reduction kernel in this file does.
    threadgroup float red[256];
    threadgroup float t[256]; // R <= loraRMax (256, model.go)
    float sc = asc[0];
    for (uint r = 0; r < R; r++) {
        device const half* Ar = A + r*K;
        float part = 0.0f;
        for (uint k = tid; k < K; k += tgs) part += float(Ar[k]) * (float(aq[k]) * sc);
        red[tid] = part;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tgs/2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
        if (tid == 0) t[r] = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint row = tgid*tgs + tid;
    if (row < Out) {
        device const half* Br = B + row*R;
        float acc = 0.0f;
        for (uint r = 0; r < R; r++) acc += float(Br[r]) * t[r];
        out[row] += scale*acc;
    }
}

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

// copy_f32 copies N elements from src to dst. Used by batched ForwardN to stage each
// token's input embedding on-device into r.x without host-GPU synchronization.
kernel void copy_f32(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
    constant uint& N [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < N) dst[i] = src[i];
}
` + moeKernels + gemma4MoeKernels + deltaNetKernels
