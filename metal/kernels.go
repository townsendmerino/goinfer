//go:build darwin

package metal

// allKernels is the full dense-decode-layer MSL kernel set in one library (W8A8 path —
// W4A8 is validated separately; this proves ASSEMBLY, not the int4 packing again).
const allKernels = `
#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm_quant(device const float* x[[buffer(0)]], device const float* w[[buffer(1)]],
    device char* aq[[buffer(2)]], device float* asc[[buffer(3)]], constant uint& H[[buffer(4)]],
    constant float& eps[[buffer(5)]], uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256]; float ss=0;
    for(uint i=tid;i<H;i+=tgs) ss+=x[i]*x[i];
    red[tid]=ss; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=tgs/2;s>0;s>>=1){ if(tid<s) red[tid]+=red[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup);}
    float rms=rsqrt(red[0]/float(H)+eps); threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx=0; for(uint i=tid;i<H;i+=tgs) mx=max(mx,fabs(x[i]*rms*w[i]));
    red[tid]=mx; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=tgs/2;s>0;s>>=1){ if(tid<s) red[tid]=max(red[tid],red[tid+s]); threadgroup_barrier(mem_flags::mem_threadgroup);}
    float sc=red[0]/127.0f; if(sc==0)sc=1; if(tid==0)asc[0]=sc; float inv=1/sc;
    for(uint i=tid;i<H;i+=tgs) aq[i]=char(clamp(int(round(x[i]*rms*w[i]*inv)),-127,127));
}
kernel void quant_vec(device const float* x[[buffer(0)]], device char* aq[[buffer(1)]],
    device float* asc[[buffer(2)]], constant uint& H[[buffer(3)]],
    uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256]; float mx=0;
    for(uint i=tid;i<H;i+=tgs) mx=max(mx,fabs(x[i]));
    red[tid]=mx; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=tgs/2;s>0;s>>=1){ if(tid<s) red[tid]=max(red[tid],red[tid+s]); threadgroup_barrier(mem_flags::mem_threadgroup);}
    float sc=red[0]/127.0f; if(sc==0)sc=1; if(tid==0)asc[0]=sc; float inv=1/sc;
    for(uint i=tid;i<H;i+=tgs) aq[i]=char(clamp(int(round(x[i]*inv)),-127,127));
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
    int acc = 0;
    for (uint k = lid; k < K; k += 32u) acc += int(aq[k]) * int(brow[k]);
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
    device const float* srow = bsc + (uint)gid*(K/32u); \
    float acc = 0.0f; \
    for (uint wi = lid; wi < wpr; wi += 32u) { \
        uint x = brow[wi]; device const char* a = aq + wi*8u; \
        int gi = (int((x)&0xF)-8)*int(a[0]) + (int((x>>4)&0xF)-8)*int(a[1]) \
               + (int((x>>8)&0xF)-8)*int(a[2]) + (int((x>>12)&0xF)-8)*int(a[3]) \
               + (int((x>>16)&0xF)-8)*int(a[4]) + (int((x>>20)&0xF)-8)*int(a[5]) \
               + (int((x>>24)&0xF)-8)*int(a[6]) + (int((x>>28)&0xF)-8)*int(a[7]); \
        acc += float(gi) * srow[wi>>2]; \
    } \
    acc = simd_sum(acc);

// Launch total = N*32, tg = 32. int4 weights = half the bytes of int8 (the target-quant
// bandwidth win). _coal is the plain projection; _bias/_resid fuse an epilogue.
kernel void gemv_w4a8_coal(device const uint* bq[[buffer(0)]], device const float* bsc[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], uint gid[[threadgroup_position_in_grid]], uint lid[[thread_index_in_threadgroup]]) {
    W4A8_BODY
    if (lid == 0) out[gid] = acc * asc[0];
}
kernel void gemv_w4a8_bias(device const uint* bq[[buffer(0)]], device const float* bsc[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    device const float* bias[[buffer(5)]], constant uint& K[[buffer(6)]],
    uint gid[[threadgroup_position_in_grid]], uint lid[[thread_index_in_threadgroup]]) {
    W4A8_BODY
    if (lid == 0) out[gid] = acc*asc[0] + bias[gid];
}
kernel void gemv_w4a8_resid(device const uint* bq[[buffer(0)]], device const float* bsc[[buffer(1)]],
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
    threadgroup short As[1536]; \
    for (uint i=tid;i<K;i+=tgs) As[i]=short(aq[i]); \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    uint G = K>>5u; \
    uint row = tgid*(tgs>>5u) + sgid; \
    device const uint4* wr = wq + (uint)row*G; \
    device const float* sr = sct + (uint)row*G; \
    float acc = 0.0f; \
    for (uint g=lane; g<G; g+=32u) { \
        uint4 w = wr[g]; threadgroup const short* a = As + g*32u; \
        int gi = UNP8(w.x,a) + UNP8(w.y,a+8) + UNP8(w.z,a+16) + UNP8(w.w,a+24); \
        acc += float(gi) * sr[g]; \
    } \
    acc = simd_sum(acc);
kernel void gemv_w4a8_sa(device const uint4* wq[[buffer(0)]], device const float* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    SA_BODY
    if (lane==0) out[row] = acc*asc[0];
}
kernel void gemv_w4a8_sa_bias(device const uint4* wq[[buffer(0)]], device const float* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    device const float* bias[[buffer(5)]], constant uint& K[[buffer(6)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    SA_BODY
    if (lane==0) out[row] = acc*asc[0] + bias[row];
}
kernel void gemv_w4a8_sa_resid(device const uint4* wq[[buffer(0)]], device const float* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    SA_BODY
    if (lane==0) out[row] += acc*asc[0];
}

// Fused block-argmax lm-head (Fable): computes the SAME logits as gemv_w4a8_sa but never
// materializes them — each threadgroup emits (maxLogit, rowIndex) over its 8 rows; a tiny
// second pass (argmax_finish) reduces the tiles to one token. Kills the 608KB logit readback
// + CPU scan. Merge key (v, -idx) is a commutative monoid → order-independent, tie-broken
// identically to a CPU first-max-wins scan (strict >, lower index wins).
struct AmaxPart { float v; uint i; };
kernel void gemv_w4a8_sa_amax(device const uint4* wq[[buffer(0)]], device const float* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device AmaxPart* part[[buffer(4)]],
    constant uint& K[[buffer(5)]], uint tgid[[threadgroup_position_in_grid]],
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
kernel void rope(device float* x[[buffer(0)]], device const float* invf[[buffer(1)]],
    constant uint& hd[[buffer(2)]], constant uint& pos[[buffer(3)]], constant uint& total[[buffer(4)]],
    uint gid[[thread_position_in_grid]]) {
    if(gid>=total) return; uint hlf=hd/2; uint head=gid/hlf; uint dd=gid%hlf; uint base=head*hd;
    float th=float(pos)*invf[dd]; float c=cos(th),s=sin(th);
    float x0=x[base+dd],x1=x[base+hlf+dd]; x[base+dd]=x0*c-x1*s; x[base+hlf+dd]=x0*s+x1*c;
}
kernel void kv_store(device const float* k[[buffer(0)]], device const float* v[[buffer(1)]],
    device float* kc[[buffer(2)]], device float* vc[[buffer(3)]], constant uint& kvDim[[buffer(4)]],
    constant uint& pos[[buffer(5)]], uint i[[thread_position_in_grid]]) {
    kc[pos*kvDim+i]=k[i]; vc[pos*kvDim+i]=v[i];
}
// One THREADGROUP (128 threads) per query head — vs the old 1-thread-per-head (12 threads
// total = 68% of decode time from underutilization). Scores parallel over keys, softmax via
// threadgroup reduction, output parallel over head dims. nKeys ≤ metalCtxCap (4096).
kernel void attention(device const float* q[[buffer(0)]], device const float* kc[[buffer(1)]],
    device const float* vc[[buffer(2)]], device float* out[[buffer(3)]], constant uint& nH[[buffer(4)]],
    constant uint& nKV[[buffer(5)]], constant uint& hd[[buffer(6)]], constant uint& nKeys[[buffer(7)]],
    constant float& scale[[buffer(8)]], uint qh[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    uint kvDim = nKV*hd; uint kvh = qh/(nH/nKV);
    device const float* qr = q + qh*hd;
    device const float* kb = kc + kvh*hd;
    device const float* vb = vc + kvh*hd;
    threadgroup float sc[4096];
    threadgroup float red[128];
    for (uint s=tid; s<nKeys; s+=tgs) {
        float a=0; device const float* k=kb+s*kvDim;
        for (uint d=0; d<hd; d++) a += qr[d]*k[d];
        sc[s]=a*scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m=-INFINITY; for (uint s=tid;s<nKeys;s+=tgs) m=max(m,sc[s]);
    red[tid]=m; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st=tgs/2; st>0; st>>=1){ if(tid<st) red[tid]=max(red[tid],red[tid+st]); threadgroup_barrier(mem_flags::mem_threadgroup); }
    float mx=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    float ls=0; for (uint s=tid;s<nKeys;s+=tgs){ float p=exp(sc[s]-mx); sc[s]=p; ls+=p; }
    red[tid]=ls; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st=tgs/2; st>0; st>>=1){ if(tid<st) red[tid]+=red[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float sum=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d=tid; d<hd; d+=tgs){ float a=0; for(uint s=0;s<nKeys;s++) a += sc[s]*vb[s*kvDim+d]; out[qh*hd+d]=a/sum; }
}
kernel void swiglu_quant(device const float* g[[buffer(0)]], device const float* u[[buffer(1)]],
    device char* dq[[buffer(2)]], device float* ds[[buffer(3)]], constant uint& I[[buffer(4)]],
    uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256]; float mx=0;
    for(uint i=tid;i<I;i+=tgs){ float s=(g[i]/(1+exp(-g[i])))*u[i]; mx=max(mx,fabs(s)); }
    red[tid]=mx; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=tgs/2;s>0;s>>=1){ if(tid<s) red[tid]=max(red[tid],red[tid+s]); threadgroup_barrier(mem_flags::mem_threadgroup);}
    float sc=red[0]/127.0f; if(sc==0)sc=1; if(tid==0)ds[0]=sc; float inv=1/sc;
    for(uint i=tid;i<I;i+=tgs){ float s=(g[i]/(1+exp(-g[i])))*u[i]; dq[i]=char(clamp(int(round(s*inv)),-127,127)); }
}
kernel void residual(device float* x[[buffer(0)]], device const float* y[[buffer(1)]], uint i[[thread_position_in_grid]]) { x[i]+=y[i]; }
`
