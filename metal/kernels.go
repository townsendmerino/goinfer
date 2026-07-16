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
// Coalesced + ILP W4A8 GEMV: ONE simdgroup (32 lanes) per output row. int4 weights = half
// the bytes of int8 (the target-quant bandwidth win). Each lane strides over the row's
// 32-nibble groups; the 8-nibble inner loop is fully unrolled (ILP), f32 group scale
// folded per group; simd_sum reduces the 32 lane partials. Launch total = N*32, tg = 32.
kernel void gemv_w4a8_coal(device const uint* bq[[buffer(0)]], device const float* bsc[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], uint gid[[threadgroup_position_in_grid]], uint lid[[thread_index_in_threadgroup]]) {
    uint gpr = K/32u;                       // groups per row
    device const uint*  brow = bq  + (uint)gid*(K/8u);
    device const float* srow = bsc + (uint)gid*gpr;
    float acc = 0.0f;
    for (uint g = lid; g < gpr; g += 32u) {
        device const uint* gw = brow + g*4u;  // 4 words = 32 nibbles
        device const char* ga = aq + g*32u;
        int gi = 0;
        for (uint w = 0; w < 4u; w++) {
            uint x = gw[w];
            device const char* a = ga + w*8u;
            gi += (int((x)      & 0xF)-8)*int(a[0]) + (int((x>>4)  & 0xF)-8)*int(a[1])
                + (int((x>>8)   & 0xF)-8)*int(a[2]) + (int((x>>12) & 0xF)-8)*int(a[3])
                + (int((x>>16)  & 0xF)-8)*int(a[4]) + (int((x>>20) & 0xF)-8)*int(a[5])
                + (int((x>>24)  & 0xF)-8)*int(a[6]) + (int((x>>28) & 0xF)-8)*int(a[7]);
        }
        acc += float(gi) * srow[g];
    }
    acc = simd_sum(acc);
    if (lid == 0) out[gid] = acc * asc[0];
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
kernel void attention(device const float* q[[buffer(0)]], device const float* kc[[buffer(1)]],
    device const float* vc[[buffer(2)]], device float* out[[buffer(3)]], constant uint& nH[[buffer(4)]],
    constant uint& nKV[[buffer(5)]], constant uint& hd[[buffer(6)]], constant uint& nKeys[[buffer(7)]],
    constant float& scale[[buffer(8)]], uint qh[[thread_position_in_grid]]) {
    if(qh>=nH) return; uint kvDim=nKV*hd; uint kvh=qh/(nH/nKV); uint qb=qh*hd; uint kb=kvh*hd;
    float acc[128]; for(uint d=0;d<hd;d++) acc[d]=0; float m=-INFINITY,l=0;
    for(uint s=0;s<nKeys;s++){ float sc=0; for(uint d=0;d<hd;d++) sc+=q[qb+d]*kc[s*kvDim+kb+d]; sc*=scale;
        float mn=max(m,sc); float a=exp(m-mn),p=exp(sc-mn); l=l*a+p;
        for(uint d=0;d<hd;d++) acc[d]=acc[d]*a+p*vc[s*kvDim+kb+d]; m=mn; }
    for(uint d=0;d<hd;d++) out[qb+d]=acc[d]/l;
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
