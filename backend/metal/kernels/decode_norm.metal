// addOne selects Gemma's (1+w) RMS offset vs plain w — mirrors decoder's rmsnorm.go, which
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
// (gemv_w4f16_sa* above is the weight-stream half). Same
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
// mean-centered (decoder's rmsnorm.go's layerNorm): y = (x-mean)/sqrt(var+eps)*w + b, then
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
