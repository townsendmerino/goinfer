// glu_act selects the gated MLP's activation. The ordinals are deliberately decoder.ActKind's
// iota (ActGeluTanh=0, ActSiLU=1) so the host passes int(m.GatedActResident()) straight through.
// GELU-tanh matches decoder's rmsnorm.go geluTanh; both compute in f32 where the CPU reference
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
// swiglu_f32: swiglu_quant's activation with no quantization (the q4k lane's down-projection input), grid = I.
kernel void swiglu_f32(device const float* g[[buffer(0)]], device const float* u[[buffer(1)]],
    device float* out[[buffer(2)]], constant uint& I[[buffer(3)]], constant uint& act[[buffer(4)]],
    uint i[[thread_position_in_grid]]) {
    if (i < I) out[i] = glu_act_pinned(g[i], act) * u[i];
}
// act_quant: FeatNonGatedMLP's fused activation+quant — up->act->down (GPT-2, Nemotron relu²),
// no gate multiply, unlike swiglu_quant. Its activation is nongated_act: glu_act's GELU-tanh and
// SiLU (inheriting the GELU-tanh clamp fix), plus squared ReLU. Exact-erf GELU (decoder's ActGelu,
// HF's plain "gelu") is NOT implemented — GPT-2's real released checkpoints use "gelu_new"
// (ActGeluTanh), which this covers.
// nongated_act is act_quant's activation: glu_act's ordinals, plus squared ReLU at decoder.ActReLU2's ordinal (2), which only
// a non-gated MLP takes (Nemotron-H), so glu_act's other callers keep their compiled code.
#define ACT_RELU2 2u
inline float nongated_act(float x, uint act) {
    if (act == ACT_RELU2) { float r = max(x, 0.0f); return r*r; }
    return glu_act(x, act);
}
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
        float s0=nongated_act(uv.x,act), s1=nongated_act(uv.y,act), s2=nongated_act(uv.z,act), s3=nongated_act(uv.w,act);
        mx=max(mx,fabs(s0)); mx=max(mx,fabs(s1)); mx=max(mx,fabs(s2)); mx=max(mx,fabs(s3));
    }
    for (uint i=(I4<<2u)+tid; i<I; i+=tgs){ float s=nongated_act(u[i],act); mx=max(mx,fabs(s)); }
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
        float s0=nongated_act(uv.x,act), s1=nongated_act(uv.y,act), s2=nongated_act(uv.z,act), s3=nongated_act(uv.w,act);
        char q0=char(clamp(int(round(s0*inv)),-127,127));
        char q1=char(clamp(int(round(s1*inv)),-127,127));
        char q2=char(clamp(int(round(s2*inv)),-127,127));
        char q3=char(clamp(int(round(s3*inv)),-127,127));
        dq4[i4]=char4(q0,q1,q2,q3);
    }
    for (uint i=(I4<<2u)+tid; i<I; i+=tgs){ float s=nongated_act(u[i],act); dq[i]=char(clamp(int(round(s*inv)),-127,127)); }
}
kernel void residual(device float* x[[buffer(0)]], device const float* y[[buffer(1)]], uint i[[thread_position_in_grid]]) { x[i]+=y[i]; }
// Gemma 4's per-layer output scalar on a dense layer: x *= s (decoder's forward_gemma4.go; the MoE join has its own).
kernel void layer_scale(device float* x[[buffer(0)]], device const float* s[[buffer(1)]], uint i[[thread_position_in_grid]]) { x[i]*=s[0]; }
// Gemma 4 E-model PLE branch (S1.5): g = gelu_tanh(PLEGate·x) * this layer's per-layer input, in place over P
// entries — decoder's forward_gemma4.go's geluTanh(px[i]) * pl[i], with glu_act's clamped tanh.
kernel void ple_gelu_mul(device float* g[[buffer(0)]], device const float* ple[[buffer(1)]], constant uint& n[[buffer(2)]],
    uint i[[thread_position_in_grid]]) { if (i < n) g[i] = glu_act(g[i], ACT_GELU_TANH) * ple[i]; }

