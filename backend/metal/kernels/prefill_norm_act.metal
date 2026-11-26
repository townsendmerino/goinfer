// rmsnorm_quant_f16: RMSNorm a single f16 row and fused-quantize it to int8 + f32 scale — the
// f16-input twin of the decode path's rmsnorm_quant. Prefill's LM head is the SAME int8-pinned,
// logit-critical head the decode path uses (weights int8, not int4), so the last token's
// final-normed activation must arrive as int8 for gemv_w8a8 — feeding f16/int4 assumptions into
// it is what produced NaN. addOne selects Gemma's (1+w). x already points at the target row.
kernel void rmsnorm_quant_f16(device const half* x[[buffer(0)]], device const float* w[[buffer(1)]],
    device char* aq[[buffer(2)]], device float* asc[[buffer(3)]], constant uint& H[[buffer(4)]],
    constant float& eps[[buffer(5)]], constant uint& addOne[[buffer(6)]],
    uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256]; float ss=0;
    for(uint i=tid;i<H;i+=tgs){ float v=float(x[i]); ss+=v*v; }
    red[tid]=ss; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=tgs/2u;s>0u;s>>=1u){ if(tid<s) red[tid]+=red[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup);}
    float rms=rsqrt(red[0]/float(H)+eps); threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx=0; for(uint i=tid;i<H;i+=tgs){ float g=addOne!=0u?(1.0f+w[i]):w[i]; mx=max(mx,fabs(float(x[i])*rms*g)); }
    mx = simd_max(mx);
    uint sgid = tid >> 5u, lane = tid & 31u;
    if (lane == 0) red[sgid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nsg = tgs >> 5u;
    if (tid == 0) { float v = red[0]; for (uint k=1; k<nsg; k++) v = max(v, red[k]); red[0] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sc=red[0]/127.0f; if(sc==0)sc=1; if(tid==0)asc[0]=sc; float inv=1/sc;
    for(uint i=tid;i<H;i+=tgs){ float g=addOne!=0u?(1.0f+w[i]):w[i]; aq[i]=char(clamp(int(round(float(x[i])*rms*g*inv)),-127,127)); }
}

// rmsnorm_f16: one threadgroup per row. out[m] = x[m]*rsqrt(mean(x[m]²)+eps)*w (w f32).
// addOne selects Gemma's (1+w) RMS offset vs plain w — mirrors decoder's rmsnorm.go / the decode
// path's rmsnorm_quant (kernels.go). G8: added so this kernel
// can serve BOTH a GEMV-input norm (Llama/Qwen, addOne=0) and Gemma's sandwich norm on a sublayer
// OUTPUT (addOne=1) — same math either way, only the weight convention differs.
kernel void rmsnorm_f16(device const half* x[[buffer(0)]], device const float* w[[buffer(1)]],
    device half* out[[buffer(2)]], constant uint& H[[buffer(3)]], constant float& eps[[buffer(4)]],
    constant uint& addOne[[buffer(5)]],
    uint row[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]],
    uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256];
    device const half* xr = x + row*H; device half* orow = out + row*H;
    float ss=0; for(uint i=tid;i<H;i+=tgs){ float v=float(xr[i]); ss+=v*v; }
    red[tid]=ss; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=tgs/2u;s>0u;s>>=1u){ if(tid<s) red[tid]+=red[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float rms=rsqrt(red[0]/float(H)+eps);
    for(uint i=tid;i<H;i+=tgs){ float g=addOne!=0u?(1.0f+w[i]):w[i]; orow[i]=half(float(xr[i])*rms*g); }
}

// rmsnorm_f16_f32out is rmsnorm_f16 on one row with an f32 output, the q4k lane head's input (decode's rmsnorm_f32_out, whose
// precise rsqrt it takes).
kernel void rmsnorm_f16_f32out(device const half* x[[buffer(0)]], device const float* w[[buffer(1)]],
    device float* out[[buffer(2)]], constant uint& H[[buffer(3)]], constant float& eps[[buffer(4)]],
    constant uint& addOne[[buffer(5)]], uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256];
    float ss=0; for(uint i=tid;i<H;i+=tgs){ float v=float(x[i]); ss+=v*v; }
    red[tid]=ss; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=tgs/2u;s>0u;s>>=1u){ if(tid<s) red[tid]+=red[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float rms=precise::rsqrt(red[0]/float(H)+eps);
    for(uint i=tid;i<H;i+=tgs){ float g=addOne!=0u?(1.0f+w[i]):w[i]; out[i]=float(x[i])*rms*g; }
}

// layernorm_f16 / layernorm_quant_f16: rmsnorm_f16 and
// rmsnorm_quant_f16 for a bias-free LayerNorm (Cohere / Command-R7B): out = (x - mean) * rsqrt(var + eps) * w, the math of
// the decode path's layernorm_quant with hasBias 0. Same signatures as the RMSNorm twins (addOne is accepted and unused:
// no LayerNorm family carries Gemma's 1+w), so a dispatch site only swaps the pipeline. A biased LayerNorm (GPT-2) does not
// take the pass (batchedPrefill declines it).
kernel void layernorm_f16(device const half* x[[buffer(0)]], device const float* w[[buffer(1)]],
    device half* out[[buffer(2)]], constant uint& H[[buffer(3)]], constant float& eps[[buffer(4)]],
    constant uint& addOne[[buffer(5)]],
    uint row[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]],
    uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256];
    device const half* xr = x + row*H; device half* orow = out + row*H;
    float s=0; for(uint i=tid;i<H;i+=tgs) s+=float(xr[i]);
    red[tid]=s; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint st=tgs/2u;st>0u;st>>=1u){ if(tid<st) red[tid]+=red[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float mean=red[0]/float(H); threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss=0; for(uint i=tid;i<H;i+=tgs){ float d=float(xr[i])-mean; ss+=d*d; }
    red[tid]=ss; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint st=tgs/2u;st>0u;st>>=1u){ if(tid<st) red[tid]+=red[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv=rsqrt(red[0]/float(H)+eps);
    for(uint i=tid;i<H;i+=tgs) orow[i]=half((float(xr[i])-mean)*inv*w[i]);
}
kernel void layernorm_quant_f16(device const half* x[[buffer(0)]], device const float* w[[buffer(1)]],
    device char* aq[[buffer(2)]], device float* asc[[buffer(3)]], constant uint& H[[buffer(4)]],
    constant float& eps[[buffer(5)]], constant uint& addOne[[buffer(6)]],
    uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256];
    float s=0; for(uint i=tid;i<H;i+=tgs) s+=float(x[i]);
    red[tid]=s; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint st=tgs/2u;st>0u;st>>=1u){ if(tid<st) red[tid]+=red[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float mean=red[0]/float(H); threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss=0; for(uint i=tid;i<H;i+=tgs){ float d=float(x[i])-mean; ss+=d*d; }
    red[tid]=ss; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint st=tgs/2u;st>0u;st>>=1u){ if(tid<st) red[tid]+=red[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv=rsqrt(red[0]/float(H)+eps); threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx=0; for(uint i=tid;i<H;i+=tgs) mx=max(mx,fabs((float(x[i])-mean)*inv*w[i]));
    mx = simd_max(mx);
    uint sgid = tid >> 5u, lane = tid & 31u;
    if (lane == 0) red[sgid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nsg = tgs >> 5u;
    if (tid == 0) { float v = red[0]; for (uint k=1; k<nsg; k++) v = max(v, red[k]); red[0] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sc=red[0]/127.0f; if(sc==0)sc=1; if(tid==0)asc[0]=sc; float qi=1/sc;
    for(uint i=tid;i<H;i+=tgs) aq[i]=char(clamp(int(round((float(x[i])-mean)*inv*w[i]*qi)),-127,127));
}

// residual_f16: x += y (element-wise, grid = M*H).
kernel void residual_f16(device half* x[[buffer(0)]], device const half* y[[buffer(1)]],
    uint i[[thread_position_in_grid]]) { x[i]=half(float(x[i])+float(y[i])); }
// residual_f16_scaled: x += y*inv, inv = 1/s a power of two: the add into a residual stored scaled by 1/s (S17,
// resident.prefillResidScale).
kernel void residual_f16_scaled(device half* x[[buffer(0)]], device const half* y[[buffer(1)]], constant float& inv[[buffer(2)]],
    uint i[[thread_position_in_grid]]) { x[i]=half(float(x[i])+float(y[i])*inv); }

// residual_f16_from_f32: x += y, x is f16, y is f32. The
// batched-prefill MoE row loop's expert-combine dispatches (encodeMoEExperts/
// encodeMoESharedExpert, metal/moe.go) are F32-only — reused UNCHANGED for the per-row loop by
// pointing their accumulate target at an isolated F32 scratch buffer instead of prefill's own F16
// residual, then folding that scratch into the real F16 residual row with this kernel, once.
kernel void residual_f16_from_f32(device half* x[[buffer(0)]], device const float* y[[buffer(1)]],
    uint i[[thread_position_in_grid]]) { x[i]=half(float(x[i])+y[i]); }

// zero_f32: x[i] = 0 (grid = N). The MoE row loop's F32 scratch (see residual_f16_from_f32 above)
// must start at zero before encodeMoEExperts' k-expert accumulate loop — unlike decode's own use
// of a scratch buffer (always fully OVERWRITTEN before being read), this one is accumulated into
// by potentially several dispatches (k routed experts plus a shared expert) that all assume
// their target already holds whatever the previous accumulation left, same as r.x does for decode.
kernel void zero_f32(device float* x[[buffer(0)]], uint i[[thread_position_in_grid]]) { x[i] = 0.0; }

// glu_act_f16: SiLU (act=1, decoder.ActKind's ActSiLU) or GELU-tanh (act=0, ActGeluTanh — Gemma).
// The tanh argument is CLAMPED to ±15 — same fix as kernels.go's glu_act (a massive-activation
// gate, e.g. Gemma's <bos>, overflows MSL's tanh to NaN otherwise; duplicated here rather than
// shared since this file compiles as its own separate library (prefillKernels), not allKernels.
inline float glu_act_f16(float x, uint act) {
    if (act == 1u) return x/(1.0f+exp(-x));
    float a = 0.7978845608028654f*(x+0.044715f*x*x*x);
    return 0.5f*x*(1.0f+tanh(clamp(a, -15.0f, 15.0f)));
}
// swiglu_f16: out[m][i] = glu_act_f16(gu[m][i]) * gu[m][I+i], gu is [M×2I]. grid = M*I.
kernel void swiglu_f16(device const half* gu[[buffer(0)]], device half* out[[buffer(1)]],
    constant uint& I[[buffer(2)]], constant uint& act[[buffer(3)]], uint gid[[thread_position_in_grid]]) {
    uint m = gid / I, i = gid % I;
    float g=float(gu[m*2u*I + i]), u=float(gu[m*2u*I + I + i]);
    out[gid]=half(glu_act_f16(g, act)*u);
}

// ple_gelu_mul_f16 (S9 step 2): the batched PLE gate, decode's ple_gelu_mul per row: g[m][j] = gelu_tanh(g[m][j]) times
// row m's own per-layer input ple[m*LP + off + j] (off = l*P). grid = M*P.
kernel void ple_gelu_mul_f16(device half* g[[buffer(0)]], device const float* ple[[buffer(1)]], constant uint& P[[buffer(2)]],
    constant uint& LP[[buffer(3)]], constant uint& off[[buffer(4)]], uint gid[[thread_position_in_grid]]) {
    uint m = gid / P, j = gid % P;
    g[gid] = half(glu_act_f16(float(g[gid]), 0u) * ple[m*LP + off + j]);
}

// layer_scale_f16 (S9 step 2): x *= s[0] over the f16 residual, Gemma 4's dense layer scalar. grid = M*H.
kernel void layer_scale_f16(device half* x[[buffer(0)]], device const float* s[[buffer(1)]], uint i[[thread_position_in_grid]]) {
    x[i] = half(float(x[i]) * s[0]);
}

