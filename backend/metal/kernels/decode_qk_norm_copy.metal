// qk_norm: per-head RMSNorm on Q and K in the fused qkv buffer, applied BEFORE RoPE (Qwen3,
// Gemma3). ONE threadgroup per head: head < nH is a Q head (weight qn), else a K head (weight
// kn, index head-nH). Norm over the head dim (hd, a power of 2). addOne selects Gemma's (1+w)
// vs plain w. Matches decoder's rmsnorm.go rmsNorm(q/k, QNorm/KNorm, ..., RMSAddOne).
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

// copy_u32 is copy_f32 for integer words: an expert id read as a float is a denormal, which a float copy may flush to
// zero. Tests only (D-G01's routing capture, resident.moeCap).
kernel void copy_u32(device const uint* src [[buffer(0)]], device uint* dst [[buffer(1)]],
    constant uint& N [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < N) dst[i] = src[i];
}
