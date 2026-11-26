// rope_f16: NeoX half-split, per-row position. Rotates a [M × total] region with row stride
// 'stride' starting at byte-free offset 'base0' (element offset into each row). positions[m].
kernel void rope_f16(device half* x[[buffer(0)]], device const float* invf[[buffer(1)]],
    constant uint& hd[[buffer(2)]], device const uint* positions[[buffer(3)]],
    constant uint& total[[buffer(4)]], constant uint& stride[[buffer(5)]],
    constant uint& base0[[buffer(6)]], constant uint& rhalf[[buffer(7)]],
    constant float& scale[[buffer(8)]],
    uint gid[[thread_position_in_grid]]) {
    uint pairsPerRow = (total/hd) * rhalf;   // nHeads*half; half=rotaryDim/2 (<hd/2 = partial rotary)
    uint m = gid / pairsPerRow, p = gid % pairsPerRow;
    uint head = p/rhalf, dd = p%rhalf;
    uint base = m*stride + base0 + head*hd;
    float th = float(positions[m]) * invf[dd]; float c=cos(th)*scale, s=sin(th)*scale;
    float x0=float(x[base+dd]), x1=float(x[base+rhalf+dd]);
    x[base+dd]=half(x0*c-x1*s); x[base+rhalf+dd]=half(x0*s+x1*c);
}

// rope_mrope_f16: rope_f16 for Qwen's multimodal RoPE. Each row has
// three positions (temporal, height, width) in pos3[m*3..], and rotary pair dd turns by pos3[m*3 + axis[dd]]. axis is the
// host's own table from decoder's mropeComponent / mropeComponentInterleaved (contiguous sections for Qwen2.5-VL, strided
// for Qwen3.5+/Qwen3-VL), built once per model, so the kernel carries no layout formula of its own. A text row has three
// equal positions and reduces to rope_f16 at that position exactly.
kernel void rope_mrope_f16(device half* x[[buffer(0)]], device const float* invf[[buffer(1)]],
    constant uint& hd[[buffer(2)]], device const uint* pos3[[buffer(3)]],
    constant uint& total[[buffer(4)]], constant uint& stride[[buffer(5)]],
    constant uint& base0[[buffer(6)]], constant uint& rhalf[[buffer(7)]],
    constant float& scale[[buffer(8)]], device const uint* axis[[buffer(9)]],
    uint gid[[thread_position_in_grid]]) {
    uint pairsPerRow = (total/hd) * rhalf;
    uint m = gid / pairsPerRow, p = gid % pairsPerRow;
    uint head = p/rhalf, dd = p%rhalf;
    uint base = m*stride + base0 + head*hd;
    float th = float(pos3[m*3u + axis[dd]]) * invf[dd]; float c=cos(th)*scale, s=sin(th)*scale;
    float x0=float(x[base+dd]), x1=float(x[base+rhalf+dd]);
    x[base+dd]=half(x0*c-x1*s); x[base+rhalf+dd]=half(x0*s+x1*c);
}

// rope_f16_pw / rope_mrope_f16_pw: rope_f16 / rope_mrope_f16 for GPT-J
// PAIRWISE rotation, pair dd = dims (2dd, 2dd+1), as decoder's rope.go's applyRoPEInterleaved and applyMRoPEPairwise (GLM-OCR's
// contiguous m-RoPE sections come in through the same host axis table). Same arguments and grid; bound for a model whose
// decoder.Model.BackendView().PairwiseRoPE() is true, and nowhere else.
kernel void rope_f16_pw(device half* x[[buffer(0)]], device const float* invf[[buffer(1)]],
    constant uint& hd[[buffer(2)]], device const uint* positions[[buffer(3)]],
    constant uint& total[[buffer(4)]], constant uint& stride[[buffer(5)]],
    constant uint& base0[[buffer(6)]], constant uint& rhalf[[buffer(7)]],
    constant float& scale[[buffer(8)]],
    uint gid[[thread_position_in_grid]]) {
    uint pairsPerRow = (total/hd) * rhalf;
    uint m = gid / pairsPerRow, p = gid % pairsPerRow;
    uint head = p/rhalf, dd = p%rhalf;
    uint base = m*stride + base0 + head*hd + 2u*dd;
    float th = float(positions[m]) * invf[dd]; float c=cos(th)*scale, s=sin(th)*scale;
    float x0=float(x[base]), x1=float(x[base+1u]);
    x[base]=half(x0*c-x1*s); x[base+1u]=half(x0*s+x1*c);
}
kernel void rope_mrope_f16_pw(device half* x[[buffer(0)]], device const float* invf[[buffer(1)]],
    constant uint& hd[[buffer(2)]], device const uint* pos3[[buffer(3)]],
    constant uint& total[[buffer(4)]], constant uint& stride[[buffer(5)]],
    constant uint& base0[[buffer(6)]], constant uint& rhalf[[buffer(7)]],
    constant float& scale[[buffer(8)]], device const uint* axis[[buffer(9)]],
    uint gid[[thread_position_in_grid]]) {
    uint pairsPerRow = (total/hd) * rhalf;
    uint m = gid / pairsPerRow, p = gid % pairsPerRow;
    uint head = p/rhalf, dd = p%rhalf;
    uint base = m*stride + base0 + head*hd + 2u*dd;
    float th = float(pos3[m*3u + axis[dd]]) * invf[dd]; float c=cos(th)*scale, s=sin(th)*scale;
    float x0=float(x[base]), x1=float(x[base+1u]);
    x[base]=half(x0*c-x1*s); x[base+1u]=half(x0*s+x1*c);
}

// qk_norm_f16: per-head Q/K RMSNorm (Qwen3) on the f16 fused qkv[M×stride], before RoPE. One
// threadgroup per (row m, head): head<nH is Q (weight qn), else K (kn, head-nH). Norm over hd.
kernel void qk_norm_f16(device half* qkv[[buffer(0)]], device const float* qn[[buffer(1)]],
    device const float* kn[[buffer(2)]], constant uint& nH[[buffer(3)]], constant uint& nKV[[buffer(4)]],
    constant uint& hd[[buffer(5)]], constant uint& nHhd[[buffer(6)]], constant uint& stride[[buffer(7)]],
    constant float& eps[[buffer(8)]], constant uint& addOne[[buffer(9)]],
    uint gid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]],
    uint tgs[[threads_per_threadgroup]]) {
    uint nHeads = nH + nKV;
    uint m = gid / nHeads, head = gid % nHeads;
    threadgroup float red[128];
    bool isQ = head < nH;
    device const float* w = isQ ? qn : kn;
    uint base = m*stride + (isQ ? head*hd : nHhd + (head-nH)*hd);
    device half* x = qkv + base;
    float ss=0; for(uint i=tid;i<hd;i+=tgs){ float v=float(x[i]); ss+=v*v; }
    red[tid]=ss; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=tgs/2u;s>0u;s>>=1u){ if(tid<s) red[tid]+=red[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float rms=rsqrt(red[0]/float(hd)+eps);
    for(uint i=tid;i<hd;i+=tgs){ float wt = addOne!=0u ? (1.0f+w[i]) : w[i]; x[i]=half(float(x[i])*rms*wt); }
}

