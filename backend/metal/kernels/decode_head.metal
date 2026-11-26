// The q4k lane's LM head: gemv_w8a8_coal's and gemv_w8a8_amax's buffer order and launch shapes, so every head dispatch
// site runs unchanged when buildResident swaps the pipelines, but slot 0 holds the f32 final-norm output (the trunk's
// rmsnorm_f32_out writes it into r.aq, sized 4·H bytes on this lane) and slot 1 is unread.
kernel void gemv_w8f32_head(device const float* ax[[buffer(0)]], device const float* unused[[buffer(1)]],
    device const char* bq[[buffer(2)]], device const float* bsc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], uint gid[[threadgroup_position_in_grid]], uint lid[[thread_index_in_threadgroup]]) {
    device const char4* wr = (device const char4*)(bq + (ulong)gid*(ulong)K);
    device const float4* a4 = (device const float4*)ax;
    float acc = 0.0f;
    for (uint g=lid; g<(K>>2u); g+=32u) acc += dot(float4(wr[g]), a4[g]);
    acc = simd_sum(acc);
    if (lid==0) out[gid] = acc*bsc[gid];
}
kernel void gemv_w8f32_amax(device const float* ax[[buffer(0)]], device const float* unused[[buffer(1)]],
    device const char* bq[[buffer(2)]], device const float* bsc[[buffer(3)]], device AmaxPart* part[[buffer(4)]],
    constant uint& K[[buffer(5)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    uint row = tgid*(tgs>>5u) + sgid;
    device const char4* wr = (device const char4*)(bq + (ulong)row*(ulong)K);
    device const float4* a4 = (device const float4*)ax;
    float acc = 0.0f;
    for (uint g=lane; g<(K>>2u); g+=32u) acc += dot(float4(wr[g]), a4[g]);
    acc = simd_sum(acc);
    threadgroup float tv[8]; threadgroup uint ti[8];
    if (lane==0) { tv[sgid] = acc*bsc[row]; ti[sgid] = row; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid==0) {
        // NaN is skipped, as argmax_rows_part skips it: a NaN leading the tile would otherwise win it, hiding the tile's maximum.
        // A tile of NaN alone writes (-inf, 0xFFFFFFFF), and argmax_finish then writes 0xFFFFFFFF only for a row that is NaN
        // throughout, whose host argmax is 0 (headID).
        uint nsg = tgs>>5u; float bv = -INFINITY; uint bi = 0xFFFFFFFFu;
        for (uint s=0u; s<nsg; s++) if (!isnan(tv[s]) && (tv[s]>bv || (tv[s]==bv && ti[s]<bi))) { bv=tv[s]; bi=ti[s]; }
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
        // NaN is skipped, as argmax_rows_part skips it: a NaN leading the tile would otherwise win it, hiding the tile's maximum.
        // A tile of NaN alone writes (-inf, 0xFFFFFFFF), and argmax_finish then writes 0xFFFFFFFF only for a row that is NaN
        // throughout, whose host argmax is 0 (headID).
        uint nsg = tgs>>5u; float bv = -INFINITY; uint bi = 0xFFFFFFFFu;
        for (uint s=0u; s<nsg; s++) if (!isnan(tv[s]) && (tv[s]>bv || (tv[s]==bv && ti[s]<bi))) { bv=tv[s]; bi=ti[s]; }
        part[tgid].v = bv; part[tgid].i = bi;
    }
}
// argmax_rows_part (E-P08, the batched step's greedy rows): one f32 logits row's partial argmaxes, threadgroup g over
// elements g, g+P*256, ... in a grid-stride loop, reduced as argmax_finish reduces (the greater value, or the equal one
// at the lower index), so argmax_finish over the P partials gives the row's first maximum. NaN is skipped: the host's
// argmaxF32 never picks a NaN unless the row starts with one, and the caller checks element 0 for that.
kernel void argmax_rows_part(device const float* row[[buffer(0)]], device AmaxPart* part[[buffer(1)]],
    constant uint& V[[buffer(2)]], constant uint& P[[buffer(3)]], uint g[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    float v=-INFINITY; uint idx=0xFFFFFFFFu;
    for (uint i=g*256u+tid; i<V; i+=P*256u) { float x=row[i]; if (!isnan(x) && (x>v||(x==v&&i<idx))) { v=x; idx=i; } }
    for (uint off=16u; off>0u; off>>=1u) { float ov=simd_shuffle_down(v,off); uint oi=simd_shuffle_down(idx,off); if(ov>v||(ov==v&&oi<idx)){v=ov;idx=oi;} }
    threadgroup float tv[8]; threadgroup uint ti[8];
    if (lane==0u){tv[sgid]=v;ti[sgid]=idx;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid==0u){ float bv=tv[0];uint bi=ti[0]; for(uint s=1u;s<8u;s++) if(tv[s]>bv||(tv[s]==bv&&ti[s]<bi)){bv=tv[s];bi=ti[s];} part[g].v=bv; part[g].i=bi; }
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
// embed_gather_i8 (C-B01, the greedy chain, greedy_chain.go): x[i] = float(q[tok*H + i]) * scale[tok], the host's
// int8 Embed.Row (linalg.DequantizeRowInt8: float32(q)*scale, one rounded multiply per element) for the token id the
// previous command buffer's argmax_finish (or gumbel_stage2) wrote, so the next token's embedding never leaves the GPU.
// Only for an int8 table with no embedding multiplier (greedyChainWhyNot), so there is nothing else to apply.
kernel void embed_gather_i8(device const char* q[[buffer(0)]], device const float* sc[[buffer(1)]],
    device const uint* tok[[buffer(2)]], device float* x[[buffer(3)]], constant uint& H[[buffer(4)]],
    constant uint& V[[buffer(5)]], uint i[[thread_position_in_grid]]) {
    if (i >= H) return;
    const uint t = tok[0];
    if (t >= V) { x[i] = 0.0f; return; } // a sampled draw's -1 (no finite logit): the chain re-commits this buffer
    x[i] = float(q[ulong(t)*ulong(H) + ulong(i)]) * sc[t];
}
