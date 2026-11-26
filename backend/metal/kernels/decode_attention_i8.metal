// attention_i8: reads int8 KV cache with per-(position,KV-head) f32 scales (ks, vs).
// Halves KV memory bandwidth and cache memory footprint vs f16 KV.
kernel void attention_i8(device const float* q[[buffer(0)]], device const char* kc[[buffer(1)]],
    device const char* vc[[buffer(2)]], device const float* ks[[buffer(3)]],
    device const float* vs[[buffer(4)]], device float* out[[buffer(5)]],
    constant uint& nH[[buffer(6)]], constant uint& nKV[[buffer(7)]], constant uint& hd[[buffer(8)]],
    constant uint& nKeys[[buffer(9)]], constant float& scale[[buffer(10)]],
    constant uint& window[[buffer(11)]], device const float* sinks[[buffer(12)]],
    constant uint& hasSink[[buffer(13)]], device const uint* kt[[buffer(14)]],
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
            uint kRow = kv_row(kt,s);
            float a=0; device const char* k=kb+kRow*kvDim; uint d=0;
            float k_scale = ks[kRow*nKV + kvh];
            if ((hd&3u)==0u) for (; d<hd; d+=4u){ char4 k4=*((device const char4*)(k+d)); a+=qr[d]*float(k4.x); a+=qr[d+1u]*float(k4.y); a+=qr[d+2u]*float(k4.z); a+=qr[d+3u]*float(k4.w); }
            for (; d<hd; d++) a += qr[d]*float(k[d]);
            sc[s - winStart]=attn_score((a * k_scale)*scale);
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
            uint vBlock = 0xffffffffu, vBase = 0u; // the table entry of the block in hand
            for (; s<nMain; s+=8u) {
                uint sc_idx = s - winStart;
                float vs0, vs1, vs2, vs3, vs4, vs5, vs6, vs7;
                float v0, v1, v2, v3, v4, v5, v6, v7;
                if ((s & 63u) <= 56u) { // the tile is inside one block
                    if ((s >> 6u) != vBlock) { vBlock = s >> 6u; vBase = kt[vBlock]*64u; }
                    uint r = (vBase + (s & 63u))*kvDim + d;
                    uint rs = (vBase + (s & 63u))*nKV + kvh;
                    vs0=vs[rs];
                    vs1=vs[rs+1u*nKV];
                    vs2=vs[rs+2u*nKV];
                    vs3=vs[rs+3u*nKV];
                    vs4=vs[rs+4u*nKV];
                    vs5=vs[rs+5u*nKV];
                    vs6=vs[rs+6u*nKV];
                    vs7=vs[rs+7u*nKV];
                    v0=float(vb[r])*vs0;
                    v1=float(vb[r+1u*kvDim])*vs1;
                    v2=float(vb[r+2u*kvDim])*vs2;
                    v3=float(vb[r+3u*kvDim])*vs3;
                    v4=float(vb[r+4u*kvDim])*vs4;
                    v5=float(vb[r+5u*kvDim])*vs5;
                    v6=float(vb[r+6u*kvDim])*vs6;
                    v7=float(vb[r+7u*kvDim])*vs7;
                } else {
                    vs0=vs[kv_row(kt,s+0u)*nKV+kvh];
                    vs1=vs[kv_row(kt,s+1u)*nKV+kvh];
                    vs2=vs[kv_row(kt,s+2u)*nKV+kvh];
                    vs3=vs[kv_row(kt,s+3u)*nKV+kvh];
                    vs4=vs[kv_row(kt,s+4u)*nKV+kvh];
                    vs5=vs[kv_row(kt,s+5u)*nKV+kvh];
                    vs6=vs[kv_row(kt,s+6u)*nKV+kvh];
                    vs7=vs[kv_row(kt,s+7u)*nKV+kvh];
                    v0=float(vb[kv_row(kt,s+0u)*kvDim+d])*vs0;
                    v1=float(vb[kv_row(kt,s+1u)*kvDim+d])*vs1;
                    v2=float(vb[kv_row(kt,s+2u)*kvDim+d])*vs2;
                    v3=float(vb[kv_row(kt,s+3u)*kvDim+d])*vs3;
                    v4=float(vb[kv_row(kt,s+4u)*kvDim+d])*vs4;
                    v5=float(vb[kv_row(kt,s+5u)*kvDim+d])*vs5;
                    v6=float(vb[kv_row(kt,s+6u)*kvDim+d])*vs6;
                    v7=float(vb[kv_row(kt,s+7u)*kvDim+d])*vs7;
                }
                a += sc[sc_idx+0u]*v0; a += sc[sc_idx+1u]*v1; a += sc[sc_idx+2u]*v2; a += sc[sc_idx+3u]*v3;
                a += sc[sc_idx+4u]*v4; a += sc[sc_idx+5u]*v5; a += sc[sc_idx+6u]*v6; a += sc[sc_idx+7u]*v7;
            }
            for (; s<nKeys; s++) a += sc[s - winStart]*(float(vb[kv_row(kt,s)*kvDim+d])*vs[kv_row(kt,s)*nKV+kvh]);
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
            float a = 0; device const char* k = kb + kv_row(kt,s)*kvDim; uint d = 0;
            float k_scale = ks[kv_row(kt,s)*nKV + kvh];
            if ((hd&3u) == 0u) for (; d < hd; d += 4u) {
                char4 k4 = *((device const char4*)(k+d));
                a += qr[d]*float(k4.x);
                a += qr[d+1u]*float(k4.y);
                a += qr[d+2u]*float(k4.z);
                a += qr[d+3u]*float(k4.w);
            }
            for (; d < hd; d++) a += qr[d]*float(k[d]);
            sc[s - tileBase] = attn_score((a * k_scale) * scale);
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
            uint vBlock = 0xffffffffu, vBase = 0u; // the table entry of the block in hand
            for (; s < nMain; s += 8u) {
                uint sc_idx = s - tileBase;
                float vs0, vs1, vs2, vs3, vs4, vs5, vs6, vs7;
                float v0, v1, v2, v3, v4, v5, v6, v7;
                if ((s & 63u) <= 56u) { // the tile is inside one block
                    if ((s >> 6u) != vBlock) { vBlock = s >> 6u; vBase = kt[vBlock]*64u; }
                    uint r = (vBase + (s & 63u))*kvDim + d;
                    uint rs = (vBase + (s & 63u))*nKV + kvh;
                    vs0=vs[rs];
                    vs1=vs[rs+1u*nKV];
                    vs2=vs[rs+2u*nKV];
                    vs3=vs[rs+3u*nKV];
                    vs4=vs[rs+4u*nKV];
                    vs5=vs[rs+5u*nKV];
                    vs6=vs[rs+6u*nKV];
                    vs7=vs[rs+7u*nKV];
                    v0=float(vb[r])*vs0;
                    v1=float(vb[r+1u*kvDim])*vs1;
                    v2=float(vb[r+2u*kvDim])*vs2;
                    v3=float(vb[r+3u*kvDim])*vs3;
                    v4=float(vb[r+4u*kvDim])*vs4;
                    v5=float(vb[r+5u*kvDim])*vs5;
                    v6=float(vb[r+6u*kvDim])*vs6;
                    v7=float(vb[r+7u*kvDim])*vs7;
                } else {
                    vs0=vs[kv_row(kt,s+0u)*nKV+kvh];
                    vs1=vs[kv_row(kt,s+1u)*nKV+kvh];
                    vs2=vs[kv_row(kt,s+2u)*nKV+kvh];
                    vs3=vs[kv_row(kt,s+3u)*nKV+kvh];
                    vs4=vs[kv_row(kt,s+4u)*nKV+kvh];
                    vs5=vs[kv_row(kt,s+5u)*nKV+kvh];
                    vs6=vs[kv_row(kt,s+6u)*nKV+kvh];
                    vs7=vs[kv_row(kt,s+7u)*nKV+kvh];
                    v0=float(vb[kv_row(kt,s+0u)*kvDim+d])*vs0;
                    v1=float(vb[kv_row(kt,s+1u)*kvDim+d])*vs1;
                    v2=float(vb[kv_row(kt,s+2u)*kvDim+d])*vs2;
                    v3=float(vb[kv_row(kt,s+3u)*kvDim+d])*vs3;
                    v4=float(vb[kv_row(kt,s+4u)*kvDim+d])*vs4;
                    v5=float(vb[kv_row(kt,s+5u)*kvDim+d])*vs5;
                    v6=float(vb[kv_row(kt,s+6u)*kvDim+d])*vs6;
                    v7=float(vb[kv_row(kt,s+7u)*kvDim+d])*vs7;
                }
                a += sc[sc_idx+0u]*v0; a += sc[sc_idx+1u]*v1; a += sc[sc_idx+2u]*v2; a += sc[sc_idx+3u]*v3;
                a += sc[sc_idx+4u]*v4; a += sc[sc_idx+5u]*v5; a += sc[sc_idx+6u]*v6; a += sc[sc_idx+7u]*v7;
            }
            for (; s < tileEnd; s++) {
                a += sc[s - tileBase]*(float(vb[kv_row(kt,s)*kvDim+d])*vs[kv_row(kt,s)*nKV+kvh]);
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
