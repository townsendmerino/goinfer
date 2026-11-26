// ---- attention_fa_blk begin (R17; see TestAttnFABlkIsTheGradedKernel) ----
// R17 (docs/completed/red-october.md; docs/measurements/metal-decode-attn-r17-2026-09-25.md): attention_fa's first
// pass in the shape of llama.cpp's flash_attn_ext_vec — keys in blocks of 32 (32 coalesced K-row loads, per head 32
// independent simd_sums landing one score per lane), ONE online-softmax step per block per head, V weighted by
// simd_shuffle(p, j), G a template parameter so every per-head loop has a constant bound. Same signature, grid and
// partial layout as attention_fa; attention_fa_combine merges it unchanged. Graded 2026-09-25 at G = 6 and 7 with a
// fixed split count of 16 (attnFABlkSplit): 3.50x in-sequence at 3900 keys on the 1.5B, ~3x closer to float64 than
// the shipped per-query-head kernel, fidelity gate PASSES on set B. Instantiated only for the graded group sizes.
#define ATTN_FA_BLK_UNROLL _Pragma("clang loop unroll(full)")

template <uint G>
kernel void attention_fa_blk(
    device const float* q[[buffer(0)]], device const half* kc[[buffer(1)]],
    device const half* vc[[buffer(2)]], device float* partial[[buffer(3)]],
    constant uint& nKV[[buffer(4)]], constant uint& Gu[[buffer(5)]],
    constant uint& nKeys[[buffer(6)]], constant float& scale[[buffer(7)]],
    constant uint& window[[buffer(8)]], constant uint& nSplit[[buffer(9)]], device const uint* kt[[buffer(10)]],
    threadgroup float* shm[[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]]) {
    (void)Gu;
    const uint hd = 128u;
    const uint kvDim = nKV * hd;
    const uint kvh = tgid / nSplit, split = tgid % nSplit;
    const uint winStart = (window > 0u && nKeys > window) ? nKeys - window : 0u;
    const uint nWin = nKeys - winStart;
    const uint chunkLen = (nWin + nSplit - 1u) / nSplit;
    const uint chunkStart = winStart + split * chunkLen;
    const uint chunkEnd = min(chunkStart + chunkLen, nKeys);

    device const float* qr = q + kvh * G * hd;
    device const half*  kb = kc + kvh * hd + lane*4u;
    device const half*  vb = vc + kvh * hd + lane*4u;

    float4 qs[G];
    float m[G], l[G];
    float4 acc[G];
    ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) {
        qs[g] = *((device const float4*)(qr + g*hd + lane*4u)) * scale;
        m[g] = -INFINITY; l[g] = 0.0f; acc[g] = float4(0.0f);
    }

    // simdgroup sgid takes blocks sgid, sgid+4, sgid+8, ... of 32 consecutive keys
    for (uint b0 = chunkStart + sgid*32u; b0 < chunkEnd; b0 += 128u) {
        const uint nb = min(32u, chunkEnd - b0);
        const uint edge = 64u - (b0 & 63u), base0 = kt[b0 >> 6u]*64u + (b0 & 63u); // a run crosses one block edge at most
        const uint base1 = (edge < nb) ? kt[(b0 + edge) >> 6u]*64u - edge : base0;
        float s[G];
        ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) s[g] = -INFINITY;
        if (nb == 32u) {
            ATTN_FA_BLK_UNROLL for (uint j = 0; j < 32u; j++) {
                float4 k4 = float4(*((device const half4*)(kb + (j < edge ? base0 + j : base1 + j)*kvDim)));
                ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) {
                    float d = attn_score(simd_sum(dot(qs[g], k4)));
                    s[g] = (lane == j) ? d : s[g];
                }
            }
        } else {
            for (uint j = 0; j < nb; j++) {
                float4 k4 = float4(*((device const half4*)(kb + (j < edge ? base0 + j : base1 + j)*kvDim)));
                ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) {
                    float d = attn_score(simd_sum(dot(qs[g], k4)));
                    s[g] = (lane == j) ? d : s[g];
                }
            }
        }
        float p[G];
        ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) {
            float mb = simd_max(s[g]);
            float mn = max(m[g], mb);
            float alpha = (m[g] == -INFINITY) ? 0.0f : exp(m[g] - mn);
            p[g] = (s[g] == -INFINITY) ? 0.0f : exp(s[g] - mn);
            l[g] = l[g]*alpha + simd_sum(p[g]);
            acc[g] *= alpha;
            m[g] = mn;
        }
        if (nb == 32u) {
            ATTN_FA_BLK_UNROLL for (uint j = 0; j < 32u; j++) {
                float4 v4 = float4(*((device const half4*)(vb + (j < edge ? base0 + j : base1 + j)*kvDim)));
                ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) acc[g] += simd_shuffle(p[g], j) * v4;
            }
        } else {
            for (uint j = 0; j < nb; j++) {
                float4 v4 = float4(*((device const half4*)(vb + (j < edge ? base0 + j : base1 + j)*kvDim)));
                ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) acc[g] += simd_shuffle(p[g], j) * v4;
            }
        }
    }

    // merge the 4 simdgroups and write the partial — attention_fa's exact layout, so attention_fa_combine is unchanged
    const uint stride = 6u*G;
    threadgroup float* row = shm + tid*stride;
    ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) {
        row[g] = m[g]; row[G+g] = l[g];
        row[2u*G + g*4u+0u] = acc[g].x; row[2u*G + g*4u+1u] = acc[g].y;
        row[2u*G + g*4u+2u] = acc[g].z; row[2u*G + g*4u+3u] = acc[g].w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        float fm[G], fl[G]; float4 fa[G];
        ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) { fm[g] = -INFINITY; fl[g] = 0.0f; fa[g] = float4(0.0f); }
        for (uint sg = 0; sg < 4u; sg++) {
            threadgroup float* r = shm + (sg*32u + lane)*stride;
            ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) {
                float m2 = r[g], l2 = r[G+g];
                float mn = max(fm[g], m2);
                float a1 = (fm[g] == -INFINITY) ? 0.0f : exp(fm[g] - mn);
                float a2 = (m2 == -INFINITY) ? 0.0f : exp(m2 - mn);
                fl[g] = fl[g]*a1 + l2*a2;
                fa[g] = fa[g]*a1 + float4(r[2u*G+g*4u+0u], r[2u*G+g*4u+1u], r[2u*G+g*4u+2u], r[2u*G+g*4u+3u])*a2;
                fm[g] = mn;
            }
        }
        const uint pStride = G * (hd + 2u);
        device float* pout = partial + (kvh*nSplit + split) * pStride;
        ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) {
            device float* og = pout + g*(hd+2u);
            og[0] = fm[g]; og[1] = fl[g];
            og[2u+lane*4u+0u] = fa[g].x; og[2u+lane*4u+1u] = fa[g].y;
            og[2u+lane*4u+2u] = fa[g].z; og[2u+lane*4u+3u] = fa[g].w;
        }
    }
}

typedef decltype(attention_fa_blk<6>) attention_fa_blk_t;
template [[host_name("attention_fa_blk_g6")]] kernel attention_fa_blk_t attention_fa_blk<6>;
template [[host_name("attention_fa_blk_g7")]] kernel attention_fa_blk_t attention_fa_blk<7>;
// ---- attention_fa_blk end ----
// B-P02: the same kernel at the other dense group sizes. Per head the arithmetic is G-independent (every per-head
// statement touches head g's state alone; G is a loop bound), so each agrees with g7 head for head, bit for bit
// (TestAttnFABlk_anyGMatchesG7); their speed is graded per model before attnFABlkAnyG turns them on.
template [[host_name("attention_fa_blk_g2")]] kernel attention_fa_blk_t attention_fa_blk<2>;
template [[host_name("attention_fa_blk_g3")]] kernel attention_fa_blk_t attention_fa_blk<3>;
template [[host_name("attention_fa_blk_g4")]] kernel attention_fa_blk_t attention_fa_blk<4>;
template [[host_name("attention_fa_blk_g5")]] kernel attention_fa_blk_t attention_fa_blk<5>;
template [[host_name("attention_fa_blk_g8")]] kernel attention_fa_blk_t attention_fa_blk<8>;

// attention_fa_blk64 (B-P01, docs/audit-metal-2026-09-30.md): attention_fa_blk at head dim 64, the 0.5B's (Qwen2.5-0.5B,
// G = 7). The block kernel's shape exactly, with 2 dims per lane where hd = 128 has 4: half2 K/V loads (128 B per key
// row per simdgroup), float2 q and accumulators, and a 4*G-float threadgroup row per lane in place of 6*G. Same
// signature, grid, split rule (attnFABlkSplit) and partial layout G*(hd+2), so attention_fa_combine merges it
// unchanged. Not bit-identical to the per-query-head kernel it replaces (the block softmax reassociates, as at hd = 128);
// fidelity-gated. Instantiated only for the group size graded.
template <uint G>
kernel void attention_fa_blk64(
    device const float* q[[buffer(0)]], device const half* kc[[buffer(1)]],
    device const half* vc[[buffer(2)]], device float* partial[[buffer(3)]],
    constant uint& nKV[[buffer(4)]], constant uint& Gu[[buffer(5)]],
    constant uint& nKeys[[buffer(6)]], constant float& scale[[buffer(7)]],
    constant uint& window[[buffer(8)]], constant uint& nSplit[[buffer(9)]], device const uint* kt[[buffer(10)]],
    threadgroup float* shm[[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]]) {
    (void)Gu;
    const uint hd = 64u;
    const uint kvDim = nKV * hd;
    const uint kvh = tgid / nSplit, split = tgid % nSplit;
    const uint winStart = (window > 0u && nKeys > window) ? nKeys - window : 0u;
    const uint nWin = nKeys - winStart;
    const uint chunkLen = (nWin + nSplit - 1u) / nSplit;
    const uint chunkStart = winStart + split * chunkLen;
    const uint chunkEnd = min(chunkStart + chunkLen, nKeys);

    device const float* qr = q + kvh * G * hd;
    device const half*  kb = kc + kvh * hd + lane*2u;
    device const half*  vb = vc + kvh * hd + lane*2u;

    float2 qs[G];
    float m[G], l[G];
    float2 acc[G];
    ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) {
        qs[g] = *((device const float2*)(qr + g*hd + lane*2u)) * scale;
        m[g] = -INFINITY; l[g] = 0.0f; acc[g] = float2(0.0f);
    }

    for (uint b0 = chunkStart + sgid*32u; b0 < chunkEnd; b0 += 128u) {
        const uint nb = min(32u, chunkEnd - b0);
        const uint edge = 64u - (b0 & 63u), base0 = kt[b0 >> 6u]*64u + (b0 & 63u); // a run crosses one block edge at most
        const uint base1 = (edge < nb) ? kt[(b0 + edge) >> 6u]*64u - edge : base0;
        float s[G];
        ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) s[g] = -INFINITY;
        if (nb == 32u) {
            ATTN_FA_BLK_UNROLL for (uint j = 0; j < 32u; j++) {
                float2 k2 = float2(*((device const half2*)(kb + (j < edge ? base0 + j : base1 + j)*kvDim)));
                ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) {
                    float d = attn_score(simd_sum(dot(qs[g], k2)));
                    s[g] = (lane == j) ? d : s[g];
                }
            }
        } else {
            for (uint j = 0; j < nb; j++) {
                float2 k2 = float2(*((device const half2*)(kb + (j < edge ? base0 + j : base1 + j)*kvDim)));
                ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) {
                    float d = attn_score(simd_sum(dot(qs[g], k2)));
                    s[g] = (lane == j) ? d : s[g];
                }
            }
        }
        float p[G];
        ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) {
            float mb = simd_max(s[g]);
            float mn = max(m[g], mb);
            float alpha = (m[g] == -INFINITY) ? 0.0f : exp(m[g] - mn);
            p[g] = (s[g] == -INFINITY) ? 0.0f : exp(s[g] - mn);
            l[g] = l[g]*alpha + simd_sum(p[g]);
            acc[g] *= alpha;
            m[g] = mn;
        }
        if (nb == 32u) {
            ATTN_FA_BLK_UNROLL for (uint j = 0; j < 32u; j++) {
                float2 v2 = float2(*((device const half2*)(vb + (j < edge ? base0 + j : base1 + j)*kvDim)));
                ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) acc[g] += simd_shuffle(p[g], j) * v2;
            }
        } else {
            for (uint j = 0; j < nb; j++) {
                float2 v2 = float2(*((device const half2*)(vb + (j < edge ? base0 + j : base1 + j)*kvDim)));
                ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) acc[g] += simd_shuffle(p[g], j) * v2;
            }
        }
    }

    const uint stride = 4u*G;
    threadgroup float* row = shm + tid*stride;
    ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) {
        row[g] = m[g]; row[G+g] = l[g];
        row[2u*G + g*2u+0u] = acc[g].x; row[2u*G + g*2u+1u] = acc[g].y;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        float fm[G], fl[G]; float2 fa[G];
        ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) { fm[g] = -INFINITY; fl[g] = 0.0f; fa[g] = float2(0.0f); }
        for (uint sg = 0; sg < 4u; sg++) {
            threadgroup float* r = shm + (sg*32u + lane)*stride;
            ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) {
                float m2 = r[g], l2 = r[G+g];
                float mn = max(fm[g], m2);
                float a1 = (fm[g] == -INFINITY) ? 0.0f : exp(fm[g] - mn);
                float a2 = (m2 == -INFINITY) ? 0.0f : exp(m2 - mn);
                fl[g] = fl[g]*a1 + l2*a2;
                fa[g] = fa[g]*a1 + float2(r[2u*G+g*2u+0u], r[2u*G+g*2u+1u])*a2;
                fm[g] = mn;
            }
        }
        const uint pStride = G * (hd + 2u);
        device float* pout = partial + (kvh*nSplit + split) * pStride;
        ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) {
            device float* og = pout + g*(hd+2u);
            og[0] = fm[g]; og[1] = fl[g];
            og[2u+lane*2u+0u] = fa[g].x; og[2u+lane*2u+1u] = fa[g].y;
        }
    }
}

typedef decltype(attention_fa_blk64<7>) attention_fa_blk64_t;
template [[host_name("attention_fa_blk64_g7")]] kernel attention_fa_blk64_t attention_fa_blk64<7>;
// attention_fa_blk96: attention_fa_blk64's shape at head dim 96 (Phi-3 mini, MHA, G = 1): 3 dims per lane, packed_half3 K/V
// loads (192 B per key row per simdgroup), float3 q and accumulators, and a 5*G-float threadgroup row per lane. Same
// signature, grid, split rule (attnFABlkSplit) and partial layout G*(hd+2), so attention_fa_combine merges it unchanged.
// Not bit-identical to the per-query-head kernel it replaces; fidelity-gated, and dispatched only under
// Options.FlashDecodeHD96 (attnFAHeadDimOK): docs/measurements/metal-fa-hd96-phi3-2026-11-23/README.md.
template <uint G>
kernel void attention_fa_blk96(
    device const float* q[[buffer(0)]], device const half* kc[[buffer(1)]],
    device const half* vc[[buffer(2)]], device float* partial[[buffer(3)]],
    constant uint& nKV[[buffer(4)]], constant uint& Gu[[buffer(5)]],
    constant uint& nKeys[[buffer(6)]], constant float& scale[[buffer(7)]],
    constant uint& window[[buffer(8)]], constant uint& nSplit[[buffer(9)]], device const uint* kt[[buffer(10)]],
    threadgroup float* shm[[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]]) {
    (void)Gu;
    const uint hd = 96u;
    const uint kvDim = nKV * hd;
    const uint kvh = tgid / nSplit, split = tgid % nSplit;
    const uint winStart = (window > 0u && nKeys > window) ? nKeys - window : 0u;
    const uint nWin = nKeys - winStart;
    const uint chunkLen = (nWin + nSplit - 1u) / nSplit;
    const uint chunkStart = winStart + split * chunkLen;
    const uint chunkEnd = min(chunkStart + chunkLen, nKeys);

    device const float* qr = q + kvh * G * hd;
    device const half*  kb = kc + kvh * hd + lane*3u;
    device const half*  vb = vc + kvh * hd + lane*3u;

    float3 qs[G];
    float m[G], l[G];
    float3 acc[G];
    ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) {
        qs[g] = float3(*((device const packed_float3*)(qr + g*hd + lane*3u))) * scale;
        m[g] = -INFINITY; l[g] = 0.0f; acc[g] = float3(0.0f);
    }

    for (uint b0 = chunkStart + sgid*32u; b0 < chunkEnd; b0 += 128u) {
        const uint nb = min(32u, chunkEnd - b0);
        const uint edge = 64u - (b0 & 63u), base0 = kt[b0 >> 6u]*64u + (b0 & 63u); // a run crosses one block edge at most
        const uint base1 = (edge < nb) ? kt[(b0 + edge) >> 6u]*64u - edge : base0;
        float s[G];
        ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) s[g] = -INFINITY;
        if (nb == 32u) {
            ATTN_FA_BLK_UNROLL for (uint j = 0; j < 32u; j++) {
                float3 k3 = float3(*((device const packed_half3*)(kb + (j < edge ? base0 + j : base1 + j)*kvDim)));
                ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) {
                    float d = attn_score(simd_sum(dot(qs[g], k3)));
                    s[g] = (lane == j) ? d : s[g];
                }
            }
        } else {
            for (uint j = 0; j < nb; j++) {
                float3 k3 = float3(*((device const packed_half3*)(kb + (j < edge ? base0 + j : base1 + j)*kvDim)));
                ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) {
                    float d = attn_score(simd_sum(dot(qs[g], k3)));
                    s[g] = (lane == j) ? d : s[g];
                }
            }
        }
        float p[G];
        ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) {
            float mb = simd_max(s[g]);
            float mn = max(m[g], mb);
            float alpha = (m[g] == -INFINITY) ? 0.0f : exp(m[g] - mn);
            p[g] = (s[g] == -INFINITY) ? 0.0f : exp(s[g] - mn);
            l[g] = l[g]*alpha + simd_sum(p[g]);
            acc[g] *= alpha;
            m[g] = mn;
        }
        if (nb == 32u) {
            ATTN_FA_BLK_UNROLL for (uint j = 0; j < 32u; j++) {
                float3 v3 = float3(*((device const packed_half3*)(vb + (j < edge ? base0 + j : base1 + j)*kvDim)));
                ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) acc[g] += simd_shuffle(p[g], j) * v3;
            }
        } else {
            for (uint j = 0; j < nb; j++) {
                float3 v3 = float3(*((device const packed_half3*)(vb + (j < edge ? base0 + j : base1 + j)*kvDim)));
                ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) acc[g] += simd_shuffle(p[g], j) * v3;
            }
        }
    }

    const uint stride = 5u*G;
    threadgroup float* row = shm + tid*stride;
    ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) {
        row[g] = m[g]; row[G+g] = l[g];
        row[2u*G + g*3u+0u] = acc[g].x; row[2u*G + g*3u+1u] = acc[g].y; row[2u*G + g*3u+2u] = acc[g].z;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        float fm[G], fl[G]; float3 fa[G];
        ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) { fm[g] = -INFINITY; fl[g] = 0.0f; fa[g] = float3(0.0f); }
        for (uint sg = 0; sg < 4u; sg++) {
            threadgroup float* r = shm + (sg*32u + lane)*stride;
            ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) {
                float m2 = r[g], l2 = r[G+g];
                float mn = max(fm[g], m2);
                float a1 = (fm[g] == -INFINITY) ? 0.0f : exp(fm[g] - mn);
                float a2 = (m2 == -INFINITY) ? 0.0f : exp(m2 - mn);
                fl[g] = fl[g]*a1 + l2*a2;
                fa[g] = fa[g]*a1 + float3(r[2u*G+g*3u+0u], r[2u*G+g*3u+1u], r[2u*G+g*3u+2u])*a2;
                fm[g] = mn;
            }
        }
        const uint pStride = G * (hd + 2u);
        device float* pout = partial + (kvh*nSplit + split) * pStride;
        ATTN_FA_BLK_UNROLL for (uint g = 0; g < G; g++) {
            device float* og = pout + g*(hd+2u);
            og[0] = fm[g]; og[1] = fl[g];
            og[2u+lane*3u+0u] = fa[g].x; og[2u+lane*3u+1u] = fa[g].y; og[2u+lane*3u+2u] = fa[g].z;
        }
    }
}

typedef decltype(attention_fa_blk96<1>) attention_fa_blk96_t;
template [[host_name("attention_fa_blk96_g1")]] kernel attention_fa_blk96_t attention_fa_blk96<1>;
