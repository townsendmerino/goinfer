// --- the q4k lane: float activations, no quantization ---
// A model loaded at --quant q4k keeps its Q4_K tensors as GGUF super-blocks and its other tensors as int8 (per-row
// scale); every projection reads the f32 activation straight from device memory, so nothing rounds Phi-3's activation
// outliers. One simdgroup per output row, 8 rows per 256-thread threadgroup, row bound N checked (the grid is N*32).
//
// rmsnorm_f32_out: rmsnorm_f16_act's reduction, f32 out (out-of-place).
kernel void rmsnorm_f32_out(device const float* x[[buffer(0)]], device const float* w[[buffer(1)]],
    device float* out[[buffer(2)]], constant uint& H[[buffer(3)]],
    constant float& eps[[buffer(4)]], constant uint& addOne[[buffer(5)]],
    uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256]; float ss=0;
    for(uint i=tid;i<H;i+=tgs) ss+=x[i]*x[i];
    red[tid]=ss; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=tgs/2;s>0;s>>=1){ if(tid<s) red[tid]+=red[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup);}
    float rms=precise::rsqrt(red[0]/float(H)+eps);
    for(uint i=tid;i<H;i+=tgs){ float g=addOne!=0u?(1.0f+w[i]):w[i]; out[i]=x[i]*rms*g; }
}
// q4k_scale_min: ggml's get_scale_min_k4, the j-th 6-bit scale and min of a super-block's 12-byte scales array.
inline void q4k_scale_min(uint j, device const uchar* q, thread float& sc, thread float& m) {
    if (j < 4u) { sc = float(q[j] & 63u); m = float(q[j+4u] & 63u); }
    else { sc = float((q[j+4u] & 15u) | ((q[j-4u] >> 6u) << 4u)); m = float((q[j+4u] >> 4u) | ((q[j] >> 6u) << 4u)); }
}
// q4k_scale_min2: q4k_scale_min for sub-blocks 2j and 2j+1 (j < 4), from the 12 bytes read as three words and decoded with
// masks and shifts. It has no branch on j, so lanes holding different j do not split the simdgroup; the values are
// q4k_scale_min's. q must be 4-byte aligned (a super-block's scales are at offset 4).
inline void q4k_scale_min2(uint j, device const uchar* q, thread float& sc0, thread float& m0, thread float& sc1,
    thread float& m1) {
    device const uint* w = (device const uint*)q;
    uint s0 = w[0], s1 = w[1], s2 = w[2], sh = ((2u*j) & 3u)*8u;
    bool hi = j >= 2u;
    uint scw = hi ? ((s2 & 0x0F0F0F0Fu) | ((s0 >> 2u) & 0x30303030u)) : (s0 & 0x3F3F3F3Fu);
    uint mw = hi ? (((s2 >> 4u) & 0x0F0F0F0Fu) | ((s1 >> 2u) & 0x30303030u)) : (s1 & 0x3F3F3F3Fu);
    sc0 = float((scw >> sh) & 0xFFu); sc1 = float((scw >> (sh + 8u)) & 0xFFu);
    m0 = float((mw >> sh) & 0xFFu); m1 = float((mw >> (sh + 8u)) & 0xFFu);
}
// Per 256 weights (144 bytes): d (f16), dmin (f16), scales[12], qs[128]; w = d*sc*q - dmin*m per 32-weight sub-block;
// byte qs[32j+l] holds element 64j+l (low nibble) and 64j+32+l (high).
// KS-C1: the q4k lane's GEMVs at R rows a simdgroup (grid ceil(N/R)*32, tg 256; rows past N are skipped).
// q4k_rows_acc puts 8 lanes on a super-block and 4 super-blocks in flight (lane l takes super-blocks l/8, l/8+4, ...). Within
// one, lane l mod 8 takes chunk j = (l mod 8)/2 and half h = l mod 2: one uint4 of quants (bytes 32j+16h..+15, whose low
// nibbles are elements 64j+16h..+15 of sub-block 2j and high ones 64j+32+16h..+15 of 2j+1) against 8 float4 activations loaded
// once for the R rows; d*sc*dot(q, a) per sub-block and the min term as dmin*m*sum(a), the scales decoded once per 32 weights.
// Not gemv_q4k_f32's order: G-Q1's bar and G-Q2 grade it. The batched step's mc3_gemv_q4k_f32_rows (batch_rows.go) repeats
// this loop per activation row, so it stays decode's bits.
template <uint R>
inline void q4k_rows_acc(device const uchar* wq, device const float* ax, uint K, uint N, uint row0, uint lane,
    thread float* acc) {
#pragma clang fp reassociate(off)
    const uint NB = K >> 8u, ix = lane >> 3u, j = (lane & 7u) >> 1u, h = lane & 1u;
    SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) acc[r] = 0.0f;
    for (uint b = ix; b < NB; b += 4u) {
        device const float4* alo4 = (device const float4*)(ax + b*256u + j*64u + h*16u);
        float4 al[4], ah[4];
        float slo = 0.0f, shi = 0.0f;
        for (uint i = 0u; i < 4u; i++) {
            al[i] = alo4[i]; ah[i] = alo4[8u + i];
            slo += (al[i].x + al[i].y) + (al[i].z + al[i].w);
            shi += (ah[i].x + ah[i].y) + (ah[i].z + ah[i].w);
        }
        SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) {
            if (row0 + r >= N) break;
            device const uchar* blk = wq + (ulong)(row0 + r)*(ulong)NB*144ul + b*144u;
            half2 dd = as_type<half2>(*(device const uint*)blk);
            float sc0, m0, sc1, m1; q4k_scale_min2(j, blk + 4, sc0, m0, sc1, m1);
            uint4 q = *(device const uint4*)(blk + 16u + j*32u + h*16u);
            float dlo = 0.0f, dhi = 0.0f;
            for (uint i = 0u; i < 4u; i++) {
                uint w = q[i];
                dlo += dot(float4(float(w & 15u), float((w >> 8u) & 15u), float((w >> 16u) & 15u), float((w >> 24u) & 15u)), al[i]);
                dhi += dot(float4(float((w >> 4u) & 15u), float((w >> 12u) & 15u), float((w >> 20u) & 15u), float(w >> 28u)), ah[i]);
            }
            float d = float(dd.x), dm = float(dd.y);
            acc[r] += d*sc0*dlo + d*sc1*dhi - dm*(m0*slo + m1*shi);
        }
    }
    SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) acc[r] = simd_sum(acc[r]);
}
template <uint R>
kernel void gemv_q4k_f32_rows(device const uchar* wq[[buffer(0)]], device const float* ax[[buffer(1)]],
    device float* out[[buffer(2)]], constant uint& K[[buffer(3)]], constant uint& N[[buffer(4)]],
    uint tgid[[threadgroup_position_in_grid]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]]) {
    uint row0 = (tgid*8u + sgid)*R;
    if (row0 >= N) return;
    float acc[R]; q4k_rows_acc<R>(wq, ax, K, N, row0, lane, acc);
    if (lane==0) { SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) if (row0 + r < N) out[row0 + r] = acc[r]; }
}
template <uint R>
kernel void gemv_q4k_f32_resid_rows(device const uchar* wq[[buffer(0)]], device const float* ax[[buffer(1)]],
    device float* out[[buffer(2)]], constant uint& K[[buffer(3)]], constant uint& N[[buffer(4)]],
    uint tgid[[threadgroup_position_in_grid]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]]) {
    uint row0 = (tgid*8u + sgid)*R;
    if (row0 >= N) return;
    float acc[R]; q4k_rows_acc<R>(wq, ax, K, N, row0, lane, acc);
    if (lane==0) { SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) if (row0 + r < N) out[row0 + r] += acc[r]; }
}
template <uint R>
kernel void gemv_q4k_f32_bias_rows(device const uchar* wq[[buffer(0)]], device const float* ax[[buffer(1)]],
    device float* out[[buffer(2)]], constant uint& K[[buffer(3)]], constant uint& N[[buffer(4)]],
    device const float* bias[[buffer(5)]], uint tgid[[threadgroup_position_in_grid]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    uint row0 = (tgid*8u + sgid)*R;
    if (row0 >= N) return;
    float acc[R]; q4k_rows_acc<R>(wq, ax, K, N, row0, lane, acc);
    if (lane==0) { SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) if (row0 + r < N) out[row0 + r] = acc[r] + bias[row0 + r]; }
}
// gemv_w8_f32 at R rows a simdgroup: each row's lane loop, simd_sum and epilogue are W8_F32_BODY's statement for statement,
// with the lane's activation load shared, so each output is gemv_w8_f32's bits (KS-C1 (b)).
template <uint R>
inline void w8f32_rows_acc(device const char* wq, device const float* sc, device const float* ax, uint K, uint N, uint row0,
    uint lane, thread float* acc) {
    device const float4* a4 = (device const float4*)ax;
    SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) acc[r] = 0.0f;
    for (uint g = lane; g < (K >> 2u); g += 32u) {
        float4 a = a4[g];
        SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) {
            if (row0 + r >= N) break;
            acc[r] += dot(float4(((device const char4*)(wq + (ulong)(row0 + r)*(ulong)K))[g]), a);
        }
    }
    SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) acc[r] = simd_sum(acc[r]) * sc[min(row0 + r, N - 1u)];
}
template <uint R>
kernel void gemv_w8_f32_rows(device const char* wq[[buffer(0)]], device const float* sc[[buffer(1)]],
    device const float* ax[[buffer(2)]], device float* out[[buffer(3)]], constant uint& K[[buffer(4)]],
    constant uint& N[[buffer(5)]], uint tgid[[threadgroup_position_in_grid]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    uint row0 = (tgid*8u + sgid)*R;
    if (row0 >= N) return;
    float acc[R]; w8f32_rows_acc<R>(wq, sc, ax, K, N, row0, lane, acc);
    if (lane==0) { SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) if (row0 + r < N) out[row0 + r] = acc[r]; }
}
template <uint R>
kernel void gemv_w8_f32_resid_rows(device const char* wq[[buffer(0)]], device const float* sc[[buffer(1)]],
    device const float* ax[[buffer(2)]], device float* out[[buffer(3)]], constant uint& K[[buffer(4)]],
    constant uint& N[[buffer(5)]], uint tgid[[threadgroup_position_in_grid]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    uint row0 = (tgid*8u + sgid)*R;
    if (row0 >= N) return;
    float acc[R]; w8f32_rows_acc<R>(wq, sc, ax, K, N, row0, lane, acc);
    if (lane==0) { SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) if (row0 + r < N) out[row0 + r] += acc[r]; }
}
template <uint R>
kernel void gemv_w8_f32_bias_rows(device const char* wq[[buffer(0)]], device const float* sc[[buffer(1)]],
    device const float* ax[[buffer(2)]], device float* out[[buffer(3)]], constant uint& K[[buffer(4)]],
    constant uint& N[[buffer(5)]], device const float* bias[[buffer(6)]], uint tgid[[threadgroup_position_in_grid]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    uint row0 = (tgid*8u + sgid)*R;
    if (row0 >= N) return;
    float acc[R]; w8f32_rows_acc<R>(wq, sc, ax, K, N, row0, lane, acc);
    if (lane==0) { SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) if (row0 + r < N) out[row0 + r] = acc[r] + bias[row0 + r]; }
}
template [[host_name("gemv_q4k_f32_rows1")]] kernel decltype(gemv_q4k_f32_rows<1>) gemv_q4k_f32_rows<1>;
template [[host_name("gemv_q4k_f32_rows2")]] kernel decltype(gemv_q4k_f32_rows<2>) gemv_q4k_f32_rows<2>;
template [[host_name("gemv_q4k_f32_rows4")]] kernel decltype(gemv_q4k_f32_rows<4>) gemv_q4k_f32_rows<4>;
template [[host_name("gemv_q4k_f32_rows8")]] kernel decltype(gemv_q4k_f32_rows<8>) gemv_q4k_f32_rows<8>;
template [[host_name("gemv_q4k_f32_resid_rows1")]] kernel decltype(gemv_q4k_f32_resid_rows<1>) gemv_q4k_f32_resid_rows<1>;
template [[host_name("gemv_q4k_f32_resid_rows2")]] kernel decltype(gemv_q4k_f32_resid_rows<2>) gemv_q4k_f32_resid_rows<2>;
template [[host_name("gemv_q4k_f32_resid_rows4")]] kernel decltype(gemv_q4k_f32_resid_rows<4>) gemv_q4k_f32_resid_rows<4>;
template [[host_name("gemv_q4k_f32_resid_rows8")]] kernel decltype(gemv_q4k_f32_resid_rows<8>) gemv_q4k_f32_resid_rows<8>;
template [[host_name("gemv_q4k_f32_bias_rows1")]] kernel decltype(gemv_q4k_f32_bias_rows<1>) gemv_q4k_f32_bias_rows<1>;
template [[host_name("gemv_q4k_f32_bias_rows2")]] kernel decltype(gemv_q4k_f32_bias_rows<2>) gemv_q4k_f32_bias_rows<2>;
template [[host_name("gemv_q4k_f32_bias_rows4")]] kernel decltype(gemv_q4k_f32_bias_rows<4>) gemv_q4k_f32_bias_rows<4>;
template [[host_name("gemv_q4k_f32_bias_rows8")]] kernel decltype(gemv_q4k_f32_bias_rows<8>) gemv_q4k_f32_bias_rows<8>;
template [[host_name("gemv_w8_f32_rows1")]] kernel decltype(gemv_w8_f32_rows<1>) gemv_w8_f32_rows<1>;
template [[host_name("gemv_w8_f32_rows2")]] kernel decltype(gemv_w8_f32_rows<2>) gemv_w8_f32_rows<2>;
template [[host_name("gemv_w8_f32_rows4")]] kernel decltype(gemv_w8_f32_rows<4>) gemv_w8_f32_rows<4>;
template [[host_name("gemv_w8_f32_resid_rows1")]] kernel decltype(gemv_w8_f32_resid_rows<1>) gemv_w8_f32_resid_rows<1>;
template [[host_name("gemv_w8_f32_resid_rows2")]] kernel decltype(gemv_w8_f32_resid_rows<2>) gemv_w8_f32_resid_rows<2>;
template [[host_name("gemv_w8_f32_resid_rows4")]] kernel decltype(gemv_w8_f32_resid_rows<4>) gemv_w8_f32_resid_rows<4>;
template [[host_name("gemv_w8_f32_bias_rows1")]] kernel decltype(gemv_w8_f32_bias_rows<1>) gemv_w8_f32_bias_rows<1>;
template [[host_name("gemv_w8_f32_bias_rows2")]] kernel decltype(gemv_w8_f32_bias_rows<2>) gemv_w8_f32_bias_rows<2>;
template [[host_name("gemv_w8_f32_bias_rows4")]] kernel decltype(gemv_w8_f32_bias_rows<4>) gemv_w8_f32_bias_rows<4>;

// --- KS-O1: native Q5_K and Q6_K on the q4k lane ---
// q4k_lane_act loads lane (j, h)'s 16 low and 16 high activations of super-block b on q4k_rows_acc's lane map, and their sums.
inline void q4k_lane_act(device const float* ax, uint b, uint j, uint h, thread float4* al, thread float4* ah,
    thread float& slo, thread float& shi) {
#pragma clang fp reassociate(off)
    device const float4* alo4 = (device const float4*)(ax + b*256u + j*64u + h*16u);
    slo = 0.0f; shi = 0.0f;
    for (uint i = 0u; i < 4u; i++) {
        al[i] = alo4[i]; ah[i] = alo4[8u + i];
        slo += (al[i].x + al[i].y) + (al[i].z + al[i].w);
        shi += (ah[i].x + ah[i].y) + (ah[i].z + ah[i].w);
    }
}
// Q5_K, 176 bytes per 256 weights: d, dmin (f16), scales[12] (Q4_K's packing), qh[32], qs[128]. Element 64j+l (l < 32) is
// qs[32j+l]'s low nibble plus 16 if bit 2j of qh[l] is set; element 64j+32+l is its high nibble and bit 2j+1. q5k_blk is lane
// (j, h)'s share of one block: its 16 low and 16 high weights against q4k_lane_act's activations.
inline float q5k_blk(device const uchar* blk, uint j, uint h, thread const float4* al, thread const float4* ah, float slo,
    float shi) {
#pragma clang fp reassociate(off)
    half2 dd = as_type<half2>(*(device const uint*)blk);
    float sc0, m0, sc1, m1; q4k_scale_min2(j, blk + 4, sc0, m0, sc1, m1);
    uint4 q = *(device const uint4*)(blk + 48u + j*32u + h*16u);
    uint4 qh = *(device const uint4*)(blk + 16u + h*16u);
    float dlo = 0.0f, dhi = 0.0f;
    for (uint i = 0u; i < 4u; i++) {
        uint lo = (q[i] & 0x0F0F0F0Fu) | (((qh[i] >> (2u*j)) & 0x01010101u) << 4u);
        uint hi = ((q[i] >> 4u) & 0x0F0F0F0Fu) | (((qh[i] >> (2u*j + 1u)) & 0x01010101u) << 4u);
        dlo += dot(float4(as_type<uchar4>(lo)), al[i]);
        dhi += dot(float4(as_type<uchar4>(hi)), ah[i]);
    }
    float d = float(dd.x), dm = float(dd.y);
    return d*sc0*dlo + d*sc1*dhi - dm*(m0*slo + m1*shi);
}
// Q6_K, 210 bytes per 256 weights, is uploaded with a segment's T = N*NB blocks split into planes (metal's q6kPlanar): T*128
// bytes of ql, T*64 of qh, T*16 of int8 scales, T halves of d, block g = row*NB + b at g*128, g*64, g*16 and g*2 of its plane,
// so every load is aligned for any NB. ggml's element 128n+32t+l (n < 2, t < 4, l < 32) is ql[64n+l] (t = 0, 2) or
// ql[64n+32+l] (t = 1, 3), its low nibble for t < 2 and high for t >= 2, plus bits 2t, 2t+1 of qh[32n+l] as 16 and 32, minus
// 32, times d and int8 scale 8n + l/16 + 2t. Lane k of a block (lanes 8i..8i+7) takes half n = k/4 and l = 8(k mod 4)..+7
// of each group t; q6k_lane_act loads its 32 activations, a[2t] and a[2t+1] for group t.
inline void q6k_lane_act(device const float* ax, uint b, uint n, uint l0, thread float4* a) {
    device const float4* y4 = (device const float4*)(ax + b*256u + n*128u + l0);
    for (uint t = 0u; t < 4u; t++) { a[2u*t] = y4[8u*t]; a[2u*t + 1u] = y4[8u*t + 1u]; }
}
// q6k_s6: four 6-bit codes (bytes 0..63) minus 32 as signed bytes, with no borrow between bytes.
inline float4 q6k_s6(uint t) { return float4(as_type<char4>(((t | 0x80808080u) - 0x20202020u) ^ 0x80808080u)); }
inline float q6k_blk(device const uchar* wq, ulong T, ulong g, uint n, uint l0, thread const float4* a) {
#pragma clang fp reassociate(off)
    uint2 qa = *(device const uint2*)(wq + g*128ul + n*64u + l0);
    uint2 qb = *(device const uint2*)(wq + g*128ul + n*64u + 32u + l0);
    uint2 qh = *(device const uint2*)(wq + T*128ul + g*64ul + n*32u + l0);
    device const char* sc = (device const char*)(wq + T*192ul + g*16ul + n*8u + (l0 >> 4u));
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    for (uint w = 0u; w < 2u; w++) {
        d0 += dot(q6k_s6((qa[w] & 0x0F0F0F0Fu) | ((qh[w] & 0x03030303u) << 4u)), a[w]);
        d1 += dot(q6k_s6((qb[w] & 0x0F0F0F0Fu) | (((qh[w] >> 2u) & 0x03030303u) << 4u)), a[2u + w]);
        d2 += dot(q6k_s6(((qa[w] >> 4u) & 0x0F0F0F0Fu) | (((qh[w] >> 4u) & 0x03030303u) << 4u)), a[4u + w]);
        d3 += dot(q6k_s6(((qb[w] >> 4u) & 0x0F0F0F0Fu) | (((qh[w] >> 6u) & 0x03030303u) << 4u)), a[6u + w]);
    }
    float d = float(*(device const half*)(wq + T*208ul + g*2ul));
    return d*((float(sc[0])*d0 + float(sc[2])*d1) + (float(sc[4])*d2 + float(sc[6])*d3));
}
// The q4k lane's Q5_K and Q6_K GEMVs at R rows a simdgroup (grid ceil(N/R)*32, tg 256; rows past N are skipped), 8 lanes a
// super-block and 4 in flight as q4k_rows_acc. The batched step's mc3_gemv_q5k_f32_rows and mc3_gemv_q6k_f32_rows
// (batch_rows.go) call the same q5k_blk and q6k_blk per activation row, so the step stays decode's bits.
template <uint R>
inline void q5k_rows_acc(device const uchar* wq, device const float* ax, uint K, uint N, uint row0, uint lane,
    thread float* acc) {
#pragma clang fp reassociate(off)
    const uint NB = K >> 8u, ix = lane >> 3u, j = (lane & 7u) >> 1u, h = lane & 1u;
    SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) acc[r] = 0.0f;
    for (uint b = ix; b < NB; b += 4u) {
        float4 al[4], ah[4]; float slo, shi;
        q4k_lane_act(ax, b, j, h, al, ah, slo, shi);
        SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) {
            if (row0 + r >= N) break;
            acc[r] += q5k_blk(wq + (ulong)(row0 + r)*(ulong)NB*176ul + b*176u, j, h, al, ah, slo, shi);
        }
    }
    SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) acc[r] = simd_sum(acc[r]);
}
template <uint R>
inline void q6k_rows_acc(device const uchar* wq, device const float* ax, uint K, uint N, uint row0, uint lane,
    thread float* acc) {
#pragma clang fp reassociate(off)
    const uint NB = K >> 8u, ix = lane >> 3u, n = (lane & 7u) >> 2u, l0 = (lane & 3u)*8u;
    SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) acc[r] = 0.0f;
    for (uint b = ix; b < NB; b += 4u) {
        float4 a[8];
        q6k_lane_act(ax, b, n, l0, a);
        SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) {
            if (row0 + r >= N) break;
            acc[r] += q6k_blk(wq, (ulong)N*NB, (ulong)(row0 + r)*NB + b, n, l0, a);
        }
    }
    SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) acc[r] = simd_sum(acc[r]);
}
template <uint R>
kernel void gemv_q5k_f32_rows(device const uchar* wq[[buffer(0)]], device const float* ax[[buffer(1)]],
    device float* out[[buffer(2)]], constant uint& K[[buffer(3)]], constant uint& N[[buffer(4)]],
    uint tgid[[threadgroup_position_in_grid]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]]) {
    uint row0 = (tgid*8u + sgid)*R;
    if (row0 >= N) return;
    float acc[R]; q5k_rows_acc<R>(wq, ax, K, N, row0, lane, acc);
    if (lane==0) { SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) if (row0 + r < N) out[row0 + r] = acc[r]; }
}
template <uint R>
kernel void gemv_q5k_f32_resid_rows(device const uchar* wq[[buffer(0)]], device const float* ax[[buffer(1)]],
    device float* out[[buffer(2)]], constant uint& K[[buffer(3)]], constant uint& N[[buffer(4)]],
    uint tgid[[threadgroup_position_in_grid]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]]) {
    uint row0 = (tgid*8u + sgid)*R;
    if (row0 >= N) return;
    float acc[R]; q5k_rows_acc<R>(wq, ax, K, N, row0, lane, acc);
    if (lane==0) { SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) if (row0 + r < N) out[row0 + r] += acc[r]; }
}
template <uint R>
kernel void gemv_q5k_f32_bias_rows(device const uchar* wq[[buffer(0)]], device const float* ax[[buffer(1)]],
    device float* out[[buffer(2)]], constant uint& K[[buffer(3)]], constant uint& N[[buffer(4)]],
    device const float* bias[[buffer(5)]],
    uint tgid[[threadgroup_position_in_grid]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]]) {
    uint row0 = (tgid*8u + sgid)*R;
    if (row0 >= N) return;
    float acc[R]; q5k_rows_acc<R>(wq, ax, K, N, row0, lane, acc);
    if (lane==0) { SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) if (row0 + r < N) out[row0 + r] = acc[r] + bias[row0 + r]; }
}
template <uint R>
kernel void gemv_q6k_f32_rows(device const uchar* wq[[buffer(0)]], device const float* ax[[buffer(1)]],
    device float* out[[buffer(2)]], constant uint& K[[buffer(3)]], constant uint& N[[buffer(4)]],
    uint tgid[[threadgroup_position_in_grid]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]]) {
    uint row0 = (tgid*8u + sgid)*R;
    if (row0 >= N) return;
    float acc[R]; q6k_rows_acc<R>(wq, ax, K, N, row0, lane, acc);
    if (lane==0) { SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) if (row0 + r < N) out[row0 + r] = acc[r]; }
}
template <uint R>
kernel void gemv_q6k_f32_resid_rows(device const uchar* wq[[buffer(0)]], device const float* ax[[buffer(1)]],
    device float* out[[buffer(2)]], constant uint& K[[buffer(3)]], constant uint& N[[buffer(4)]],
    uint tgid[[threadgroup_position_in_grid]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]]) {
    uint row0 = (tgid*8u + sgid)*R;
    if (row0 >= N) return;
    float acc[R]; q6k_rows_acc<R>(wq, ax, K, N, row0, lane, acc);
    if (lane==0) { SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) if (row0 + r < N) out[row0 + r] += acc[r]; }
}
template <uint R>
kernel void gemv_q6k_f32_bias_rows(device const uchar* wq[[buffer(0)]], device const float* ax[[buffer(1)]],
    device float* out[[buffer(2)]], constant uint& K[[buffer(3)]], constant uint& N[[buffer(4)]],
    device const float* bias[[buffer(5)]],
    uint tgid[[threadgroup_position_in_grid]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]]) {
    uint row0 = (tgid*8u + sgid)*R;
    if (row0 >= N) return;
    float acc[R]; q6k_rows_acc<R>(wq, ax, K, N, row0, lane, acc);
    if (lane==0) { SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) if (row0 + r < N) out[row0 + r] = acc[r] + bias[row0 + r]; }
}
template [[host_name("gemv_q5k_f32_rows4")]] kernel decltype(gemv_q5k_f32_rows<4>) gemv_q5k_f32_rows<4>;
template [[host_name("gemv_q5k_f32_resid_rows4")]] kernel decltype(gemv_q5k_f32_resid_rows<4>) gemv_q5k_f32_resid_rows<4>;
template [[host_name("gemv_q5k_f32_bias_rows4")]] kernel decltype(gemv_q5k_f32_bias_rows<4>) gemv_q5k_f32_bias_rows<4>;
template [[host_name("gemv_q6k_f32_rows4")]] kernel decltype(gemv_q6k_f32_rows<4>) gemv_q6k_f32_rows<4>;
template [[host_name("gemv_q6k_f32_resid_rows4")]] kernel decltype(gemv_q6k_f32_resid_rows<4>) gemv_q6k_f32_resid_rows<4>;
template [[host_name("gemv_q6k_f32_bias_rows4")]] kernel decltype(gemv_q6k_f32_bias_rows<4>) gemv_q6k_f32_bias_rows<4>;

