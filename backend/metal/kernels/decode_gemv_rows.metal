// R18 (docs/measurements/metal-decode-gemv-r18-2026-09-26.md): the SA family at R rows per simdgroup, each lane reusing
// its group's activations for R rows. R18b (docs/measurements/metal-decode-gemv-r18b-2026-09-26.md) changed the inner
// arithmetic to MLX's masked form: activations are staged as HALF pre-scaled by 16^-(k mod 4) — exact, |a| <= 127 and
// 127*2^-12 is a normal half — in the same K*2 bytes the short staging used. Weight nibbles are masked in place per
// 16-bit half-word, converted once per (row, half-word), and dotted with the four pre-scaled activations; the group's
// sum(a), for the -8 fold, comes from the same staged halves (one exact dot per half-word). Every product and partial
// sum is an integer below 2^24, so each group sum equals SA_BODY's integer gi exactly, and float(gi)*scale accumulates
// per row in SA_BODY's lane order -> bit-identical to gemv_w4a8_sa / _bias / _resid. Grid rows*32/R, tg 256. row0
// assumes FULL threadgroups (a partial last one reports a smaller threads_per_threadgroup), so the host takes this path
// only when rows % (8R) == 0 (gemvRowsFor).
#define SA_ROWS_UNROLL _Pragma("clang loop unroll(full)")
template <uint R>
inline void sa_rows_acc(device const uint4* wq, device const half* sct, device const char* aq, uint K,
    threadgroup half* Ah, uint tgid, uint tid, uint tgs, uint sgid, uint lane, thread float* acc, thread uint& row0) {
    constexpr float P4[4] = {1.0f, 0.0625f, 0.00390625f, 0.000244140625f};
    constexpr float4 U4 = float4(1.0f, 16.0f, 256.0f, 4096.0f); // undoes the pre-scale, exactly
    for (uint i=tid;i<K;i+=tgs) Ah[i] = half(float(aq[i]) * P4[i & 3u]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint G = K>>5u;
    row0 = (tgid*(tgs>>5u) + sgid)*R;
    SA_ROWS_UNROLL for (uint r=0;r<R;r++) acc[r]=0.0f;
    for (uint g=lane; g<G; g+=32u) {
        float gf[R];
        uint4 w[R];
        SA_ROWS_UNROLL for (uint r=0;r<R;r++) { gf[r] = 0.0f; w[r] = wq[(row0+r)*G + g]; }
        threadgroup const half4* a4 = reinterpret_cast<threadgroup const half4*>(Ah + g*32u);
        float sa = 0.0f;
        SA_ROWS_UNROLL for (uint t=0;t<8u;t++) {
            float4 x = float4(a4[t]);
            sa += dot(x, U4);
            SA_ROWS_UNROLL for (uint r=0;r<R;r++) {
                uint word = (t < 2u) ? w[r].x : (t < 4u) ? w[r].y : (t < 6u) ? w[r].z : w[r].w;
                uint u = (t & 1u) ? (word >> 16) : (word & 0xFFFFu);
                gf[r] += dot(float4(float(u & 0xFu), float(u & 0xF0u), float(u & 0xF00u), float(u & 0xF000u)), x);
            }
        }
        SA_ROWS_UNROLL for (uint r=0;r<R;r++) acc[r] += (gf[r] - 8.0f*sa) * float(sct[(row0+r)*G + g]);
    }
    SA_ROWS_UNROLL for (uint r=0;r<R;r++) acc[r] = simd_sum(acc[r]);
}
template <uint R>
kernel void gemv_w4a8_sa_rows(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], threadgroup half* Ah [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    float acc[R]; uint row0;
    sa_rows_acc<R>(wq, sct, aq, K, Ah, tgid, tid, tgs, sgid, lane, acc, row0);
    if (lane==0) { SA_ROWS_UNROLL for (uint r=0;r<R;r++) out[row0+r] = acc[r]*asc[0]; }
}
template <uint R>
kernel void gemv_w4a8_sa_bias_rows(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    device const float* bias[[buffer(5)]], constant uint& K[[buffer(6)]], threadgroup half* Ah [[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    float acc[R]; uint row0;
    sa_rows_acc<R>(wq, sct, aq, K, Ah, tgid, tid, tgs, sgid, lane, acc, row0);
    if (lane==0) { SA_ROWS_UNROLL for (uint r=0;r<R;r++) out[row0+r] = acc[r]*asc[0] + bias[row0+r]; }
}
template <uint R>
kernel void gemv_w4a8_sa_resid_rows(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], threadgroup half* Ah [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    float acc[R]; uint row0;
    sa_rows_acc<R>(wq, sct, aq, K, Ah, tgid, tid, tgs, sgid, lane, acc, row0);
    if (lane==0) { SA_ROWS_UNROLL for (uint r=0;r<R;r++) out[row0+r] += acc[r]*asc[0]; }
}
template [[host_name("gemv_w4a8_sa_rows2")]] kernel decltype(gemv_w4a8_sa_rows<2>) gemv_w4a8_sa_rows<2>;
template [[host_name("gemv_w4a8_sa_rows4")]] kernel decltype(gemv_w4a8_sa_rows<4>) gemv_w4a8_sa_rows<4>;
template [[host_name("gemv_w4a8_sa_bias_rows2")]] kernel decltype(gemv_w4a8_sa_bias_rows<2>) gemv_w4a8_sa_bias_rows<2>;
template [[host_name("gemv_w4a8_sa_bias_rows4")]] kernel decltype(gemv_w4a8_sa_bias_rows<4>) gemv_w4a8_sa_bias_rows<4>;
// E-P02 (docs/audit-metal-2026-09-30.md): gemv_w4a8_sa_rows / _bias_rows over B activation rows in ONE dispatch, the B
// threadgroups that read one weight tile adjacent in launch order (threadgroup tg serves batch row tg % B, tile tg / B),
// so a row's threadgroup can find the tile in cache. Each (row, tile) runs sa_rows_acc exactly as the per-row dispatch
// does for that row alone: bit-identical. Rows are K int8s / one scale / N floats apart, as the batched step lays them out.
template <uint R>
kernel void mc3_gemv_w4a8_sa_rows_adj(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq_b[[buffer(2)]], device const float* asc_b[[buffer(3)]], device float* out_b[[buffer(4)]],
    constant uint& K[[buffer(5)]], constant uint& N[[buffer(6)]], constant uint& B[[buffer(7)]],
    threadgroup half* Ah [[threadgroup(0)]], uint tg_b[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    uint b = tg_b % B, tgid = tg_b / B;
    device const char* aq = aq_b + b*K; device const float* asc = asc_b + b; device float* out = out_b + b*N;
    float acc[R]; uint row0;
    sa_rows_acc<R>(wq, sct, aq, K, Ah, tgid, tid, tgs, sgid, lane, acc, row0);
    if (lane==0) { SA_ROWS_UNROLL for (uint r=0;r<R;r++) out[row0+r] = acc[r]*asc[0]; }
}
template <uint R>
kernel void mc3_gemv_w4a8_sa_bias_rows_adj(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq_b[[buffer(2)]], device const float* asc_b[[buffer(3)]], device float* out_b[[buffer(4)]],
    device const float* bias[[buffer(5)]], constant uint& K[[buffer(6)]], constant uint& N[[buffer(7)]],
    constant uint& B[[buffer(8)]], threadgroup half* Ah [[threadgroup(0)]], uint tg_b[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    uint b = tg_b % B, tgid = tg_b / B;
    device const char* aq = aq_b + b*K; device const float* asc = asc_b + b; device float* out = out_b + b*N;
    float acc[R]; uint row0;
    sa_rows_acc<R>(wq, sct, aq, K, Ah, tgid, tid, tgs, sgid, lane, acc, row0);
    if (lane==0) { SA_ROWS_UNROLL for (uint r=0;r<R;r++) out[row0+r] = acc[r]*asc[0] + bias[row0+r]; }
}
template [[host_name("mc3_gemv_w4a8_sa_rows_adj2")]] kernel decltype(mc3_gemv_w4a8_sa_rows_adj<2>) mc3_gemv_w4a8_sa_rows_adj<2>;
template [[host_name("mc3_gemv_w4a8_sa_rows_adj4")]] kernel decltype(mc3_gemv_w4a8_sa_rows_adj<4>) mc3_gemv_w4a8_sa_rows_adj<4>;
template [[host_name("mc3_gemv_w4a8_sa_bias_rows_adj2")]] kernel decltype(mc3_gemv_w4a8_sa_bias_rows_adj<2>) mc3_gemv_w4a8_sa_bias_rows_adj<2>;
template [[host_name("mc3_gemv_w4a8_sa_bias_rows_adj4")]] kernel decltype(mc3_gemv_w4a8_sa_bias_rows_adj<4>) mc3_gemv_w4a8_sa_bias_rows_adj<4>;
template [[host_name("gemv_w4a8_sa_resid_rows2")]] kernel decltype(gemv_w4a8_sa_resid_rows<2>) gemv_w4a8_sa_resid_rows<2>;
template [[host_name("gemv_w4a8_sa_resid_rows4")]] kernel decltype(gemv_w4a8_sa_resid_rows<4>) gemv_w4a8_sa_resid_rows<4>;

// KS-C3: the native int8 twins of R18's rows kernels, each taking its int4 twin's arguments in the same order at the same
// launch shape (grid rows*32/R, tg 256, full threadgroups only; gemvRowsFor). The activation is staged as uint4 words and
// each lane reads uint4 words of R rows, so K must be a multiple of 16. The int32 dot is exact in any order and the epilogue
// is W8A8_SA_BODY's (simd_broadcast_first included), so each output is the shipped one-row kernel's bits.
template <uint R>
inline void w8_rows_acc(device const uint4* wq, device const float* sct, device const char* aq, device const float* asc,
    uint K, threadgroup uint4* As, uint tgid, uint tid, uint tgs, uint sgid, uint lane, thread float* y, thread uint& row0) {
    device const uint4* a4 = reinterpret_cast<device const uint4*>(aq);
    uint G = K >> 4u;
    for (uint i = tid; i < G; i += tgs) As[i] = a4[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    row0 = (tgid*(tgs>>5u) + sgid)*R;
    int acc[R];
    SA_ROWS_UNROLL for (uint r=0;r<R;r++) acc[r]=0;
    for (uint g = lane; g < G; g += 32u) {
        uint4 a = As[g];
        SA_ROWS_UNROLL for (uint r=0;r<R;r++) {
            uint4 w = wq[(row0+r)*G + g];
            acc[r] += DOT4I8(a.x, w.x) + DOT4I8(a.y, w.y) + DOT4I8(a.z, w.z) + DOT4I8(a.w, w.w);
        }
    }
    SA_ROWS_UNROLL for (uint r=0;r<R;r++) y[r] = simd_broadcast_first(float(simd_sum(acc[r])) * asc[0] * sct[row0+r]);
}
template <uint R>
kernel void gemv_w8a8_sa_rows(device const uint4* wq[[buffer(0)]], device const float* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], threadgroup uint4* As [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    float y[R]; uint row0;
    w8_rows_acc<R>(wq, sct, aq, asc, K, As, tgid, tid, tgs, sgid, lane, y, row0);
    if (lane==0) { SA_ROWS_UNROLL for (uint r=0;r<R;r++) out[row0+r] = y[r]; }
}
template <uint R>
kernel void gemv_w8a8_sa_bias_rows(device const uint4* wq[[buffer(0)]], device const float* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    device const float* bias[[buffer(5)]], constant uint& K[[buffer(6)]], threadgroup uint4* As [[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    float y[R]; uint row0;
    w8_rows_acc<R>(wq, sct, aq, asc, K, As, tgid, tid, tgs, sgid, lane, y, row0);
    if (lane==0) { SA_ROWS_UNROLL for (uint r=0;r<R;r++) out[row0+r] = y[r] + bias[row0+r]; }
}
template <uint R>
kernel void gemv_w8a8_sa_resid_rows(device const uint4* wq[[buffer(0)]], device const float* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], threadgroup uint4* As [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    float y[R]; uint row0;
    w8_rows_acc<R>(wq, sct, aq, asc, K, As, tgid, tid, tgs, sgid, lane, y, row0);
    if (lane==0) { SA_ROWS_UNROLL for (uint r=0;r<R;r++) out[row0+r] += y[r]; }
}
// The down projection's twin of gemv_w4a8_resid_staged: gemv_w8a8_body_resid's output, the same rows kernel.
template <uint R>
kernel void gemv_w8a8_body_resid_staged(device const uint4* bq[[buffer(0)]], device const float* bsc[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], threadgroup uint4* As [[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    float y[R]; uint row0;
    w8_rows_acc<R>(bq, bsc, aq, asc, K, As, tgid, tid, tgs, sgid, lane, y, row0);
    if (lane==0) { SA_ROWS_UNROLL for (uint r=0;r<R;r++) out[row0+r] += y[r]; }
}
template [[host_name("gemv_w8a8_sa_rows2")]] kernel decltype(gemv_w8a8_sa_rows<2>) gemv_w8a8_sa_rows<2>;
template [[host_name("gemv_w8a8_sa_rows4")]] kernel decltype(gemv_w8a8_sa_rows<4>) gemv_w8a8_sa_rows<4>;
template [[host_name("gemv_w8a8_sa_bias_rows2")]] kernel decltype(gemv_w8a8_sa_bias_rows<2>) gemv_w8a8_sa_bias_rows<2>;
template [[host_name("gemv_w8a8_sa_bias_rows4")]] kernel decltype(gemv_w8a8_sa_bias_rows<4>) gemv_w8a8_sa_bias_rows<4>;
template [[host_name("gemv_w8a8_sa_resid_rows2")]] kernel decltype(gemv_w8a8_sa_resid_rows<2>) gemv_w8a8_sa_resid_rows<2>;
template [[host_name("gemv_w8a8_sa_resid_rows4")]] kernel decltype(gemv_w8a8_sa_resid_rows<4>) gemv_w8a8_sa_resid_rows<4>;
template [[host_name("gemv_w8a8_body_resid_staged2")]] kernel decltype(gemv_w8a8_body_resid_staged<2>) gemv_w8a8_body_resid_staged<2>;
template [[host_name("gemv_w8a8_body_resid_staged4")]] kernel decltype(gemv_w8a8_body_resid_staged<4>) gemv_w8a8_body_resid_staged<4>;
// KS-C2: a word's eight-nibble sum in R18b's masked form for the staged coal kernels below. x0 and x1 are the word's eight
// int8 activations scaled by 16^-(k mod 4); each 16-bit half is masked in place (nibble k as nib*16^(k mod 4)) and dotted with
// its four. Every partial is an integer below 2^24, so the result less 8*sum(a) is W4A8_BODY's integer gi exactly.
// KSC2_OLD_DOWN (a test-only define, ksc2OldDownForTest) compiles the per-word integer form back.
constant float4 W4M_P4 = float4(1.0f, 0.0625f, 0.00390625f, 0.000244140625f);
constant float4 W4M_U4 = float4(1.0f, 16.0f, 256.0f, 4096.0f);
inline float w4_masked_word(uint x, float4 x0, float4 x1) {
    uint lo = x & 0xFFFFu, hi = x >> 16;
    return dot(float4(float(lo & 0xFu), float(lo & 0xF0u), float(lo & 0xF00u), float(lo & 0xF000u)), x0)
         + dot(float4(float(hi & 0xFu), float(hi & 0xF0u), float(hi & 0xF00u), float(hi & 0xF000u)), x1);
}

// R18: gemv_w4a8_resid (the coal down projection) with its int8 activations staged ONCE per threadgroup (K bytes of
// threadgroup memory; the host checks it fits) and R rows per simdgroup. Lane l still owns words l, l+32, ... of each
// row, in order, with W4A8_BODY's per-word integer sum (taken in w4_masked_word's form) and float(gi)*scale accumulation ->
// bit-identical to gemv_w4a8_resid. Grid rows*32/R, tg 256, rows % (8R) == 0 (gemvRowsFor), as for gemv_w4a8_sa_rows.
template <uint R>
kernel void gemv_w4a8_resid_staged(device const uint* bq[[buffer(0)]], device const half* bsc[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], threadgroup char* As [[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    device const uint* aq4 = reinterpret_cast<device const uint*>(aq);
    threadgroup uint* As4 = reinterpret_cast<threadgroup uint*>(As);
    for (uint i=tid;i<(K>>2u);i+=tgs) As4[i]=aq4[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint wpr = K/8u, spr = K/32u;
    uint row0 = (tgid*(tgs>>5u) + sgid)*R;
    float acc[R];
    SA_ROWS_UNROLL for (uint r=0;r<R;r++) acc[r]=0.0f;
    threadgroup const char4* A = reinterpret_cast<threadgroup const char4*>(As);
    for (uint wi = lane; wi < wpr; wi += 32u) {
#ifdef KSC2_OLD_DOWN
        char4 a0 = A[wi*2u], a1 = A[wi*2u+1u];
        SA_ROWS_UNROLL for (uint r=0;r<R;r++) {
            uint x = bq[(row0+r)*wpr + wi];
            int gi = (int((x)&0xF)-8)*int(a0.x) + (int((x>>4)&0xF)-8)*int(a0.y)
                   + (int((x>>8)&0xF)-8)*int(a0.z) + (int((x>>12)&0xF)-8)*int(a0.w)
                   + (int((x>>16)&0xF)-8)*int(a1.x) + (int((x>>20)&0xF)-8)*int(a1.y)
                   + (int((x>>24)&0xF)-8)*int(a1.z) + (int((x>>28)&0xF)-8)*int(a1.w);
            acc[r] += float(gi) * float(bsc[(row0+r)*spr + (wi>>2u)]);
        }
#else
        float4 x0 = float4(A[wi*2u]) * W4M_P4, x1 = float4(A[wi*2u+1u]) * W4M_P4;
        float s8 = 8.0f * (dot(x0, W4M_U4) + dot(x1, W4M_U4));
        SA_ROWS_UNROLL for (uint r=0;r<R;r++) {
            acc[r] += (w4_masked_word(bq[(row0+r)*wpr + wi], x0, x1) - s8) * float(bsc[(row0+r)*spr + (wi>>2u)]);
        }
#endif
    }
    SA_ROWS_UNROLL for (uint r=0;r<R;r++) { float s = simd_sum(acc[r]); if (lane==0) out[row0+r] += s*asc[0]; }
}
template [[host_name("gemv_w4a8_resid_staged2")]] kernel decltype(gemv_w4a8_resid_staged<2>) gemv_w4a8_resid_staged<2>;
template [[host_name("gemv_w4a8_resid_staged4")]] kernel decltype(gemv_w4a8_resid_staged<4>) gemv_w4a8_resid_staged<4>;
// gemv_w4a8_coal_staged (D-B04): gemv_w4a8_resid_staged with gemv_w4a8_coal's epilogue (out = s * asc, no residual):
// the same staged activations, rows per simdgroup, lane-strided words and per-word sum, so bit-identical to
// gemv_w4a8_coal. For the plain coal projections R18 never reached: DeltaNet's qkv and z, the gated shared expert's down.
template <uint R>
kernel void gemv_w4a8_coal_staged(device const uint* bq[[buffer(0)]], device const half* bsc[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], threadgroup char* As [[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    device const uint* aq4 = reinterpret_cast<device const uint*>(aq);
    threadgroup uint* As4 = reinterpret_cast<threadgroup uint*>(As);
    for (uint i=tid;i<(K>>2u);i+=tgs) As4[i]=aq4[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint wpr = K/8u, spr = K/32u;
    uint row0 = (tgid*(tgs>>5u) + sgid)*R;
    float acc[R];
    SA_ROWS_UNROLL for (uint r=0;r<R;r++) acc[r]=0.0f;
    threadgroup const char4* A = reinterpret_cast<threadgroup const char4*>(As);
    for (uint wi = lane; wi < wpr; wi += 32u) {
#ifdef KSC2_OLD_DOWN
        char4 a0 = A[wi*2u], a1 = A[wi*2u+1u];
        SA_ROWS_UNROLL for (uint r=0;r<R;r++) {
            uint x = bq[(row0+r)*wpr + wi];
            int gi = (int((x)&0xF)-8)*int(a0.x) + (int((x>>4)&0xF)-8)*int(a0.y)
                   + (int((x>>8)&0xF)-8)*int(a0.z) + (int((x>>12)&0xF)-8)*int(a0.w)
                   + (int((x>>16)&0xF)-8)*int(a1.x) + (int((x>>20)&0xF)-8)*int(a1.y)
                   + (int((x>>24)&0xF)-8)*int(a1.z) + (int((x>>28)&0xF)-8)*int(a1.w);
            acc[r] += float(gi) * float(bsc[(row0+r)*spr + (wi>>2u)]);
        }
#else
        float4 x0 = float4(A[wi*2u]) * W4M_P4, x1 = float4(A[wi*2u+1u]) * W4M_P4;
        float s8 = 8.0f * (dot(x0, W4M_U4) + dot(x1, W4M_U4));
        SA_ROWS_UNROLL for (uint r=0;r<R;r++) {
            acc[r] += (w4_masked_word(bq[(row0+r)*wpr + wi], x0, x1) - s8) * float(bsc[(row0+r)*spr + (wi>>2u)]);
        }
#endif
    }
    SA_ROWS_UNROLL for (uint r=0;r<R;r++) { float s = simd_sum(acc[r]); if (lane==0) out[row0+r] = s*asc[0]; }
}
template [[host_name("gemv_w4a8_coal_staged2")]] kernel decltype(gemv_w4a8_coal_staged<2>) gemv_w4a8_coal_staged<2>;
template [[host_name("gemv_w4a8_coal_staged4")]] kernel decltype(gemv_w4a8_coal_staged<4>) gemv_w4a8_coal_staged<4>;
