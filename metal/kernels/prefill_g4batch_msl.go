package kernels

const MoeBatchKernels = `
// moe_batch_gemv<R,B>: for one routed expert and up to B pairs routed to it (entry = slot, count, B pair ids), the
// expert's rows [slot*rowsPerExpert, +rows) dotted with each pair's int8 activation (pair p reads activation p/actDiv),
// R rows a simdgroup as sa_rows_acc. Every (b, r) accumulator runs sa_rows_acc's operations in sa_rows_acc's order, so
// each output equals the single-activation kernel's bit for bit. scaleOut: out = acc*asc[act] (gate|up, as
// gemv_w4a8_moe_rows), else out = acc (the down projection's raw sum, combined by moe_batch_combine). Grid
// nEntries*tiles threadgroups of 256, tiles = rowsPerExpert/(8R) (full threadgroups only, as sa_rows_acc).
template <uint R, uint B>
kernel void moe_batch_gemv(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq_all[[buffer(2)]], device const float* asc_all[[buffer(3)]], device float* out_all[[buffer(4)]],
    constant uint& K[[buffer(5)]], device const uint* ent[[buffer(6)]], constant uint& rowsPerExpert[[buffer(7)]],
    constant uint& tiles[[buffer(8)]], constant uint& actDiv[[buffer(9)]], constant uint& scaleOut[[buffer(10)]],
    threadgroup half* Ah[[threadgroup(0)]], uint tg_b[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    constexpr float P4[4] = {1.0f, 0.0625f, 0.00390625f, 0.000244140625f};
    constexpr float4 U4 = float4(1.0f, 16.0f, 256.0f, 4096.0f);
    uint e = tg_b / tiles, tgid = tg_b % tiles;
    device const uint* en = ent + e*(2u + B);
    uint slot = en[0], cnt = en[1];
    uint G = K>>5u;
    device const uint4* w0 = wq + slot*rowsPerExpert*G;
    device const half* s0 = sct + slot*rowsPerExpert*G;
    for (uint b=0;b<cnt;b++) {
        device const char* aq = aq_all + (en[2u+b]/actDiv)*K;
        for (uint i=tid;i<K;i+=tgs) Ah[b*K+i] = half(float(aq[i]) * P4[i & 3u]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint row0 = (tgid*(tgs>>5u) + sgid)*R;
    float acc[B][R];
    SA_ROWS_UNROLL for (uint b=0;b<B;b++) SA_ROWS_UNROLL for (uint r=0;r<R;r++) acc[b][r]=0.0f;
    for (uint g=lane; g<G; g+=32u) {
        uint4 w[R];
        SA_ROWS_UNROLL for (uint r=0;r<R;r++) w[r] = w0[(row0+r)*G + g];
        SA_ROWS_UNROLL for (uint b=0;b<B;b++) {
            if (b < cnt) {
                threadgroup const half4* a4 = reinterpret_cast<threadgroup const half4*>(Ah + b*K + g*32u);
                float gf[R];
                SA_ROWS_UNROLL for (uint r=0;r<R;r++) gf[r] = 0.0f;
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
                SA_ROWS_UNROLL for (uint r=0;r<R;r++) acc[b][r] += (gf[r] - 8.0f*sa) * float(s0[(row0+r)*G + g]);
            }
        }
    }
    SA_ROWS_UNROLL for (uint b=0;b<B;b++) SA_ROWS_UNROLL for (uint r=0;r<R;r++) acc[b][r] = simd_sum(acc[b][r]);
    if (lane==0) {
        SA_ROWS_UNROLL for (uint b=0;b<B;b++) {
            if (b < cnt) {
                uint p = en[2u+b];
                device float* out = out_all + p*rowsPerExpert;
                if (scaleOut != 0u) {
                    float sc = asc_all[p/actDiv];
                    SA_ROWS_UNROLL for (uint r=0;r<R;r++) out[row0+r] = acc[b][r]*sc;
                } else {
                    SA_ROWS_UNROLL for (uint r=0;r<R;r++) out[row0+r] = acc[b][r];
                }
            }
        }
    }
}
template [[host_name("moe_batch_gemv2")]] kernel decltype(moe_batch_gemv<2,4>) moe_batch_gemv<2,4>;
template [[host_name("moe_batch_gemv4")]] kernel decltype(moe_batch_gemv<4,4>) moe_batch_gemv<4,4>;

// moe_batch_combine: a row's routed experts into its own H floats, in route order j = 0..k-1, with gemv_w4a8_moe_wacc's
// epilogue fma(wgt*acc, asc, out) starting from zero, as encodeG4Phase2Paged's zero_vec and k wacc dispatches leave it.
// Pair p = row*k + j: wgt[p] its router weight, dacc[p*H + h] its raw down sum, dsc[p] its activation scale.
kernel void moe_batch_combine(device const float* dacc[[buffer(0)]], device const float* wgt[[buffer(1)]],
    device const float* dsc[[buffer(2)]], device float* out[[buffer(3)]], constant uint& H[[buffer(4)]],
    constant uint& k[[buffer(5)]], constant uint& n[[buffer(6)]], uint i[[thread_position_in_grid]]) {
    if (i >= n*H) return;
    uint row = i / H, h = i % H;
    float o = 0.0f;
    for (uint j=0;j<k;j++) {
        uint p = row*k + j;
        o = fma(wgt[p]*dacc[p*H + h], dsc[p], o);
    }
    out[i] = o;
}

// moe_batch_combine_acc (KS-D9): moe_batch_combine for the generic pager, whose per-row phase 2 accumulates each routed
// expert straight into the row's residual: out holds the rows' residuals [n x H] on entry, and each takes its k experts in route
// order with gemv_w4a8_moe_wacc's epilogue fma(wgt*acc, asc, out).
kernel void moe_batch_combine_acc(device const float* dacc[[buffer(0)]], device const float* wgt[[buffer(1)]],
    device const float* dsc[[buffer(2)]], device float* out[[buffer(3)]], constant uint& H[[buffer(4)]],
    constant uint& k[[buffer(5)]], constant uint& n[[buffer(6)]], uint i[[thread_position_in_grid]]) {
    if (i >= n*H) return;
    uint row = i / H, h = i % H;
    float o = out[i];
    for (uint j=0;j<k;j++) {
        uint p = row*k + j;
        o = fma(wgt[p]*dacc[p*H + h], dsc[p], o);
    }
    out[i] = o;
}
`
