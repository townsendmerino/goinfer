// gemv_w4f16_* — R1: a decode GEMV in the f16-FMA regime MLX and
// llama.cpp use, instead of gemv_w4a8_sa's int8-activation W4A8 regime. NOT bit-identical to the
// shipped path (no int8 activation quantization at all — arguably HIGHER fidelity, not lower;
// still a fidelity-gated lane per the 2026-09-18 owner decision, §4). First slice only: the plain
// dense (non-MoE, non-paged, non-sandwich, non-DeltaNet) QKV/o-proj/gate-up path — the down-proj
// (gemv_w4a8_resid, the "coal" family, a different kernel shape entirely) stays W4A8 for now.
//
// nib2half: the exponent-bias dequant trick named in the brief. A raw 4-bit nibble packed into a
// half's mantissa bits [9:6] with a FIXED exponent field (biased 15 = 2^0) gives the half value
// 1.0 + nibble/16 (nibble<<6 never overflows the 10-bit mantissa: max 15<<6=960<1024) — no
// int-to-float convert instruction. Subtracting 1.5h centers it ((nibble-8)/16) and *16 recovers
// nibble-8 exactly (small integers are exact in half). Verified against a scalar reference for
// all 16 nibble values before this kernel is dispatched anywhere — see gemv_w4f16_test.go.
inline half nib2half(uint nibble) {
    half v = as_type<half>(ushort(0x3C00u | (nibble << 6)));
    return (v - half(1.5h)) * half(16.0h);
}
// UNP8HV: the half-arithmetic analogue of UNP8V above. Each nibble's dequant is promoted to
// float and the 8-term dot for one uint word is accumulated in float — not left in half — so a
// 32-element group's within-group sum does not inherit half's ~3-decimal-digit precision; only
// the per-group SCALE multiply (sr[g], already half in the existing buffer layout, unchanged)
// stays at its existing precision, matching gemv_w4a8_sa's own scale handling exactly.
#define UNP8HV(xw, a4) ( \
    float(nib2half((xw)&0xFu))      *float((a4)[0].x) + float(nib2half(((xw)>>4)&0xFu)) *float((a4)[0].y) \
  + float(nib2half(((xw)>>8)&0xFu)) *float((a4)[0].z) + float(nib2half(((xw)>>12)&0xFu))*float((a4)[0].w) \
  + float(nib2half(((xw)>>16)&0xFu))*float((a4)[1].x) + float(nib2half(((xw)>>20)&0xFu))*float((a4)[1].y) \
  + float(nib2half(((xw)>>24)&0xFu))*float((a4)[1].z) + float(nib2half(((xw)>>28)&0xFu))*float((a4)[1].w) )
// SA_F16_BODY: gemv_w4a8_sa's SA_BODY with the activation staged as half (from a norm producer
// with NO quantization step, e.g. rmsnorm_f16_act) instead of int8, and no separate activation
// scale — the f16 activation values are already the true values; only the per-group WEIGHT
// scale (sr[g]) still applies, exactly as it does in the int8 kernels.
#define SA_F16_BODY \
    for (uint i=tid;i<K;i+=tgs) As[i]=ax[i]; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    uint G = K>>5u; \
    uint row = tgid*(tgs>>5u) + sgid; \
    device const uint4* wr = wq + (uint)row*G; \
    device const half*  sr = sct + (uint)row*G; \
    float acc = 0.0f; \
    for (uint g=lane; g<G; g+=32u) { \
        uint4 w = wr[g]; threadgroup const half4* a4 = reinterpret_cast<threadgroup const half4*>(As + g*32u); \
        float gi = UNP8HV(w.x,a4) + UNP8HV(w.y,a4+2) + UNP8HV(w.z,a4+4) + UNP8HV(w.w,a4+6); \
        acc += gi * float(sr[g]); \
    } \
    acc = simd_sum(acc);
// gemv_w4f16_sa: base, overwrite-out epilogue (gate|up, no bias — Qwen2.5's shape).
kernel void gemv_w4f16_sa(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const half* ax[[buffer(2)]], device float* out[[buffer(3)]],
    constant uint& K[[buffer(4)]], threadgroup half* As [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    SA_F16_BODY
    if (lane==0) out[row] = acc;
}
// gemv_w4f16_sa_bias: additive per-row bias, overwrite-out (QKV's shape).
kernel void gemv_w4f16_sa_bias(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const half* ax[[buffer(2)]], device const float* bias[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], threadgroup half* As [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    SA_F16_BODY
    if (lane==0) out[row] = acc + bias[row];
}
// gemv_w4f16_sa_resid: accumulate into out (o-proj + residual's shape — out IS x at the call site).
kernel void gemv_w4f16_sa_resid(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const half* ax[[buffer(2)]], device float* out[[buffer(3)]],
    constant uint& K[[buffer(4)]], threadgroup half* As [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    SA_F16_BODY
    if (lane==0) out[row] += acc;
}

