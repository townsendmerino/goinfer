//go:build darwin && goinfer_testhooks

package metal

import "strings"

// ksc1OldKernels is the q4k lane's GEMVs as they stood before KS-C1 (one row a simdgroup; gemv_q4k_f32, gemv_w8_f32 and their
// residual and bias variants), the old arm of KS-C1's tests. ksc1OldLib puts them back where they stood in allKernels: under
// fast math the compiler's contraction of the residual and bias epilogues depends on the surrounding source, so appending them
// elsewhere compiles different bits.
const ksc1OldKernels = `
// Per 256 weights (144 bytes): d (f16), dmin (f16), scales[12], qs[128]; w = d*sc*q - dmin*m per 32-weight sub-block;
// byte qs[32j+l] holds element 64j+l (low nibble) and 64j+32+l (high). Lane l takes byte l of each 32-byte group.
#define Q4K_F32_BODY \
    uint row = tgid*8u + sgid; \
    if (row >= N) return; \
    uint NB = K >> 8u; \
    device const uchar* wr = wq + (ulong)row*(ulong)NB*144ul; \
    float acc = 0.0f; \
    for (uint b=0u; b<NB; b++) { \
        device const uchar* blk = wr + b*144u; \
        float d  = float(as_type<half>(ushort(uint(blk[0]) | (uint(blk[1])<<8u)))); \
        float dm = float(as_type<half>(ushort(uint(blk[2]) | (uint(blk[3])<<8u)))); \
        device const uchar* scs = blk + 4; device const uchar* qs = blk + 16; \
        device const float* a = ax + b*256u; \
        for (uint j=0u; j<4u; j++) { \
            float sc0, m0, sc1, m1; q4k_scale_min(2u*j, scs, sc0, m0); q4k_scale_min(2u*j+1u, scs, sc1, m1); \
            uint q = uint(qs[j*32u + lane]); \
            float a0 = a[j*64u + lane], a1 = a[j*64u + 32u + lane]; \
            acc += (d*sc0*float(q & 15u) - dm*m0) * a0 + (d*sc1*float(q >> 4u) - dm*m1) * a1; \
        } \
    } \
    acc = simd_sum(acc);
kernel void gemv_q4k_f32(device const uchar* wq[[buffer(0)]], device const float* ax[[buffer(1)]],
    device float* out[[buffer(2)]], constant uint& K[[buffer(3)]], constant uint& N[[buffer(4)]],
    uint tgid[[threadgroup_position_in_grid]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]]) {
    Q4K_F32_BODY
    if (lane==0) out[row] = acc;
}
kernel void gemv_q4k_f32_resid(device const uchar* wq[[buffer(0)]], device const float* ax[[buffer(1)]],
    device float* out[[buffer(2)]], constant uint& K[[buffer(3)]], constant uint& N[[buffer(4)]],
    uint tgid[[threadgroup_position_in_grid]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]]) {
    Q4K_F32_BODY
    if (lane==0) out[row] += acc;
}
kernel void gemv_q4k_f32_bias(device const uchar* wq[[buffer(0)]], device const float* ax[[buffer(1)]],
    device float* out[[buffer(2)]], constant uint& K[[buffer(3)]], constant uint& N[[buffer(4)]],
    device const float* bias[[buffer(5)]], uint tgid[[threadgroup_position_in_grid]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    Q4K_F32_BODY
    if (lane==0) out[row] = acc + bias[row];
}
// int8 weights (per-row scale) over f32 activations. K a multiple of 4.
#define W8_F32_BODY \
    uint row = tgid*8u + sgid; \
    if (row >= N) return; \
    device const char4* wr = (device const char4*)(wq + (ulong)row*(ulong)K); \
    device const float4* a4 = (device const float4*)ax; \
    float acc = 0.0f; \
    for (uint g=lane; g<(K>>2u); g+=32u) acc += dot(float4(wr[g]), a4[g]); \
    acc = simd_sum(acc) * sc[row];
kernel void gemv_w8_f32(device const char* wq[[buffer(0)]], device const float* sc[[buffer(1)]],
    device const float* ax[[buffer(2)]], device float* out[[buffer(3)]], constant uint& K[[buffer(4)]],
    constant uint& N[[buffer(5)]], uint tgid[[threadgroup_position_in_grid]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    W8_F32_BODY
    if (lane==0) out[row] = acc;
}
kernel void gemv_w8_f32_resid(device const char* wq[[buffer(0)]], device const float* sc[[buffer(1)]],
    device const float* ax[[buffer(2)]], device float* out[[buffer(3)]], constant uint& K[[buffer(4)]],
    constant uint& N[[buffer(5)]], uint tgid[[threadgroup_position_in_grid]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    W8_F32_BODY
    if (lane==0) out[row] += acc;
}
kernel void gemv_w8_f32_bias(device const char* wq[[buffer(0)]], device const float* sc[[buffer(1)]],
    device const float* ax[[buffer(2)]], device float* out[[buffer(3)]], constant uint& K[[buffer(4)]],
    constant uint& N[[buffer(5)]], device const float* bias[[buffer(6)]], uint tgid[[threadgroup_position_in_grid]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    W8_F32_BODY
    if (lane==0) out[row] = acc + bias[row];
}

`

// ksc1OldLib is allKernels with ksc1OldKernels at their former place, just before KS-C1's rows kernels.
func ksc1OldLib() string {
	const at = "// KS-C1: the q4k lane's GEMVs at R rows a simdgroup"
	i := strings.Index(allKernels, at)
	if i < 0 {
		panic("ksc1OldLib: the rows kernels' comment moved")
	}
	return allKernels[:i] + ksc1OldKernels[1:] + allKernels[i:]
}
