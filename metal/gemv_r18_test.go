//go:build darwin

package metal

import (
	"fmt"
	"math"
	"os"
	"sort"
	"testing"
	"time"
)

// r18Kernels are R18's prototypes (docs/tasks/red-october.md R18; docs/measurements/metal-decode-gemv-s0-2026-09-26.md),
// TEST-ONLY. Step 0: gemv_w4a8_sa with R rows per simdgroup — each lane reads its group's 32 staged activations ONCE into
// registers and reuses them for R rows; the integer math (UNP8V) and, per row, the lane-to-group assignment and the
// float(gi)*scale accumulation order are the shipped kernel's, so the output is bit-identical by construction.
// Instantiated for R = 1, 2, 4 with the SA family's three epilogues (base = overwrite, bias, resid).
const r18Kernels = `
#define R18_UNROLL _Pragma("clang loop unroll(full)")

template <uint R, uint EPI>   // EPI: 0 overwrite, 1 + bias, 2 += (residual)
kernel void gemv_w4a8_sa_rr(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], device const float* bias[[buffer(6)]], threadgroup short* As [[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    for (uint i=tid;i<K;i+=tgs) As[i]=short(aq[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint G = K>>5u;
    uint row0 = (tgid*(tgs>>5u) + sgid)*R;
    float acc[R];
    R18_UNROLL for (uint r=0;r<R;r++) acc[r]=0.0f;
    for (uint g=lane; g<G; g+=32u) {
        threadgroup const short4* a4s = reinterpret_cast<threadgroup const short4*>(As + g*32u);
        short4 a[8];
        R18_UNROLL for (uint j=0;j<8u;j++) a[j]=a4s[j];
        R18_UNROLL for (uint r=0;r<R;r++) {
            uint4 w = wq[(row0+r)*G + g];
            int gi = UNP8V(w.x,a) + UNP8V(w.y,a+2) + UNP8V(w.z,a+4) + UNP8V(w.w,a+6);
            acc[r] += float(gi) * float(sct[(row0+r)*G + g]);
        }
    }
    R18_UNROLL for (uint r=0;r<R;r++) {
        float s = simd_sum(acc[r]);
        if (lane==0) {
            if (EPI == 0u) out[row0+r] = s*asc[0];
            else if (EPI == 1u) out[row0+r] = s*asc[0] + bias[row0+r];
            else out[row0+r] += s*asc[0];
        }
    }
}
// Step 1: the shift-free f32 form (MLX qmv's). Activations staged once per threadgroup as f32 PRE-SCALED by
// 16^-j for nibble slot j of their word (k mod 8), plus 8*sum(a) per group; a nibble then costs an AND, a convert
// and an FMA: float(w & (0xF << 4j)) * (a / 16^j) == n_j * a_j exactly. Every product and partial sum is an integer
// below 2^24, so the group sum equals the shipped integer gi exactly, and float(gi)*scale accumulates per row in the
// shipped lane order -> bit-identical by construction.
#define R18_DOTM(wd, xa, xb) ( \
    float((wd)&0xFu)*(xa).x + float((wd)&0xF0u)*(xa).y + float((wd)&0xF00u)*(xa).z + float((wd)&0xF000u)*(xa).w \
  + float((wd)&0xF0000u)*(xb).x + float((wd)&0xF00000u)*(xb).y + float((wd)&0xF000000u)*(xb).z + float((wd)&0xF0000000u)*(xb).w )

template <uint R, uint EPI>
kernel void gemv_w4a8_sa_fm(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], device const float* bias[[buffer(6)]], threadgroup float* sh [[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    uint G = K>>5u;
    for (uint i=tid;i<K;i+=tgs) sh[i] = float(aq[i]) * exp2(-4.0f*float(i&7u));
    for (uint g=tid;g<G;g+=tgs) {
        int sa = 0;
        for (uint i=0;i<32u;i++) sa += int(aq[g*32u+i]);
        sh[K+g] = 8.0f*float(sa);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint row0 = (tgid*(tgs>>5u) + sgid)*R;
    float acc[R];
    R18_UNROLL for (uint r=0;r<R;r++) acc[r]=0.0f;
    for (uint g=lane; g<G; g+=32u) {
        threadgroup const float4* xs = reinterpret_cast<threadgroup const float4*>(sh + g*32u);
        float4 x[8];
        R18_UNROLL for (uint j=0;j<8u;j++) x[j]=xs[j];
        float sa8 = sh[K+g];
        R18_UNROLL for (uint r=0;r<R;r++) {
            uint4 w = wq[(row0+r)*G + g];
            float gi = (R18_DOTM(w.x,x[0],x[1]) + R18_DOTM(w.y,x[2],x[3]) + R18_DOTM(w.z,x[4],x[5]) + R18_DOTM(w.w,x[6],x[7])) - sa8;
            acc[r] += gi * float(sct[(row0+r)*G + g]);
        }
    }
    R18_UNROLL for (uint r=0;r<R;r++) {
        float s = simd_sum(acc[r]);
        if (lane==0) {
            if (EPI == 0u) out[row0+r] = s*asc[0];
            else if (EPI == 1u) out[row0+r] = s*asc[0] + bias[row0+r];
            else out[row0+r] += s*asc[0];
        }
    }
}
// Step 1, variant B (integer, exact): R rows, activations in registers, the -8 folded per group:
// gi = sum(n*a) - 8*sum(a), sum(a) summed once per group in registers and reused for the R rows.
#define R18_UNPN(xw, a4) ( \
    int((xw)&0xF)*int((a4)[0].x) + int(((xw)>>4)&0xF)*int((a4)[0].y) \
  + int(((xw)>>8)&0xF)*int((a4)[0].z) + int(((xw)>>12)&0xF)*int((a4)[0].w) \
  + int(((xw)>>16)&0xF)*int((a4)[1].x) + int(((xw)>>20)&0xF)*int((a4)[1].y) \
  + int(((xw)>>24)&0xF)*int((a4)[1].z) + int(((xw)>>28)&0xF)*int((a4)[1].w) )
template <uint R>
kernel void gemv_w4a8_sa_ib(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], device const float* bias[[buffer(6)]], threadgroup short* As [[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    for (uint i=tid;i<K;i+=tgs) As[i]=short(aq[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint G = K>>5u;
    uint row0 = (tgid*(tgs>>5u) + sgid)*R;
    float acc[R];
    R18_UNROLL for (uint r=0;r<R;r++) acc[r]=0.0f;
    for (uint g=lane; g<G; g+=32u) {
        threadgroup const short4* a4s = reinterpret_cast<threadgroup const short4*>(As + g*32u);
        short4 a[8];
        int sa = 0;
        R18_UNROLL for (uint j=0;j<8u;j++) { a[j]=a4s[j]; sa += int(a[j].x)+int(a[j].y)+int(a[j].z)+int(a[j].w); }
        int sa8 = 8*sa;
        R18_UNROLL for (uint r=0;r<R;r++) {
            uint4 w = wq[(row0+r)*G + g];
            int gi = R18_UNPN(w.x,a) + R18_UNPN(w.y,a+2) + R18_UNPN(w.z,a+4) + R18_UNPN(w.w,a+6) - sa8;
            acc[r] += float(gi) * float(sct[(row0+r)*G + g]);
        }
    }
    R18_UNROLL for (uint r=0;r<R;r++) { float s = simd_sum(acc[r]); if (lane==0) out[row0+r] = s*asc[0]; }
}
// Step 1, variant C (float, shift-free): R rows, pre-scaling IN REGISTERS from the staged shorts with exact
// power-of-two constants (1, 2^-4, ..., 2^-28), -8 folded per group, no float staging.
constant float R18_P[8] = {1.0f, 0.0625f, 0.00390625f, 0.000244140625f, 1.52587890625e-05f,
    9.5367431640625e-07f, 5.9604644775390625e-08f, 3.7252902984619140625e-09f};
template <uint R>
kernel void gemv_w4a8_sa_fr(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], device const float* bias[[buffer(6)]], threadgroup short* As [[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    for (uint i=tid;i<K;i+=tgs) As[i]=short(aq[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint G = K>>5u;
    uint row0 = (tgid*(tgs>>5u) + sgid)*R;
    float acc[R];
    R18_UNROLL for (uint r=0;r<R;r++) acc[r]=0.0f;
    for (uint g=lane; g<G; g+=32u) {
        threadgroup const short4* a4s = reinterpret_cast<threadgroup const short4*>(As + g*32u);
        float4 x[8];
        int sa = 0;
        R18_UNROLL for (uint j=0;j<8u;j++) {
            short4 v = a4s[j];
            sa += int(v.x)+int(v.y)+int(v.z)+int(v.w);
            uint b = (j&1u)*4u;
            x[j] = float4(float(v.x)*R18_P[b], float(v.y)*R18_P[b+1u], float(v.z)*R18_P[b+2u], float(v.w)*R18_P[b+3u]);
        }
        float sa8 = float(8*sa);
        R18_UNROLL for (uint r=0;r<R;r++) {
            uint4 w = wq[(row0+r)*G + g];
            float gi = (R18_DOTM(w.x,x[0],x[1]) + R18_DOTM(w.y,x[2],x[3]) + R18_DOTM(w.z,x[4],x[5]) + R18_DOTM(w.w,x[6],x[7])) - sa8;
            acc[r] += gi * float(sct[(row0+r)*G + g]);
        }
    }
    R18_UNROLL for (uint r=0;r<R;r++) { float s = simd_sum(acc[r]); if (lane==0) out[row0+r] = s*asc[0]; }
}
// Variant U: ib (int, -8 folded) with the group loop unrolled by 2 — lane l issues the weight loads of groups g and
// g+32 for all R rows before computing, then accumulates g then g+32 per row (the shipped order) -> bit-identical.
template <uint R>
kernel void gemv_w4a8_sa_iu(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], device const float* bias[[buffer(6)]], threadgroup short* As [[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    for (uint i=tid;i<K;i+=tgs) As[i]=short(aq[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint G = K>>5u;
    uint row0 = (tgid*(tgs>>5u) + sgid)*R;
    float acc[R];
    R18_UNROLL for (uint r=0;r<R;r++) acc[r]=0.0f;
    uint g = lane;
    for (; g+32u < G; g += 64u) {
        uint4 w0[R], w1[R]; float s0[R], s1[R];
        R18_UNROLL for (uint r=0;r<R;r++) {
            w0[r] = wq[(row0+r)*G + g]; w1[r] = wq[(row0+r)*G + g+32u];
            s0[r] = float(sct[(row0+r)*G + g]); s1[r] = float(sct[(row0+r)*G + g+32u]);
        }
        threadgroup const short4* p0 = reinterpret_cast<threadgroup const short4*>(As + g*32u);
        threadgroup const short4* p1 = reinterpret_cast<threadgroup const short4*>(As + (g+32u)*32u);
        short4 a[8], b[8]; int sa = 0, sb = 0;
        R18_UNROLL for (uint j=0;j<8u;j++) {
            a[j]=p0[j]; b[j]=p1[j];
            sa += int(a[j].x)+int(a[j].y)+int(a[j].z)+int(a[j].w);
            sb += int(b[j].x)+int(b[j].y)+int(b[j].z)+int(b[j].w);
        }
        R18_UNROLL for (uint r=0;r<R;r++) {
            int gi = R18_UNPN(w0[r].x,a) + R18_UNPN(w0[r].y,a+2) + R18_UNPN(w0[r].z,a+4) + R18_UNPN(w0[r].w,a+6) - 8*sa;
            acc[r] += float(gi) * s0[r];
            int gj = R18_UNPN(w1[r].x,b) + R18_UNPN(w1[r].y,b+2) + R18_UNPN(w1[r].z,b+4) + R18_UNPN(w1[r].w,b+6) - 8*sb;
            acc[r] += float(gj) * s1[r];
        }
    }
    if (g < G) {
        threadgroup const short4* p0 = reinterpret_cast<threadgroup const short4*>(As + g*32u);
        short4 a[8]; int sa = 0;
        R18_UNROLL for (uint j=0;j<8u;j++) { a[j]=p0[j]; sa += int(a[j].x)+int(a[j].y)+int(a[j].z)+int(a[j].w); }
        R18_UNROLL for (uint r=0;r<R;r++) {
            uint4 w = wq[(row0+r)*G + g];
            int gi = R18_UNPN(w.x,a) + R18_UNPN(w.y,a+2) + R18_UNPN(w.z,a+4) + R18_UNPN(w.w,a+6) - 8*sa;
            acc[r] += float(gi) * float(sct[(row0+r)*G + g]);
        }
    }
    R18_UNROLL for (uint r=0;r<R;r++) { float s = simd_sum(acc[r]); if (lane==0) out[row0+r] = s*asc[0]; }
}
typedef decltype(gemv_w4a8_sa_iu<4>) gemv_w4a8_sa_iu_t;
template [[host_name("gemv_w4a8_sa_iu2")]] kernel gemv_w4a8_sa_iu_t gemv_w4a8_sa_iu<2>;
template [[host_name("gemv_w4a8_sa_iu4")]] kernel gemv_w4a8_sa_iu_t gemv_w4a8_sa_iu<4>;

typedef decltype(gemv_w4a8_sa_ib<4>) gemv_w4a8_sa_ib_t;
template [[host_name("gemv_w4a8_sa_ib8")]] kernel gemv_w4a8_sa_ib_t gemv_w4a8_sa_ib<8>;
template [[host_name("gemv_w4a8_sa_ib4")]] kernel gemv_w4a8_sa_ib_t gemv_w4a8_sa_ib<4>;
template [[host_name("gemv_w4a8_sa_ib2")]] kernel gemv_w4a8_sa_ib_t gemv_w4a8_sa_ib<2>;
typedef decltype(gemv_w4a8_sa_fr<4>) gemv_w4a8_sa_fr_t;
template [[host_name("gemv_w4a8_sa_fr4")]] kernel gemv_w4a8_sa_fr_t gemv_w4a8_sa_fr<4>;
template [[host_name("gemv_w4a8_sa_fr2")]] kernel gemv_w4a8_sa_fr_t gemv_w4a8_sa_fr<2>;

// Step 2: the down projection (coal family: gemv_w4a8_resid, one simdgroup per row, lane-strided WORDS, per-word
// float(gi)*scale). Activations staged ONCE per threadgroup as int8 (K <= ~32 K fits), read per word as two char4
// into registers and reused for R rows; lane l still owns words l, l+32, ... of each row in order, with the per-word
// integer sum and accumulation of W4A8_BODY -> bit-identical to gemv_w4a8_resid.
template <uint R>
kernel void gemv_w4a8_resid_st(device const uint* bq[[buffer(0)]], device const half* bsc[[buffer(1)]],
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
    R18_UNROLL for (uint r=0;r<R;r++) acc[r]=0.0f;
    threadgroup const char4* A = reinterpret_cast<threadgroup const char4*>(As);
    for (uint wi = lane; wi < wpr; wi += 32u) {
        char4 a0 = A[wi*2u], a1 = A[wi*2u+1u];
        R18_UNROLL for (uint r=0;r<R;r++) {
            uint x = bq[(row0+r)*wpr + wi];
            int gi = (int((x)&0xF)-8)*int(a0.x) + (int((x>>4)&0xF)-8)*int(a0.y)
                   + (int((x>>8)&0xF)-8)*int(a0.z) + (int((x>>12)&0xF)-8)*int(a0.w)
                   + (int((x>>16)&0xF)-8)*int(a1.x) + (int((x>>20)&0xF)-8)*int(a1.y)
                   + (int((x>>24)&0xF)-8)*int(a1.z) + (int((x>>28)&0xF)-8)*int(a1.w);
            acc[r] += float(gi) * float(bsc[(row0+r)*spr + (wi>>2u)]);
        }
    }
    R18_UNROLL for (uint r=0;r<R;r++) { float s = simd_sum(acc[r]); if (lane==0) out[row0+r] += s*asc[0]; }
}
typedef decltype(gemv_w4a8_resid_st<1>) gemv_w4a8_resid_st_t;
template [[host_name("gemv_w4a8_resid_st1")]] kernel gemv_w4a8_resid_st_t gemv_w4a8_resid_st<1>;
template [[host_name("gemv_w4a8_resid_st2")]] kernel gemv_w4a8_resid_st_t gemv_w4a8_resid_st<2>;
template [[host_name("gemv_w4a8_resid_st4")]] kernel gemv_w4a8_resid_st_t gemv_w4a8_resid_st<4>;
template [[host_name("gemv_w4a8_resid_st8")]] kernel gemv_w4a8_resid_st_t gemv_w4a8_resid_st<8>;

typedef decltype(gemv_w4a8_sa_fm<1,0>) gemv_w4a8_sa_fm_t;
template [[host_name("gemv_w4a8_sa_fm1")]] kernel gemv_w4a8_sa_fm_t gemv_w4a8_sa_fm<1,0>;
template [[host_name("gemv_w4a8_sa_fm2")]] kernel gemv_w4a8_sa_fm_t gemv_w4a8_sa_fm<2,0>;
template [[host_name("gemv_w4a8_sa_fm4")]] kernel gemv_w4a8_sa_fm_t gemv_w4a8_sa_fm<4,0>;
template [[host_name("gemv_w4a8_sa_fm8")]] kernel gemv_w4a8_sa_fm_t gemv_w4a8_sa_fm<8,0>;

typedef decltype(gemv_w4a8_sa_rr<1,0>) gemv_w4a8_sa_rr_t;
template [[host_name("gemv_w4a8_sa_r1")]] kernel gemv_w4a8_sa_rr_t gemv_w4a8_sa_rr<1,0>;
template [[host_name("gemv_w4a8_sa_r2")]] kernel gemv_w4a8_sa_rr_t gemv_w4a8_sa_rr<2,0>;
template [[host_name("gemv_w4a8_sa_r4")]] kernel gemv_w4a8_sa_rr_t gemv_w4a8_sa_rr<4,0>;
template [[host_name("gemv_w4a8_sa_bias_r2")]] kernel gemv_w4a8_sa_rr_t gemv_w4a8_sa_rr<2,1>;
template [[host_name("gemv_w4a8_sa_bias_r4")]] kernel gemv_w4a8_sa_rr_t gemv_w4a8_sa_rr<4,1>;
template [[host_name("gemv_w4a8_sa_resid_r2")]] kernel gemv_w4a8_sa_rr_t gemv_w4a8_sa_rr<2,2>;
template [[host_name("gemv_w4a8_sa_resid_r4")]] kernel gemv_w4a8_sa_rr_t gemv_w4a8_sa_rr<4,2>;

// Variant D (MLX's activation path): no threadgroup staging at all — each lane reads its group's 32 int8 activations
// straight from device memory (1.5-3.5 KB, L1/SLC-hot) as two uint4, reinterpreted as char4 (free), and reuses them
// for R rows. F = 0: UNP8V integer math; F = 1: exact power-of-two pre-scaled floats, 32-bit masks; F = 2: 16-bit masks
// (MLX's form: ushort & 0x000f/0x00f0/0x0f00/0xf000 against a, a/16, a/256, a/4096). All bit-identical.
constant float R18_Q[4] = {1.0f, 0.0625f, 0.00390625f, 0.000244140625f};
template <uint R, uint F>
kernel void gemv_w4a8_sa_dv(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], device const float* bias[[buffer(6)]],
    uint tgid[[threadgroup_position_in_grid]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    uint G = K>>5u;
    uint row0 = (tgid*(tgs>>5u) + sgid)*R;
    device const uint4* a16 = reinterpret_cast<device const uint4*>(aq);
    float acc[R];
    R18_UNROLL for (uint r=0;r<R;r++) acc[r]=0.0f;
    for (uint g=lane; g<G; g+=32u) {
        uint4 u0 = a16[2u*g], u1 = a16[2u*g+1u];
        char4 a[8] = {as_type<char4>(u0.x), as_type<char4>(u0.y), as_type<char4>(u0.z), as_type<char4>(u0.w),
                      as_type<char4>(u1.x), as_type<char4>(u1.y), as_type<char4>(u1.z), as_type<char4>(u1.w)};
        if (F == 0u) {
            R18_UNROLL for (uint r=0;r<R;r++) {
                uint4 w = wq[(row0+r)*G + g];
                int gi = UNP8V(w.x,a) + UNP8V(w.y,a+2) + UNP8V(w.z,a+4) + UNP8V(w.w,a+6);
                acc[r] += float(gi) * float(sct[(row0+r)*G + g]);
            }
        } else if (F == 1u) {
            float4 x[8]; int sa = 0;
            R18_UNROLL for (uint j=0;j<8u;j++) {
                char4 v = a[j];
                sa += int(v.x)+int(v.y)+int(v.z)+int(v.w);
                uint b = (j&1u)*4u;
                x[j] = float4(float(v.x)*R18_P[b], float(v.y)*R18_P[b+1u], float(v.z)*R18_P[b+2u], float(v.w)*R18_P[b+3u]);
            }
            float sa8 = float(8*sa);
            R18_UNROLL for (uint r=0;r<R;r++) {
                uint4 w = wq[(row0+r)*G + g];
                float gi = (R18_DOTM(w.x,x[0],x[1]) + R18_DOTM(w.y,x[2],x[3]) + R18_DOTM(w.z,x[4],x[5]) + R18_DOTM(w.w,x[6],x[7])) - sa8;
                acc[r] += gi * float(sct[(row0+r)*G + g]);
            }
        } else {
            float4 x[8]; int sa = 0;
            R18_UNROLL for (uint j=0;j<8u;j++) {
                char4 v = a[j];
                sa += int(v.x)+int(v.y)+int(v.z)+int(v.w);
                x[j] = float4(float(v.x)*R18_Q[0], float(v.y)*R18_Q[1], float(v.z)*R18_Q[2], float(v.w)*R18_Q[3]);
            }
            float sa8 = float(8*sa);
            R18_UNROLL for (uint r=0;r<R;r++) {
                uint4 w = wq[(row0+r)*G + g];
                ushort2 h[4] = {as_type<ushort2>(w.x), as_type<ushort2>(w.y), as_type<ushort2>(w.z), as_type<ushort2>(w.w)};
                float gi = -sa8;
                float part = 0.0f;
                R18_UNROLL for (uint k=0;k<4u;k++) {
                    ushort lo = h[k].x, hi = h[k].y;
                    float4 xa = x[2u*k], xb = x[2u*k+1u];
                    part += float(lo&0x000Fu)*xa.x + float(lo&0x00F0u)*xa.y + float(lo&0x0F00u)*xa.z + float(lo&0xF000u)*xa.w
                          + float(hi&0x000Fu)*xb.x + float(hi&0x00F0u)*xb.y + float(hi&0x0F00u)*xb.z + float(hi&0xF000u)*xb.w;
                }
                gi += part;
                acc[r] += gi * float(sct[(row0+r)*G + g]);
            }
        }
    }
    R18_UNROLL for (uint r=0;r<R;r++) { float s = simd_sum(acc[r]); if (lane==0) out[row0+r] = s*asc[0]; }
}
typedef decltype(gemv_w4a8_sa_dv<4,0>) gemv_w4a8_sa_dv_t;
template [[host_name("gemv_w4a8_sa_dvi2")]] kernel gemv_w4a8_sa_dv_t gemv_w4a8_sa_dv<2,0>;
template [[host_name("gemv_w4a8_sa_dvi4")]] kernel gemv_w4a8_sa_dv_t gemv_w4a8_sa_dv<4,0>;
template [[host_name("gemv_w4a8_sa_dvf4")]] kernel gemv_w4a8_sa_dv_t gemv_w4a8_sa_dv<4,1>;
template [[host_name("gemv_w4a8_sa_dvs4")]] kernel gemv_w4a8_sa_dv_t gemv_w4a8_sa_dv<4,2>;

// In-sequence candidates (TestR18InSequence): the SA family's three kernels with the PRODUCTION buffer signatures
// (gemv_w4a8_sa / _bias — bias at 5, K at 6 — / _resid), so they drop into the shipped dispatch sites. F = 0 is
// step 0 (UNP8V integer math, unchanged); F = 1 is step 1 (shift-free f32 with exact power-of-two constants, -8*sum(a)
// folded per group). Both keep each row's lane-to-group order and float(gi)*scale accumulation -> bit-identical.
template <uint R, uint F>
inline void r18_sa_acc(device const uint4* wq, device const half* sct, device const char* aq, uint K,
    threadgroup short* As, uint tgid, uint tid, uint tgs, uint sgid, uint lane, thread float* acc, thread uint& row0) {
    for (uint i=tid;i<K;i+=tgs) As[i]=short(aq[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint G = K>>5u;
    row0 = (tgid*(tgs>>5u) + sgid)*R;
    R18_UNROLL for (uint r=0;r<R;r++) acc[r]=0.0f;
    for (uint g=lane; g<G; g+=32u) {
        threadgroup const short4* a4s = reinterpret_cast<threadgroup const short4*>(As + g*32u);
        if (F == 0u) {
            short4 a[8];
            R18_UNROLL for (uint j=0;j<8u;j++) a[j]=a4s[j];
            R18_UNROLL for (uint r=0;r<R;r++) {
                uint4 w = wq[(row0+r)*G + g];
                int gi = UNP8V(w.x,a) + UNP8V(w.y,a+2) + UNP8V(w.z,a+4) + UNP8V(w.w,a+6);
                acc[r] += float(gi) * float(sct[(row0+r)*G + g]);
            }
        } else {
            float4 x[8];
            int sa = 0;
            R18_UNROLL for (uint j=0;j<8u;j++) {
                short4 v = a4s[j];
                sa += int(v.x)+int(v.y)+int(v.z)+int(v.w);
                uint b = (j&1u)*4u;
                x[j] = float4(float(v.x)*R18_P[b], float(v.y)*R18_P[b+1u], float(v.z)*R18_P[b+2u], float(v.w)*R18_P[b+3u]);
            }
            float sa8 = float(8*sa);
            R18_UNROLL for (uint r=0;r<R;r++) {
                uint4 w = wq[(row0+r)*G + g];
                float gi = (R18_DOTM(w.x,x[0],x[1]) + R18_DOTM(w.y,x[2],x[3]) + R18_DOTM(w.z,x[4],x[5]) + R18_DOTM(w.w,x[6],x[7])) - sa8;
                acc[r] += gi * float(sct[(row0+r)*G + g]);
            }
        }
    }
    R18_UNROLL for (uint r=0;r<R;r++) acc[r] = simd_sum(acc[r]);
}
template <uint R, uint F>
kernel void r18_sa(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], threadgroup short* As [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    float acc[R]; uint row0;
    r18_sa_acc<R,F>(wq, sct, aq, K, As, tgid, tid, tgs, sgid, lane, acc, row0);
    if (lane==0) { R18_UNROLL for (uint r=0;r<R;r++) out[row0+r] = acc[r]*asc[0]; }
}
template <uint R, uint F>
kernel void r18_sa_bias(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    device const float* bias[[buffer(5)]], constant uint& K[[buffer(6)]], threadgroup short* As [[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    float acc[R]; uint row0;
    r18_sa_acc<R,F>(wq, sct, aq, K, As, tgid, tid, tgs, sgid, lane, acc, row0);
    if (lane==0) { R18_UNROLL for (uint r=0;r<R;r++) out[row0+r] = acc[r]*asc[0] + bias[row0+r]; }
}
template <uint R, uint F>
kernel void r18_sa_resid(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], threadgroup short* As [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    float acc[R]; uint row0;
    r18_sa_acc<R,F>(wq, sct, aq, K, As, tgid, tid, tgs, sgid, lane, acc, row0);
    if (lane==0) { R18_UNROLL for (uint r=0;r<R;r++) out[row0+r] += acc[r]*asc[0]; }
}
// R18b candidates (docs/tasks/red-october.md R18b): the SA rows kernels in MLX's masked, half-staged form, production
// signatures. Activations staged as HALF pre-scaled by 16^-(k mod 4) — exact, |a| <= 127 and 127*2^-12 is a normal half
// — in the same K*2 bytes the shipped short staging uses. Each lane takes its group's sum(a), for the -8 fold, from the
// staged halves it already reads (one exact dot per half-word), so neither extra threadgroup memory nor a device
// re-read is needed. (The first cut re-read the int8 activations from device per lane: 0.94x / 0.98x in sequence.) Weight nibbles are masked in place per
// 16-bit half-word, converted once per (row, half-word), and dotted with the four pre-scaled activations. Every product
// and partial sum is an integer below 2^24, so the group sum equals the shipped integer gi exactly, and float(gi)*scale
// accumulates per row in the shipped lane order -> bit-identical to gemv_w4a8_sa_rows.
template <uint R>
inline void r18_sa_h_acc(device const uint4* wq, device const half* sct, device const char* aq, uint K,
    threadgroup half* Ah, uint tgid, uint tid, uint tgs, uint sgid, uint lane, thread float* acc, thread uint& row0) {
    constexpr float P4[4] = {1.0f, 0.0625f, 0.00390625f, 0.000244140625f};
    for (uint i=tid;i<K;i+=tgs) Ah[i] = half(float(aq[i]) * P4[i & 3u]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint G = K>>5u;
    row0 = (tgid*(tgs>>5u) + sgid)*R;
    R18_UNROLL for (uint r=0;r<R;r++) acc[r]=0.0f;
    constexpr float4 U4 = float4(1.0f, 16.0f, 256.0f, 4096.0f); // undoes the pre-scale: exact, the result is a
    for (uint g=lane; g<G; g+=32u) {
        float gf[R];
        uint4 w[R];
        R18_UNROLL for (uint r=0;r<R;r++) { gf[r] = 0.0f; w[r] = wq[(row0+r)*G + g]; }
        threadgroup const half4* a4 = reinterpret_cast<threadgroup const half4*>(Ah + g*32u);
        float sa = 0.0f; // sum(a) over the group, exact (integers below 2^24), from the staged halves already read
        R18_UNROLL for (uint t=0;t<8u;t++) {
            float4 x = float4(a4[t]);
            sa += dot(x, U4);
            R18_UNROLL for (uint r=0;r<R;r++) {
                uint word = (t < 2u) ? w[r].x : (t < 4u) ? w[r].y : (t < 6u) ? w[r].z : w[r].w;
                uint u = (t & 1u) ? (word >> 16) : (word & 0xFFFFu);
                gf[r] += dot(float4(float(u & 0xFu), float(u & 0xF0u), float(u & 0xF00u), float(u & 0xF000u)), x);
            }
        }
        R18_UNROLL for (uint r=0;r<R;r++) acc[r] += (gf[r] - 8.0f*sa) * float(sct[(row0+r)*G + g]);
    }
    R18_UNROLL for (uint r=0;r<R;r++) acc[r] = simd_sum(acc[r]);
}
template <uint R>
kernel void r18_sa_h(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], threadgroup half* Ah [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    float acc[R]; uint row0;
    r18_sa_h_acc<R>(wq, sct, aq, K, Ah, tgid, tid, tgs, sgid, lane, acc, row0);
    if (lane==0) { R18_UNROLL for (uint r=0;r<R;r++) out[row0+r] = acc[r]*asc[0]; }
}
template <uint R>
kernel void r18_sa_bias_h(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    device const float* bias[[buffer(5)]], constant uint& K[[buffer(6)]], threadgroup half* Ah [[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    float acc[R]; uint row0;
    r18_sa_h_acc<R>(wq, sct, aq, K, Ah, tgid, tid, tgs, sgid, lane, acc, row0);
    if (lane==0) { R18_UNROLL for (uint r=0;r<R;r++) out[row0+r] = acc[r]*asc[0] + bias[row0+r]; }
}
template <uint R>
kernel void r18_sa_resid_h(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], threadgroup half* Ah [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    float acc[R]; uint row0;
    r18_sa_h_acc<R>(wq, sct, aq, K, Ah, tgid, tid, tgs, sgid, lane, acc, row0);
    if (lane==0) { R18_UNROLL for (uint r=0;r<R;r++) out[row0+r] += acc[r]*asc[0]; }
}
template [[host_name("r18_sa_h1")]] kernel decltype(r18_sa_h<1>) r18_sa_h<1>;
template [[host_name("r18_sa_bias_h1")]] kernel decltype(r18_sa_bias_h<1>) r18_sa_bias_h<1>;
template [[host_name("r18_sa_resid_h1")]] kernel decltype(r18_sa_resid_h<1>) r18_sa_resid_h<1>;
template [[host_name("r18_sa_h2")]] kernel decltype(r18_sa_h<2>) r18_sa_h<2>;
template [[host_name("r18_sa_bias_h2")]] kernel decltype(r18_sa_bias_h<2>) r18_sa_bias_h<2>;
template [[host_name("r18_sa_resid_h2")]] kernel decltype(r18_sa_resid_h<2>) r18_sa_resid_h<2>;
template [[host_name("r18_sa_h4")]] kernel decltype(r18_sa_h<4>) r18_sa_h<4>;
template [[host_name("r18_sa_bias_h4")]] kernel decltype(r18_sa_bias_h<4>) r18_sa_bias_h<4>;
template [[host_name("r18_sa_resid_h4")]] kernel decltype(r18_sa_resid_h<4>) r18_sa_resid_h<4>;

#define R18_SA_INST(R, F, tag) \
    template [[host_name("r18_sa_" tag)]] kernel decltype(r18_sa<R,F>) r18_sa<R,F>; \
    template [[host_name("r18_sa_bias_" tag)]] kernel decltype(r18_sa_bias<R,F>) r18_sa_bias<R,F>; \
    template [[host_name("r18_sa_resid_" tag)]] kernel decltype(r18_sa_resid<R,F>) r18_sa_resid<R,F>;
R18_SA_INST(1, 0, "i1")
R18_SA_INST(2, 0, "i2")
R18_SA_INST(4, 0, "i4")
R18_SA_INST(1, 1, "f1")
R18_SA_INST(2, 1, "f2")
R18_SA_INST(4, 1, "f4")
`

// TestR18Step0Standalone: EXPLORATORY (R18 grades in sequence, not here). For each SA-family decode shape of the 1.5B
// and 7B, the shipped gemv_w4a8_sa against the step-0 candidates (R = 1, 2, 4 rows per simdgroup) — bit-identity of
// the output on the same inputs first, then time per dispatch with the S0 bench's SLC-defeating rotation.
//
//	GOINFER_METAL_R18=1 go test -run '^TestR18Step0Standalone$' -v ./metal/
func TestR18Step0Standalone(t *testing.T) {
	if os.Getenv("GOINFER_METAL_R18") != "1" {
		t.Skip("set GOINFER_METAL_R18=1 (allocates ~0.5 GB of GPU buffers)")
	}
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer d.ReleaseAll()
	lib, err := d.CompileLibrary(allKernels+"\n"+r18Kernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe := func(name string) Pipeline {
		p, err := d.NewComputePipeline(lib, name)
		if err != nil {
			t.Fatalf("pipeline %s: %v", name, err)
		}
		return p
	}
	cq := d.NewCommandQueue()
	t0 := time.Now()
	hb := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[r18-s0 %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	seed := uint32(88172645)
	rnd := func() uint32 { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed }
	const reps = 7
	med := func(xs []float64) float64 { s := append([]float64(nil), xs...); sort.Float64s(s); return s[len(s)/2] }
	timeIt := func(per int, enc func(e *Encoder, i int)) float64 {
		var ms []float64
		for r := range reps + 1 {
			e := cq.Begin()
			for i := range per {
				enc(e, i)
			}
			e.End()
			if err := e.Err(); err != nil {
				t.Fatalf("dispatch: %v", err)
			}
			if r > 0 {
				ms = append(ms, (e.GPUEnd()-e.GPUStart())*1e3/float64(per))
			}
		}
		return med(ms)
	}
	type shape struct {
		model, cat string
		N, K       int
	}
	shapes := []shape{
		{"1.5B", "qkv", 2048, 1536}, {"1.5B", "o", 1536, 1536}, {"1.5B", "gate/up", 17920, 1536},
		{"7B", "qkv", 4608, 3584}, {"7B", "o", 3584, 3584}, {"7B", "gate/up", 37888, 3584},
	}
	arms := []struct {
		name string
		R    int
	}{{"gemv_w4a8_sa", 1}, {"gemv_w4a8_sa_r2", 2}, {"gemv_w4a8_sa_r4", 4},
		{"gemv_w4a8_sa_fr4", 4}, {"gemv_w4a8_sa_dvi2", 2}, {"gemv_w4a8_sa_dvi4", 4}, {"gemv_w4a8_sa_dvf4", 4}, {"gemv_w4a8_sa_dvs4", 4}}
	shmFor := func(name string, K int) int {
		if len(name) > len("gemv_w4a8_sa_fm") && name[:len("gemv_w4a8_sa_fm")] == "gemv_w4a8_sa_fm" {
			return (K + K/32) * 4
		}
		return K * 2
	}
	for _, s := range shapes {
		bytes := s.N*s.K/2 + s.N*(s.K/32)*2
		copies := min(64, max(1, int(math.Ceil(float64(256<<20)/float64(bytes)))))
		aqb := d.NewBufferBytes(s.K)
		av := aqb.Int8s()[:s.K]
		for i := range av {
			av[i] = int8(rnd()%255) - 127
		}
		asc := NewBufferFloats(d, []float32{0.0123})
		bias := d.NewBufferLen(s.N)
		uK := NewBufferU32(d, uint32(s.K))
		var ws, ss []Buffer
		for range copies {
			nw := s.N * s.K / 8
			wb := d.NewBufferLen(nw)
			v := wb.U32s()[:nw]
			for i := range v {
				v[i] = rnd()
			}
			sc := make([]uint16, s.N*(s.K/32))
			for i := range sc {
				sc[i] = f32ToF16(float32(rnd()%1000+1) * 1e-5)
			}
			ws, ss = append(ws, wb), append(ss, NewBufferU16s(d, sc))
		}
		// bit-identity on copy 0
		var ref []float32
		outs := make([]Buffer, len(arms))
		for ai, a := range arms {
			outs[ai] = d.NewBufferLen(s.N)
			p := pipe(a.name)
			e := cq.Begin()
			if a.name == "gemv_w4a8_sa" {
				e.DispatchTG(p, s.N*32, 256, s.K*2, ws[0], ss[0], aqb, asc, outs[ai], uK)
			} else {
				e.DispatchTG(p, s.N*32/a.R, 256, shmFor(a.name, s.K), ws[0], ss[0], aqb, asc, outs[ai], uK, bias)
			}
			e.End()
			got := append([]float32(nil), outs[ai].Floats()[:s.N]...)
			if ai == 0 {
				ref = got
				continue
			}
			diff := 0
			for i := range got {
				if math.Float32bits(got[i]) != math.Float32bits(ref[i]) {
					diff++
				}
			}
			if diff != 0 {
				t.Errorf("%s %s %s: %d of %d outputs differ from gemv_w4a8_sa", s.model, s.cat, a.name, diff, s.N)
			}
		}
		per := max(8, 2*copies)
		line := fmt.Sprintf("%-4s %-8s N=%-6d K=%-5d %6.1f MB:", s.model, s.cat, s.N, s.K, float64(bytes)/1e6)
		var base float64
		for ai, a := range arms {
			p := pipe(a.name)
			out := outs[ai]
			var ms float64
			if a.name == "gemv_w4a8_sa" {
				ms = timeIt(per, func(e *Encoder, i int) {
					e.DispatchTG(p, s.N*32, 256, s.K*2, ws[i%copies], ss[i%copies], aqb, asc, out, uK)
				})
				base = ms
			} else {
				R, shm := a.R, shmFor(a.name, s.K)
				ms = timeIt(per, func(e *Encoder, i int) {
					e.DispatchTG(p, s.N*32/R, 256, shm, ws[i%copies], ss[i%copies], aqb, asc, out, uK, bias)
				})
			}
			line += fmt.Sprintf("  %s %.3f ms (%.0f GB/s, %.2fx)", a.name[len("gemv_w4a8_"):], ms, float64(bytes)/1e6/ms, base/ms)
		}
		hb("%s", line)
		d.ReleaseAll()
	}
	// step 2: the down projection against gemv_w4a8_resid (tg 32, one row per threadgroup)
	for _, s := range []shape{{"1.5B", "down", 1536, 8960}, {"7B", "down", 3584, 18944}} {
		bytes := s.N*s.K/2 + s.N*(s.K/32)*2
		copies := min(64, max(1, int(math.Ceil(float64(256<<20)/float64(bytes)))))
		aqb := d.NewBufferBytes(s.K)
		av := aqb.Int8s()[:s.K]
		for i := range av {
			av[i] = int8(rnd()%255) - 127
		}
		asc := NewBufferFloats(d, []float32{0.0123})
		uK := NewBufferU32(d, uint32(s.K))
		var ws, ss []Buffer
		for range copies {
			nw := s.N * s.K / 8
			wb := d.NewBufferLen(nw)
			v := wb.U32s()[:nw]
			for i := range v {
				v[i] = rnd()
			}
			sc := make([]uint16, s.N*(s.K/32))
			for i := range sc {
				sc[i] = f32ToF16(float32(rnd()%1000+1) * 1e-5)
			}
			ws, ss = append(ws, wb), append(ss, NewBufferU16s(d, sc))
		}
		cArms := []struct {
			name string
			R    int
		}{{"gemv_w4a8_resid", 0}, {"gemv_w4a8_resid_st1", 1}, {"gemv_w4a8_resid_st2", 2}, {"gemv_w4a8_resid_st4", 4}, {"gemv_w4a8_resid_st8", 8}}
		disp := func(e *Encoder, name string, R int, c int, out Buffer) {
			p := pipe(name)
			if R == 0 {
				e.Dispatch(p, s.N*32, 32, ws[c], ss[c], aqb, asc, out, uK)
			} else {
				e.DispatchTG(p, s.N*32/R, 256, s.K, ws[c], ss[c], aqb, asc, out, uK)
			}
		}
		var ref []float32
		outs := make([]Buffer, len(cArms))
		for ai, a := range cArms {
			outs[ai] = d.NewBufferLen(s.N) // zero: the residual starts at 0 for the identity check
			e := cq.Begin()
			disp(e, a.name, a.R, 0, outs[ai])
			e.End()
			got := append([]float32(nil), outs[ai].Floats()[:s.N]...)
			if ai == 0 {
				ref = got
				continue
			}
			diff := 0
			for i := range got {
				if math.Float32bits(got[i]) != math.Float32bits(ref[i]) {
					diff++
				}
			}
			if diff != 0 {
				t.Errorf("%s %s %s: %d of %d outputs differ from gemv_w4a8_resid", s.model, s.cat, a.name, diff, s.N)
			}
		}
		per := max(8, 2*copies)
		line := fmt.Sprintf("%-4s %-8s N=%-6d K=%-5d %6.1f MB:", s.model, s.cat, s.N, s.K, float64(bytes)/1e6)
		var base float64
		for ai, a := range cArms {
			a, out := a, outs[ai]
			ms := timeIt(per, func(e *Encoder, i int) { disp(e, a.name, a.R, i%copies, out) })
			if ai == 0 {
				base = ms
			}
			line += fmt.Sprintf("  %s %.3f ms (%.0f GB/s, %.2fx)", a.name[len("gemv_w4a8_"):], ms, float64(bytes)/1e6/ms, base/ms)
		}
		hb("%s", line)
		d.ReleaseAll()
	}
}
