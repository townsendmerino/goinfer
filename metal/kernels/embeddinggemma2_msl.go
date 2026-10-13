package kernels

const Eg2MSL = `
#include <metal_stdlib>
using namespace metal;

#define EG2_TG 128
#define EG2_MAXHD 512

static inline float eg2_gelu(float v) {
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * v * (1.0f + precise::tanh(c * (v + 0.044715f * v * v * v)));
}

// x[i] = gelu_tanh(x[i]) * u[i]
kernel void eg2_gelu_mul(device float* x [[buffer(0)]], device const float* u [[buffer(1)]],
    constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]])
{
    if (gid < n) x[gid] = eg2_gelu(x[gid]) * u[gid];
}

// The PLE block's gate: g[t*P + j] = gelu_tanh(g[t*P + j]) * ple[(t*L + li)*P + j].
kernel void eg2_gelu_mul_ple(device float* g [[buffer(0)]], device const float* ple [[buffer(1)]],
    constant uint& n [[buffer(2)]], constant uint& P [[buffer(3)]], constant uint& L [[buffer(4)]],
    constant uint& li [[buffer(5)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    uint t = gid / P, j = gid % P;
    g[gid] = eg2_gelu(g[gid]) * ple[(t * L + li) * P + j];
}

kernel void eg2_add(device float* x [[buffer(0)]], device const float* y [[buffer(1)]],
    constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]])
{
    if (gid < n) x[gid] = x[gid] + y[gid];
}

// The layer's last step: x = (x + y) * s.
kernel void eg2_add_scale(device float* x [[buffer(0)]], device const float* y [[buffer(1)]],
    constant uint& n [[buffer(2)]], constant float& s [[buffer(3)]], uint gid [[thread_position_in_grid]])
{
    if (gid < n) x[gid] = (x[gid] + y[gid]) * s;
}

// Rotate-half RoPE in place over x [T, heads, hd], from host tables cos/sin [T, hd/2].
kernel void eg2_rope(device float* x [[buffer(0)]], device const float* cs [[buffer(1)]],
    device const float* sn [[buffer(2)]], constant uint& heads [[buffer(3)]], constant uint& hd [[buffer(4)]],
    constant uint& n [[buffer(5)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return; // n = T * heads * hd/2
    uint half_ = hd / 2u;
    uint i = gid % half_, th = gid / half_, t = th / heads;
    device float* r = x + th * hd;
    float x1 = r[i], x2 = r[i + half_];
    float c = cs[t * half_ + i], s = sn[t * half_ + i];
    r[i] = x1 * c - x2 * s;
    r[i + half_] = x2 * c + x1 * s;
}

// Attention as matmuls (the GEMMs are aikit's): these kernels only move data and run the softmax.
// Head-major copy: x [T, nh, hd] -> out [nh, T, hd].
kernel void eg2_headmajor(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
    constant uint& T [[buffer(2)]], constant uint& nh [[buffer(3)]], constant uint& hd [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = T * nh * hd;
    if (gid >= n) return;
    uint d = gid % hd, th = gid / hd, h = th % nh, t = th / nh;
    out[(h * T + t) * hd + d] = x[gid];
}

// One block's values transposed: vh [nkv, T, hd] head kvh, rows [klo, klo+kc) -> vt [hd, kc].
kernel void eg2_vt_block(device const float* vh [[buffer(0)]], device float* vt [[buffer(1)]],
    constant uint& T [[buffer(2)]], constant uint& hd [[buffer(3)]], constant uint& kvh [[buffer(4)]],
    constant uint& klo [[buffer(5)]], constant uint& kc [[buffer(6)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= hd * kc) return;
    uint j = gid % kc, d = gid / kc;
    vt[d * kc + j] = vh[(kvh * T + klo + j) * hd + d];
}

// Row softmax over a block of scores [rows, kc] (scale 1.0): row r is query i0+r, column c is key klo+c. Keys
// outside |i - j| <= W (all keys when W == 0) get weight 0. One threadgroup of EG2_TG per row.
kernel void eg2_softmax_rows(device float* sc [[buffer(0)]], constant uint& kc [[buffer(1)]],
    constant uint& i0 [[buffer(2)]], constant uint& klo [[buffer(3)]], constant uint& W [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]], uint r [[threadgroup_position_in_grid]])
{
    threadgroup float red[EG2_TG];
    device float* row = sc + r * kc;
    uint i = i0 + r;
    uint lo = 0u, hi = kc;
    if (W > 0u) {
        uint a = i > W ? i - W : 0u;
        lo = a > klo ? a - klo : 0u;
        hi = min(kc, i + W + 1u - klo);
    }
    float m = -INFINITY;
    for (uint c = lo + tid; c < hi; c += EG2_TG) m = max(m, row[c]);
    red[tid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = EG2_TG / 2u; o > 0u; o >>= 1) {
        if (tid < o) red[tid] = max(red[tid], red[tid + o]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    m = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint c = lo + tid; c < hi; c += EG2_TG) {
        float e = exp(row[c] - m);
        row[c] = e;
        sum += e;
    }
    red[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = EG2_TG / 2u; o > 0u; o >>= 1) {
        if (tid < o) red[tid] += red[tid + o];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0f / red[0];
    for (uint c = tid; c < kc; c += EG2_TG) row[c] = (c >= lo && c < hi) ? row[c] * inv : 0.0f;
}

// A block's output [rows, hd] into ctx [T, nH*hd] at head h, rows from i0.
kernel void eg2_scatter_head(device const float* ob [[buffer(0)]], device float* ctx [[buffer(1)]],
    constant uint& rows [[buffer(2)]], constant uint& hd [[buffer(3)]], constant uint& nH [[buffer(4)]],
    constant uint& h [[buffer(5)]], constant uint& i0 [[buffer(6)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= rows * hd) return;
    uint d = gid % hd, r = gid / hd;
    ctx[((i0 + r) * nH + h) * hd + d] = ob[gid];
}
`
