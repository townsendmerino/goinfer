package kernels

const G4vMSL = `
// x[i, d] += X[pos[i].x, d] + Y[pos[i].y, d]: the patch embedder's two position tables.
kernel void g4v_pos_add(device float* x [[buffer(0)]], device const float* X [[buffer(1)]],
    device const float* Y [[buffer(2)]], device const uint* pos [[buffer(3)]], constant uint& H [[buffer(4)]],
    constant uint& n [[buffer(5)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    uint i = gid / H, d = gid % H;
    x[gid] = x[gid] + (X[pos[2u * i] * H + d] + Y[pos[2u * i + 1u] * H + d]);
}

static inline float g4v_clampf(float v, float lo, float hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

// A ClippableLinear's output clamp, in place.
kernel void g4v_clamp(device float* x [[buffer(0)]], constant uint& n [[buffer(1)]], constant float& lo [[buffer(2)]],
    constant float& hi [[buffer(3)]], uint gid [[thread_position_in_grid]])
{
    if (gid < n) x[gid] = g4v_clampf(x[gid], lo, hi);
}

// Its input clamp, into a copy (q, k and v read the same normed input under different bounds).
kernel void g4v_clamp_copy(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]], constant float& lo [[buffer(3)]], constant float& hi [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) out[gid] = g4v_clampf(x[gid], lo, hi);
}

// Axial 2-D RoPE in place over x [T, heads, hd] from aikit's tables cos/sin [T, hd]: each half of a head is rotated
// rotate-half within itself (pairs d, d + hd/4), the first half by the patch's x and the second by its y.
kernel void g4v_rope(device float* x [[buffer(0)]], device const float* cs [[buffer(1)]],
    device const float* sn [[buffer(2)]], constant uint& heads [[buffer(3)]], constant uint& hd [[buffer(4)]],
    constant uint& n [[buffer(5)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return; // n = T * heads * hd/2
    uint half_ = hd / 2u, q = hd / 4u;
    uint p = gid % half_, th = gid / half_, t = th / heads;
    uint base = (p / q) * half_, d = p % q;
    device float* r = x + th * hd + base;
    device const float* c = cs + t * hd + base;
    device const float* s = sn + t * hd + base;
    float a = r[d], b = r[d + q];
    r[d] = a * c[d] - b * s[d];
    r[d + q] = b * c[d + q] + a * s[d + q];
}
`
