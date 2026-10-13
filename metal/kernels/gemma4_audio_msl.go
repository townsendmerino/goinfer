package kernels

const G4aMSL = `
#define G4A_MAXTG 1024
#define G4A_MAXW 64

kernel void g4a_silu(device float* x [[buffer(0)]], constant uint& n [[buffer(1)]], uint gid [[thread_position_in_grid]])
{
    if (gid < n) { float v = x[gid]; x[gid] = v / (1.0f + exp(-v)); }
}

// x += a * y
kernel void g4a_axpy(device float* x [[buffer(0)]], device const float* y [[buffer(1)]], constant uint& n [[buffer(2)]],
    constant float& a [[buffer(3)]], uint gid [[thread_position_in_grid]])
{
    if (gid < n) x[gid] = x[gid] + a * y[gid];
}

// out[t, j] = g[t, j] * sigmoid(g[t, H + j]) over g [n, 2H].
kernel void g4a_glu(device const float* g [[buffer(0)]], device float* out [[buffer(1)]], constant uint& H [[buffer(2)]],
    constant uint& n [[buffer(3)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    uint t = gid / H, j = gid % H;
    float a = g[t * 2u * H + j], b = g[t * 2u * H + H + j];
    out[gid] = a * (1.0f / (1.0f + exp(-b)));
}

// Causal depthwise conv over time: out[t, c] = sum_k w[c, k] * y[t - (K-1) + k, c], y before 0 being 0.
kernel void g4a_dwconv(device const float* y [[buffer(0)]], device const float* w [[buffer(1)]], device float* out [[buffer(2)]],
    constant uint& H [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint& n [[buffer(5)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    uint t = gid / H, c = gid % H;
    float s = 0.0f;
    for (uint k = 0; k < K; k++) {
        int ti = int(t) - int(K - 1u) + int(k);
        if (ti >= 0) s += w[c * K + k] * y[uint(ti) * H + c];
    }
    out[gid] = s;
}

// q[i] *= qs[i % hd]; k[i] *= ks.
kernel void g4a_scale_qk(device float* q [[buffer(0)]], device float* k [[buffer(1)]], device const float* qs [[buffer(2)]],
    constant float& ks [[buffer(3)]], constant uint& hd [[buffer(4)]], constant uint& n [[buffer(5)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    q[gid] = q[gid] * qs[gid % hd];
    k[gid] = k[gid] * ks;
}

// Sliding-window attention: threadgroup (t, h), one thread per head dim d. Key j is seen iff 0 <= t - j < W; its logit
// is q·k_j + q·relk_{t-j}, softcapped by cap, then a softmax over the window and the weighted sum of v.
kernel void g4a_attn(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]], device const float* v [[buffer(2)]],
    device const float* relk [[buffer(3)]], device float* ctx [[buffer(4)]], constant uint& H [[buffer(5)]],
    constant uint& nH [[buffer(6)]], constant uint& W [[buffer(7)]], constant float& cap [[buffer(8)]],
    uint d [[thread_position_in_threadgroup]], uint tgsz [[threads_per_threadgroup]], uint g [[threadgroup_position_in_grid]],
    uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float part[G4A_MAXTG / 32];
    threadgroup float lg[G4A_MAXW];
    uint t = g / nH, h = g % nH, hd = tgsz;
    uint off = h * hd + d;
    float qd = q[t * H + off];
    uint lo = t + 1u > W ? t + 1u - W : 0u;
    uint nsg = (tgsz + 31u) / 32u;
    for (uint j = lo; j <= t; j++) {
        float p = qd * k[j * H + off] + qd * relk[(t - j) * H + off];
        p = simd_sum(p);
        if (sl == 0u) part[sg] = p;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (d == 0u) {
            float s = 0.0f;
            for (uint i = 0; i < nsg; i++) s += part[i];
            lg[j - lo] = cap * precise::tanh(s / cap);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint m = t - lo + 1u;
    float mx = -INFINITY;
    for (uint i = 0; i < m; i++) mx = max(mx, lg[i]);
    float z = 0.0f;
    for (uint i = 0; i < m; i++) z += exp(lg[i] - mx);
    float acc = 0.0f;
    for (uint j = lo; j <= t; j++) acc += (exp(lg[j - lo] - mx) / z) * v[j * H + off];
    ctx[t * H + off] = acc;
}
`
