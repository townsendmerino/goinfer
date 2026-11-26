// qgate_split_f16 (D-B01): a gated-attention layer's q_proj emits [query ‖ gate] PER HEAD at double width (Qwen3.5;
// decode's delta_qsplit). For M rows of 2*nHhd it writes the query into qkv's Q slot (row stride qStride) and the gate
// into gate rows of nHhd.
kernel void qgate_split_f16(device const half* qg[[buffer(0)]], device half* qkv[[buffer(1)]], device half* gate[[buffer(2)]],
    constant uint& nHhd[[buffer(3)]], constant uint& hd[[buffer(4)]], constant uint& qStride[[buffer(5)]],
    constant uint& M[[buffer(6)]], uint i[[thread_position_in_grid]]) {
    if (i >= M*nHhd) return;
    uint m = i / nHhd, t = i % nHhd;
    uint h = t / hd, d = t % hd;
    uint base = m*2u*nHhd + h*2u*hd + d;
    qkv[m*qStride + t] = qg[base];
    gate[i] = qg[base+hd];
}

// copy_cols_f16 (D-B01): rows of n halves into columns off..off+n of rows dstStride wide (the K‖V projection of a gated
// layer into qkv's K and V slots).
kernel void copy_cols_f16(device const half* src[[buffer(0)]], device half* dst[[buffer(1)]], constant uint& n[[buffer(2)]],
    constant uint& dstStride[[buffer(3)]], constant uint& off[[buffer(4)]], constant uint& M[[buffer(5)]],
    uint i[[thread_position_in_grid]]) {
    if (i >= M*n) return;
    uint m = i / n, j = i % n;
    dst[m*dstStride + off + j] = src[i];
}

// attn_gate_f16 (D-B01): ctx *= sigmoid(gate) in place, M rows of n, before o_proj (decode's delta_attn_gate).
kernel void attn_gate_f16(device half* ctx[[buffer(0)]], device const half* gate[[buffer(1)]], constant uint& n[[buffer(2)]],
    uint i[[thread_position_in_grid]]) {
    if (i >= n) return;
    ctx[i] = half(float(ctx[i]) / (1.0f + exp(-float(gate[i]))));
}

// shared_gate_add_f16: qwen2_moe gated shared expert — residual[m*H+h] += sigmoid(gl[m]) * src[m*H+h].
kernel void shared_gate_add_f16(
    device const half* src[[buffer(0)]],
    device half* residual[[buffer(1)]],
    device const float* gl[[buffer(2)]],
    constant uint& H[[buffer(3)]],
    constant uint& M[[buffer(4)]],
    uint tid[[thread_position_in_grid]]) {
    uint total = M * H;
    if (tid >= total) return;
    uint m = tid / H;
    uint h = tid % H;
    float sig = 1.0f / (1.0f + exp(-gl[m]));
    float val = float(src[m * H + h]) * sig;
    residual[m * H + h] += half(val);
}
