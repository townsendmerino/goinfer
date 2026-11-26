// gemm_f32_sg — the PRODUCTION f32 GEMM, on simdgroup_matrix (the Apple GPU's cooperative
// 8×8 matrix ALU), vs gemm_f32_tiled's scalar accumulation. Same C[m,n]=dot(A[m,:],B[n,:])
// with B row-major [N,K].
//
// Layout: each threadgroup computes a BM×BN = 32×32 output tile with 4 simdgroups (2×2),
// each owning a 16×16 sub-tile = 2×2 accumulator fragments. Per K-chunk (BK=8) the 128
// threads cooperatively stage A[32×8] and Bᵀ[8×32] into threadgroup memory, ZERO-PADDED on
// the M/N/K edges, so simdgroup_load never reads out of bounds and ANY shape is handled;
// each simdgroup then loads its A/B fragments from that staged tile and issues four
// simdgroup_multiply_accumulate. Results stage back through Cs and are written bounds-checked.
//
// B is loaded pre-transposed into Bs[k][n] (a plain scatter during staging, not a device
// transpose), so the multiply consumes an [m,k]·[k,n] pair directly. Launch via Run2D with
// grid = (ceil(N/32), ceil(M/32)) threadgroups of 128 threads (SGDims).
#define SGBM 32
#define SGBN 32
#define SGBK 8

kernel void gemm_f32_sg(
    device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]], constant int& N [[buffer(4)]], constant int& K [[buffer(5)]],
    uint2 tgpos [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float As[SGBM][SGBK];
    threadgroup float Bs[SGBK][SGBN];
    threadgroup float Cs[SGBM][SGBN];
    int m0 = (int)tgpos.y * SGBM;
    int n0 = (int)tgpos.x * SGBN;
    int sg = (int)tid / 32;      // 0..3
    int sgr = sg / 2, sgc = sg % 2; // 2×2 simdgroup grid → each covers 16×16

    simdgroup_float8x8 acc[2][2];
    for (int f = 0; f < 2; f++)
        for (int g = 0; g < 2; g++)
            acc[f][g] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k0 = 0; k0 < K; k0 += SGBK) {
        for (int idx = (int)tid; idx < SGBM * SGBK; idx += 128) {
            int r = idx / SGBK, c = idx % SGBK;
            As[r][c] = (m0 + r < M && k0 + c < K) ? A[(uint)(m0 + r) * (uint)K + (uint)(k0 + c)] : 0.0f;
        }
        for (int idx = (int)tid; idx < SGBK * SGBN; idx += 128) {
            int k = idx / SGBN, n = idx % SGBN;
            Bs[k][n] = (n0 + n < N && k0 + k < K) ? B[(uint)(n0 + n) * (uint)K + (uint)(k0 + k)] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 a[2], b[2];
        for (int f = 0; f < 2; f++) simdgroup_load(a[f], &As[sgr * 16 + f * 8][0], SGBK);
        for (int g = 0; g < 2; g++) simdgroup_load(b[g], &Bs[0][sgc * 16 + g * 8], SGBN);
        for (int f = 0; f < 2; f++)
            for (int g = 0; g < 2; g++)
                simdgroup_multiply_accumulate(acc[f][g], a[f], b[g], acc[f][g]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int f = 0; f < 2; f++)
        for (int g = 0; g < 2; g++)
            simdgroup_store(acc[f][g], &Cs[sgr * 16 + f * 8][sgc * 16 + g * 8], SGBN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = (int)tid; idx < SGBM * SGBN; idx += 128) {
        int r = idx / SGBN, c = idx % SGBN;
        if (m0 + r < M && n0 + c < N) C[(uint)(m0 + r) * (uint)N + (uint)(n0 + c)] = Cs[r][c];
    }
}

// gemm_f32_bias — production f32 GEMM with fused bias addition on simdgroup_matrix.
// Writes C[m,n] = Cs[r,c] + bias[n0+c].
kernel void gemm_f32_bias(
    device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
    device const float* bias [[buffer(2)]], device float* C [[buffer(3)]],
    constant int& M [[buffer(4)]], constant int& N [[buffer(5)]], constant int& K [[buffer(6)]],
    uint2 tgpos [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float As[SGBM][SGBK];
    threadgroup float Bs[SGBK][SGBN];
    threadgroup float Cs[SGBM][SGBN];
    int m0 = (int)tgpos.y * SGBM;
    int n0 = (int)tgpos.x * SGBN;
    int sg = (int)tid / 32;
    int sgr = sg / 2, sgc = sg % 2;

    simdgroup_float8x8 acc[2][2];
    for (int f = 0; f < 2; f++)
        for (int g = 0; g < 2; g++)
            acc[f][g] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k0 = 0; k0 < K; k0 += SGBK) {
        for (int idx = (int)tid; idx < SGBM * SGBK; idx += 128) {
            int r = idx / SGBK, c = idx % SGBK;
            As[r][c] = (m0 + r < M && k0 + c < K) ? A[(uint)(m0 + r) * (uint)K + (uint)(k0 + c)] : 0.0f;
        }
        for (int idx = (int)tid; idx < SGBK * SGBN; idx += 128) {
            int k = idx / SGBN, n = idx % SGBN;
            Bs[k][n] = (n0 + n < N && k0 + k < K) ? B[(uint)(n0 + n) * (uint)K + (uint)(k0 + k)] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 a[2], b[2];
        for (int f = 0; f < 2; f++) simdgroup_load(a[f], &As[sgr * 16 + f * 8][0], SGBK);
        for (int g = 0; g < 2; g++) simdgroup_load(b[g], &Bs[0][sgc * 16 + g * 8], SGBN);
        for (int f = 0; f < 2; f++)
            for (int g = 0; g < 2; g++)
                simdgroup_multiply_accumulate(acc[f][g], a[f], b[g], acc[f][g]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int f = 0; f < 2; f++)
        for (int g = 0; g < 2; g++)
            simdgroup_store(acc[f][g], &Cs[sgr * 16 + f * 8][sgc * 16 + g * 8], SGBN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = (int)tid; idx < SGBM * SGBN; idx += 128) {
        int r = idx / SGBN, c = idx % SGBN;
        if (m0 + r < M && n0 + c < N) {
            uint pos = (uint)(m0 + r) * (uint)N + (uint)(n0 + c);
            C[pos] = Cs[r][c] + bias[n0 + c];
        }
    }
}

// gemm_f32_bias_add — production f32 GEMM with fused bias and residual addition on simdgroup_matrix.
// Writes residual[m,n] += Cs[r,c] + bias[n0+c].
kernel void gemm_f32_bias_add(
    device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
    device const float* bias [[buffer(2)]], device float* residual [[buffer(3)]],
    constant int& M [[buffer(4)]], constant int& N [[buffer(5)]], constant int& K [[buffer(6)]],
    uint2 tgpos [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float As[SGBM][SGBK];
    threadgroup float Bs[SGBK][SGBN];
    threadgroup float Cs[SGBM][SGBN];
    int m0 = (int)tgpos.y * SGBM;
    int n0 = (int)tgpos.x * SGBN;
    int sg = (int)tid / 32;
    int sgr = sg / 2, sgc = sg % 2;

    simdgroup_float8x8 acc[2][2];
    for (int f = 0; f < 2; f++)
        for (int g = 0; g < 2; g++)
            acc[f][g] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k0 = 0; k0 < K; k0 += SGBK) {
        for (int idx = (int)tid; idx < SGBM * SGBK; idx += 128) {
            int r = idx / SGBK, c = idx % SGBK;
            As[r][c] = (m0 + r < M && k0 + c < K) ? A[(uint)(m0 + r) * (uint)K + (uint)(k0 + c)] : 0.0f;
        }
        for (int idx = (int)tid; idx < SGBK * SGBN; idx += 128) {
            int k = idx / SGBN, n = idx % SGBN;
            Bs[k][n] = (n0 + n < N && k0 + k < K) ? B[(uint)(n0 + n) * (uint)K + (uint)(k0 + k)] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 a[2], b[2];
        for (int f = 0; f < 2; f++) simdgroup_load(a[f], &As[sgr * 16 + f * 8][0], SGBK);
        for (int g = 0; g < 2; g++) simdgroup_load(b[g], &Bs[0][sgc * 16 + g * 8], SGBN);
        for (int f = 0; f < 2; f++)
            for (int g = 0; g < 2; g++)
                simdgroup_multiply_accumulate(acc[f][g], a[f], b[g], acc[f][g]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int f = 0; f < 2; f++)
        for (int g = 0; g < 2; g++)
            simdgroup_store(acc[f][g], &Cs[sgr * 16 + f * 8][sgc * 16 + g * 8], SGBN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = (int)tid; idx < SGBM * SGBN; idx += 128) {
        int r = idx / SGBN, c = idx % SGBN;
        if (m0 + r < M && n0 + c < N) {
            uint pos = (uint)(m0 + r) * (uint)N + (uint)(n0 + c);
            residual[pos] += Cs[r][c] + bias[n0 + c];
        }
    }
}

// gemm_f32_sg_big — the ALIGNED fast path of gemm_f32_sg, same 32×32 tile / 4 simdgroups /
// 2×2 fragments (so the same low register footprint the general kernel already tunes well),
// but for shapes with M%32==0, N%32==0, K%8==0 (GEMMF32Plan enforces it, else falls back to
// gemm_f32_sg). The alignment buys the removal of the general kernel's per-K-step cost:
//   - A and B fragments load DIRECTLY from device (B with the transpose flag, since it is
//     stored [N,K]) — NO threadgroup staging and, crucially, NO two barriers per K-step;
//   - NO edge bounds checks and NO output staging through Cs — every access is in range.
// Wider register blocking was tried (64×64, 16 fragments/simdgroup) and lost badly to register
// spilling and low occupancy, so the win here is dropping the barriers, not enlarging the tile.
kernel void gemm_f32_sg_big(
    device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]], constant int& N [[buffer(4)]], constant int& K [[buffer(5)]],
    uint2 tgpos [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]])
{
    int sgr = (int)sgid / 2, sgc = (int)sgid % 2; // 2×2 simdgroups, each 16×16
    int rm = (int)tgpos.y * 32 + sgr * 16;
    int cn = (int)tgpos.x * 32 + sgc * 16;

    simdgroup_float8x8 acc[2][2];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k0 = 0; k0 < K; k0 += 8) {
        simdgroup_float8x8 a[2], b[2];
        for (int i = 0; i < 2; i++)
            simdgroup_load(a[i], A + (uint)(rm + i * 8) * (uint)K + (uint)k0, K);
        for (int j = 0; j < 2; j++)
            simdgroup_load(b[j], B + (uint)(cn + j * 8) * (uint)K + (uint)k0, K, ulong2(0, 0), true);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            simdgroup_store(acc[i][j], C + (uint)(rm + i * 8) * (uint)N + (uint)(cn + j * 8), N);
}