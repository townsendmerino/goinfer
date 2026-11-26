// ---------------------------------------------------------------------------
// TILED GEMMs (throughput) — the Metal mirror of vit.cu's tiled section. The untiled
// gemm_w8a8 / gemm_f32 above are correctness-first: one thread per output, each re-reading
// a whole A row and B row from global memory. These stage a TILE×TILE block of A and of B
// through threadgroup memory per K-chunk, so each element is read once per tile.
//
// Launched via Run2D (dispatchThreadgroups — UNIFORM whole threadgroups), so the edge
// tiles are full and every thread reaches the barrier; each thread then bounds-checks its
// own (m,n,k), exactly like the CUDA kernels. This is the ONE Metal kernel family that
// keeps the CUDA bounds checks, because its dispatch is uniform, not dispatchThreads.
//
// gemm_w8a8_tiled is expected BIT-IDENTICAL to the untiled kernel: the accumulator is
// int32 and integer addition is associative, so re-chunking K cannot change the sum (no
// overflow at ViT shapes: K·127² ≈ 1.9e7 ≪ 2³¹). gemm_f32_tiled cannot claim that (f32
// addition reassociates), so it is gated on a tight relative bound vs a float64 reference.
#define TILE 16

kernel void gemm_w8a8_tiled(
    device const char* A [[buffer(0)]], device const float* aScale [[buffer(1)]],
    device const char* B [[buffer(2)]], device const float* bScale [[buffer(3)]],
    device float* C [[buffer(4)]],
    constant int& M [[buffer(5)]], constant int& N [[buffer(6)]], constant int& K [[buffer(7)]],
    uint2 tgpos [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]])
{
    threadgroup char As[TILE][TILE];
    threadgroup char Bs[TILE][TILE];
    int tx = (int)tid.x, ty = (int)tid.y;
    int m = (int)tgpos.y * TILE + ty;
    int n = (int)tgpos.x * TILE + tx;
    int bRow = (int)tgpos.x * TILE + ty; // the B row this thread STAGES (not the one it uses)
    int acc = 0;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        int k = k0 + tx;
        As[ty][tx] = (m < M && k < K) ? A[(uint)m * (uint)K + (uint)k] : (char)0;
        Bs[ty][tx] = (bRow < N && k < K) ? B[(uint)bRow * (uint)K + (uint)k] : (char)0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < TILE; kk++) acc += (int)As[ty][kk] * (int)Bs[tx][kk];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (m < M && n < N) C[(uint)m * (uint)N + (uint)n] = (float)acc * aScale[m] * bScale[n];
}

// gemm_w8a8_reg — the register-blocked, packed-word-staged twin of gemm_w8a8_tiled
// (M-15, audit-metal-2026-09-12.md), porting CUDA's already-shipped gemm_w8a8_reg
// (vit.cu, audit M-12). gemm_w8a8_tiled computes ONE output per thread with
// byte-granular threadgroup staging — 32 threadgroup loads per 4 MACs, LSU-bound
// rather than MAC-bound, the exact shape CUDA's own M-12 measured at 7% of its dp4a
// roof and retired from the ANN path; the retirement never reached the ViT path.
//
// Two changes, both mirroring gemm_w8a8_reg's CUDA playbook:
//
//   PACKED-WORD STAGING. Threadgroup memory holds int (four packed int8 lanes), not
//   char, so one load feeds a whole 4-lane MAC instead of four char loads feeding a
//   quarter of one. A/B are read through an int* view — K/4 words per row — which is
//   why K%4 is required; Metal buffer allocations are word-aligned.
//
//   4×4 REGISTER BLOCKING. Each thread owns a 4×4 micro-tile, so the 8 shared words it
//   loads per k-word-step feed 16 4-lane MACs instead of 1 — the same ratio win CUDA's
//   dp4a version gets. MSL has no simdgroup_matrix int8 form and no hardware dp4a
//   intrinsic (unlike gemm_f32_sg/sg_big above, which use simdgroup_matrix — f32 only),
//   so dp4a_manual below unpacks each packed word via arithmetic right-shift
//   (sign-extending each byte) and sums the four lane products in scalar registers —
//   this still cuts threadgroup-memory TRAFFIC 4× per word loaded and amortizes each
//   load over 4 uses in each dimension via the register tile, even without a hardware
//   4-way MAC instruction. The shift-based unpack matches how the bytes were packed:
//   both CPU and Apple GPU are little-endian, so element 0 of the original int8 array
//   is the LOW byte of the reinterpreted word, same as CUDA's convention.
//
// K steps 16 elements (IBK=4 words), not 64: SigLIP-so400m's intermediate width (4304)
// is a multiple of 16 but not of 64, so a K%64 kernel would miss the MLP projections
// that are most of a ViT's work — the same reasoning gemm_w8a8_reg's own comment gives.
//
// The epilogue is byte-for-byte gemm_w8a8_tiled's: an exact int32 accumulator scaled by
// aScale[m]*bScale[n] in the same order. int32 addition is associative (no overflow at
// ViT shapes: K·127² ≈ 1.9e7 ≪ 2³¹), so re-chunking K into words/steps cannot change the
// sum — this kernel and gemm_w8a8_tiled agree bit-for-bit regardless of grouping, and
// GEMMW8A8Plan can route between them freely.
//
// Requires only K%16==0 (IBK words of four int8). M and N are free: edge tiles stage
// zeros and skip their stores, so the inner loop stays unguarded for any M, N — matters
// because an M%64/N%64 requirement would have excluded SigLIP-so400m's MLP projections
// (N=4304, not a multiple of 64).
#define IBM 64
#define IBN 64
#define IBK 4
#define IBTM 4
#define IBTN 4

inline int dp4a_manual(int a, int b, int acc) {
    int a0 = a << 24 >> 24, a1 = a << 16 >> 24, a2 = a << 8 >> 24, a3 = a >> 24;
    int b0 = b << 24 >> 24, b1 = b << 16 >> 24, b2 = b << 8 >> 24, b3 = b >> 24;
    return acc + a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
}

kernel void gemm_w8a8_reg(
    device const int* Aw [[buffer(0)]], device const float* aScale [[buffer(1)]],
    device const int* Bw [[buffer(2)]], device const float* bScale [[buffer(3)]],
    device float* C [[buffer(4)]],
    constant int& M [[buffer(5)]], constant int& N [[buffer(6)]], constant int& K [[buffer(7)]],
    uint2 tgpos [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]])
{
    threadgroup int As[IBK][IBM + 1];
    threadgroup int Bs[IBK][IBN + 1];
    int kw = K >> 2;
    int m0 = (int)tgpos.y * IBM, n0 = (int)tgpos.x * IBN;
    int tid = (int)tid2.y * 16 + (int)tid2.x; // 0..255
    int lr = tid >> 2, lc = tid & 3;          // 64 rows x 4 k-words per pass

    int acc[IBTM][IBTN];
    for (int i = 0; i < IBTM; i++)
        for (int j = 0; j < IBTN; j++) acc[i][j] = 0;

    for (int k0 = 0; k0 < kw; k0 += IBK) {
        // Edge tiles are handled by STAGING ZEROS, not by bounds-checking the inner
        // loop — see the kernel comment above for why.
        As[lc][lr] = (m0 + lr < M) ? Aw[(uint)(m0 + lr) * (uint)kw + (uint)(k0 + lc)] : 0;
        Bs[lc][lr] = (n0 + lr < N) ? Bw[(uint)(n0 + lr) * (uint)kw + (uint)(k0 + lc)] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kk = 0; kk < IBK; kk++) {
            int a[IBTM], b[IBTN];
            for (int i = 0; i < IBTM; i++) a[i] = As[kk][(int)tid2.y * IBTM + i];
            for (int j = 0; j < IBTN; j++) b[j] = Bs[kk][(int)tid2.x * IBTN + j];
            for (int i = 0; i < IBTM; i++)
                for (int j = 0; j < IBTN; j++) acc[i][j] = dp4a_manual(a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = 0; i < IBTM; i++) {
        int m = m0 + (int)tid2.y * IBTM + i;
        if (m >= M) continue;
        for (int j = 0; j < IBTN; j++) {
            int n = n0 + (int)tid2.x * IBTN + j;
            if (n < N) C[(uint)m * (uint)N + (uint)n] = (float)acc[i][j] * aScale[m] * bScale[n];
        }
    }
}

// gemm_w8a8_bias — fused GEMM + bias addition.
// Writes C[m, n] = (acc * aScale * bScale) + bias[n], eliminating separate add_bias dispatch.
kernel void gemm_w8a8_bias(
    device const int* Aw [[buffer(0)]], device const float* aScale [[buffer(1)]],
    device const int* Bw [[buffer(2)]], device const float* bScale [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device float* C [[buffer(5)]],
    constant int& M [[buffer(6)]], constant int& N [[buffer(7)]], constant int& K [[buffer(8)]],
    uint2 tgpos [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]])
{
    threadgroup int As[IBK][IBM + 1];
    threadgroup int Bs[IBK][IBN + 1];
    int kw = K >> 2;
    int m0 = (int)tgpos.y * IBM, n0 = (int)tgpos.x * IBN;
    int tid = (int)tid2.y * 16 + (int)tid2.x;
    int lr = tid >> 2, lc = tid & 3;

    int acc[IBTM][IBTN];
    for (int i = 0; i < IBTM; i++)
        for (int j = 0; j < IBTN; j++) acc[i][j] = 0;

    for (int k0 = 0; k0 < kw; k0 += IBK) {
        As[lc][lr] = (m0 + lr < M) ? Aw[(uint)(m0 + lr) * (uint)kw + (uint)(k0 + lc)] : 0;
        Bs[lc][lr] = (n0 + lr < N) ? Bw[(uint)(n0 + lr) * (uint)kw + (uint)(k0 + lc)] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kk = 0; kk < IBK; kk++) {
            int a[IBTM], b[IBTN];
            for (int i = 0; i < IBTM; i++) a[i] = As[kk][(int)tid2.y * IBTM + i];
            for (int j = 0; j < IBTN; j++) b[j] = Bs[kk][(int)tid2.x * IBTN + j];
            for (int i = 0; i < IBTM; i++)
                for (int j = 0; j < IBTN; j++) acc[i][j] = dp4a_manual(a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = 0; i < IBTM; i++) {
        int m = m0 + (int)tid2.y * IBTM + i;
        if (m >= M) continue;
        for (int j = 0; j < IBTN; j++) {
            int n = n0 + (int)tid2.x * IBTN + j;
            if (n < N) C[(uint)m * (uint)N + (uint)n] = (float)acc[i][j] * aScale[m] * bScale[n] + bias[n];
        }
    }
}

// gemm_w8a8_bias_add — fused GEMM + bias + residual addition.
// Writes residual[m, n] += (acc * aScale * bScale) + bias[n], eliminating both add_bias and add_vec.
kernel void gemm_w8a8_bias_add(
    device const int* Aw [[buffer(0)]], device const float* aScale [[buffer(1)]],
    device const int* Bw [[buffer(2)]], device const float* bScale [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device float* residual [[buffer(5)]],
    constant int& M [[buffer(6)]], constant int& N [[buffer(7)]], constant int& K [[buffer(8)]],
    uint2 tgpos [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]])
{
    threadgroup int As[IBK][IBM + 1];
    threadgroup int Bs[IBK][IBN + 1];
    int kw = K >> 2;
    int m0 = (int)tgpos.y * IBM, n0 = (int)tgpos.x * IBN;
    int tid = (int)tid2.y * 16 + (int)tid2.x;
    int lr = tid >> 2, lc = tid & 3;

    int acc[IBTM][IBTN];
    for (int i = 0; i < IBTM; i++)
        for (int j = 0; j < IBTN; j++) acc[i][j] = 0;

    for (int k0 = 0; k0 < kw; k0 += IBK) {
        As[lc][lr] = (m0 + lr < M) ? Aw[(uint)(m0 + lr) * (uint)kw + (uint)(k0 + lc)] : 0;
        Bs[lc][lr] = (n0 + lr < N) ? Bw[(uint)(n0 + lr) * (uint)kw + (uint)(k0 + lc)] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kk = 0; kk < IBK; kk++) {
            int a[IBTM], b[IBTN];
            for (int i = 0; i < IBTM; i++) a[i] = As[kk][(int)tid2.y * IBTM + i];
            for (int j = 0; j < IBTN; j++) b[j] = Bs[kk][(int)tid2.x * IBTN + j];
            for (int i = 0; i < IBTM; i++)
                for (int j = 0; j < IBTN; j++) acc[i][j] = dp4a_manual(a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = 0; i < IBTM; i++) {
        int m = m0 + (int)tid2.y * IBTM + i;
        if (m >= M) continue;
        for (int j = 0; j < IBTN; j++) {
            int n = n0 + (int)tid2.x * IBTN + j;
            if (n < N) {
                uint idx = (uint)m * (uint)N + (uint)n;
                residual[idx] += (float)acc[i][j] * aScale[m] * bScale[n] + bias[n];
            }
        }
    }
}

kernel void gemm_f32_tiled(
    device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]], constant int& N [[buffer(4)]], constant int& K [[buffer(5)]],
    uint2 tgpos [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]])
{
    threadgroup float As[TILE][TILE];
    threadgroup float Bs[TILE][TILE];
    int tx = (int)tid.x, ty = (int)tid.y;
    int m = (int)tgpos.y * TILE + ty;
    int n = (int)tgpos.x * TILE + tx;
    int bRow = (int)tgpos.x * TILE + ty;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        int k = k0 + tx;
        As[ty][tx] = (m < M && k < K) ? A[(uint)m * (uint)K + (uint)k] : 0.0f;
        Bs[ty][tx] = (bRow < N && k < K) ? B[(uint)bRow * (uint)K + (uint)k] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < TILE; kk++) acc += As[ty][kk] * Bs[tx][kk];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (m < M && n < N) C[(uint)m * (uint)N + (uint)n] = acc;
}

