//go:build darwin

package metal

// bvkKernels — batched-M (weight-stationary) twin of decode's SA-style and COAL-style W4A8 GEMV
// kernels, prototyped for the small-M speculative-verify use case
// (docs/task-metal-batched-verify-kernel.md). Deliberately NOT part of allKernels and never
// referenced from model.go's dispatch path — decode itself is unmodified; these are compiled into
// their own library and dispatched only from metal/batched_verify_test.go.
//
// Each kernel's per-block reduction is a byte-for-byte copy of the corresponding decode kernel's
// macro (SA_BODY / W4A8_BODY, metal/kernels.go) — same lane-strided block iteration, same
// `acc += float(gi) * scale` SOURCE FORM (deliberately not fma(), unlike the pre-existing
// gemv_w4a8_sa_bk prototype). docs/task-int4-int8-exact-mma.md's investigation found that
// decode's bits depend on fast-math contraction being licensed at each call site, not guaranteed
// identical between differently-written expressions that happen to be mathematically equivalent —
// so matching decode's literal source text removes that axis of doubt instead of assuming a
// differently-shaped kernel fuses the same way. The only real change from decode's single-token
// kernels is an inner loop over M token rows that reuses each block's unpacked weight nibbles —
// the amortization this kernel exists to measure.
const bvkKernels = `
#include <metal_stdlib>
using namespace metal;

#define BVK_MAX_M 16

// gemv_w4a8_bvk_bias — batched-M twin of gemv_w4a8_sa_bias's SA_BODY reduction (QKV in-proj: the
// only dense projection with a bias epilogue). Host MUST verify kk*K*2 <=
// device.MaxThreadgroupMemoryLength() before dispatch — this kernel does not self-limit, and an
// over-budget threadgroup-memory request silently no-ops on Apple GPUs (never sets an encoder
// error), so an unchecked caller would read stale/zero output instead of failing loudly.
kernel void gemv_w4a8_bvk_bias(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    device const float* bias[[buffer(5)]], constant uint& K[[buffer(6)]], constant uint& N[[buffer(7)]],
    constant uint& kk[[buffer(8)]], threadgroup short* As [[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]],
    uint tgs[[threads_per_threadgroup]], uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    for (uint i=tid; i<kk*K; i+=tgs) As[i] = short(aq[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint G = K>>5u;
    uint row = tgid*(tgs>>5u) + sgid;
    if (row >= N) return;
    device const uint4* wr = wq + (uint)row*G;
    device const half*  sr = sct + (uint)row*G;
    float acc[BVK_MAX_M];
    for (uint j=0;j<kk;j++) acc[j]=0.0f;
    for (uint g=lane; g<G; g+=32u) {
        uint4 w = wr[g];
        int x0=int(w.x&0xF)-8, x1=int((w.x>>4)&0xF)-8, x2=int((w.x>>8)&0xF)-8, x3=int((w.x>>12)&0xF)-8,
            x4=int((w.x>>16)&0xF)-8, x5=int((w.x>>20)&0xF)-8, x6=int((w.x>>24)&0xF)-8, x7=int((w.x>>28)&0xF)-8;
        int y0=int(w.y&0xF)-8, y1=int((w.y>>4)&0xF)-8, y2=int((w.y>>8)&0xF)-8, y3=int((w.y>>12)&0xF)-8,
            y4=int((w.y>>16)&0xF)-8, y5=int((w.y>>20)&0xF)-8, y6=int((w.y>>24)&0xF)-8, y7=int((w.y>>28)&0xF)-8;
        int z0=int(w.z&0xF)-8, z1=int((w.z>>4)&0xF)-8, z2=int((w.z>>8)&0xF)-8, z3=int((w.z>>12)&0xF)-8,
            z4=int((w.z>>16)&0xF)-8, z5=int((w.z>>20)&0xF)-8, z6=int((w.z>>24)&0xF)-8, z7=int((w.z>>28)&0xF)-8;
        int u0=int(w.w&0xF)-8, u1=int((w.w>>4)&0xF)-8, u2=int((w.w>>8)&0xF)-8, u3=int((w.w>>12)&0xF)-8,
            u4=int((w.w>>16)&0xF)-8, u5=int((w.w>>20)&0xF)-8, u6=int((w.w>>24)&0xF)-8, u7=int((w.w>>28)&0xF)-8;
        float sc = float(sr[g]);
        for (uint j=0;j<kk;j++) {
            threadgroup const short* a = As + j*K + g*32u;
            int gi = x0*int(a[0])+x1*int(a[1])+x2*int(a[2])+x3*int(a[3])+x4*int(a[4])+x5*int(a[5])+x6*int(a[6])+x7*int(a[7])
                   + y0*int(a[8])+y1*int(a[9])+y2*int(a[10])+y3*int(a[11])+y4*int(a[12])+y5*int(a[13])+y6*int(a[14])+y7*int(a[15])
                   + z0*int(a[16])+z1*int(a[17])+z2*int(a[18])+z3*int(a[19])+z4*int(a[20])+z5*int(a[21])+z6*int(a[22])+z7*int(a[23])
                   + u0*int(a[24])+u1*int(a[25])+u2*int(a[26])+u3*int(a[27])+u4*int(a[28])+u5*int(a[29])+u6*int(a[30])+u7*int(a[31]);
            acc[j] += float(gi) * sc;
        }
    }
    for (uint j=0;j<kk;j++) {
        float s = simd_sum(acc[j]);
        if (lane==0) out[j*N + row] = s*asc[j] + bias[row];
    }
}

// gemv_w4a8_bvk_plain — identical reduction to gemv_w4a8_bvk_bias, no bias epilogue. Used for both
// O-proj and gate/up-proj (both are plain SA-style projections in decode; residual add is applied
// as a separate pass reusing decode's own compiled "residual" kernel, not fused here).
kernel void gemv_w4a8_bvk_plain(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], constant uint& N[[buffer(6)]], constant uint& kk[[buffer(7)]],
    threadgroup short* As [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]],
    uint tgs[[threads_per_threadgroup]], uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    for (uint i=tid; i<kk*K; i+=tgs) As[i] = short(aq[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint G = K>>5u;
    uint row = tgid*(tgs>>5u) + sgid;
    if (row >= N) return;
    device const uint4* wr = wq + (uint)row*G;
    device const half*  sr = sct + (uint)row*G;
    float acc[BVK_MAX_M];
    for (uint j=0;j<kk;j++) acc[j]=0.0f;
    for (uint g=lane; g<G; g+=32u) {
        uint4 w = wr[g];
        int x0=int(w.x&0xF)-8, x1=int((w.x>>4)&0xF)-8, x2=int((w.x>>8)&0xF)-8, x3=int((w.x>>12)&0xF)-8,
            x4=int((w.x>>16)&0xF)-8, x5=int((w.x>>20)&0xF)-8, x6=int((w.x>>24)&0xF)-8, x7=int((w.x>>28)&0xF)-8;
        int y0=int(w.y&0xF)-8, y1=int((w.y>>4)&0xF)-8, y2=int((w.y>>8)&0xF)-8, y3=int((w.y>>12)&0xF)-8,
            y4=int((w.y>>16)&0xF)-8, y5=int((w.y>>20)&0xF)-8, y6=int((w.y>>24)&0xF)-8, y7=int((w.y>>28)&0xF)-8;
        int z0=int(w.z&0xF)-8, z1=int((w.z>>4)&0xF)-8, z2=int((w.z>>8)&0xF)-8, z3=int((w.z>>12)&0xF)-8,
            z4=int((w.z>>16)&0xF)-8, z5=int((w.z>>20)&0xF)-8, z6=int((w.z>>24)&0xF)-8, z7=int((w.z>>28)&0xF)-8;
        int u0=int(w.w&0xF)-8, u1=int((w.w>>4)&0xF)-8, u2=int((w.w>>8)&0xF)-8, u3=int((w.w>>12)&0xF)-8,
            u4=int((w.w>>16)&0xF)-8, u5=int((w.w>>20)&0xF)-8, u6=int((w.w>>24)&0xF)-8, u7=int((w.w>>28)&0xF)-8;
        float sc = float(sr[g]);
        for (uint j=0;j<kk;j++) {
            threadgroup const short* a = As + j*K + g*32u;
            int gi = x0*int(a[0])+x1*int(a[1])+x2*int(a[2])+x3*int(a[3])+x4*int(a[4])+x5*int(a[5])+x6*int(a[6])+x7*int(a[7])
                   + y0*int(a[8])+y1*int(a[9])+y2*int(a[10])+y3*int(a[11])+y4*int(a[12])+y5*int(a[13])+y6*int(a[14])+y7*int(a[15])
                   + z0*int(a[16])+z1*int(a[17])+z2*int(a[18])+z3*int(a[19])+z4*int(a[20])+z5*int(a[21])+z6*int(a[22])+z7*int(a[23])
                   + u0*int(a[24])+u1*int(a[25])+u2*int(a[26])+u3*int(a[27])+u4*int(a[28])+u5*int(a[29])+u6*int(a[30])+u7*int(a[31]);
            acc[j] += float(gi) * sc;
        }
    }
    for (uint j=0;j<kk;j++) {
        float s = simd_sum(acc[j]);
        if (lane==0) out[j*N + row] = s*asc[j];
    }
}

// gemv_w4a8_bvk_plain_resid — same reduction as gemv_w4a8_bvk_plain, but a FUSED "+=" epilogue
// matching gemv_w4a8_sa_resid exactly (decode's actual non-sandwich O-proj kernel): out[j*N+row]
// is read-modify-written in one op, not written by this kernel and added by a separate pass. This
// distinction is load-bearing, not stylistic: under fast-math (goinfer's default compile mode,
// see docs/task-int4-int8-exact-mma.md), a source-level "out[i] += a*b" licenses the compiler to
// fuse it into one fma (a*b computed at full precision, rounded once against the addition) —
// whereas writing "a*b" to memory and adding it in a SEPARATE kernel forces an intermediate
// rounding boundary the fused form doesn't have. The first prototype of this kernel used the
// split form for the non-sandwich residual add and measurably diverged from decode by 1-4 ULP,
// compounding across KV positions — this fused form is the fix (see the parity test's bisection).
// Only valid when N equals the residual stream width (H) — out IS the M x H residual buffer.
kernel void gemv_w4a8_bvk_plain_resid(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], constant uint& N[[buffer(6)]], constant uint& kk[[buffer(7)]],
    threadgroup short* As [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]],
    uint tgs[[threads_per_threadgroup]], uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    for (uint i=tid; i<kk*K; i+=tgs) As[i] = short(aq[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint G = K>>5u;
    uint row = tgid*(tgs>>5u) + sgid;
    if (row >= N) return;
    device const uint4* wr = wq + (uint)row*G;
    device const half*  sr = sct + (uint)row*G;
    float acc[BVK_MAX_M];
    for (uint j=0;j<kk;j++) acc[j]=0.0f;
    for (uint g=lane; g<G; g+=32u) {
        uint4 w = wr[g];
        int x0=int(w.x&0xF)-8, x1=int((w.x>>4)&0xF)-8, x2=int((w.x>>8)&0xF)-8, x3=int((w.x>>12)&0xF)-8,
            x4=int((w.x>>16)&0xF)-8, x5=int((w.x>>20)&0xF)-8, x6=int((w.x>>24)&0xF)-8, x7=int((w.x>>28)&0xF)-8;
        int y0=int(w.y&0xF)-8, y1=int((w.y>>4)&0xF)-8, y2=int((w.y>>8)&0xF)-8, y3=int((w.y>>12)&0xF)-8,
            y4=int((w.y>>16)&0xF)-8, y5=int((w.y>>20)&0xF)-8, y6=int((w.y>>24)&0xF)-8, y7=int((w.y>>28)&0xF)-8;
        int z0=int(w.z&0xF)-8, z1=int((w.z>>4)&0xF)-8, z2=int((w.z>>8)&0xF)-8, z3=int((w.z>>12)&0xF)-8,
            z4=int((w.z>>16)&0xF)-8, z5=int((w.z>>20)&0xF)-8, z6=int((w.z>>24)&0xF)-8, z7=int((w.z>>28)&0xF)-8;
        int u0=int(w.w&0xF)-8, u1=int((w.w>>4)&0xF)-8, u2=int((w.w>>8)&0xF)-8, u3=int((w.w>>12)&0xF)-8,
            u4=int((w.w>>16)&0xF)-8, u5=int((w.w>>20)&0xF)-8, u6=int((w.w>>24)&0xF)-8, u7=int((w.w>>28)&0xF)-8;
        float sc = float(sr[g]);
        for (uint j=0;j<kk;j++) {
            threadgroup const short* a = As + j*K + g*32u;
            int gi = x0*int(a[0])+x1*int(a[1])+x2*int(a[2])+x3*int(a[3])+x4*int(a[4])+x5*int(a[5])+x6*int(a[6])+x7*int(a[7])
                   + y0*int(a[8])+y1*int(a[9])+y2*int(a[10])+y3*int(a[11])+y4*int(a[12])+y5*int(a[13])+y6*int(a[14])+y7*int(a[15])
                   + z0*int(a[16])+z1*int(a[17])+z2*int(a[18])+z3*int(a[19])+z4*int(a[20])+z5*int(a[21])+z6*int(a[22])+z7*int(a[23])
                   + u0*int(a[24])+u1*int(a[25])+u2*int(a[26])+u3*int(a[27])+u4*int(a[28])+u5*int(a[29])+u6*int(a[30])+u7*int(a[31]);
            acc[j] += float(gi) * sc;
        }
    }
    for (uint j=0;j<kk;j++) {
        float s = simd_sum(acc[j]);
        if (lane==0) out[j*N + row] += s*asc[j];
    }
}

// gemv_w4a8_bvk_coal_resid — same as gemv_w4a8_bvk_coal, fused "+=" epilogue matching
// gemv_w4a8_resid exactly (decode's actual non-sandwich down-proj kernel). See
// gemv_w4a8_bvk_plain_resid's comment for why the fused form, not a split write-then-add, is
// required for bit-identity under fast-math.
kernel void gemv_w4a8_bvk_coal_resid(device const uint* bq[[buffer(0)]], device const half* bsc[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], constant uint& N[[buffer(6)]], constant uint& kk[[buffer(7)]],
    uint gid[[threadgroup_position_in_grid]], uint lid[[thread_index_in_threadgroup]]) {
    uint wpr = K/8u;
    device const uint* brow = bq + (uint)gid*wpr;
    device const half* srow = bsc + (uint)gid*(K/32u);
    float acc[BVK_MAX_M];
    for (uint j=0;j<kk;j++) acc[j]=0.0f;
    for (uint wi = lid; wi < wpr; wi += 32u) {
        uint x = brow[wi];
        int n0=int(x&0xF)-8, n1=int((x>>4)&0xF)-8, n2=int((x>>8)&0xF)-8, n3=int((x>>12)&0xF)-8,
            n4=int((x>>16)&0xF)-8, n5=int((x>>20)&0xF)-8, n6=int((x>>24)&0xF)-8, n7=int((x>>28)&0xF)-8;
        float sc = float(srow[wi>>2]);
        for (uint j=0;j<kk;j++) {
            device const char* a = aq + j*K + wi*8u;
            int gi = n0*int(a[0])+n1*int(a[1])+n2*int(a[2])+n3*int(a[3])+n4*int(a[4])+n5*int(a[5])+n6*int(a[6])+n7*int(a[7]);
            acc[j] += float(gi) * sc;
        }
    }
    for (uint j=0;j<kk;j++) {
        float s = simd_sum(acc[j]);
        if (lid==0) out[j*N + gid] += s*asc[j];
    }
}

// gemv_w4a8_bvk_coal — batched-M twin of gemv_w4a8_coal/_resid's W4A8_BODY reduction (down-proj).
// Reads activations from DEVICE memory (no threadgroup staging), matching gemv_w4a8_coal's own
// design exactly — so, unlike the two kernels above, this one has NO threadgroup-memory ceiling on
// M: it scales register-only (an M-element accumulator array per lane).
kernel void gemv_w4a8_bvk_coal(device const uint* bq[[buffer(0)]], device const half* bsc[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], constant uint& N[[buffer(6)]], constant uint& kk[[buffer(7)]],
    uint gid[[threadgroup_position_in_grid]], uint lid[[thread_index_in_threadgroup]]) {
    uint wpr = K/8u;
    device const uint* brow = bq + (uint)gid*wpr;
    device const half* srow = bsc + (uint)gid*(K/32u);
    float acc[BVK_MAX_M];
    for (uint j=0;j<kk;j++) acc[j]=0.0f;
    for (uint wi = lid; wi < wpr; wi += 32u) {
        uint x = brow[wi];
        int n0=int(x&0xF)-8, n1=int((x>>4)&0xF)-8, n2=int((x>>8)&0xF)-8, n3=int((x>>12)&0xF)-8,
            n4=int((x>>16)&0xF)-8, n5=int((x>>20)&0xF)-8, n6=int((x>>24)&0xF)-8, n7=int((x>>28)&0xF)-8;
        float sc = float(srow[wi>>2]);
        for (uint j=0;j<kk;j++) {
            device const char* a = aq + j*K + wi*8u;
            int gi = n0*int(a[0])+n1*int(a[1])+n2*int(a[2])+n3*int(a[3])+n4*int(a[4])+n5*int(a[5])+n6*int(a[6])+n7*int(a[7]);
            acc[j] += float(gi) * sc;
        }
    }
    for (uint j=0;j<kk;j++) {
        float s = simd_sum(acc[j]);
        if (lid==0) out[j*N + gid] = s*asc[j];
    }
}
`

// bvkThreadgroupBytes returns the threadgroup staging footprint (bytes) gemv_w4a8_bvk_bias/_plain
// need to stage M rows of a K-wide activation block — 2 bytes/element (short-staged), the same
// convention maxThreadgroupStageBytes (metal/model.go) uses for the M=1 case. gemv_w4a8_bvk_coal
// needs none (it reads activations from device memory, not threadgroup memory).
func bvkThreadgroupBytes(K, M int) int { return 2 * K * M }
