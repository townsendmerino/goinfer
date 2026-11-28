package wgsl

import "fmt"

// MatmulTiledW8A8KernelWGSL — Tiled W8A8 GEMM (the prefill, M>1 kernel). The naive matmul re-streams a full
// weight row and activation row from global memory per output element — fine when
// bandwidth-bound at M=1, but at prefill it's compute/traffic-bound and wastes
// reuse. This stages 16×16 int8 tiles of the activation and weight into workgroup
// shared memory, so each loaded element is reused across the tile (arithmetic
// intensity ↑). One workgroup computes a 16×16 output block; K is walked in
// 16-wide strips (kp is a multiple of 16). Math identical to MatmulBTW8A8.
//
// The inner product over each 4-int8-packed word is factored out as dot4 so the
// tiling/scheduling code is written once and shared by two dot4 implementations
// (see below) rather than duplicating the whole kernel.
const MatmulTiledW8A8KernelWGSL = `
struct Dims { m: u32, kp: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       aq:      array<u32>;  // [M, kp/4] packed int8
@group(0) @binding(1) var<storage, read>       bq:      array<u32>;  // [N, kp/4]
@group(0) @binding(2) var<storage, read>       aScales: array<f32>;  // [M]
@group(0) @binding(3) var<storage, read>       bScales: array<f32>;  // [N]
@group(0) @binding(4) var<storage, read_write> dst:     array<f32>;  // [M, N]
@group(0) @binding(5) var<uniform>             dims:    Dims;

// Tile PACKED WORDS, not int8: each u32 (4 int8 = 4 K) is loaded into shared once
// and reused across the 16×16 output tile, then unpacked in the inner product.
// (Loading individual int8 would re-fetch each word 4× and lose to the naive
// kernel, which already gets free 4-way reuse per word.) Strip = 16 words = 64 K.
const TS: u32 = 16u;
var<workgroup> As: array<u32, 256>;  // [16 rows][16 words]
var<workgroup> Bs: array<u32, 256>;  // [16 cols][16 words]

%s

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let kw = dims.kp / 4u;                 // words per row
    let row = wid.y * TS + lid.y;          // output row (m)
    let col = wid.x * TS + lid.x;          // output col (n)
    let nStrips = (kw + TS - 1u) / TS;     // 16 words per strip
    var acc: i32 = 0;
    for (var s: u32 = 0u; s < nStrips; s = s + 1u) {
        let wcol = s * TS + lid.x;         // word column in this strip
        var aw: u32 = 0u;
        if (row < dims.m && wcol < kw) { aw = aq[row * kw + wcol]; }
        As[lid.y * TS + lid.x] = aw;
        let brow = wid.x * TS + lid.y;     // Bs[lid.y][lid.x] = bq word (colBase+lid.y, wcol)
        var bw: u32 = 0u;
        if (brow < dims.n && wcol < kw) { bw = bq[brow * kw + wcol]; }
        Bs[lid.y * TS + lid.x] = bw;
        workgroupBarrier();
        for (var w: u32 = 0u; w < TS; w = w + 1u) {
            acc = acc + dot4(As[lid.y * TS + w], Bs[lid.x * TS + w]);
        }
        workgroupBarrier();
    }
    if (row < dims.m && col < dims.n) {
        dst[row * dims.n + col] = f32(acc) * aScales[row] * bScales[col];
    }
}
`

// Dot4UnpackedWGSL — every backend accepts this — four sign-extended int8 unpacked
// from the unpackged word and multiply-accumulated in scalar ALU ops.
const Dot4UnpackedWGSL = `
fn unpack_i8x4(w: u32) -> vec4<i32> {
    return vec4<i32>(i32(w << 24u) >> 24u, i32(w << 16u) >> 24u, i32(w << 8u) >> 24u, i32(w) >> 24u);
}
fn dot4(a: u32, b: u32) -> i32 {
    let av = unpack_i8x4(a);
    let bv = unpack_i8x4(b);
    return av.x*bv.x + av.y*bv.y + av.z*bv.z + av.w*bv.w;
}
`

// Dot4PackedWGSL — the native WGSL builtin (gfx-rs/wgpu PR #7494/#7595), only used
// when Context.hasDP4A (probed live in New()) — on a backend that lowers it to real
// hardware DP4A (e.g. Vulkan's VK_KHR_shader_integer_dot_product on a DP4A-capable
// NVIDIA part), this is what clears the compute/bandwidth wall the tiled GEMM was
// gated on. Same math as the unpacked fallback —
// dot4I8Packed treats the u32 as four little-endian sign-extended int8 lanes,
// verified against the unpacked path's parity gate (TestMatmulW8A8Tiled_dp4aParity).
const Dot4PackedWGSL = `
fn dot4(a: u32, b: u32) -> i32 {
    return dot4I8Packed(a, b);
}
`

var (
	MatmulTiledW8A8ShaderWGSL     = fmt.Sprintf(MatmulTiledW8A8KernelWGSL, Dot4UnpackedWGSL)
	MatmulTiledW8A8DP4AShaderWGSL = fmt.Sprintf(MatmulTiledW8A8KernelWGSL, Dot4PackedWGSL)
)

// MatmulTiledW8A8BiasKernelWGSL is MatmulTiledW8A8KernelWGSL with one addition: a per-output-column bias added in the SAME
// expression as the dequant multiply (`f32(acc) * aScales[row] * bScales[col] + bias[col]`), with textually identical operand
// order to GemvW8A8BiasShaderWGSL's epilogue (gemv.go). That is the point: a multiply and an add in two dispatches go
// through an f32 round-trip through memory, while one WGSL expression may be FMA-contracted, and the last-bit differences flip
// int8 requantization buckets and compound across layers into a real logit divergence. PrefillLastW8A8 uses this kernel to
// dispatch the identical epilogue gemvBias does, for M rows at once; do not split the epilogue back into a separate add.
// History and figures: docs/code-notes/gpu.md#MatmulTiledW8A8BiasKernelWGSL.
const MatmulTiledW8A8BiasKernelWGSL = `
struct Dims { m: u32, kp: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       aq:      array<u32>;  // [M, kp/4] packed int8
@group(0) @binding(1) var<storage, read>       bq:      array<u32>;  // [N, kp/4]
@group(0) @binding(2) var<storage, read>       aScales: array<f32>;  // [M]
@group(0) @binding(3) var<storage, read>       bScales: array<f32>;  // [N]
@group(0) @binding(4) var<storage, read_write> dst:     array<f32>;  // [M, N]
@group(0) @binding(5) var<uniform>             dims:    Dims;
@group(0) @binding(6) var<storage, read>       bias:    array<f32>;  // [N]

const TS: u32 = 16u;
var<workgroup> As: array<u32, 256>;
var<workgroup> Bs: array<u32, 256>;

%s

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let kw = dims.kp / 4u;
    let row = wid.y * TS + lid.y;
    let col = wid.x * TS + lid.x;
    let nStrips = (kw + TS - 1u) / TS;
    var acc: i32 = 0;
    for (var s: u32 = 0u; s < nStrips; s = s + 1u) {
        let wcol = s * TS + lid.x;
        var aw: u32 = 0u;
        if (row < dims.m && wcol < kw) { aw = aq[row * kw + wcol]; }
        As[lid.y * TS + lid.x] = aw;
        let brow = wid.x * TS + lid.y;
        var bw: u32 = 0u;
        if (brow < dims.n && wcol < kw) { bw = bq[brow * kw + wcol]; }
        Bs[lid.y * TS + lid.x] = bw;
        workgroupBarrier();
        for (var w: u32 = 0u; w < TS; w = w + 1u) {
            acc = acc + dot4(As[lid.y * TS + w], Bs[lid.x * TS + w]);
        }
        workgroupBarrier();
    }
    if (row < dims.m && col < dims.n) {
        dst[row * dims.n + col] = f32(acc) * aScales[row] * bScales[col] + bias[col];
    }
}
`

var (
	MatmulTiledW8A8BiasShaderWGSL     = fmt.Sprintf(MatmulTiledW8A8BiasKernelWGSL, Dot4UnpackedWGSL)
	MatmulTiledW8A8DP4ABiasShaderWGSL = fmt.Sprintf(MatmulTiledW8A8BiasKernelWGSL, Dot4PackedWGSL)
)

// MatmulRB64W8A8KernelWGSL — Register-blocked tiled W8A8 GEMM (docs/measurements/webgpu-prefill-profile-2026-09-22.md).
//
// The 16×16 kernel (MatmulTiledW8A8KernelWGSL) gives each thread ONE output: per packed word of K it does two
// shared-memory loads for one dot4. This kernel keeps the same staging idea but gives each thread a 4×4 block of outputs in
// a 64×64 workgroup tile: per word of K, eight shared loads feed sixteen dot4s, four times the arithmetic per byte moved
// through shared memory, with the accumulators in registers (four vec4<i32>).
//
// The output assignment is STRIDED, not contiguous: thread (lid.x, lid.y) owns rows lid.y + {0,16,32,48} and columns
// lid.x + {0,16,32,48}, so that a warp's shared reads (Bs[w][lid.x + 16j]) and its dst writes are consecutive across lanes,
// and the shared tiles are stored word-major with a padded row stride (LD = 65) so the staging stores (consecutive lanes
// write consecutive w for one r) do not bank-conflict.
//
// BIT-IDENTICAL TO THE 16×16 KERNEL BY CONSTRUCTION. The K reduction is an exact i32 sum (|acc| ≤ K·127², far under 2³¹),
// summed in the same ascending-K order, and the dequant epilogue is the identical single expression
// (`f32(acc) * aScales[row] * bScales[col]`, `+ bias[col]` in the bias form). Pinned by TestTiledRB64_bitIdentical across
// aligned and ragged shapes. Profile figures: docs/code-notes/gpu.md#Register-blocked tiled W8A8 GEMM.
const MatmulRB64W8A8KernelWGSL = `
struct Dims { m: u32, kp: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       aq:      array<u32>;  // [M, kp/4] packed int8
@group(0) @binding(1) var<storage, read>       bq:      array<u32>;  // [N, kp/4]
@group(0) @binding(2) var<storage, read>       aScales: array<f32>;  // [M]
@group(0) @binding(3) var<storage, read>       bScales: array<f32>;  // [N]
@group(0) @binding(4) var<storage, read_write> dst:     array<f32>;  // [M, N]
@group(0) @binding(5) var<uniform>             dims:    Dims;

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;   // packed words per K strip = 64 K
const LD: u32 = 65u;   // padded row stride of the word-major shared tiles
var<workgroup> As: array<u32, 1040>;  // [BK words][LD]: As[w*LD + r] = aq(rowBase+r, strip*BK+w)
var<workgroup> Bs: array<u32, 1040>;  // [BK words][LD]: Bs[w*LD + c] = bq(colBase+c, strip*BK+w)

%s

fn store4(acc: vec4<i32>, row: u32, col0: u32) {
    if (row >= dims.m) { return; }
    let base = row * dims.n;
    let sa = aScales[row];
    if (col0        < dims.n) { dst[base + col0]        = f32(acc.x) * sa * bScales[col0]; }
    if (col0 + 16u  < dims.n) { dst[base + col0 + 16u]  = f32(acc.y) * sa * bScales[col0 + 16u]; }
    if (col0 + 32u  < dims.n) { dst[base + col0 + 32u]  = f32(acc.z) * sa * bScales[col0 + 32u]; }
    if (col0 + 48u  < dims.n) { dst[base + col0 + 48u]  = f32(acc.w) * sa * bScales[col0 + 48u]; }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let kw = dims.kp / 4u;
    let rowBase = wid.y * BM;
    let colBase = wid.x * BN;
    let t = lid.y * 16u + lid.x;
    let nStrips = (kw + BK - 1u) / BK;
    var acc0 = vec4<i32>(0);
    var acc1 = vec4<i32>(0);
    var acc2 = vec4<i32>(0);
    var acc3 = vec4<i32>(0);
    for (var s: u32 = 0u; s < nStrips; s = s + 1u) {
        let w0 = s * BK;
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let idx = t + q * 256u;
            let r = idx >> 4u;
            let w = idx & 15u;
            let wcol = w0 + w;
            var aw: u32 = 0u;
            let ar = rowBase + r;
            if (ar < dims.m && wcol < kw) { aw = aq[ar * kw + wcol]; }
            As[w * LD + r] = aw;
            var bw: u32 = 0u;
            let br = colBase + r;
            if (br < dims.n && wcol < kw) { bw = bq[br * kw + wcol]; }
            Bs[w * LD + r] = bw;
        }
        workgroupBarrier();
        for (var w: u32 = 0u; w < BK; w = w + 1u) {
            let ab = w * LD + lid.y;
            let bb = w * LD + lid.x;
            let a0 = As[ab];
            let a1 = As[ab + 16u];
            let a2 = As[ab + 32u];
            let a3 = As[ab + 48u];
            let b0 = Bs[bb];
            let b1 = Bs[bb + 16u];
            let b2 = Bs[bb + 32u];
            let b3 = Bs[bb + 48u];
            acc0 = acc0 + vec4<i32>(dot4(a0, b0), dot4(a0, b1), dot4(a0, b2), dot4(a0, b3));
            acc1 = acc1 + vec4<i32>(dot4(a1, b0), dot4(a1, b1), dot4(a1, b2), dot4(a1, b3));
            acc2 = acc2 + vec4<i32>(dot4(a2, b0), dot4(a2, b1), dot4(a2, b2), dot4(a2, b3));
            acc3 = acc3 + vec4<i32>(dot4(a3, b0), dot4(a3, b1), dot4(a3, b2), dot4(a3, b3));
        }
        workgroupBarrier();
    }
    let col0 = colBase + lid.x;
    store4(acc0, rowBase + lid.y,       col0);
    store4(acc1, rowBase + lid.y + 16u, col0);
    store4(acc2, rowBase + lid.y + 32u, col0);
    store4(acc3, rowBase + lid.y + 48u, col0);
}
`

// MatmulRB64W8A8BiasKernelWGSL is the register-blocked kernel with the fused per-column
// bias epilogue — the same single expression MatmulTiledW8A8BiasKernelWGSL uses
// (`f32(acc) * aScales[row] * bScales[col] + bias[col]`), for the same reason.
const MatmulRB64W8A8BiasKernelWGSL = `
struct Dims { m: u32, kp: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       aq:      array<u32>;
@group(0) @binding(1) var<storage, read>       bq:      array<u32>;
@group(0) @binding(2) var<storage, read>       aScales: array<f32>;
@group(0) @binding(3) var<storage, read>       bScales: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst:     array<f32>;
@group(0) @binding(5) var<uniform>             dims:    Dims;
@group(0) @binding(6) var<storage, read>       bias:    array<f32>;  // [N]

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;
const LD: u32 = 65u;
var<workgroup> As: array<u32, 1040>;
var<workgroup> Bs: array<u32, 1040>;

%s

fn store4(acc: vec4<i32>, row: u32, col0: u32) {
    if (row >= dims.m) { return; }
    let base = row * dims.n;
    let sa = aScales[row];
    if (col0        < dims.n) { dst[base + col0]        = f32(acc.x) * sa * bScales[col0]        + bias[col0]; }
    if (col0 + 16u  < dims.n) { dst[base + col0 + 16u]  = f32(acc.y) * sa * bScales[col0 + 16u]  + bias[col0 + 16u]; }
    if (col0 + 32u  < dims.n) { dst[base + col0 + 32u]  = f32(acc.z) * sa * bScales[col0 + 32u]  + bias[col0 + 32u]; }
    if (col0 + 48u  < dims.n) { dst[base + col0 + 48u]  = f32(acc.w) * sa * bScales[col0 + 48u]  + bias[col0 + 48u]; }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let kw = dims.kp / 4u;
    let rowBase = wid.y * BM;
    let colBase = wid.x * BN;
    let t = lid.y * 16u + lid.x;
    let nStrips = (kw + BK - 1u) / BK;
    var acc0 = vec4<i32>(0);
    var acc1 = vec4<i32>(0);
    var acc2 = vec4<i32>(0);
    var acc3 = vec4<i32>(0);
    for (var s: u32 = 0u; s < nStrips; s = s + 1u) {
        let w0 = s * BK;
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let idx = t + q * 256u;
            let r = idx >> 4u;
            let w = idx & 15u;
            let wcol = w0 + w;
            var aw: u32 = 0u;
            let ar = rowBase + r;
            if (ar < dims.m && wcol < kw) { aw = aq[ar * kw + wcol]; }
            As[w * LD + r] = aw;
            var bw: u32 = 0u;
            let br = colBase + r;
            if (br < dims.n && wcol < kw) { bw = bq[br * kw + wcol]; }
            Bs[w * LD + r] = bw;
        }
        workgroupBarrier();
        for (var w: u32 = 0u; w < BK; w = w + 1u) {
            let ab = w * LD + lid.y;
            let bb = w * LD + lid.x;
            let a0 = As[ab];
            let a1 = As[ab + 16u];
            let a2 = As[ab + 32u];
            let a3 = As[ab + 48u];
            let b0 = Bs[bb];
            let b1 = Bs[bb + 16u];
            let b2 = Bs[bb + 32u];
            let b3 = Bs[bb + 48u];
            acc0 = acc0 + vec4<i32>(dot4(a0, b0), dot4(a0, b1), dot4(a0, b2), dot4(a0, b3));
            acc1 = acc1 + vec4<i32>(dot4(a1, b0), dot4(a1, b1), dot4(a1, b2), dot4(a1, b3));
            acc2 = acc2 + vec4<i32>(dot4(a2, b0), dot4(a2, b1), dot4(a2, b2), dot4(a2, b3));
            acc3 = acc3 + vec4<i32>(dot4(a3, b0), dot4(a3, b1), dot4(a3, b2), dot4(a3, b3));
        }
        workgroupBarrier();
    }
    let col0 = colBase + lid.x;
    store4(acc0, rowBase + lid.y,       col0);
    store4(acc1, rowBase + lid.y + 16u, col0);
    store4(acc2, rowBase + lid.y + 32u, col0);
    store4(acc3, rowBase + lid.y + 48u, col0);
}
`

var (
	MatmulRB64W8A8ShaderWGSL         = fmt.Sprintf(MatmulRB64W8A8KernelWGSL, Dot4UnpackedWGSL)
	MatmulRB64W8A8DP4AShaderWGSL     = fmt.Sprintf(MatmulRB64W8A8KernelWGSL, Dot4PackedWGSL)
	MatmulRB64W8A8BiasShaderWGSL     = fmt.Sprintf(MatmulRB64W8A8BiasKernelWGSL, Dot4UnpackedWGSL)
	MatmulRB64W8A8DP4ABiasShaderWGSL = fmt.Sprintf(MatmulRB64W8A8BiasKernelWGSL, Dot4PackedWGSL)
)

const GemmRowW8A8ShaderWGSL = `
struct Dims { m: u32, kp: u32, n: u32, pad: u32 };

@group(0) @binding(0) var<storage, read>       aq:      array<vec4<u32>>;  // [M, kp/16]
@group(0) @binding(1) var<storage, read>       bq:      array<vec4<u32>>;  // [N, kp/16]
@group(0) @binding(2) var<storage, read>       aScales: array<f32>;        // [M]
@group(0) @binding(3) var<storage, read>       bScales: array<f32>;        // [N]
@group(0) @binding(4) var<storage, read_write> dst:     array<f32>;        // [M*N] row-major (m*N + n)
@group(0) @binding(5) var<uniform>             dims:    Dims;

fn unpack_i8x4(w: u32) -> vec4<i32> {
    return vec4<i32>(i32(w << 24u) >> 24u, i32(w << 16u) >> 24u, i32(w << 8u) >> 24u, i32(w) >> 24u);
}
fn dotw(a: u32, b: u32) -> i32 {
    let av = unpack_i8x4(a);
    let bv = unpack_i8x4(b);
    return av.x*bv.x + av.y*bv.y + av.z*bv.z + av.w*bv.w;
}

var<workgroup> partial: array<i32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = wid.x + wid.y * 32768u;   // 2D grid: N can exceed the 65535/dim cap
    if (n >= dims.n) { return; }      // n is workgroup-uniform ⇒ barrier-safe
    let t = lid.x;
    let kv = dims.kp / 16u;
    let bBase = n * kv;
    let M = dims.m;                   // uniform ⇒ the M-loops are uniform control flow

    var acc: array<i32, 16>;          // gemmRowMaxM accumulators
    for (var m: u32 = 0u; m < M; m = m + 1u) { acc[m] = 0; }

    for (var v: u32 = t; v < kv; v = v + 64u) {   // warp reads consecutive weight vec4s ⇒ coalesced
        let b4 = bq[bBase + v];                    // weight loaded ONCE, reused across all M rows
        for (var m: u32 = 0u; m < M; m = m + 1u) {
            let a4 = aq[m * kv + v];
            acc[m] = acc[m] + dotw(a4.x, b4.x) + dotw(a4.y, b4.y) + dotw(a4.z, b4.z) + dotw(a4.w, b4.w);
        }
    }

    for (var m: u32 = 0u; m < M; m = m + 1u) {   // reduce each row, reusing the one shared buffer
        partial[t] = acc[m];
        workgroupBarrier();
        var stride: u32 = 32u;
        loop {
            if (stride == 0u) { break; }
            if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
            workgroupBarrier();
            stride = stride / 2u;
        }
        if (t == 0u) {
            dst[m * dims.n + n] = f32(partial[0]) * aScales[m] * bScales[n];
        }
        workgroupBarrier();
    }
}
`

// GemmRowsW4A8ShaderWGSL is GemvW4A8ShaderWGSL with one accumulator per row: a workgroup owns output column n, each of
// its 64 lanes loads a weight group once and multiplies it into up to w4MultiMaxRows activation rows. Every row's sum is
// the single-row kernel's own expression in the same order (per lane over its groups, then the lane tree, then the row's
// activation scale), so a row's output equals GemvW4A8ShaderWGSL's bit for bit.
const GemmRowsW4A8ShaderWGSL = `
struct Dims { m: u32, kp: u32, n: u32, row0: u32 };  // m rows this dispatch, starting at row0

@group(0) @binding(0) var<storage, read>       aq:      array<vec4<u32>>;  // [rows, kp/16] int8 act, 16/vec4
@group(0) @binding(1) var<storage, read>       bq:      array<vec4<u32>>;  // [N, kp/32] nibbles, 32/vec4
@group(0) @binding(2) var<storage, read>       aScales: array<f32>;        // [rows]
@group(0) @binding(3) var<storage, read>       bScales: array<u32>;        // [N*kp/32] f16 group scales, 2/u32
@group(0) @binding(4) var<storage, read_write> dst:     array<f32>;        // [rows, N]
@group(0) @binding(5) var<uniform>             dims:    Dims;

fn unpack_i8x4(w: u32) -> vec4<i32> {
    return vec4<i32>(i32(w << 24u) >> 24u, i32(w << 16u) >> 24u, i32(w << 8u) >> 24u, i32(w) >> 24u);
}
fn dot8(ww: u32, alo: u32, ahi: u32) -> i32 {
    let al = unpack_i8x4(alo);
    let ah = unpack_i8x4(ahi);
    var s: i32 = 0;
    s = s + (i32((ww >>  0u) & 0xFu) - 8) * al.x;
    s = s + (i32((ww >>  4u) & 0xFu) - 8) * al.y;
    s = s + (i32((ww >>  8u) & 0xFu) - 8) * al.z;
    s = s + (i32((ww >> 12u) & 0xFu) - 8) * al.w;
    s = s + (i32((ww >> 16u) & 0xFu) - 8) * ah.x;
    s = s + (i32((ww >> 20u) & 0xFu) - 8) * ah.y;
    s = s + (i32((ww >> 24u) & 0xFu) - 8) * ah.z;
    s = s + (i32((ww >> 28u) & 0xFu) - 8) * ah.w;
    return s;
}

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = wid.x + wid.y * 32768u;
    if (n >= dims.n) { return; }   // workgroup-uniform, so the barriers below stay in uniform control flow
    let t = lid.x;
    let ng = dims.kp / 32u;
    let wBase = n * ng;
    let M = dims.m;
    var acc: array<f32, 16>;
    for (var m: u32 = 0u; m < M; m = m + 1u) { acc[m] = 0.0; }
    for (var v: u32 = t; v < ng; v = v + 64u) {
        let w4 = bq[wBase + v];
        let s = wBase + v;
        let pair = unpack2x16float(bScales[s >> 1u]);
        let sc = select(pair.x, pair.y, (s & 1u) == 1u);
        for (var m: u32 = 0u; m < M; m = m + 1u) {
            let ab = (dims.row0 + m) * ng * 2u + v * 2u;
            let a0 = aq[ab];
            let a1 = aq[ab + 1u];
            var idot: i32 = 0;
            idot = idot + dot8(w4.x, a0.x, a0.y);
            idot = idot + dot8(w4.y, a0.z, a0.w);
            idot = idot + dot8(w4.z, a1.x, a1.y);
            idot = idot + dot8(w4.w, a1.z, a1.w);
            acc[m] = acc[m] + f32(idot) * sc;
        }
    }
    for (var m: u32 = 0u; m < M; m = m + 1u) {
        partial[t] = acc[m];
        workgroupBarrier();
        var stride: u32 = 32u;
        loop {
            if (stride == 0u) { break; }
            if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
            workgroupBarrier();
            stride = stride / 2u;
        }
        if (t == 0u) {
            let row = dims.row0 + m;
            dst[row * dims.n + n] = partial[0] * aScales[row];
        }
        workgroupBarrier();
    }
}
`

// MatmulTiledW4A8KernelWGSL is MatmulRB64W8A8KernelWGSL's shape for int4 weights: a 64×64 output tile, each of 256
// threads owning a strided 4×4 block of it, both operands staged into workgroup memory once per 32-wide weight group.
// The nibbles are unpacked to packed int8 as they are staged, so the inner loop is the int8 kernel's dot4; a group's
// integer sum is exact and is scaled by that group's f16 scale as the group ends.
//
// The groups are visited in the order that reproduces GemvW4A8ShaderWGSL's float sum. That kernel gives lane t the
// groups t, t+64, …, summed in turn, then adds the 64 lanes pairwise: lanes 32 apart first, then 16, down to 1. Here the
// lanes are taken in bit-reversed order, which is that tree's leaves from left to right, and a seven-level stack adds
// each finished subtree to its sibling. Every output is then the single-row kernel's own additions in its own pairing,
// so a row's output equals GemvW4A8ShaderWGSL's bit for bit.
const MatmulTiledW4A8KernelWGSL = `
struct Dims { m: u32, kp: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       aq:      array<u32>;  // [M, kp/4] packed int8
@group(0) @binding(1) var<storage, read>       bq:      array<u32>;  // [N, kp/8] packed nibbles
@group(0) @binding(2) var<storage, read>       aScales: array<f32>;  // [M]
@group(0) @binding(3) var<storage, read>       bScales: array<u32>;  // [N, kp/32] f16 group scales, 2/u32
@group(0) @binding(4) var<storage, read_write> dst:     array<f32>;  // [M, N]
@group(0) @binding(5) var<uniform>             dims:    Dims;

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 8u;    // packed int8 words per weight group
const LD: u32 = 65u;   // padded row stride of the word-major shared tiles
var<workgroup> As: array<u32, 520>;  // [BK words][LD]: As[w*LD + r]
var<workgroup> Bs: array<u32, 520>;  // [BK words][LD]: Bs[w*LD + c], the nibbles unpacked to int8
var<workgroup> Ss: array<f32, 64>;   // the group's scale for each of the tile's columns

%s

// nib4 turns four nibbles (value + 8, in the low 16 bits) into four little-endian int8 bytes.
fn nib4(x16: u32) -> u32 {
    var x = (x16 | (x16 << 8u)) & 0x00FF00FFu;
    x = (x | (x << 4u)) & 0x0F0F0F0Fu;
    let y = x ^ 0x08080808u;
    return y | (((y >> 3u) & 0x01010101u) * 0xF0u);
}

fn store4(acc: vec4<f32>, row: u32, col0: u32) {
    if (row >= dims.m) { return; }
    let base = row * dims.n;
    let sa = aScales[row];
    if (col0        < dims.n) { dst[base + col0]        = acc.x * sa; }
    if (col0 + 16u  < dims.n) { dst[base + col0 + 16u]  = acc.y * sa; }
    if (col0 + 32u  < dims.n) { dst[base + col0 + 32u]  = acc.z * sa; }
    if (col0 + 48u  < dims.n) { dst[base + col0 + 48u]  = acc.w * sa; }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let kw = dims.kp / 4u;    // packed int8 words in an activation row
    let nwr = dims.kp / 8u;   // nibble words in a weight row
    let ng = dims.kp / 32u;   // groups in a weight row
    let rowBase = wid.y * BM;
    let colBase = wid.x * BN;
    let t = lid.y * 16u + lid.x;
    var st: array<vec4<f32>, 28>;   // [7 levels][4 row blocks]: finished subtrees waiting for their siblings
    for (var leaf: u32 = 0u; leaf < 64u; leaf = leaf + 1u) {
        let lane = reverseBits(leaf) >> 26u;
        var f0 = vec4<f32>(0.0);
        var f1 = vec4<f32>(0.0);
        var f2 = vec4<f32>(0.0);
        var f3 = vec4<f32>(0.0);
        for (var g: u32 = lane; g < ng; g = g + 64u) {
            let w0 = g * BK;
            for (var q: u32 = 0u; q < 2u; q = q + 1u) {
                let idx = t + q * 256u;
                let r = idx >> 3u;
                let w = idx & 7u;
                var aw: u32 = 0u;
                let ar = rowBase + r;
                if (ar < dims.m) { aw = aq[ar * kw + w0 + w]; }
                As[w * LD + r] = aw;
            }
            {
                let c = t >> 2u;
                let nw = t & 3u;
                var lo: u32 = 0u;
                var hi: u32 = 0u;
                let bc = colBase + c;
                if (bc < dims.n) {
                    let ww = bq[bc * nwr + g * 4u + nw];
                    lo = nib4(ww & 0xFFFFu);
                    hi = nib4(ww >> 16u);
                }
                Bs[(2u * nw) * LD + c] = lo;
                Bs[(2u * nw + 1u) * LD + c] = hi;
            }
            if (t < 64u) {
                let bc = colBase + t;
                var sc: f32 = 0.0;
                if (bc < dims.n) {
                    let si = bc * ng + g;
                    let pair = unpack2x16float(bScales[si >> 1u]);
                    sc = select(pair.x, pair.y, (si & 1u) == 1u);
                }
                Ss[t] = sc;
            }
            workgroupBarrier();
            var i0 = vec4<i32>(0);
            var i1 = vec4<i32>(0);
            var i2 = vec4<i32>(0);
            var i3 = vec4<i32>(0);
            for (var w: u32 = 0u; w < BK; w = w + 1u) {
                let ab = w * LD + lid.y;
                let bb = w * LD + lid.x;
                let a0 = As[ab];
                let a1 = As[ab + 16u];
                let a2 = As[ab + 32u];
                let a3 = As[ab + 48u];
                let b0 = Bs[bb];
                let b1 = Bs[bb + 16u];
                let b2 = Bs[bb + 32u];
                let b3 = Bs[bb + 48u];
                i0 = i0 + vec4<i32>(dot4(a0, b0), dot4(a0, b1), dot4(a0, b2), dot4(a0, b3));
                i1 = i1 + vec4<i32>(dot4(a1, b0), dot4(a1, b1), dot4(a1, b2), dot4(a1, b3));
                i2 = i2 + vec4<i32>(dot4(a2, b0), dot4(a2, b1), dot4(a2, b2), dot4(a2, b3));
                i3 = i3 + vec4<i32>(dot4(a3, b0), dot4(a3, b1), dot4(a3, b2), dot4(a3, b3));
            }
            let s4 = vec4<f32>(Ss[lid.x], Ss[lid.x + 16u], Ss[lid.x + 32u], Ss[lid.x + 48u]);
            f0 = f0 + vec4<f32>(i0) * s4;
            f1 = f1 + vec4<f32>(i1) * s4;
            f2 = f2 + vec4<f32>(i2) * s4;
            f3 = f3 + vec4<f32>(i3) * s4;
            workgroupBarrier();
        }
        // This lane is a finished leaf: add it to every finished sibling subtree of its own size, smallest first.
        var cnt = leaf + 1u;
        var lvl: u32 = 0u;
        loop {
            if ((cnt & 1u) != 0u) { break; }
            f0 = st[lvl * 4u] + f0;
            f1 = st[lvl * 4u + 1u] + f1;
            f2 = st[lvl * 4u + 2u] + f2;
            f3 = st[lvl * 4u + 3u] + f3;
            lvl = lvl + 1u;
            cnt = cnt >> 1u;
        }
        st[lvl * 4u] = f0;
        st[lvl * 4u + 1u] = f1;
        st[lvl * 4u + 2u] = f2;
        st[lvl * 4u + 3u] = f3;
    }
    let col0 = colBase + lid.x;
    store4(st[24], rowBase + lid.y,       col0);
    store4(st[25], rowBase + lid.y + 16u, col0);
    store4(st[26], rowBase + lid.y + 32u, col0);
    store4(st[27], rowBase + lid.y + 48u, col0);
}
`

// GemvW8A8ShaderWGSL — Coalesced W8A8 GEMV (the M=1 decode kernel). The naive matmul assigns thread
// (m,n) to one output and has it stream a whole weight row — so adjacent threads
// (n, n+1) read rows kw words apart: UNCOALESCED, ~6% of bandwidth, slower than
// the CPU at decode. This kernel instead assigns one WORKGROUP per output column
// n: its 64 lanes cooperatively stride the contiguous weight row bq[n], so a warp
// reads consecutive words (coalesced), then a tree-reduce sums the partials.
// Decode is bandwidth-bound, so coalescing the weight stream is the actual unlock.
const GemvW8A8ShaderWGSL = `
struct Dims { m: u32, kp: u32, n: u32, addResidual: u32 };  // addResidual=1 ⇒ dst[n] += result (fused residual epilogue)

// vec4<u32> view of the packed int8 (16 int8 / 16-byte transaction): wider loads
// raise memory throughput vs scalar u32. kp is a multiple of 16, so kp/16 vec4s.
@group(0) @binding(0) var<storage, read>       aq:      array<vec4<u32>>;  // [1, kp/16]
@group(0) @binding(1) var<storage, read>       bq:      array<vec4<u32>>;  // [N, kp/16]
@group(0) @binding(2) var<storage, read>       aScales: array<f32>;        // [1]
@group(0) @binding(3) var<storage, read>       bScales: array<f32>;        // [N]
@group(0) @binding(4) var<storage, read_write> dst:     array<f32>;        // [N]
@group(0) @binding(5) var<uniform>             dims:    Dims;

fn unpack_i8x4(w: u32) -> vec4<i32> {
    return vec4<i32>(i32(w << 24u) >> 24u, i32(w << 16u) >> 24u, i32(w << 8u) >> 24u, i32(w) >> 24u);
}
fn dotw(a: u32, b: u32) -> i32 {
    let av = unpack_i8x4(a);
    let bv = unpack_i8x4(b);
    return av.x*bv.x + av.y*bv.y + av.z*bv.z + av.w*bv.w;
}

var<workgroup> partial: array<i32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = wid.x + wid.y * 32768u;  // 2D grid: vocab-sized N exceeds the 65535/dim cap
    if (n >= dims.n) { return; }
    let t = lid.x;
    let kv = dims.kp / 16u;     // vec4s per row
    let bBase = n * kv;
    var acc: i32 = 0;
    for (var v: u32 = t; v < kv; v = v + 64u) {   // warp reads consecutive vec4s ⇒ coalesced
        let a4 = aq[v];
        let b4 = bq[bBase + v];
        acc = acc + dotw(a4.x, b4.x) + dotw(a4.y, b4.y) + dotw(a4.z, b4.z) + dotw(a4.w, b4.w);
    }
    partial[t] = acc;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) {
        let r = f32(partial[0]) * aScales[0] * bScales[n];
        // addResidual folds the residual-add into the matmul epilogue: dst is the
        // running hidden state, dst[n] += r deletes a standalone residual dispatch
        // (and its barrier link). Default 0 ⇒ overwrite, unchanged for all other
        // callers (they pass dims._pad/addResidual = 0).
        if (dims.addResidual == 1u) { dst[n] = dst[n] + r; } else { dst[n] = r; }
    }
}
`

// GemvW8A8BiasShaderWGSL is gemvW8A8 with a per-output bias folded into the epilogue:
// dst[n] = scale·acc + bias[n]. It deletes the standalone biasAdd dispatch (and its
// barrier link) for the q/k/v projections of bias models (Qwen2/Qwen2.5) — a
// dependent-chain fusion (biasAdd reads the GEMV output), so it saves barrier cost,
// not just launch cost. W8A8 only (the resident bias models are int8); the body is
// byte-identical to gemvW8A8 except for the extra binding and the `+ bias[n]`.
const GemvW8A8BiasShaderWGSL = `
struct Dims { m: u32, kp: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       aq:      array<vec4<u32>>;  // [1, kp/16]
@group(0) @binding(1) var<storage, read>       bq:      array<vec4<u32>>;  // [N, kp/16]
@group(0) @binding(2) var<storage, read>       aScales: array<f32>;        // [1]
@group(0) @binding(3) var<storage, read>       bScales: array<f32>;        // [N]
@group(0) @binding(4) var<storage, read_write> dst:     array<f32>;        // [N]
@group(0) @binding(5) var<uniform>             dims:    Dims;
@group(0) @binding(6) var<storage, read>       bias:    array<f32>;        // [N]

fn unpack_i8x4(w: u32) -> vec4<i32> {
    return vec4<i32>(i32(w << 24u) >> 24u, i32(w << 16u) >> 24u, i32(w << 8u) >> 24u, i32(w) >> 24u);
}
fn dotw(a: u32, b: u32) -> i32 {
    let av = unpack_i8x4(a);
    let bv = unpack_i8x4(b);
    return av.x*bv.x + av.y*bv.y + av.z*bv.z + av.w*bv.w;
}

var<workgroup> partial: array<i32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = wid.x + wid.y * 32768u;
    if (n >= dims.n) { return; }
    let t = lid.x;
    let kv = dims.kp / 16u;
    let bBase = n * kv;
    var acc: i32 = 0;
    for (var v: u32 = t; v < kv; v = v + 64u) {
        let a4 = aq[v];
        let b4 = bq[bBase + v];
        acc = acc + dotw(a4.x, b4.x) + dotw(a4.y, b4.y) + dotw(a4.z, b4.z) + dotw(a4.w, b4.w);
    }
    partial[t] = acc;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) {
        dst[n] = f32(partial[0]) * aScales[0] * bScales[n] + bias[n];
    }
}
`

const GemvW4A8ShaderWGSL = `
struct Dims { m: u32, kp: u32, n: u32, addResidual: u32 };  // kp = K padded to 32; addResidual=1 ⇒ dst[n]+=result

@group(0) @binding(0) var<storage, read>       aq:      array<vec4<u32>>;  // [kp/16] int8 act, 16/vec4
@group(0) @binding(1) var<storage, read>       bq:      array<vec4<u32>>;  // [N*kp/32] nibbles, 32/vec4
@group(0) @binding(2) var<storage, read>       aScale:  array<f32>;        // [1]
@group(0) @binding(3) var<storage, read>       bScales: array<u32>;        // [N*kp/32] f16 group scales, 2/u32
@group(0) @binding(4) var<storage, read_write> dst:     array<f32>;        // [N]
@group(0) @binding(5) var<uniform>             dims:    Dims;

fn unpack_i8x4(w: u32) -> vec4<i32> {
    return vec4<i32>(i32(w << 24u) >> 24u, i32(w << 16u) >> 24u, i32(w << 8u) >> 24u, i32(w) >> 24u);
}
// dot8: 8 int4 nibbles of ww (values nibble−8) · 8 int8 (alo=elems 0–3, ahi=4–7).
fn dot8(ww: u32, alo: u32, ahi: u32) -> i32 {
    let al = unpack_i8x4(alo);
    let ah = unpack_i8x4(ahi);
    var s: i32 = 0;
    s = s + (i32((ww >>  0u) & 0xFu) - 8) * al.x;
    s = s + (i32((ww >>  4u) & 0xFu) - 8) * al.y;
    s = s + (i32((ww >>  8u) & 0xFu) - 8) * al.z;
    s = s + (i32((ww >> 12u) & 0xFu) - 8) * al.w;
    s = s + (i32((ww >> 16u) & 0xFu) - 8) * ah.x;
    s = s + (i32((ww >> 20u) & 0xFu) - 8) * ah.y;
    s = s + (i32((ww >> 24u) & 0xFu) - 8) * ah.z;
    s = s + (i32((ww >> 28u) & 0xFu) - 8) * ah.w;
    return s;
}

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = wid.x + wid.y * 32768u;
    if (n >= dims.n) { return; }
    let t = lid.x;
    let ng = dims.kp / 32u;        // groups per row = weight vec4s per row
    let wBase = n * ng;            // bq vec4 base for this row
    let sBase = n * ng;            // bScales base for this row
    var acc: f32 = 0.0;
    for (var v: u32 = t; v < ng; v = v + 64u) {   // one lane per group, coalesced
        let w4 = bq[wBase + v];                   // 32 nibbles
        let a0 = aq[v * 2u];                       // act elems 0–15 of this group
        let a1 = aq[v * 2u + 1u];                  // act elems 16–31
        var idot: i32 = 0;
        idot = idot + dot8(w4.x, a0.x, a0.y);      // elems 0–7
        idot = idot + dot8(w4.y, a0.z, a0.w);      // elems 8–15
        idot = idot + dot8(w4.z, a1.x, a1.y);      // elems 16–23
        idot = idot + dot8(w4.w, a1.z, a1.w);      // elems 24–31
        let s = sBase + v;                          // f16 scales, 2 per u32
        let pair = unpack2x16float(bScales[s >> 1u]);
        acc = acc + f32(idot) * select(pair.x, pair.y, (s & 1u) == 1u);
    }
    partial[t] = acc;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) {
        let r = partial[0] * aScale[0];
        if (dims.addResidual == 1u) { dst[n] = dst[n] + r; } else { dst[n] = r; }
    }
}
`
