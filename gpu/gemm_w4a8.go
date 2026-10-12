//go:build gpu

package gpu

import (
	"fmt"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// Multi-row int4 projections: M quantized activation rows against one ResidentW4A8, the weight streamed once for all of
// them. Two kernels compute the same product, each bit for bit the single-row kernel's. Both read the activations in the layout quantizeShaderWGSL writes with
// np = the weight's kPad() ([M, kp/4] packed int8 words, one scale a row) and write dst [M, N] row-major.

// w4MultiMaxRows is the most rows one dispatch of the row-shaped kernel takes.
const w4MultiMaxRows = 16

// gemmRowsW4A8ShaderWGSL is gemvW4A8ShaderWGSL with one accumulator per row: a workgroup owns output column n, each of
// its 64 lanes loads a weight group once and multiplies it into up to w4MultiMaxRows activation rows. Every row's sum is
// the single-row kernel's own expression in the same order (per lane over its groups, then the lane tree, then the row's
// activation scale), so a row's output equals gemvW4A8ShaderWGSL's bit for bit.
const gemmRowsW4A8ShaderWGSL = `
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

// matmulTiledW4A8KernelWGSL is matmulRB64W8A8KernelWGSL's shape for int4 weights: a 64×64 output tile, each of 256
// threads owning a strided 4×4 block of it, both operands staged into workgroup memory once per 32-wide weight group.
// The nibbles are unpacked to packed int8 as they are staged, so the inner loop is the int8 kernel's dot4; a group's
// integer sum is exact and is scaled by that group's f16 scale as the group ends.
//
// The groups are visited in the order that reproduces gemvW4A8ShaderWGSL's float sum. That kernel gives lane t the
// groups t, t+64, …, summed in turn, then adds the 64 lanes pairwise: lanes 32 apart first, then 16, down to 1. Here the
// lanes are taken in bit-reversed order, which is that tree's leaves from left to right, and a seven-level stack adds
// each finished subtree to its sibling. Every output is then the single-row kernel's own additions in its own pairing,
// so a row's output equals gemvW4A8ShaderWGSL's bit for bit.
const matmulTiledW4A8KernelWGSL = `
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

// w4MultiKind names one of the two multi-row int4 kernels.
type w4MultiKind int

const (
	w4MultiRows  w4MultiKind = iota // gemmRowsW4A8ShaderWGSL
	w4MultiTiled                    // matmulTiledW4A8KernelWGSL
)

func (k w4MultiKind) String() string {
	if k == w4MultiTiled {
		return "tiled"
	}
	return "rows"
}

// w4MultiVerified lists the WebGPU backends on which the multi-row int4 kernels have been checked bit for bit against
// the single-row kernel and the batched pass against the row-by-row walk (TestW4Multi_matchesSingleRow,
// TestPrefillInt4_matchesWalk). Each kernel's output is a float sum in a fixed order, and a backend's shader compiler
// may evaluate the same expression differently in two kernels, so the identity is a property of a backend, not of the
// source. The batched pass takes an int4 model only on a backend listed here; add one after those tests pass on it.
var w4MultiVerified = map[string]bool{"metal": true, "vulkan": true}

// w4MultiFor is the kernel for a projection over M rows: the row-shaped one while a single dispatch of it holds the
// rows, the tiled one beyond. The tiled kernel computes whole 64-row tiles however few rows there are, so below a
// block of rows it does more work than the row-shaped one. The choice changes speed only: both give the same bits.
func (c *Context) w4MultiFor(M int) w4MultiKind {
	if c.w4MultiPin != 0 {
		return w4MultiKind(c.w4MultiPin - 1)
	}
	if M <= w4MultiMaxRows {
		return w4MultiRows
	}
	return w4MultiTiled
}

// w4MultiSource is kind's shader at this context's dot4 flavour.
func (c *Context) w4MultiSource(kind w4MultiKind) (code, label string) {
	if kind == w4MultiRows {
		return gemmRowsW4A8ShaderWGSL, "gemmRowsW4A8"
	}
	if c.hasDP4A {
		return fmt.Sprintf(matmulTiledW4A8KernelWGSL, dot4PackedWGSL), "matmulTiledW4A8DP4A"
	}
	return fmt.Sprintf(matmulTiledW4A8KernelWGSL, dot4UnpackedWGSL), "matmulTiledW4A8"
}

// ensureW4Multi compiles both multi-row int4 kernels.
func (c *Context) ensureW4Multi() error {
	for kind := range c.w4Multi {
		if c.w4Multi[kind].pl != nil {
			continue
		}
		code, label := c.w4MultiSource(w4MultiKind(kind))
		sh, pl, layout, err := c.mkPipeline(label, code)
		if err != nil {
			return err
		}
		c.w4Multi[kind] = tiledPipes{sh: sh, pl: pl, layout: layout}
	}
	return nil
}

// encodeW4Multi encodes dst[M, N] = aq[M, kp/4] · rmᵀ with kind's kernel into disp. p is kind's pipeline and layout
// (c.w4Multi[kind], or a test's own build of the source). uni makes a uniform buffer and bind a bind group, as the
// caller's encoder does; aq and aScales are the quantized rows, dst is at least M*N floats.
func encodeW4Multi(kind w4MultiKind, p tiledPipes, rm *ResidentW4A8, M int, aq, aScales, dst *wgpu.Buffer,
	uni func([]uint32) *wgpu.Buffer, bind func(*wgpu.BindGroupLayout, ...*wgpu.Buffer) *wgpu.BindGroup,
	disp func(*wgpu.ComputePipeline, *wgpu.BindGroup, uint32, uint32)) {
	N := rm.rows
	if kind == w4MultiTiled {
		d := uni([]uint32{uint32(M), uint32(rm.kp), uint32(N), 0})
		disp(p.pl, bind(p.layout, aq, rm.bq, aScales, rm.bScales, dst, d), (uint32(N)+63)/64, (uint32(M)+63)/64)
		return
	}
	gx, gy := gemvGrid(N)
	for row0 := 0; row0 < M; row0 += w4MultiMaxRows {
		d := uni([]uint32{uint32(min(M-row0, w4MultiMaxRows)), uint32(rm.kp), uint32(N), uint32(row0)})
		disp(p.pl, bind(p.layout, aq, rm.bq, aScales, rm.bScales, dst, d), gx, gy)
	}
}
