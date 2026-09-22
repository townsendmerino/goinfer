//go:build gpu

package gpu

import (
	"fmt"
	"os"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// Register-blocked tiled W8A8 GEMM (R10, docs/measurements/webgpu-prefill-profile-2026-09-22.md).
//
// The 16×16 kernel (matmulTiledW8A8KernelWGSL) gives each thread ONE output: per packed
// word of K it does two shared-memory loads for one dot4, and profiled at a flat ~1 TFLOPS
// — 81–94% of batched prefill at ~11% of this card's f32 peak. This kernel keeps the same
// staging idea but gives each thread a 4×4 block of outputs in a 64×64 workgroup tile: per
// word of K, eight shared loads feed sixteen dot4s, four times the arithmetic per byte
// moved through shared memory, with the accumulators in registers (four vec4<i32>).
//
// The output assignment is STRIDED, not contiguous — thread (lid.x, lid.y) owns rows
// lid.y + {0,16,32,48} and columns lid.x + {0,16,32,48} — so that a warp's shared reads
// (Bs[w][lid.x + 16j]) and its dst writes are consecutive across lanes, and the shared
// tiles are stored word-major with a padded row stride (LD = 65) so the staging stores
// (consecutive lanes write consecutive w for one r) do not bank-conflict.
//
// BIT-IDENTICAL TO THE 16×16 KERNEL BY CONSTRUCTION. The K reduction is an exact i32 sum
// (|acc| ≤ K·127² ≈ 1.4e8 at K = 8960, far under 2³¹), summed in the same ascending-K
// order, and the dequant epilogue is the identical single expression
// (`f32(acc) * aScales[row] * bScales[col]`, `+ bias[col]` in the bias form). Pinned by
// TestTiledRB64_bitIdentical across aligned and ragged shapes.
const matmulRB64W8A8KernelWGSL = `
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

// matmulRB64W8A8BiasKernelWGSL is the register-blocked kernel with the fused per-column
// bias epilogue — the same single expression matmulTiledW8A8BiasKernelWGSL uses
// (`f32(acc) * aScales[row] * bScales[col] + bias[col]`), for the same reason.
const matmulRB64W8A8BiasKernelWGSL = `
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
	matmulRB64W8A8ShaderWGSL         = fmt.Sprintf(matmulRB64W8A8KernelWGSL, dot4UnpackedWGSL)
	matmulRB64W8A8DP4AShaderWGSL     = fmt.Sprintf(matmulRB64W8A8KernelWGSL, dot4PackedWGSL)
	matmulRB64W8A8BiasShaderWGSL     = fmt.Sprintf(matmulRB64W8A8BiasKernelWGSL, dot4UnpackedWGSL)
	matmulRB64W8A8DP4ABiasShaderWGSL = fmt.Sprintf(matmulRB64W8A8BiasKernelWGSL, dot4PackedWGSL)
)

// gemmTileDefault picks the prefill GEMM tile: 64 (the register-blocked kernel) unless
// GOINFER_WEBGPU_GEMM=tiled16 asks for the 16×16 kernel it replaced.
func gemmTileDefault() int {
	if os.Getenv("GOINFER_WEBGPU_GEMM") == "tiled16" {
		return 16
	}
	return 64
}

// tiledPipes is one compiled tiled-GEMM variant (plain or bias) for one tile size.
type tiledPipes struct {
	sh     *wgpu.ShaderModule
	pl     *wgpu.ComputePipeline
	layout *wgpu.BindGroupLayout
}

// tiledSource returns the shader for (tile, bias) at this context's dot4 flavour.
func (c *Context) tiledSource(tile int, bias bool) (code, label string) {
	switch {
	case tile == 64 && bias:
		code, label = matmulRB64W8A8BiasShaderWGSL, "matmulRB64W8A8Bias"
		if c.hasDP4A {
			code, label = matmulRB64W8A8DP4ABiasShaderWGSL, "matmulRB64W8A8DP4ABias"
		}
	case tile == 64:
		code, label = matmulRB64W8A8ShaderWGSL, "matmulRB64W8A8"
		if c.hasDP4A {
			code, label = matmulRB64W8A8DP4AShaderWGSL, "matmulRB64W8A8DP4A"
		}
	case bias:
		code, label = matmulTiledW8A8BiasShaderWGSL, "matmulTiledW8A8Bias"
		if c.hasDP4A {
			code, label = matmulTiledW8A8DP4ABiasShaderWGSL, "matmulTiledW8A8DP4ABias"
		}
	default:
		code, label = matmulTiledW8A8ShaderWGSL, "matmulTiledW8A8"
		if c.hasDP4A {
			code, label = matmulTiledW8A8DP4AShaderWGSL, "matmulTiledW8A8DP4A"
		}
	}
	return code, label
}

// compileTiled builds one variant and registers its releases.
func (c *Context) compileTiled(tile int, bias bool) (tiledPipes, error) {
	code, label := c.tiledSource(tile, bias)
	sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:      label,
		WGSLSource: &wgpu.ShaderSourceWGSL{Code: code},
	})
	if err != nil {
		return tiledPipes{}, fmt.Errorf("gpu: compile %s: %w", label, err)
	}
	pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   label,
		Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return tiledPipes{}, fmt.Errorf("gpu: create %s pipeline: %w", label, err)
	}
	c.track(sh.Release, pl.Release) // audit C-26: register at creation
	return tiledPipes{sh: sh, pl: pl, layout: c.bgl(pl)}, nil
}

// gemmGrid is the tiled GEMM's workgroup grid for an [M, N] output at the active tile.
func (c *Context) gemmGrid(N, M int) (gx, gy uint32) {
	t := uint32(c.gemmTile)
	return (uint32(N) + t - 1) / t, (uint32(M) + t - 1) / t
}

// SetGEMMTileForTest switches the active prefill GEMM kernel between the 16×16 (the
// pre-R10 kernel) and the 64×64 register-blocked one, for an A/B on one loaded model.
// Both variants stay compiled; only the active pointers and the grid change.
func (c *Context) SetGEMMTileForTest(tile int) error {
	if tile != 16 && tile != 64 {
		return fmt.Errorf("gpu: GEMM tile %d: want 16 or 64", tile)
	}
	c.gemmTile = tile
	if c.tiledPipeline != nil {
		c.tiledPipeline = nil
		if err := c.ensureTiled(); err != nil {
			return err
		}
	}
	if c.tiledBiasPipeline != nil {
		c.tiledBiasPipeline = nil
		if err := c.ensureTiledBias(); err != nil {
			return err
		}
	}
	return nil
}

// GEMMTile reports the active prefill GEMM tile (16 or 64).
func (c *Context) GEMMTile() int { return c.gemmTile }
