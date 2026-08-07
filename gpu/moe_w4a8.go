//go:build gpu

package gpu

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"
)

// moeExpertGEMVW4WGSL is the int4 (W4A8) twin of moeExpertGEMVWGSL: ONE indexed
// stacked expert's group-wise int4 weight × int8 activation. The stacking + index
// addressing (e = idx[slot], row = e*N + n) match the int8 kernel; the int4
// nibble/group-scale unpack matches gemvW4A8ShaderWGSL (group = 32 = one vec4<u32>,
// nibble value = nibble−8, f16 group scales 2-per-u32). Activations are int8 padded
// to a multiple of 32 (the same assumption the dense W4A8 GEMV makes).
const moeExpertGEMVW4WGSL = `
struct Dims { kp: u32, n: u32, slot: u32, mode: u32 };
@group(0) @binding(0) var<storage, read>       aq:      array<vec4<u32>>;  // [kp/16] int8 act, 16/vec4
@group(0) @binding(1) var<storage, read>       bq:      array<vec4<u32>>;  // [nE*N, kp/32] nibbles, 32/vec4
@group(0) @binding(2) var<storage, read>       aScales: array<f32>;        // [1]
@group(0) @binding(3) var<storage, read>       bScales: array<u32>;        // [nE*N, kp/32] f16 group scales, 2/u32
@group(0) @binding(4) var<storage, read_write> dst:     array<f32>;        // [N]
@group(0) @binding(5) var<storage, read>       idx:     array<u32>;        // [k] chosen expert indices
@group(0) @binding(6) var<storage, read>       wgt:     array<f32>;        // [k] chosen expert weights
@group(0) @binding(7) var<uniform>             dims:    Dims;
fn unpack_i8x4w(w: u32) -> vec4<i32> {
    return vec4<i32>(i32(w << 24u) >> 24u, i32(w << 16u) >> 24u, i32(w << 8u) >> 24u, i32(w) >> 24u);
}
// dot8w: 8 int4 nibbles of ww (value nibble−8) · 8 int8 (alo=elems 0–3, ahi=4–7).
fn dot8w(ww: u32, alo: u32, ahi: u32) -> i32 {
    let al = unpack_i8x4w(alo);
    let ah = unpack_i8x4w(ahi);
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
var<workgroup> partw: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = wid.x + wid.y * 32768u;
    if (n >= dims.n) { return; }
    let e = idx[dims.slot];
    let t = lid.x;
    let ng = dims.kp / 32u;             // groups per row
    let row = e * dims.n + n;           // stacked: expert e, output row n
    let wBase = row * ng;
    let sBase = row * ng;
    var acc: f32 = 0.0;
    for (var v: u32 = t; v < ng; v = v + 64u) {
        let w4 = bq[wBase + v];
        let a0 = aq[v * 2u];
        let a1 = aq[v * 2u + 1u];
        var idot: i32 = 0;
        idot = idot + dot8w(w4.x, a0.x, a0.y);
        idot = idot + dot8w(w4.y, a0.z, a0.w);
        idot = idot + dot8w(w4.z, a1.x, a1.y);
        idot = idot + dot8w(w4.w, a1.z, a1.w);
        let s = sBase + v;
        let pair = unpack2x16float(bScales[s >> 1u]);
        acc = acc + f32(idot) * select(pair.x, pair.y, (s & 1u) == 1u);
    }
    partw[t] = acc;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { partw[t] = partw[t] + partw[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) {
        let r = partw[0] * aScales[0];
        if (dims.mode == 1u) { dst[n] = dst[n] + wgt[dims.slot] * r; } else { dst[n] = r; }
    }
}
`

func (c *Context) ensureMoEExpertW4() error {
	if c.moeExpertW4Pipeline != nil {
		return nil
	}
	sh, err := c.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "moeExpertGEMVW4", WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: moeExpertGEMVW4WGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile moeExpertGEMVW4: %w", err)
	}
	pl, err := c.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "moeExpertGEMVW4", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline moeExpertGEMVW4: %w", err)
	}
	c.track(sh.Release, pl.Release) // audit C-26: register at creation
	c.moeExpertW4Shader, c.moeExpertW4Pipeline, c.moeExpertW4Layout = sh, pl, c.bgl(pl)
	return nil
}

// UploadStackedExpertsInt4 packs nE experts' int4 group-wise weights (each [N,K]
// as nibbles 0..15 = value+8, plus per-group f32 scales [N*nGroups], group=32) into
// one resident buffer (expert e's row n at packed-row e*N+n), the layout the indexed
// W4A8 expert GEMV reads. The returned stacked buffer carries w4=true so moeExpert
// dispatches the int4 kernel. K must be a multiple of 32 (the W4A8 group).
func (c *Context) UploadStackedExpertsInt4(nib [][]uint8, scales [][]float32, nE, N, K int) (*ResidentStackedW8A8, error) {
	if nE <= 0 || N <= 0 || K <= 0 || len(nib) < nE || len(scales) < nE {
		return nil, fmt.Errorf("gpu: UploadStackedExpertsInt4 bad dims nE=%d N=%d K=%d", nE, N, K)
	}
	kp := padK32(K)
	ng := kp / w4a8GroupSize // groups per row
	wpr := kp / 8            // u32 words per row (nibbles, 8/u32)
	packed := make([]uint32, nE*N*wpr)
	allScales := make([]float32, nE*N*ng)
	for e := range nE {
		if len(nib[e]) < N*K || len(scales[e]) < N*ng {
			return nil, fmt.Errorf("gpu: UploadStackedExpertsInt4 expert %d too small (nib %d<%d, sc %d<%d)", e, len(nib[e]), N*K, len(scales[e]), N*ng)
		}
		copy(packed[e*N*wpr:(e+1)*N*wpr], packNibbles(nib[e], N, K))
		copy(allScales[e*N*ng:(e+1)*N*ng], scales[e][:N*ng])
	}
	bq, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "moe-experts-w4", Contents: wgpu.ToBytes(packed), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, fmt.Errorf("gpu: stacked w4 experts buffer: %w", err)
	}
	sc, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "moe-expert-w4-scales", Contents: wgpu.ToBytes(packF16Pairs(allScales)), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		bq.Release()
		return nil, fmt.Errorf("gpu: stacked w4 scales buffer: %w", err)
	}
	return &ResidentStackedW8A8{bq: bq, bScales: sc, nE: nE, rows: N, cols: K, kp: kp, w4: true}, nil
}

// UploadStackedExpertsInt4Packed is the fast path of UploadStackedExpertsInt4: each
// expert's q4 is ALREADY the GPU packed layout (decoder int4 == packNibbles when K%32==0,
// TestInt4LayoutMatch), so it just concatenates the per-expert byte slices (one memcpy each,
// ~10× a nibble unpack+repack) and CreateBufferInits — no per-element shuffle. q4[e] is
// expert e's decoder int4 bytes (≥ N*K/2); scales[e] its per-group f32 (≥ N*K/32).
func (c *Context) UploadStackedExpertsInt4Packed(q4 [][]byte, scales [][]float32, nE, N, K int) (*ResidentStackedW8A8, error) {
	if nE <= 0 || N <= 0 || K <= 0 || len(q4) < nE || len(scales) < nE {
		return nil, fmt.Errorf("gpu: UploadStackedExpertsInt4Packed bad dims nE=%d N=%d K=%d", nE, N, K)
	}
	if K%w4a8GroupSize != 0 {
		return nil, fmt.Errorf("gpu: UploadStackedExpertsInt4Packed needs K%%%d==0, got K=%d", w4a8GroupSize, K)
	}
	kp := K
	ng := kp / w4a8GroupSize
	bpe := N * kp / 2 // bytes per expert
	packed := make([]byte, nE*bpe)
	allScales := make([]float32, nE*N*ng)
	for e := range nE {
		if len(q4[e]) < bpe || len(scales[e]) < N*ng {
			return nil, fmt.Errorf("gpu: UploadStackedExpertsInt4Packed expert %d too small (q4 %d<%d, sc %d<%d)", e, len(q4[e]), bpe, len(scales[e]), N*ng)
		}
		copy(packed[e*bpe:(e+1)*bpe], q4[e][:bpe])
		copy(allScales[e*N*ng:(e+1)*N*ng], scales[e][:N*ng])
	}
	bq, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "moe-experts-w4", Contents: packed, Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, fmt.Errorf("gpu: stacked w4 experts buffer: %w", err)
	}
	sc, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "moe-expert-w4-scales", Contents: wgpu.ToBytes(packF16Pairs(allScales)), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		bq.Release()
		return nil, fmt.Errorf("gpu: stacked w4 scales buffer: %w", err)
	}
	return &ResidentStackedW8A8{bq: bq, bScales: sc, nE: nE, rows: N, cols: K, kp: kp, w4: true}, nil
}
