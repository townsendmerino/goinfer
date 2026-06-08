//go:build gpu

package gpu

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"
)

// W4A8 decode GEMV: int4 group-wise weights × int8 activation. The decode
// roofline is weight bytes/token; int8 streams ~1.55 GB, int4 ~0.97 GB (½ the
// nibbles + ⅛ the f32 group scales), so this is the lever on the 4.3 ms gemv
// floor (docs/gpu-assessment.md §0.0, W4A8). Format matches aikit's int4-resident
// (`internal/linalg`, group=32, GGUF Q4_K granularity): per row, K/32 groups each
// of 32 nibbles + one f32 scale; element i's nibble = (word>>4i)&0xF, value
// (nibble−8). group=32 nibbles = 16 bytes = exactly one vec4<u32>, so one group
// is one coalesced load. Activations are int8 (per-row aScale, same as W8A8).
const w4a8GroupSize = 32

const gemvW4A8ShaderWGSL = `
struct Dims { m: u32, kp: u32, n: u32, _pad: u32 };  // kp = K padded to mult of 32

@group(0) @binding(0) var<storage, read>       aq:      array<vec4<u32>>;  // [kp/16] int8 act, 16/vec4
@group(0) @binding(1) var<storage, read>       bq:      array<vec4<u32>>;  // [N*kp/32] nibbles, 32/vec4
@group(0) @binding(2) var<storage, read>       aScale:  array<f32>;        // [1]
@group(0) @binding(3) var<storage, read>       bScales: array<f32>;        // [N*kp/32] per-group
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
        acc = acc + f32(idot) * bScales[sBase + v];
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
        dst[n] = partial[0] * aScale[0];
    }
}
`

// ResidentW4A8 is an int4 group-wise weight matrix resident on the GPU.
type ResidentW4A8 struct {
	bq      *wgpu.Buffer // [N, kp/32] vec4<u32> packed nibbles
	bScales *wgpu.Buffer // [N, kp/32] f32 per-group scales
	rows    int          // N
	cols    int          // K (unpadded)
	kp      int          // K padded to mult of 32
	nGroups int          // kp/32
}

// Release frees the resident GPU buffers.
func (rm *ResidentW4A8) Release() {
	if rm.bq != nil {
		rm.bq.Release()
		rm.bq = nil
	}
	if rm.bScales != nil {
		rm.bScales.Release()
		rm.bScales = nil
	}
}

func padK32(k int) int { return (k + 31) &^ 31 }

// packNibbles packs a [rows, cols] int4 matrix (values 0..15, already nibbles)
// into [rows, kp/8] u32 words: element k of a row goes to word (k%kp)/8 at nibble
// (k%8), i.e. word |= nib << (4*(k%8)). Rows zero-padded to kp (mult of 32).
func packNibbles(nib []uint8, rows, cols int) []uint32 {
	kp := padK32(cols)
	wpr := kp / 8 // u32 words per row
	out := make([]uint32, rows*wpr)
	for r := 0; r < rows; r++ {
		src := nib[r*cols : r*cols+cols]
		dst := out[r*wpr : r*wpr+wpr]
		for k := 0; k < cols; k++ {
			dst[k/8] |= uint32(src[k]&0xF) << (4 * (k % 8))
		}
	}
	return out
}

// UploadW4A8 uploads int4 group-wise weights [N,K] (nibbles 0..15 = value−8) plus
// per-group scales [N, ceil(K/32)] to resident GPU buffers. Group size is fixed at
// 32. The caller must Release the result.
func (c *Context) UploadW4A8(nib []uint8, scales []float32, N, K int) (*ResidentW4A8, error) {
	if N <= 0 || K <= 0 {
		return nil, fmt.Errorf("gpu: UploadW4A8 non-positive dim N=%d K=%d", N, K)
	}
	kp := padK32(K)
	nGroups := kp / w4a8GroupSize
	if len(nib) < N*K {
		return nil, fmt.Errorf("gpu: UploadW4A8 nibbles too small: %d < %d", len(nib), N*K)
	}
	if len(scales) < N*nGroups {
		return nil, fmt.Errorf("gpu: UploadW4A8 scales too small: %d < %d", len(scales), N*nGroups)
	}
	packed := packNibbles(nib, N, K)
	bq, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "w4a8-weight", Contents: wgpu.ToBytes(packed), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: create W4A8 weight buffer: %w", err)
	}
	// pad scales to N*nGroups (the GPU reads a full nGroups per row)
	sc := make([]float32, N*nGroups)
	for r := 0; r < N; r++ {
		copy(sc[r*nGroups:(r+1)*nGroups], scales[r*nGroups:(r+1)*nGroups])
	}
	bs, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "w4a8-bscales", Contents: wgpu.ToBytes(sc), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		bq.Release()
		return nil, fmt.Errorf("gpu: create W4A8 scales buffer: %w", err)
	}
	return &ResidentW4A8{bq: bq, bScales: bs, rows: N, cols: K, kp: kp, nGroups: nGroups}, nil
}

func (c *Context) ensureGEMVW4() error {
	if c.gemvW4Pipeline != nil {
		return nil
	}
	sh, err := c.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "gemvW4A8", WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: gemvW4A8ShaderWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile W4A8 shader: %w", err)
	}
	pl, err := c.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "gemvW4A8", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: create W4A8 pipeline: %w", err)
	}
	c.gemvW4Shader = sh
	c.gemvW4Pipeline = pl
	c.gemvW4Layout = pl.GetBindGroupLayout(0)
	return nil
}
