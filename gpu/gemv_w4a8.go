//go:build gpu

package gpu

import (
	"fmt"
	"math"

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

// f32to16 rounds float32 → IEEE-754 half (round-half-up). The per-group scales
// are ~⅕ of the W4A8 stream at f32; half width halves that. Scales are small
// positives — inf/subnormal/sign paths are handled but not exercised here.
func f32to16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16((b >> 16) & 0x8000)
	exp := int32((b>>23)&0xff) - 127 + 15
	mant := b & 0x7fffff
	if exp >= 0x1f {
		return sign | 0x7c00
	}
	if exp <= 0 {
		return sign
	}
	half := sign | uint16(exp<<10) | uint16(mant>>13)
	if mant&0x1000 != 0 { // round up (carry propagates mantissa→exponent)
		half++
	}
	return half
}

// f16to32 expands an IEEE-754 half to float32 (so the parity reference can
// dequantize with the exact f16 the GPU reads).
func f16to32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		e := uint32(127 - 15 + 1)
		for mant&0x400 == 0 {
			mant <<= 1
			e--
		}
		return math.Float32frombits(sign | e<<23 | (mant&0x3ff)<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | mant<<13)
	default:
		return math.Float32frombits(sign | (exp-15+127)<<23 | mant<<13)
	}
}

// packF16Pairs packs []float32 into u32 words, two f16 per word (even index in
// the low half), for the GPU's unpack2x16float reads. Odd length zero-pads.
func packF16Pairs(f []float32) []uint32 {
	out := make([]uint32, (len(f)+1)/2)
	for i, v := range f {
		h := uint32(f32to16(v))
		if i&1 == 0 {
			out[i/2] |= h
		} else {
			out[i/2] |= h << 16
		}
	}
	return out
}

// packNibbles packs a [rows, cols] int4 matrix (values 0..15, already nibbles)
// into [rows, kp/8] u32 words: element k of a row goes to word (k%kp)/8 at nibble
// (k%8), i.e. word |= nib << (4*(k%8)). Rows zero-padded to kp (mult of 32).
func packNibbles(nib []uint8, rows, cols int) []uint32 {
	kp := padK32(cols)
	wpr := kp / 8 // u32 words per row
	out := make([]uint32, rows*wpr)
	for r := range rows {
		src := nib[r*cols : r*cols+cols]
		dst := out[r*wpr : r*wpr+wpr]
		for k := range cols {
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
	// pad scales to N*nGroups, store at f16 (2 per u32) — the scale bytes are ~⅕
	// of the stream; f16 halves that.
	sc := make([]float32, N*nGroups)
	for r := range N {
		copy(sc[r*nGroups:(r+1)*nGroups], scales[r*nGroups:(r+1)*nGroups])
	}
	bs, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "w4a8-bscales", Contents: wgpu.ToBytes(packF16Pairs(sc)), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		bq.Release()
		return nil, fmt.Errorf("gpu: create W4A8 scales buffer: %w", err)
	}
	return &ResidentW4A8{bq: bq, bScales: bs, rows: N, cols: K, kp: kp, nGroups: nGroups}, nil
}

// UploadW4A8Packed is the fast path of UploadW4A8: it uploads int4 weights whose bytes are
// ALREADY in the GPU packed layout. The decoder's int4 storage (2 nibbles/byte) is
// byte-identical to packNibbles' output when K is a multiple of the 32-wide group (no row
// padding — proven by TestInt4LayoutMatch), so the resident upload is a straight
// CreateBufferInit of the decoder bytes — skipping the per-element unpack + packNibbles
// re-pack that costs ~30 s on a 12 B model (docs/task-mellum2-fast-load.md). q4 is the
// decoder int4 bytes (≥ N*K/2); scales the per-group f32 (≥ N*K/32). Requires K%32==0;
// callers fall back to UploadW4A8 otherwise. Same nibble value convention (value+8).
func (c *Context) UploadW4A8Packed(q4 []byte, scales []float32, N, K int) (*ResidentW4A8, error) {
	if N <= 0 || K <= 0 {
		return nil, fmt.Errorf("gpu: UploadW4A8Packed non-positive dim N=%d K=%d", N, K)
	}
	if K%w4a8GroupSize != 0 {
		return nil, fmt.Errorf("gpu: UploadW4A8Packed needs K%%%d==0, got K=%d", w4a8GroupSize, K)
	}
	kp := K // padK32(K) == K when K%32==0
	nGroups := kp / w4a8GroupSize
	wantBytes := N * kp / 2
	if len(q4) < wantBytes {
		return nil, fmt.Errorf("gpu: UploadW4A8Packed q4 too small: %d < %d", len(q4), wantBytes)
	}
	if len(scales) < N*nGroups {
		return nil, fmt.Errorf("gpu: UploadW4A8Packed scales too small: %d < %d", len(scales), N*nGroups)
	}
	bq, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "w4a8-weight", Contents: q4[:wantBytes], Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: create W4A8 weight buffer: %w", err)
	}
	bs, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "w4a8-bscales", Contents: wgpu.ToBytes(packF16Pairs(scales[:N*nGroups])), Usage: wgpu.BufferUsageStorage,
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

// decodeWeight is a resident projection matrix the DecodeRunner can GEMV —
// either W8A8 or W4A8. Both kernels share the same 6-binding layout
// (aq, weights, aScale, scales, dst, dims) and the dims.addResidual epilogue, so
// the runner's gemv/gemvAdd builders are uniform across precisions; only the
// pipeline/layout and the weight/scale buffers differ.
type decodeWeight interface {
	gPipe(c *Context) *wgpu.ComputePipeline
	gLayout(c *Context) *wgpu.BindGroupLayout
	wbuf() *wgpu.Buffer // packed weights (int8 or int4 nibbles)
	sbuf() *wgpu.Buffer // scales (f32 per-row, or f16 per-group)
	nRows() int         // output features N
	kPad() int          // K padded (mult of 4 for int8, 32 for int4)
}

func (rm *ResidentW8A8) gPipe(c *Context) *wgpu.ComputePipeline   { return c.gemvPipeline }
func (rm *ResidentW8A8) gLayout(c *Context) *wgpu.BindGroupLayout { return c.gemvLayout }
func (rm *ResidentW8A8) wbuf() *wgpu.Buffer                       { return rm.bq }
func (rm *ResidentW8A8) sbuf() *wgpu.Buffer                       { return rm.bScales }
func (rm *ResidentW8A8) nRows() int                               { return rm.rows }
func (rm *ResidentW8A8) kPad() int                                { return rm.kp }

func (rm *ResidentW4A8) gPipe(c *Context) *wgpu.ComputePipeline   { return c.gemvW4Pipeline }
func (rm *ResidentW4A8) gLayout(c *Context) *wgpu.BindGroupLayout { return c.gemvW4Layout }
func (rm *ResidentW4A8) wbuf() *wgpu.Buffer                       { return rm.bq }
func (rm *ResidentW4A8) sbuf() *wgpu.Buffer                       { return rm.bScales }
func (rm *ResidentW4A8) nRows() int                               { return rm.rows }
func (rm *ResidentW4A8) kPad() int                                { return rm.kp }
