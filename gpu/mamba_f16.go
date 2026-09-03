//go:build gpu

package gpu

import (
	"fmt"
	"math"

	"github.com/cogentcore/webgpu/wgpu"
)

// f16 mixed-precision path for the Mamba projections (in/out_proj). The R2 control
// localized granite's int8 quality loss to the SSM projections (exp(dt·A) amplifies int8
// dt/B/C error → flipped MoE selection); f16 weights + f32 activation recover near-f32
// quality at ~2× the bandwidth of int8 on those two GEMVs. Everything else (experts,
// attention, MoE) stays int8. f16 is stored 2-per-u32 and read with the core
// unpack2x16float builtin — no shader-f16 device feature (same trick as the f16 KV cache).

// f32ToF16 rounds an f32 to IEEE half (round-to-nearest-even), handling normals,
// subnormals, overflow→inf. Weights sit in the normal range so this is exact to f16.
// f32ToF16 is THE float32 → IEEE-754 half converter for this package, byte-for-byte identical to
// decoder.f32ToF16bits — the canonical resident-backend representation that cuda/kernels.go's
// f32tof16, metal/pack.go and the GOINFER_INT4_F16_SCALES CPU diagnostic all replicate.
// Round-half-up plus gradual underflow to subnormals.
//
// N-04: this package had TWO converters and neither matched. gemv_w4a8.go's flushed the whole
// subnormal range (so an int4 group scale below 2^-14 read as all-zero on WebGPU only), and this
// one used round-to-nearest-EVEN in the normal range where every other backend rounds half up —
// so identical inputs could produce different halves on exact ties. One converter now, and
// TestF32ToF16_N04 pins it against a local copy of the canonical algorithm, the same way
// cuda/f16_convert_test.go does for C-15.
//
// NOT RNE: a lone RNE here would re-introduce the divergence C-15 closed.
func f32ToF16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16((b >> 16) & 0x8000)
	e := int32((b>>23)&0xFF) - 112
	m := b & 0x7FFFFF
	switch {
	case (b>>23)&0xFF == 0xFF:
		if m != 0 {
			return sign | 0x7E00
		}
		return sign | 0x7C00
	case e >= 0x1F:
		return sign | 0x7C00
	case e <= 0:
		if e < -10 {
			return sign
		}
		m |= 0x800000
		sh := uint32(14 - e)
		return sign | uint16((m+(1<<(sh-1)))>>sh)
	default:
		half := sign | uint16(e<<10) | uint16(m>>13)
		if m&0x1000 != 0 {
			half++
		}
		return half
	}
}

// packF16 packs a row-major [N, K] f32 weight into [N, ceil(K/2)] u32 (2 f16/word, low
// then high, the unpack2x16float layout), optionally pre-scaled by mult (folds ResidMul
// into the out_proj weights). K is even for granite.
func packF16(w []float32, N, K int, mult float32) []uint32 {
	kw := (K + 1) / 2
	out := make([]uint32, N*kw)
	for r := range N {
		for v := range kw {
			lo := f32ToF16(w[r*K+2*v] * mult)
			var hi uint16
			if 2*v+1 < K {
				hi = f32ToF16(w[r*K+2*v+1] * mult)
			}
			out[r*kw+v] = uint32(lo) | uint32(hi)<<16
		}
	}
	return out
}

// UploadF16Weight uploads a packed-f16 [N,K] weight (build-once resident buffer).
func (c *Context) UploadF16Weight(w []float32, N, K int, mult float32) (*wgpu.Buffer, error) {
	packed := packF16(w, N, K, mult)
	return c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "mamba-f16", Contents: wgpu.ToBytes(packed), Usage: wgpu.BufferUsageStorage,
	})
}

// mambaGemvF16 = f16 weight × f32 activation → f32, one workgroup per output column
// (coalesced over k/2 words). residual=1 ⇒ dst[n] += result (out_proj residual epilogue).
const mambaGemvF16WGSL = `
struct Dims { k: u32, n: u32, residual: u32, _p: u32 };
@group(0) @binding(0) var<storage, read>       act: array<f32>;  // [k] f32 activation
@group(0) @binding(1) var<storage, read>       w:   array<u32>;  // [n, k/2] f16-packed
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;  // [n]
@group(0) @binding(3) var<uniform>             dims: Dims;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = wid.x + wid.y * 32768u;
    if (n >= dims.n) { return; }
    let t = lid.x;
    let kw = dims.k / 2u;
    let base = n * kw;
    var acc: f32 = 0.0;
    for (var v: u32 = t; v < kw; v = v + 64u) {
        let pair = unpack2x16float(w[base + v]);
        acc = acc + pair.x * act[2u * v] + pair.y * act[2u * v + 1u];
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
        if (dims.residual == 1u) { dst[n] = dst[n] + partial[0]; } else { dst[n] = partial[0]; }
    }
}
`

func (c *Context) ensureMambaF16() error {
	if c.mambaF16Pipeline != nil {
		return nil
	}
	sh, err := c.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "mambaGemvF16", WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: mambaGemvF16WGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile mambaGemvF16: %w", err)
	}
	pl, err := c.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "mambaGemvF16", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: pipeline mambaGemvF16: %w", err)
	}
	c.track(sh.Release, pl.Release) // audit C-26: register at creation
	c.mambaF16Shader, c.mambaF16Pipeline, c.mambaF16Layout = sh, pl, c.bgl(pl)
	return nil
}
