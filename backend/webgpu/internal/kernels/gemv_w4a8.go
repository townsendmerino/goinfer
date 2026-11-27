//go:build gpu

package kernels

import (
	"fmt"
	"math"

	"github.com/townsendmerino/goinfer/backend/webgpu/internal/wgsl"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// W4a8GroupSize is the group of the W4A8 decode GEMV (int4 group-wise weights × int8 activation). The decode roofline is weight bytes/token, and int4 streams
// about 0.6× the bytes of int8 (½ the nibbles + ⅛ the f32 group scales). Format matches aikit's int4-resident
// (`internal/linalg`, group=32, GGUF Q4_K granularity): per row, K/32 groups each of 32 nibbles + one f32 scale; element i's
// nibble = (word>>4i)&0xF, value (nibble−8). group=32 nibbles = 16 bytes = exactly one vec4<u32>, so one group is one
// coalesced load. Activations are int8 (per-row aScale, same as W8A8).
const W4a8GroupSize = 32

// ResidentW4A8 is an int4 group-wise weight matrix resident on the GPU.
type ResidentW4A8 struct {
	ctx     *Context     // the Context that owns the buffers: Close after it is closed releases on the finalizer goroutine (releaseOwned)
	bq      *wgpu.Buffer // [N, kp/32] vec4<u32> packed nibbles
	bScales *wgpu.Buffer // [N, kp/32] f16 per-group scales, packed 2/u32 (both writers below pack via packF16Pairs, not raw f32)
	rows    int          // N
	cols    int          // K (unpadded)
	kp      int          // K padded to mult of 32
	nGroups int          // kp/32
}

// Close frees the resident GPU buffers.
func (rm *ResidentW4A8) Close() error {
	rm.ctx.releaseOwned(func() {
		traceFree(rm)
		if rm.bq != nil {
			accountFree(int64(rm.bq.GetSize()))
			releaseBuf(rm.bq)
			rm.bq = nil
		}
		if rm.bScales != nil {
			accountFree(int64(rm.bScales.GetSize()))
			releaseBuf(rm.bScales)
			rm.bScales = nil
		}
	})
	return nil
}

func PadK32(k int) int { return (k + 31) &^ 31 }

// f32to16 is an alias for the ONE float32→half converter this package uses; see f32ToF16 in f16.go. Keep it one: a second
// converter here once flushed the entire subnormal range, so an int4 group scale below 2^-14 read as all-zero on WebGPU and
// nowhere else, and made the GOINFER_INT4_F16_SCALES diagnostic (which claims to reproduce WebGPU's unpack on the CPU) lie.
// History: docs/code-notes/gpu.md#f32to16.
func f32to16(f float32) uint16 { return f32ToF16(f) }

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

// PackF16Pairs packs []float32 into u32 words, two f16 per word (even index in
// the low half), for the GPU's unpack2x16float reads. Odd length zero-pads.
func PackF16Pairs(f []float32) []uint32 {
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
	kp := PadK32(cols)
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
	kp := PadK32(K)
	nGroups := kp / W4a8GroupSize
	if len(nib) < N*K {
		return nil, fmt.Errorf("gpu: UploadW4A8 nibbles too small: %d < %d", len(nib), N*K)
	}
	if len(scales) < N*nGroups {
		return nil, fmt.Errorf("gpu: UploadW4A8 scales too small: %d < %d", len(scales), N*nGroups)
	}
	packed := packNibbles(nib, N, K)
	bq, err := c.tryCreateBufferInit(mem.LineDense, &wgpu.BufferInitDescriptor{
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
	bs, err := c.tryCreateBufferInit(mem.LineDense, &wgpu.BufferInitDescriptor{
		Label: "w4a8-bscales", Contents: wgpu.ToBytes(PackF16Pairs(sc)), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		releaseBuf(bq)
		return nil, fmt.Errorf("gpu: create W4A8 scales buffer: %w", err)
	}
	accountAlloc(int64(bq.GetSize()) + int64(bs.GetSize()))
	return traced(&ResidentW4A8{ctx: c, bq: bq, bScales: bs, rows: N, cols: K, kp: kp, nGroups: nGroups}, int64(bq.GetSize())+int64(bs.GetSize())), nil
}

// UploadW4A8Packed is the fast path of UploadW4A8: it uploads int4 weights whose bytes are ALREADY in the GPU packed layout.
// The decoder's int4 storage (2 nibbles/byte) is byte-identical to packNibbles' output when K is a multiple of the 32-wide
// group (no row padding; TestInt4LayoutMatch), so the resident upload is a straight CreateBufferInit of the decoder bytes,
// skipping the per-element unpack + packNibbles re-pack that is slow on a large model. q4 is the decoder int4 bytes
// (≥ N*K/2); scales the per-group f32 (≥ N*K/32). Requires K%32==0; callers fall back to UploadW4A8 otherwise. Same nibble
// value convention (value+8).
func (c *Context) UploadW4A8Packed(q4 []byte, scales []float32, N, K int) (*ResidentW4A8, error) {
	if N <= 0 || K <= 0 {
		return nil, fmt.Errorf("gpu: UploadW4A8Packed non-positive dim N=%d K=%d", N, K)
	}
	if K%W4a8GroupSize != 0 {
		return nil, fmt.Errorf("gpu: UploadW4A8Packed needs K%%%d==0, got K=%d", W4a8GroupSize, K)
	}
	kp := K // padK32(K) == K when K%32==0
	nGroups := kp / W4a8GroupSize
	wantBytes := N * kp / 2
	if len(q4) < wantBytes {
		return nil, fmt.Errorf("gpu: UploadW4A8Packed q4 too small: %d < %d", len(q4), wantBytes)
	}
	if len(scales) < N*nGroups {
		return nil, fmt.Errorf("gpu: UploadW4A8Packed scales too small: %d < %d", len(scales), N*nGroups)
	}
	bq, err := c.tryCreateBufferInit(mem.LineDense, &wgpu.BufferInitDescriptor{
		Label: "w4a8-weight", Contents: q4[:wantBytes], Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: create W4A8 weight buffer: %w", err)
	}
	bs, err := c.tryCreateBufferInit(mem.LineDense, &wgpu.BufferInitDescriptor{
		Label: "w4a8-bscales", Contents: wgpu.ToBytes(PackF16Pairs(scales[:N*nGroups])), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		releaseBuf(bq)
		return nil, fmt.Errorf("gpu: create W4A8 scales buffer: %w", err)
	}
	accountAlloc(int64(bq.GetSize()) + int64(bs.GetSize()))
	return traced(&ResidentW4A8{ctx: c, bq: bq, bScales: bs, rows: N, cols: K, kp: kp, nGroups: nGroups}, int64(bq.GetSize())+int64(bs.GetSize())), nil
}

func (c *Context) ensureGEMVW4() error {
	if c.gemvW4Pipeline != nil {
		return nil
	}
	sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "gemvW4A8", WGSLSource: &wgpu.ShaderSourceWGSL{Code: wgsl.GemvW4A8ShaderWGSL},
	})
	if err != nil {
		return fmt.Errorf("gpu: compile W4A8 shader: %w", err)
	}
	pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "gemvW4A8", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		sh.Release()
		return fmt.Errorf("gpu: create W4A8 pipeline: %w", err)
	}
	c.track(sh.Release, pl.Release) // registered at creation, so Close releases them
	c.gemvW4Shader = sh
	c.gemvW4Pipeline = pl
	c.gemvW4Layout = c.bgl(pl)
	return nil
}

// DecodeWeight is a resident projection matrix the DecodeRunner can GEMV —
// either W8A8 or W4A8. Both kernels share the same 6-binding layout
// (aq, weights, aScale, scales, dst, dims) and the dims.addResidual epilogue, so
// the runner's gemv/gemvAdd builders are uniform across precisions; only the
// pipeline/layout and the weight/scale buffers differ.
type DecodeWeight interface {
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
