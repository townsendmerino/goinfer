package decoder

import (
	"math"
	"os"

	"github.com/townsendmerino/aikit/linalg"
)

// GOINFER_INT4_F16_SCALES=1 is a DIAGNOSTIC (default-off, one env read per weight at LOAD only):
// when set, every int4 WeightMat's group scales are rounded through IEEE half precision before use,
// matching what the resident GPU backends STORE (CUDA `ws16`, Metal's f16 group scales, WebGPU's
// f16-group-scale unpack). A CPU forward loaded with this on therefore carries the resident
// backends' EXACT quantized weights, so a resident-vs-CPU cosine becomes single-variable (kernel
// arithmetic only) instead of confounded by the f32→f16 scale-representation gap — which floors the
// tiny gemma4 fixture's cosine at ~0.98 even for a byte-perfect port. Observe-only: unset, the int4
// path keeps its f32 scales and every consumer stays bit-identical.
//
// Read per-call (not cached at package init like fakeQuantScheme) so a test can t.Setenv it around
// a single Load. Load-time only, so the per-weight os.Getenv is off the decode hot path.
func maybeF16RoundInt4Scales(wm linalg.WeightMat) linalg.WeightMat {
	if os.Getenv("GOINFER_INT4_F16_SCALES") == "" {
		return wm
	}
	q4, q4s, group, ok := wm.Int4()
	if !ok {
		return wm
	}
	rounded := make([]float32, len(q4s))
	for i, s := range q4s {
		rounded[i] = f16RoundF32(s)
	}
	return linalg.WrapInt4(q4, rounded, wm.Rows(), wm.Cols(), group)
}

// f16RoundF32 narrows an f32 through IEEE half and back — f16ToF32(f32ToF16(f)) — the exact
// round-trip the resident backends apply to int4 group scales at upload. The two halves replicate
// metal/pack.go f32ToF16 and aikit/linalg f16ToF32 bit-for-bit so this measures THEIR representation.
func f16RoundF32(f float32) float32 { return f16bitsToF32(f32ToF16bits(f)) }

func f32ToF16bits(f float32) uint16 {
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

func f16bitsToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := int32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	var bits uint32
	switch {
	case exp == 0:
		if mant == 0 {
			bits = sign
			break
		}
		exp = 1
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		mant &= 0x3ff
		bits = sign | uint32(exp+(127-15))<<23 | mant<<13
	case exp == 0x1f:
		bits = sign | 0x7f800000 | mant<<13
	default:
		bits = sign | uint32(exp+(127-15))<<23 | mant<<13
	}
	return math.Float32frombits(bits)
}
