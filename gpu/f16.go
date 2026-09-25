//go:build gpu

package gpu

import "math"

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
