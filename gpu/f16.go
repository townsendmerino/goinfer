//go:build gpu

package gpu

import "math"

// f32ToF16 is THE float32 → IEEE-754 half converter for this package, byte-for-byte identical to decoder.f32ToF16bits, the
// canonical resident-backend representation that cuda/kernels.go's f32tof16, metal/pack.go and the
// GOINFER_INT4_F16_SCALES CPU diagnostic all replicate. Round-half-up plus gradual underflow to subnormals.
//
// NOT RNE: round-to-nearest-even here would diverge from every other backend on exact ties. Keep it the only converter in
// this package; TestF32ToF16_N04 pins it against a local copy of the canonical algorithm. History:
// docs/code-notes/gpu.md#f32ToF16.
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
