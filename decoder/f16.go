package decoder

import "math"

// F16Bits converts an f32 to the IEEE binary16 bit pattern the Metal W4A8 kernels read their group scales
// in. It is the ONE definition of that conversion: metal's f32ToF16 calls it, and a metal-target .giw
// (weights format v14) stores scales already converted by it, so a scale read from the file and one
// converted at load are the same bits — which is what keeps aliased-vs-copied logits byte-identical.
//
// Round-half-up on the dropped mantissa bits (not ties-to-even), carry into the exponent, overflow to
// ±Inf, NaN to a quiet NaN, and gradual underflow — exactly the behaviour the kernels were validated with.
func F16Bits(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16((b >> 16) & 0x8000)
	e := int32((b>>23)&0xFF) - 112 // f16 biased exp = f32biasedexp - 127 + 15
	m := b & 0x7FFFFF
	switch {
	case (b>>23)&0xFF == 0xFF: // inf / nan
		if m != 0 {
			return sign | 0x7E00
		}
		return sign | 0x7C00
	case e >= 0x1F: // overflow -> inf
		return sign | 0x7C00
	case e <= 0: // subnormal / underflow
		if e < -10 {
			return sign
		}
		m |= 0x800000
		sh := uint32(14 - e)
		return sign | uint16((m+(1<<(sh-1)))>>sh)
	default:
		half := sign | uint16(e<<10) | uint16(m>>13)
		if m&0x1000 != 0 { // round up; a mantissa carry propagates into the exp field
			half++
		}
		return half
	}
}
