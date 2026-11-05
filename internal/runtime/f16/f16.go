package f16

import "math"

// Bits converts an f32 to the IEEE binary16 bit pattern the Metal W4A8 kernels read their group scales
// in. It is the ONE definition of that conversion: decoder.F16Bits and metal's f32ToF16 call it, and a metal-target .giw
// (weights format v14) stores scales already converted by it, so a scale read from the file and one
// converted at load are the same bits — which is what keeps aliased-vs-copied logits byte-identical.
//
// Round-half-up on the dropped mantissa bits (not ties-to-even), carry into the exponent, overflow to
// ±Inf, NaN to a quiet NaN, and gradual underflow — exactly the behaviour the kernels were validated with.
func Bits(f float32) uint16 {
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

// ToF32 widens an IEEE binary16 bit pattern to f32 exactly: subnormals, ±Inf and NaN payloads included.
func ToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1F
	man := uint32(h & 0x3FF)
	if exp == 0 {
		if man == 0 {
			return math.Float32frombits(sign)
		}
		f := float32(man) / 1024 * float32(math.Exp2(-14))
		if sign != 0 {
			f = -f
		}
		return f
	}
	if exp == 0x1F {
		return math.Float32frombits(sign | 0x7F800000 | man<<13)
	}
	return math.Float32frombits(sign | (exp+112)<<23 | man<<13)
}
