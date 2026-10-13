//go:build cuda

package cuda

import "math"

// f32ToF16RNE converts an f32 to its IEEE binary16 bit pattern, rounding to nearest with ties to even: the rounding __float2half_rn applies in the
// _kv16 store kernels, so a K/V row the CPU-prefill bridge uploads and one a kernel stores are the same bits. (linalg.F32ToF16 and
// decoder.F16Bits round ties UP, which is what the W4A8 group scales were validated with and is not what the store kernels do.)
// Overflow rounds to ±Inf, NaN to a quiet NaN, and underflow is gradual.
func f32ToF16RNE(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	exp := int32(b>>23) & 0xFF
	mant := b & 0x7FFFFF
	if exp == 0xFF {
		if mant != 0 {
			return sign | 0x7E00
		}
		return sign | 0x7C00
	}
	e := exp - 127 + 15 // the binary16 biased exponent
	if e >= 0x1F {
		return sign | 0x7C00
	}
	if e <= 0 { // subnormal result, or underflow
		if e < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint32(14 - e)
		half := mant >> shift
		rem, halfway := mant&(1<<shift-1), uint32(1)<<(shift-1)
		if rem > halfway || (rem == halfway && half&1 == 1) {
			half++
		}
		return sign | uint16(half)
	}
	half := uint32(e)<<10 | mant>>13
	rem := mant & 0x1FFF
	if rem > 0x1000 || (rem == 0x1000 && half&1 == 1) {
		half++ // a mantissa carry runs into the exponent, up to ±Inf, which is the correct rounding
	}
	return sign | uint16(half)
}

// f32SliceToF16RNE converts src into dst (len(dst) >= len(src)).
func f32SliceToF16RNE(dst []uint16, src []float32) {
	for i, f := range src {
		dst[i] = f32ToF16RNE(f)
	}
}
