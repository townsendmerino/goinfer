//go:build gpu

package gpu

import (
	"math"
	"testing"
)

// canonF32ToF16 is a local copy of decoder.f32ToF16bits (unexported, another module) — THE
// canonical resident-backend f16 representation that cuda/kernels.go's f32tof16, metal/pack.go
// and the GOINFER_INT4_F16_SCALES CPU diagnostic all replicate. Same shape as
// cuda/f16_convert_test.go's copy, which gates the identical property for C-15.
func canonF32ToF16(f float32) uint16 {
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

// N-04: gpu/ carried TWO float32→half converters and neither matched the canonical one.
//
//	gemv_w4a8.go's f32to16   flushed the ENTIRE subnormal range (exp <= 0 → sign), and it is the
//	                         load-bearing one — every W4A8 group-scale upload and NewKVCacheF16.
//	                         An int4 group with scale < 2^-14 therefore read as all-zero on
//	                         WebGPU and nowhere else.
//	mamba_f16.go's f32ToF16  handled subnormals but rounded to nearest EVEN in the normal range,
//	                         where every other backend rounds half up.
//
// C-15 fixed this class in cuda/ and gpu/ was not in that disposition. This is the same gate.
func TestF32ToF16_N04(t *testing.T) {
	// (1) Bit-identity with the canonical algorithm across the exponent range that matters for
	// group scales, plus the boundaries.
	for _, f := range []float32{
		0, 1, 0.5, 2, 65504, // normal range and the f16 max
		1e-5, 6.1e-5, 6.0e-5, // just above / at / below the f16 subnormal boundary (2^-14)
		1e-7, 5.96e-8, 1e-9, // deep subnormal, the smallest f16 subnormal, and below it
		-1, -1e-5, -6.1e-5, // signs
		float32(math.Inf(1)), float32(math.Inf(-1)),
	} {
		if got, want := f32ToF16(f), canonF32ToF16(f); got != want {
			t.Errorf("f32ToF16(%g) = %#04x, canonical %#04x", f, got, want)
		}
		// f32to16 is the alias the W4A8 path uses; it must be the SAME function, not a copy
		// that can drift again.
		if got := f32to16(f); got != canonF32ToF16(f) {
			t.Errorf("f32to16(%g) = %#04x, canonical %#04x — the W4A8 upload path diverges",
				f, got, canonF32ToF16(f))
		}
	}

	// (2) A sweep over every representable exponent, so this cannot pass by naming lucky values.
	for e := -30; e <= 16; e++ {
		for _, mant := range []float64{1.0, 1.5, 1.0009765625, 1.9995117188} {
			f := float32(math.Ldexp(mant, e))
			if got, want := f32ToF16(f), canonF32ToF16(f); got != want {
				t.Fatalf("f32ToF16(%g) = %#04x, canonical %#04x (e=%d)", f, got, want, e)
			}
		}
	}

	// (3) THE DEFECT ITSELF: the subnormal range must not flush to zero. Without this the
	// bit-identity checks above could pass a converter that agreed with a canonical copy that
	// was ALSO wrong.
	small := float32(3e-8) // representable as an f16 subnormal
	if canonF32ToF16(small) == 0 {
		t.Fatalf("premise broke: %g is not representable as an f16 subnormal", small)
	}
	if f32ToF16(small) == 0 {
		t.Errorf("f32ToF16(%g) flushed a representable subnormal to zero — an int4 group with a "+
			"scale this small reads as all-zero on WebGPU only (N-04)", small)
	}
}
