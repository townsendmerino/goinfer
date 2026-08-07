//go:build cuda

package cuda

import (
	"math"
	"testing"
)

// canonF32ToF16 is a local copy of decoder.f32ToF16bits (unexported, another module) — THE canonical
// resident-backend f16 scale representation that metal/pack.go, aikit/linalg and the
// GOINFER_INT4_F16_SCALES CPU diagnostic all replicate. cuda/kernels.go's f32tof16 MUST equal it
// bit-for-bit or CUDA's int4 group scales diverge from every other backend (audit C-15).
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

// TestF32ToF16_C15 gates C-15: cuda's f32tof16 must (1) match the canonical cross-backend f16
// representation bit-for-bit — the OLD version TRUNCATED, biasing every group scale down and
// diverging from metal/aikit — and (2) no longer flush the whole f16 subnormal range to zero.
func TestF32ToF16_C15(t *testing.T) {
	for _, f := range []float32{
		0, 1, 0.5, 0.1, 0.333333, 2.0 / 3, 1.5, 1024.5, 65504, 65520, 1e30,
		0.01, 0.007, 0.002, 0.001, 3e-4, 1e-4, 6.1e-5, 3e-5, 1e-5, 2e-6, 1e-7,
		0.0009765625, 0.00006103515625,
	} {
		if got, want := f32tof16(f), canonF32ToF16(f); got != want {
			t.Errorf("f32tof16(%g) = 0x%04x, canonical = 0x%04x — CUDA f16 scales diverge from metal/aikit (C-15)", f, got, want)
		}
	}
	// The truncation bug flushed the whole e<=0 range to zero; a representable subnormal must survive.
	if f32tof16(3e-5) == 0 {
		t.Error("scale 3e-5 flushed to zero — C-15: the f16 subnormal range must be emitted, not zeroed")
	}
	// Round-half-up rounds a guard-bit-set mantissa UP; pure truncation never did. At least one value
	// in the sweep must differ from truncation, proving the downward bias is gone.
	trunc := func(f float32) uint16 {
		b := math.Float32bits(f)
		e := int32((b>>23)&0xff) - 112
		if e <= 0 || e >= 0x1f {
			return uint16((b >> 16) & 0x8000)
		}
		return uint16((b>>16)&0x8000) | uint16(e<<10) | uint16((b&0x7fffff)>>13)
	}
	differs := false
	for _, f := range []float32{0.1, 0.333333, 2.0 / 3, 0.007, 0.002, 3e-5, 1e-5} {
		if f32tof16(f) != trunc(f) {
			differs = true
		}
	}
	if !differs {
		t.Error("f32tof16 is still bit-identical to truncation across the sweep — the C-15 bias/flush was not fixed")
	}
}
