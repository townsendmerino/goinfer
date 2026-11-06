package f16

import (
	"math"
	"testing"
)

// half64 is binary16's value for h, decoded independently of ToF32: ±man·2^-24 for a subnormal, ±(1024+man)·2^(exp-25) for a
// normal, ±Inf for an all-ones exponent with no mantissa. NaN reports ok false.
func half64(h uint16) (v float64, ok bool) {
	sign := 1.0
	if h&0x8000 != 0 {
		sign = -1
	}
	exp, man := int(h>>10)&0x1F, float64(h&0x3FF)
	switch exp {
	case 0:
		return sign * math.Ldexp(man, -24), true
	case 0x1F:
		if man != 0 {
			return math.NaN(), false
		}
		return math.Inf(int(sign)), true
	}
	return sign * math.Ldexp(1024+man, exp-25), true
}

// ToF32 widens every one of the 65,536 halves exactly: each value matches an independent decode bit for bit (the sign of zero
// included), and a NaN stays a NaN with its payload in the top of the f32 mantissa.
func TestToF32_isExactOnEveryHalf(t *testing.T) {
	for i := range 1 << 16 {
		h := uint16(i)
		got := ToF32(h)
		want, ok := half64(h)
		if !ok {
			if b := math.Float32bits(got); !math.IsNaN(float64(got)) || b != uint32(h&0x8000)<<16|0x7F800000|uint32(h&0x3FF)<<13 {
				t.Fatalf("ToF32(%#04x) = %#08x, want the NaN carrying its payload", h, b)
			}
			continue
		}
		if math.Float32bits(got) != math.Float32bits(float32(want)) {
			t.Fatalf("ToF32(%#04x) = %g (%#08x), want %g", h, got, math.Float32bits(got), want)
		}
	}
}

// Every half that is not a NaN comes back from Bits(ToF32(h)) as itself (±0, subnormals and ±Inf included), and a NaN comes
// back as the quiet NaN of its sign.
func TestBits_roundTripsEveryHalf(t *testing.T) {
	for i := range 1 << 16 {
		h := uint16(i)
		want := h
		if _, ok := half64(h); !ok {
			want = h&0x8000 | 0x7E00
		}
		if got := Bits(ToF32(h)); got != want {
			t.Fatalf("Bits(ToF32(%#04x)) = %#04x, want %#04x", h, got, want)
		}
	}
}

// Bits rounds the dropped mantissa bits half up, not to even (the kernels were validated with half up): the midpoint between two
// adjacent finite halves gives the larger in magnitude, whichever is even, and the f32 just below it gives the smaller. That holds
// from the subnormal range, where the midpoint of 0 and the smallest subnormal rounds up to it, through to the midpoint above the
// largest finite half, which rounds up to Inf.
func TestBits_roundsHalfUpAcrossTheRange(t *testing.T) {
	for i := range 0x7C00 { // every positive finite half, paired with the next one up (0x7C00 is +Inf)
		lo, hi := uint16(i), uint16(i+1)
		vlo, _ := half64(lo)
		vhi, _ := half64(hi)
		if hi == 0x7C00 {
			vhi = 1 << 16 // the step past the largest finite half, where rounding up overflows to +Inf
		}
		mid := float32((vlo + vhi) / 2) // one bit past a half's mantissa: exact in f32
		below := math.Nextafter32(mid, 0)
		for _, sign := range []uint16{0, 0x8000} {
			m, b := mid, below
			if sign != 0 {
				m, b = -mid, -below
			}
			if got := Bits(m); got != sign|hi {
				t.Fatalf("Bits(%g), the midpoint of %#04x and %#04x, = %#04x, want %#04x (half up)", m, sign|lo, sign|hi, got, sign|hi)
			}
			if got := Bits(b); got != sign|lo {
				t.Fatalf("Bits(%g), just below the midpoint of %#04x and %#04x, = %#04x, want %#04x", b, sign|lo, sign|hi, got, sign|lo)
			}
		}
	}
	if got := Bits(math.MaxFloat32); got != 0x7C00 {
		t.Errorf("Bits(MaxFloat32) = %#04x, want +Inf", got)
	}
}
