package wpack

import (
	"math"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// f16Value is the exact value of a finite binary16 bit pattern.
func f16Value(h uint16) float64 { return float64(linalg.F16ToF32(h)) }

// TestF32ToF16RNE_ties: every exact midpoint between two adjacent finite f16 values rounds to the one with an even mantissa, and anything a
// hair either side of a midpoint rounds to the nearer value. Every f16 value round-trips. Overflow and the specials behave as __float2half_rn.
func TestF32ToF16RNE_ties(t *testing.T) {
	for h := uint32(0); h < 0x7C00; h++ { // 0 .. the largest finite, positive
		lo, hi := uint16(h), uint16(h+1)
		if got := F32ToF16RNE(float32(f16Value(lo))); got != lo {
			t.Fatalf("%#04x does not round-trip: %#04x", lo, got)
		}
		mid := float32((f16Value(lo) + f16Value(hi)) / 2) // 12 significant bits at most: exact in f32
		if hi == 0x7C00 {
			mid = 65520 // halfway from the largest finite f16 (65504) to the next step of 32, where the spacing would have gone on
		}
		want := lo
		if hi&1 == 0 {
			want = hi
		}
		if hi == 0x7C00 { // the midpoint above the largest finite f16 (65520) rounds to even = infinity
			want = 0x7C00
		}
		if got := F32ToF16RNE(mid); got != want {
			t.Fatalf("midpoint of %#04x and %#04x (%g): got %#04x, want the even one %#04x", lo, hi, mid, got, want)
		}
		if got := F32ToF16RNE(math.Nextafter32(mid, 0)); got != lo {
			t.Fatalf("just below the midpoint of %#04x and %#04x: got %#04x, want %#04x", lo, hi, got, lo)
		}
		if got := F32ToF16RNE(math.Nextafter32(mid, float32(math.Inf(1)))); got != hi {
			t.Fatalf("just above the midpoint of %#04x and %#04x: got %#04x, want %#04x", lo, hi, got, hi)
		}
		if F32ToF16RNE(-float32(f16Value(lo))) != lo|0x8000 {
			t.Fatalf("sign of %#04x", lo)
		}
	}
	if F32ToF16RNE(float32(math.Inf(1))) != 0x7C00 || F32ToF16RNE(float32(math.Inf(-1))) != 0xFC00 || F32ToF16RNE(1e30) != 0x7C00 {
		t.Error("overflow does not reach infinity")
	}
	if h := F32ToF16RNE(float32(math.NaN())); h&0x7C00 != 0x7C00 || h&0x03FF == 0 {
		t.Errorf("NaN converts to %#04x", h)
	}
	if F32ToF16RNE(1e-10) != 0 || F32ToF16RNE(-1e-10) != 0x8000 {
		t.Error("underflow does not reach signed zero")
	}
	if F32ToF16RNE(float32(0x1p-25)) != 0 || F32ToF16RNE(float32(0x1p-24)) != 1 || F32ToF16RNE(float32(1.5*0x1p-24)) != 2 {
		t.Error("the smallest subnormals round wrong (half of one ulp ties to even = 0; one and a half ties to even = 2)")
	}
}
