package decoder

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestF16Rules_agree pins that aikit's int4 scale storage (linalg.F32ToF16) and goinfer's F16Bits — the rule
// every resident GPU backend converts int4 scales with — are the same conversion, bit for bit, so a CPU
// weight and a GPU upload of the same tensor carry identical scales. Before aikit stored int4 scales as
// binary16 this was a third copy (the GOINFER_INT4_F16_SCALES diagnostic's f32ToF16bits); all three were
// checked equal on this same input before that copy was deleted.
func TestF16Rules_agree(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	check := func(f float32) {
		a, c := F16Bits(f), linalg.F32ToF16(f)
		if a != c {
			t.Fatalf("%g (%#08x): F16Bits %#04x, linalg.F32ToF16 %#04x", f, math.Float32bits(f), a, c)
		}
	}
	for n := 0; n < 5_000_000; n++ {
		check(math.Float32frombits(rng.Uint32()))
	}
	for h := 0; h < 1<<16; h++ { // every binary16 value and its exact half-way neighbours
		f := linalg.F16ToF32(uint16(h))
		b := math.Float32bits(f)
		for _, d := range []int32{-0x1000, -1, 0, 1, 0x1000} {
			check(math.Float32frombits(uint32(int32(b) + d)))
		}
	}
}
