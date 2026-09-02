package decoder

import "testing"

// M-08: MinP > 1 with no TopK panicked inside the sampler.
//
// The min-p threshold is maxL + T·ln(minP); for minP > 1 that sits ABOVE maxL, so the
// candidate set comes back EMPTY and the "always keep the top token" clamps then slice ips[:1]
// on an empty slice or read ips[len(ips)-1] at index −1. Measured before the fix:
// `index out of range [-1]`, panicking in the Generate goroutine. min_p is not on the HTTP
// surface, so this reached users through the library and `goinfer-chat --min-p`.
//
// TWO INDEPENDENT DEFENCES were added, and they are tested independently — together they mask
// each other, so a single end-to-end "does not panic" test would go green with either one
// removed and prove neither.
func TestSampler_minPAboveOne(t *testing.T) {
	// Defence 1: NewSampler clamps. 1.0 is the identity for min-p (keep only what ties the
	// max), which is the nearest meaningful reading of "more than everything".
	t.Run("NewSampler clamps to 1", func(t *testing.T) {
		for _, in := range []float64{1.5, 2, 1e9} {
			if got := NewSampler(SamplingParams{MinP: in}).p.MinP; got != 1 {
				t.Errorf("MinP %v was not clamped: got %v, want 1", in, got)
			}
		}
		// Legal values are untouched — a clamp that flattened everything would pass above.
		for _, in := range []float64{0, 0.05, 1} {
			if got := NewSampler(SamplingParams{MinP: in}).p.MinP; got != in {
				t.Errorf("MinP %v was altered to %v", in, got)
			}
		}
	})

	// Defence 2: an empty filtered set yields the argmax rather than a −1 index. Driven
	// through drawFiltered directly, because with the clamp in place nothing can empty the
	// set through the public API — which is the point of it being defence in depth.
	t.Run("drawFiltered survives an empty set", func(t *testing.T) {
		s := NewSampler(SamplingParams{Seed: 1})
		if got := s.drawFiltered(nil); got != -1 {
			t.Errorf("drawFiltered(empty) = %d, want the −1 sentinel the caller converts", got)
		}
	})

	// And the behaviour that was actually broken, end to end.
	t.Run("Sample returns the argmax, not a panic", func(t *testing.T) {
		logits := []float32{0.1, 0.5, 0.2, 0.9}
		const argmaxID = 3
		for _, minP := range []float64{1.5, 2, 1e9} {
			s := NewSampler(SamplingParams{Temperature: 1.0, MinP: minP, Seed: 1})
			id, err := s.Sample(logits)
			if err != nil {
				t.Fatalf("MinP=%v: %v", minP, err)
			}
			// At minP == 1 only tokens tying the max survive, so the draw is deterministic.
			if id != argmaxID {
				t.Errorf("MinP=%v: got id %d, want %d (only the max survives min-p 1)", minP, id, argmaxID)
			}
		}
	})

	// The docstring's promise, now that it is true: min_p at ANY value is safe. A sweep, so a
	// future change that fixes only the values this test happens to name is caught.
	t.Run("no value of min_p panics", func(t *testing.T) {
		logits := []float32{0.1, 0.5, 0.2, 0.9, -3, 7}
		for _, minP := range []float64{-1, 0, 1e-12, 0.3, 0.999, 1, 1.0001, 3, 1e30} {
			for _, temp := range []float64{0.01, 1, 5} {
				s := NewSampler(SamplingParams{Temperature: temp, MinP: minP, Seed: 7})
				id, err := s.Sample(logits)
				if err != nil {
					t.Errorf("min_p=%v temp=%v: %v", minP, temp, err)
				}
				if id < 0 || id >= len(logits) {
					t.Errorf("min_p=%v temp=%v: id %d out of range", minP, temp, id)
				}
			}
		}
	})
}
