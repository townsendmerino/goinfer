package decoder

import "testing"

// N-02: a float-rounding miss must not return a MASKED token.
//
// drawChunked and drawFull walked the cumulative distribution and, if rounding left `r` past the
// final cumulative sum, fell through to the LAST INDEX. The vector is masked — top-k and top-p zero
// the excluded tail — so the last index is very often a token the filter deliberately removed, and
// returning it emits something the caller configured to be impossible. spec_sample.go's drawTree
// already walked back to the last entry with mass; these two did not.
//
// ~1e-16 per draw, so a contract nick rather than a live bug — but it is the contract that top-k
// and top-p exist to provide.
func TestLastWithMass_neverReturnsAMaskedToken(t *testing.T) {
	// A masked distribution: mass only in the first three of eight, as top-k leaves it.
	probs := []float64{0.2, 0.3, 0.5, 0, 0, 0, 0, 0}
	if got := lastWithMass(probs, 0, len(probs)); got != 2 {
		t.Errorf("fell back to index %d, want 2 — indices 3..7 were zeroed by the filter", got)
	}
	if probs[lastWithMass(probs, 0, len(probs))] == 0 {
		t.Error("returned a zero-mass token")
	}
	// Chunk-local: the bound must be respected, so a chunk of all-zeros returns its own last index
	// rather than reaching into a neighbouring chunk.
	if got := lastWithMass(probs, 4, 8); got != 7 {
		t.Errorf("all-zero chunk [4,8) returned %d, want 7 (its own last index)", got)
	}
	if got := lastWithMass(probs, 0, 3); got != 2 {
		t.Errorf("chunk [0,3) returned %d, want 2", got)
	}
	// An all-zero vector has no better answer; it must terminate in range.
	zero := make([]float64, 8)
	if got := lastWithMass(zero, 0, len(zero)); got < 0 || got >= len(zero) {
		t.Errorf("all-zero vector returned %d, out of range", got)
	}
}
