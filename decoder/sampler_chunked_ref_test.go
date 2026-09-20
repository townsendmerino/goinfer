package decoder

// The LEGACY temperature-only sampler: inverse CDF in vocabulary index order over a parallel chunked
// normalisation (P2b). It was the production draw until R7b replaced it with Gumbel-max
// (sampler_gumbel.go), and is kept here, unchanged, as the REFERENCE the new draw is tested against: the two
// must sample the same DISTRIBUTION (sampler_gumbel_test.go) even though, by design, they no longer draw the
// same token for a given seed.

// drawChunked picks a token from UNNORMALIZED weights e with per-chunk sums, using a two-level walk:
// scan chunk sums ascending to find the chunk holding the target, then walk within that chunk. That
// is O(V/C + C) instead of O(V), and — importantly — it also lets the whole normalize-divide pass be
// skipped, since the target is scaled by Z rather than the weights being divided by it.
//
// The chunk-prefix grouping is PART OF THE SPEC, not an implementation detail: the cumulative value
// at a token depends on the order the partial sums were folded, so the reference walks the same way.
// Comparator is `t < cum`, matching drawFull.
func drawChunked(e []float64, sums []float64, z, r float64) int {
	t := r * z
	var cum float64
	for c := range numChunks {
		nextCum := cum + sums[c]
		if t < nextCum {
			lo, hi := chunkBounds(len(e), c)
			for i := lo; i < hi; i++ {
				cum += e[i]
				if t < cum {
					return i
				}
			}
			return lastWithMass(e, lo, hi) // N-02: not hi-1, which is often a masked token
		}
		cum = nextCum
	}
	return lastWithMass(e, 0, len(e))
}

// sampleChunked is the temperature-only draw: parallel max, parallel exp + per-chunk sums, ordered
// fold, two-level walk. One rng draw, taken by the caller.
func (s *Sampler) sampleChunked(logits []float32, temperature, r float64) int {
	if temperature <= 0 {
		temperature = 1
	}
	maxv, ok := parallelMax(logits)
	if !ok {
		// NaN present: fall back to the exact sequential path rather than reduce over NaN.
		probs := softmaxStable(logits, temperature)
		var cum float64
		for i, p := range probs {
			cum += p
			if r < cum {
				return i
			}
		}
		return len(probs) - 1
	}
	e := s.vocabBufN(len(logits))
	sums := expChunked(logits, e, maxv, temperature)
	z := foldChunkSums(sums)
	return drawChunked(e, sums, z, r)
}
