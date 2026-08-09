package decoder

import (
	"math"
	"runtime"
	"sync"
)

// P2b — deterministic parallel host normalization.
//
// The temperature-only path's cost is the full-vocabulary exp+sum, measured at ~31-34 ns per
// vocabulary entry and ~78% of the whole temperature-only penalty (bc59c56). Lazy Z tried to SKIP
// that work and was refuted; this instead does the same work in parallel, and deletes the separate
// normalize-divide pass by drawing against unnormalized weights.
//
// ── THE LOAD-BEARING DESIGN DECISION: numChunks IS A COMPILE-TIME CONSTANT ──────────────────────
//
// The vocabulary is split into a FIXED number of chunks, and Z is folded from the per-chunk partial
// sums in ASCENDING CHUNK INDEX order — never in completion order, and never over a chunk count
// derived from runtime.NumCPU()/GOMAXPROCS.
//
// This is what keeps given-seed output a property of the CODE rather than of the MACHINE. Float
// addition is not associative, so the grouping of the sum decides Z's last few ULPs, and Z decides
// which side of a token boundary a near-boundary draw falls on. If the chunk shape followed the core
// count, the same seed on the same build would emit different tokens on a 4-core and a 16-core box —
// an unreproducible-by-construction sampler.
//
// Workers may be scheduled however the runtime likes; only the REDUCTION order is pinned. Sizing the
// chunk count to cores is the obvious "optimization" a future refactor reaches for, which is why
// TestChunkedSoftmax_MachineIndependent runs the same draw at GOMAXPROCS 1/2/8 and requires
// identical output — that test is what makes the constant unwriteable, not this comment.
const numChunks = 64

// chunkBounds returns the [lo, hi) span of chunk c over n elements. Deterministic and independent of
// worker count: chunk c always covers exactly the same ids.
func chunkBounds(n, c int) (int, int) {
	lo := n * c / numChunks
	hi := n * (c + 1) / numChunks
	return lo, hi
}

// parallelMax returns the maximum logit. Parallel is exactly order-independent here: max is
// associative and commutative on floats barring NaN, so no seed effect. A NaN would break that (and
// every comparison downstream), so it is checked once, here, rather than assumed.
func parallelMax(logits []float32) (float64, bool) {
	n := len(logits)
	if n == 0 {
		return 0, true
	}
	part := make([]float64, numChunks)
	nan := make([]bool, numChunks)
	forEachChunk(n, func(c, lo, hi int) {
		m := math.Inf(-1)
		bad := false
		for _, v := range logits[lo:hi] {
			f := float64(v)
			if math.IsNaN(f) {
				bad = true
				continue
			}
			if f > m {
				m = f
			}
		}
		part[c], nan[c] = m, bad
	})
	m := math.Inf(-1)
	anyNaN := false
	for c := range numChunks { // ascending chunk order (irrelevant for max, kept for uniformity)
		if nan[c] {
			anyNaN = true
		}
		if part[c] > m {
			m = part[c]
		}
	}
	return m, !anyNaN
}

// expChunked fills out[i] = exp((x_i - m)/T) and returns the per-chunk partial sums, each summed
// SEQUENTIALLY in ascending id within its chunk. The e values are bit-identical to the old
// sequential path (same math.Exp, same operands); only the grouping of the SUM changes.
func expChunked(logits []float32, out []float64, maxv, temperature float64) []float64 {
	n := len(logits)
	sums := make([]float64, numChunks)
	forEachChunk(n, func(c, lo, hi int) {
		var s float64
		for i := lo; i < hi; i++ {
			e := math.Exp((float64(logits[i]) - maxv) / temperature)
			out[i] = e
			s += e // ascending id within the chunk
		}
		sums[c] = s
	})
	return sums
}

// foldChunkSums folds per-chunk partial sums in ASCENDING CHUNK INDEX order. This ordering is the
// spec: it is what the sequential reference reproduces and what makes Z machine-independent.
func foldChunkSums(sums []float64) float64 {
	var z float64
	for c := range numChunks {
		z += sums[c]
	}
	return z
}

// forEachChunk runs fn over all numChunks chunks, distributing them across workers. Worker count may
// vary with the machine; chunk BOUNDARIES and the later reduction order may not.
func forEachChunk(n int, fn func(c, lo, hi int)) {
	workers := runtime.GOMAXPROCS(0)
	if workers > numChunks {
		workers = numChunks
	}
	if workers <= 1 || n < numChunks*64 { // small vocab: the goroutine overhead dominates
		for c := range numChunks {
			lo, hi := chunkBounds(n, c)
			fn(c, lo, hi)
		}
		return
	}
	var wg sync.WaitGroup
	next := make(chan int, numChunks)
	for c := range numChunks {
		next <- c
	}
	close(next)
	wg.Add(workers)
	for range workers {
		go func() {
			defer wg.Done()
			for c := range next {
				lo, hi := chunkBounds(n, c)
				fn(c, lo, hi)
			}
		}()
	}
	wg.Wait()
}

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
			return hi - 1 // rounding hair inside the chunk
		}
		cum = nextCum
	}
	return len(e) - 1
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
	e := make([]float64, len(logits))
	sums := expChunked(logits, e, maxv, temperature)
	z := foldChunkSums(sums)
	return drawChunked(e, sums, z, r)
}

// chunkedZ is the top-p denominator (step 3), given the same fixed-chunk treatment so the nucleus
// cut shifts in the SAME release as the temperature-only change rather than dribbling out later.
func chunkedZ(logits []float32, maxv, temperature float64) float64 {
	tmp := make([]float64, len(logits))
	return foldChunkSums(expChunked(logits, tmp, maxv, temperature))
}
