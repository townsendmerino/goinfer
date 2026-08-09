package decoder

import (
	"math"
	"math/rand"
	"runtime"
	"testing"
)

// refChunked is the SEQUENTIAL implementation of the same chunked spec: same chunk boundaries, same
// per-chunk ascending-id summation, same ascending-chunk fold, same `t < cum` comparator. It is the
// gate target — the parallel path must reproduce it bit-for-bit, which is what proves the
// parallelism changed only scheduling and not arithmetic.
func refChunked(logits []float32, temperature, r float64) int {
	if temperature <= 0 {
		temperature = 1
	}
	maxv := math.Inf(-1)
	for _, v := range logits {
		if float64(v) > maxv {
			maxv = float64(v)
		}
	}
	n := len(logits)
	e := make([]float64, n)
	sums := make([]float64, numChunks)
	for c := range numChunks {
		lo, hi := chunkBounds(n, c)
		var s float64
		for i := lo; i < hi; i++ {
			ev := math.Exp((float64(logits[i]) - maxv) / temperature)
			e[i] = ev
			s += ev
		}
		sums[c] = s
	}
	var z float64
	for c := range numChunks {
		z += sums[c]
	}
	t := r * z
	var cum float64
	for c := range numChunks {
		lo, hi := chunkBounds(n, c)
		next := cum + sums[c]
		if t < next {
			for i := lo; i < hi; i++ {
				cum += e[i]
				if t < cum {
					return i
				}
			}
			return hi - 1
		}
		cum = next
	}
	return n - 1
}

func chunkedAt(logits []float32, temperature, r float64) int {
	s := &Sampler{p: SamplingParams{Temperature: temperature}, rng: rand.New(rand.NewSource(1))}
	return s.sampleChunked(logits, temperature, r)
}

// TestChunkedSoftmax_MatchesReference: bit-for-bit against the sequential same-spec reference across
// peaked / flat / tie-heavy distributions and both real vocab widths.
func TestChunkedSoftmax_MatchesReference(t *testing.T) {
	total := 0
	for _, V := range []int{152064, 262144} {
		for _, shape := range []string{"peaked", "flat", "ties"} {
			for seed := int64(1); seed <= 3; seed++ {
				r := rand.New(rand.NewSource(seed))
				var logits []float32
				switch shape {
				case "peaked":
					logits = randLogits(V, r)
				case "flat":
					logits = make([]float32, V)
					for i := range logits {
						logits[i] = float32(r.NormFloat64() * 0.01)
					}
				case "ties":
					logits = randLogitsWithTies(V, r)
				}
				for _, temp := range []float64{0.7, 1.0, 1.5} {
					for _, rv := range []float64{0.0, 1e-12, 0.25, 0.5, 0.75, 0.999999} {
						got, want := chunkedAt(logits, temp, rv), refChunked(logits, temp, rv)
						if got != want {
							t.Fatalf("V=%d %s seed=%d temp=%v r=%v: parallel %d, sequential-spec %d",
								V, shape, seed, temp, rv, got, want)
						}
						total++
					}
				}
			}
		}
	}
	t.Logf("parallel chunked path matched the sequential spec on %d draws", total)
}

// TestChunkedSoftmax_ChunkBoundaryDraws targets the NEW boundary surface this design introduces:
// draws whose target lands within a hair of a CHUNK prefix boundary, where the two-level walk must
// descend into the right chunk.
func TestChunkedSoftmax_ChunkBoundaryDraws(t *testing.T) {
	const V = 152064
	logits := randLogits(V, rand.New(rand.NewSource(4)))
	// Build the chunk prefix fractions, then aim draws just inside/outside each.
	maxv, _ := parallelMax(logits)
	e := make([]float64, V)
	sums := expChunked(logits, e, maxv, 1.0)
	z := foldChunkSums(sums)
	var cum float64
	checked := 0
	for c := range numChunks {
		cum += sums[c]
		frac := cum / z
		for _, d := range []float64{-1e-15, 0, 1e-15} {
			rv := frac + d
			if rv <= 0 || rv >= 1 {
				continue
			}
			if got, want := chunkedAt(logits, 1.0, rv), refChunked(logits, 1.0, rv); got != want {
				t.Fatalf("chunk %d boundary r=%.17g: parallel %d, sequential-spec %d", c, rv, got, want)
			}
			checked++
		}
	}
	t.Logf("verified %d draws at chunk-prefix boundaries", checked)
}

// TestChunkedSoftmax_MachineIndependent is the test that makes numChunks unwriteable.
//
// If anyone "optimizes" the chunk count to runtime.NumCPU()/GOMAXPROCS, or reduces in completion
// order instead of ascending chunk index, the same seed produces different tokens at different
// parallelism — an unreproducible sampler. This runs identical draws at GOMAXPROCS 1, 2 and 8 and
// requires identical output. On failure it reports the exact (seed, temp, r) so the divergence is
// reproducible rather than described.
func TestChunkedSoftmax_MachineIndependent(t *testing.T) {
	orig := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(orig)

	for _, V := range []int{152064, 262144} {
		for seed := int64(1); seed <= 3; seed++ {
			logits := randLogitsWithTies(V, rand.New(rand.NewSource(seed)))
			for _, temp := range []float64{0.8, 1.2} {
				for _, rv := range []float64{0.01, 0.3, 0.5, 0.87, 0.999} {
					runtime.GOMAXPROCS(1)
					base := chunkedAt(logits, temp, rv)
					baseZ := zAt(logits, temp)
					for _, procs := range []int{2, 8} {
						runtime.GOMAXPROCS(procs)
						if got := chunkedAt(logits, temp, rv); got != base {
							t.Fatalf("MACHINE-DEPENDENT OUTPUT: V=%d seed=%d temp=%v r=%.17g — "+
								"GOMAXPROCS=1 chose %d, GOMAXPROCS=%d chose %d. The reduction order or "+
								"chunk shape is following the core count.", V, seed, temp, rv, base, procs, got)
						}
						if gz := zAt(logits, temp); gz != baseZ {
							t.Fatalf("MACHINE-DEPENDENT Z: V=%d seed=%d temp=%v — GOMAXPROCS=1 Z=%.17g, "+
								"GOMAXPROCS=%d Z=%.17g", V, seed, temp, baseZ, procs, gz)
						}
					}
				}
			}
		}
	}
}

func zAt(logits []float32, temperature float64) float64 {
	maxv, _ := parallelMax(logits)
	tmp := make([]float64, len(logits))
	return foldChunkSums(expChunked(logits, tmp, maxv, temperature))
}

// TestChunkedSoftmax_Deterministic: repeated runs, same answer.
func TestChunkedSoftmax_Deterministic(t *testing.T) {
	logits := randLogits(262144, rand.New(rand.NewSource(6)))
	for _, rv := range []float64{0.2, 0.6, 0.95} {
		first := chunkedAt(logits, 1.0, rv)
		for rep := 0; rep < 4; rep++ {
			if got := chunkedAt(logits, 1.0, rv); got != first {
				t.Fatalf("r=%v repeat %d: %d != %d", rv, rep, got, first)
			}
		}
	}
}

// TestChunkedSoftmax_NaNFallsBack: a NaN logit must not be reduced over (max would be undefined);
// the path falls back to the exact sequential computation rather than producing garbage.
func TestChunkedSoftmax_NaNFallsBack(t *testing.T) {
	logits := randLogits(4096, rand.New(rand.NewSource(8)))
	logits[123] = float32(math.NaN())
	if _, ok := parallelMax(logits); ok {
		t.Fatal("parallelMax reported ok with a NaN present")
	}
	s := &Sampler{p: SamplingParams{Temperature: 1}, rng: rand.New(rand.NewSource(1))}
	_ = s.sampleChunked(logits, 1.0, 0.5) // must not panic
}
