package sampling

import (
	"math/rand"
	"runtime"
	"slices"
	"testing"
)

// R15's parallel topKByLogit arm, test-only until its call is made (docs/queue-performance.md, P42; the band is in
// docs/completed/red-october.md, "R15"), and the checks on the min-p scan that topFilterLogits runs, minPCandidates.

// topKByLogitParallel is topKByLogit by chunks: each chunk's own top k, then the top k of their union. The top k under
// (logit desc, id asc) is unique, and every member of it is in its own chunk's top k, so the set equals the serial one.
// The returned slice is unordered, as topKByLogit's is.
func topKByLogitParallel(logits []float32, k int) []int {
	n := len(logits)
	if k >= n {
		return topKByLogit(logits, k)
	}
	parts := make([][]int, numChunks)
	forEachChunk(n, func(c, lo, hi int) {
		local := topKByLogit(logits[lo:hi], k)
		for i := range local {
			local[i] += lo
		}
		parts[c] = local
	})
	union := make([]int, 0, numChunks*k)
	for _, p := range parts {
		union = append(union, p...)
	}
	slices.SortFunc(union, func(a, b int) int {
		if logits[a] != logits[b] {
			if logits[a] > logits[b] {
				return -1
			}
			return 1
		}
		return a - b
	})
	return union[:k]
}

// tiedLogits draws from a few distinct values, so the top k and the threshold both cut through ties.
func tiedLogits(n int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	a := make([]float32, n)
	for i := range a {
		a[i] = float32(rng.Intn(9))
	}
	return a
}

// TestTopKByLogitParallel_matchesSerial: the parallel top k is the serial set at every size the benchmarks use, at
// k from 1 past the chunk size, on spread logits and on logits where the cut falls inside a run of ties.
func TestTopKByLogitParallel_matchesSerial(t *testing.T) {
	for _, n := range []int{1, 7, 100, 8192, 32064, 151936, 262144} {
		for _, k := range []int{1, 8, 40, 5000} {
			for _, logits := range [][]float32{benchRandLogits(n, int64(n+k)), tiedLogits(n, int64(n*k))} {
				want := slices.Sorted(slices.Values(topKByLogit(logits, k)))
				got := slices.Sorted(slices.Values(topKByLogitParallel(logits, k)))
				if !slices.Equal(got, want) {
					t.Fatalf("n=%d k=%d: parallel top k differs from the serial one", n, k)
				}
			}
		}
	}
}

// TestMinPCandidates_matchesSerialScan: minPCandidates returns the serial scan's ids in the same order, on both sides of
// forEachChunk's parallel threshold (numChunks*64), with chunks smaller than one element, with the threshold cutting through
// ties, whatever the scratch holds or lacks, and on one worker as on many.
func TestMinPCandidates_matchesSerialScan(t *testing.T) {
	for _, procs := range []int{1, 8} {
		prev := runtime.GOMAXPROCS(procs)
		for _, n := range []int{1, 7, numChunks*64 - 1, numChunks * 64, 32064, 262144} {
			for _, logits := range [][]float32{benchRandLogits(n, int64(n)), tiedLogits(n, int64(n))} {
				thr := float64(maxScanSerial(logits)) - 4
				want := minPScanSerial(logits, thr, nil)
				dirty := make([]int, n)
				for i := range dirty {
					dirty[i] = -1
				}
				for name, scratch := range map[string][]int{"nil": nil, "short": make([]int, 0, n/2), "dirty": dirty, "empty": make([]int, 0, n)} {
					if got := minPCandidates(logits, thr, scratch); !slices.Equal(got, want) {
						t.Fatalf("GOMAXPROCS=%d n=%d scratch=%s: %d candidates, want %d, or a different order", procs, n, name, len(got), len(want))
					}
				}
			}
		}
		runtime.GOMAXPROCS(prev)
	}
}

func benchTopKByLogitParallel(b *testing.B, n, k int) {
	logits := benchRandLogits(n, 2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		topKByLogitParallel(logits, k)
	}
}

func BenchmarkTopKByLogit_gemmaVocab_k1_parallel(b *testing.B) {
	benchTopKByLogitParallel(b, 262144, 1)
}
func BenchmarkTopKByLogit_gemmaVocab_k8_parallel(b *testing.B) {
	benchTopKByLogitParallel(b, 262144, 8)
}
func BenchmarkTopKByLogit_gemmaVocab_k40_parallel(b *testing.B) {
	benchTopKByLogitParallel(b, 262144, 40)
}

func benchMinPCandidates(b *testing.B, n int) {
	logits := benchRandLogits(n, 3)
	thr := float64(maxScanSerial(logits)) - 4
	cand := make([]int, n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cand = minPCandidates(logits, thr, cand)
	}
}

func BenchmarkMinPScan_gemmaVocab_parallel(b *testing.B) { benchMinPCandidates(b, 262144) }
func BenchmarkMinPScan_152kVocab_parallel(b *testing.B)  { benchMinPCandidates(b, 151936) }
func BenchmarkMinPScan_32kVocab_parallel(b *testing.B)   { benchMinPCandidates(b, 32064) }
