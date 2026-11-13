package sampling

import (
	"math/rand"
	"slices"
	"testing"
)

// The parallel arms of R15's two open components, for the in-process A/B: topKByLogit and the min-p candidate scan
// over forEachChunk's fixed chunks. They are test-only until the A/B says one is worth taking
// (docs/queue-performance.md, P42); the band is in docs/completed/red-october.md, "R15".

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

// minPScanParallel is the min-p candidate scan by chunks, concatenated in chunk order: the same ids in the same
// ascending order as the serial scan, which finishFilter's (prob desc, id asc) sort does not depend on but a caller
// reading cand in order could.
func minPScanParallel(logits []float32, thr float64, cand []int) []int {
	parts := make([][]int, numChunks)
	forEachChunk(len(logits), func(c, lo, hi int) {
		var local []int
		for i := lo; i < hi; i++ {
			if float64(logits[i]) >= thr {
				local = append(local, i)
			}
		}
		parts[c] = local
	})
	cand = cand[:0]
	for _, p := range parts {
		cand = append(cand, p...)
	}
	return cand
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

// TestMinPScanParallel_matchesSerial: the parallel scan returns the serial scan's ids in the same order.
func TestMinPScanParallel_matchesSerial(t *testing.T) {
	for _, n := range []int{1, 7, 8192, 32064, 262144} {
		for _, logits := range [][]float32{benchRandLogits(n, int64(n)), tiedLogits(n, int64(n))} {
			thr := float64(maxScanSerial(logits)) - 4
			want := minPScanSerial(logits, thr, nil)
			got := minPScanParallel(logits, thr, nil)
			if !slices.Equal(got, want) {
				t.Fatalf("n=%d: parallel min-p candidates differ from the serial scan", n)
			}
		}
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

func BenchmarkMinPScan_gemmaVocab_parallel(b *testing.B) {
	logits := benchRandLogits(262144, 3)
	thr := float64(maxScanSerial(logits)) - 4
	cand := make([]int, 0, len(logits))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cand = minPScanParallel(logits, thr, cand)
	}
}
