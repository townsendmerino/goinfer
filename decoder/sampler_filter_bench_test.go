package decoder

import (
	"math/rand"
	"testing"
)

// maxScanSerial reproduces topFilterLogits's own inline max-scan (a serial, 4x-unrolled loop at the top of the function,
// not parallelMax in sampler_chunked.go) in isolation, for a same-session A/B against parallelMax; same design as the
// softcap benchmark in softcap_test.go. Task: docs/tasks/red-october.md, R15.
func maxScanSerial(logits []float32) float32 {
	maxF := logits[0]
	i := 1
	_ = logits[len(logits)-1]
	for ; i+3 < len(logits); i += 4 {
		v0, v1, v2, v3 := logits[i], logits[i+1], logits[i+2], logits[i+3]
		m0 := max(v0, v1)
		m1 := max(v2, v3)
		m := max(m0, m1)
		if m > maxF {
			maxF = m
		}
	}
	for ; i < len(logits); i++ {
		if logits[i] > maxF {
			maxF = logits[i]
		}
	}
	return maxF
}

func benchRandLogits(n int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	a := make([]float32, n)
	for i := range a {
		a[i] = float32(rng.NormFloat64()) * 8
	}
	return a
}

// TestMaxScanSerial_matchesParallelMax is the bit-exactness check the A/B below leans on: max is associative and
// commutative barring NaN, so parallelMax's chunked reduction must agree with the serial scan exactly at every vocab
// size the benchmark uses.
func TestMaxScanSerial_matchesParallelMax(t *testing.T) {
	for _, n := range []int{1, 7, 8192, 32064, 151936, 262144} {
		logits := benchRandLogits(n, int64(n))
		want := maxScanSerial(logits)
		got, ok := parallelMax(logits)
		if !ok {
			t.Fatalf("n=%d: parallelMax reported NaN on a NaN-free input", n)
		}
		if float32(got) != want {
			t.Fatalf("n=%d: parallelMax=%v serial=%v", n, got, want)
		}
	}
}

func benchMaxScan(b *testing.B, n int, parallel bool) {
	logits := benchRandLogits(n, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if parallel {
			parallelMax(logits)
		} else {
			maxScanSerial(logits)
		}
	}
}

// Vocab sizes: gemma 262144, qwen/llama-class ~152k, phi3-scale 32064, the sizes of the CUDA-side softcap vocab-scaling
// measurement.
func BenchmarkMaxScan_gemmaVocab_serial(b *testing.B)   { benchMaxScan(b, 262144, false) }
func BenchmarkMaxScan_gemmaVocab_parallel(b *testing.B) { benchMaxScan(b, 262144, true) }
func BenchmarkMaxScan_152kVocab_serial(b *testing.B)    { benchMaxScan(b, 151936, false) }
func BenchmarkMaxScan_152kVocab_parallel(b *testing.B)  { benchMaxScan(b, 151936, true) }
func BenchmarkMaxScan_32kVocab_serial(b *testing.B)     { benchMaxScan(b, 32064, false) }
func BenchmarkMaxScan_32kVocab_parallel(b *testing.B)   { benchMaxScan(b, 32064, true) }

// topKByLogit and the min-p linear scan are sized here, not parallelized: unlike the max-scan, parallelizing a
// heap-select or an order-preserving filter is real algorithmic work, and R15 asks whether either is worth it. topK
// 1/8/40 span greedy-adjacent to a typical sampling config.
func benchTopKByLogit(b *testing.B, n, k int) {
	logits := benchRandLogits(n, 2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		topKByLogit(logits, k)
	}
}

func BenchmarkTopKByLogit_gemmaVocab_k1(b *testing.B)  { benchTopKByLogit(b, 262144, 1) }
func BenchmarkTopKByLogit_gemmaVocab_k8(b *testing.B)  { benchTopKByLogit(b, 262144, 8) }
func BenchmarkTopKByLogit_gemmaVocab_k40(b *testing.B) { benchTopKByLogit(b, 262144, 40) }

func minPScanSerial(logits []float32, thr float64, cand []int) []int {
	cand = cand[:0]
	for i, v := range logits {
		if float64(v) >= thr {
			cand = append(cand, i)
		}
	}
	return cand
}

func benchMinPScan(b *testing.B, n int) {
	logits := benchRandLogits(n, 3)
	maxF := maxScanSerial(logits)
	thr := float64(maxF) - 4 // keeps a modest, realistic fraction of the vocab as candidates
	cand := make([]int, 0, n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cand = minPScanSerial(logits, thr, cand)
	}
}

func BenchmarkMinPScan_gemmaVocab(b *testing.B) { benchMinPScan(b, 262144) }

// End to end: the whole topFilterLogits call for each active-filter shape at gemma vocab, to put the per-component
// numbers above against the per-token cost that reaches a decode token's budget (Z's chunkedZ is already parallel).
func benchTopFilterLogits(b *testing.B, topK int, topP, minP float64) {
	const n = 262144
	logits := benchRandLogits(n, 4)
	vocabScratch := make([]float64, n)
	candScratch := make([]int, 0, n)
	ipsScratch := make([]indexedProb, 0, n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		topFilterLogits(logits, 1.0, topK, topP, minP, vocabScratch, candScratch, ipsScratch)
	}
}

func BenchmarkTopFilterLogits_gemmaVocab_topK40(b *testing.B) { benchTopFilterLogits(b, 40, 0, 0) }
func BenchmarkTopFilterLogits_gemmaVocab_topP(b *testing.B)   { benchTopFilterLogits(b, 0, 0.95, 0) }
func BenchmarkTopFilterLogits_gemmaVocab_minP(b *testing.B)   { benchTopFilterLogits(b, 0, 0, 0.05) }
