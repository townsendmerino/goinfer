package decoder

import (
	"math/rand"
	"testing"
)

// SAMPLER MICROBENCHMARKS IN THIS PACKAGE MAY NOT PRODUCE QUOTABLE FIGURES.
//
// They are a tool for deciding WHERE TO LOOK. No number they emit belongs in a doc, a queue item, a
// commit message, or a comparison between two commits. This is a standing prohibition, not advice,
// and it was earned twice in a single investigation (G26, 2026-08-27):
//
//   1. BenchmarkExpChunked's ~96 us was used to argue that "sampling is a low-single-digit
//      percentage of per-token time, so no sampler change can move end-to-end by 5.9%." The real
//      in-situ sampled tail is 703-950 us — 7-10x larger. The bound retired the correct hypothesis
//      for two rounds.
//   2. A whole-path Sampler.Sample benchmark reported HEAD 23% SLOWER at 152k vocab. Measured
//      end-to-end on a 151936-vocab model, HEAD is 31% FASTER. Not a mis-scaled magnitude — an
//      INVERTED SIGN, and it had already been filed as a finding before the end-to-end run.
//
// WHY the loop lies here specifically: it keeps the vocab-sized scratch hot in cache and the
// allocator warm in its free-list, whereas decode runs one draw per token behind an ~8 ms forward
// that evicts both. The benchmark measures a cache state that never occurs.
//
// MEASURE IT IN SITU INSTEAD, which costs nothing extra: the same-build end-to-end greedy vs
// temp1.0 difference, with GOINFER_NO_OPTFWD=1 so the optimistic-forward overlap cannot confound
// it. On CUDA that difference is the whole sampled tail (greedy takes ForwardArgmax's on-device
// path and never reads back the logit vector), not the sampler alone — but it is measured where the
// code runs, and both times the two disagreed, the microbenchmark was the one that was wrong.

// G26 localisation: the FULL temp1.0_notrunc sampling path, not expChunked alone.
//
// WHY THE WHOLE PATH. G26's earlier round eliminated the sampler using BenchmarkExpChunked's
// ~96 us, and that bound was wrong by 7-10x: the end-to-end greedy-vs-temp1.0 gap puts the real
// sampling step at 703 us (ca29d6c) / 950 us (HEAD) per token on phi3-mini. A microbenchmark of
// one function inside the path cannot bound the path. This benchmarks Sampler.Sample itself.
//
// The greedy arm is the control, and it is the point: subtracting it here reproduces the same
// decomposition the peer sweep produced end-to-end, so the two are comparable rather than merely
// suggestive.
//
// RESULT, 2026-08-27: THIS BENCHMARK'S 152k COMPARISON IS WRONG — do not use it as evidence.
// Measured end-to-end on qwen2.5-coder-1.5B (vocab 151936) with optFwd disabled in both arms, the
// production sampling step goes 1467 us -> 1009 us, i.e. HEAD is 31% FASTER, where this benchmark
// reports HEAD 23% SLOWER. The sign is inverted, not merely the magnitude. A tight loop keeps the
// scratch buffer hot and the allocator warm; decode does neither, running it behind an ~8 ms
// forward. Prefer the in-situ measure: the same-build greedy-vs-temp1.0 end-to-end difference with
// GOINFER_NO_OPTFWD=1. Kept only because the retraction is worth more than the file.
//
// CAVEAT, stated because it decides how a null result is read: these logits are SYNTHETIC. If the
// regression is data-dependent (a denormal-heavy tail, say), synthetic logits may not reproduce it
// -- and a flat result here is then evidence about the shape of the cause, not an absence of one.
func g26BenchLogits(n int, seed int64) []float32 {
	r := rand.New(rand.NewSource(seed))
	l := make([]float32, n)
	for i := range l {
		l[i] = float32(r.NormFloat64() * 2.0)
	}
	l[r.Intn(n)] = 14.0 // a peak, as a real next-token distribution has
	return l
}

func g26BenchSample(b *testing.B, vocab int, temp float64) {
	logits := g26BenchLogits(vocab, 42)
	scratch := make([]float32, vocab)
	s := NewSampler(SamplingParams{Temperature: temp, Seed: 7})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(scratch, logits) // Sample may modify in place; each iteration sees identical input
		if _, err := s.Sample(scratch); err != nil {
			b.Fatal(err)
		}
	}
}

// 32064 is phi3-mini's vocab -- the model G26 is about.
func BenchmarkG26SampleTemp1_32k(b *testing.B)   { g26BenchSample(b, 32064, 1.0) }
func BenchmarkG26SampleGreedy_32k(b *testing.B)  { g26BenchSample(b, 32064, 0.0) }
func BenchmarkG26SampleTemp1_152k(b *testing.B)  { g26BenchSample(b, 151936, 1.0) }
func BenchmarkG26SampleGreedy_152k(b *testing.B) { g26BenchSample(b, 151936, 0.0) }
