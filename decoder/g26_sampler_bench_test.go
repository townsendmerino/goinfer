package decoder

import (
	"math/rand"
	"testing"
)

// SAMPLER MICROBENCHMARKS IN THIS PACKAGE MAY NOT PRODUCE QUOTABLE FIGURES.
//
// They are a tool for deciding WHERE TO LOOK. No number they emit belongs in a doc, a queue item, a commit message, or a
// comparison between two commits. This is a standing prohibition, not advice: the loop keeps the vocab-sized scratch hot in
// cache and the allocator warm, whereas decode runs one draw per token behind an ~8 ms forward that evicts both, so the
// benchmark measures a cache state that never occurs. In G26 it was wrong by 7-10x in one case and had the wrong SIGN in
// another (docs/code-notes/decoder.md#g26BenchLogits.prohibition).
//
// MEASURE IT IN SITU instead: the same-build end-to-end greedy vs temp1.0 difference, with GOINFER_NO_OPTFWD=1 so the
// optimistic-forward overlap cannot confound it. On CUDA that difference is the whole sampled tail (greedy takes
// ForwardArgmax's on-device path and never reads back the logit vector), not the sampler alone, but it is measured where the
// code runs.

// G26 localisation: the FULL temp1.0_notrunc sampling path, not expChunked alone. A microbenchmark of one function inside the
// path cannot bound the path, so this benchmarks Sampler.Sample itself. The greedy arm is the control: subtracting it
// reproduces the decomposition the peer sweep produced end-to-end.
//
// RETRACTED: this benchmark's 152k comparison is WRONG (it reports HEAD slower where end-to-end measurement says faster; the
// sign is inverted, not just the magnitude). Do not use it as evidence; the figures are in
// docs/code-notes/decoder.md#g26BenchLogits.localisation.
//
// CAVEAT, because it decides how a null result is read: these logits are SYNTHETIC. If a regression is data-dependent (a
// denormal-heavy tail, say), synthetic logits may not reproduce it, and a flat result here is then evidence about the shape of
// the cause, not an absence of one.
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
