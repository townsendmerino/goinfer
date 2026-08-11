package decoder

import (
	"context"
	"math"
	"slices"
	"testing"
)

// TestSpecStepLossless is the rigorous (model-free) gate for sampled speculation:
// the rejection-sampling step must reproduce the target distribution p EXACTLY,
// for any drafted token x — accept x w.p. p[x], else draw from the residual. We
// verify the emitted-token frequencies match p over many draws, for x in-support
// (high and low prob) and out-of-support (filtered out, p[x]=0 ⇒ always reject).
func TestSpecStepLossless(t *testing.T) {
	const N = 300000
	const tol = 0.01

	cases := []struct {
		name   string
		params SamplingParams
		logits []float32
		xs     []int
	}{
		{
			name:   "pure temperature (full support)",
			params: SamplingParams{Temperature: 1, Seed: 7},
			logits: []float32{2.0, 1.0, 0.5, 0.0, -1.0, -2.0},
			xs:     []int{0, 3, 5}, // top, mid, low
		},
		{
			name:   "top-k filtered support",
			params: SamplingParams{Temperature: 1, TopK: 3, Seed: 11},
			logits: []float32{2.0, 1.0, 0.5, 0.0, -1.0, -2.0},
			xs:     []int{0, 2, 5}, // 5 is filtered out → p[5]=0 → always reject
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ref := NewSampler(tc.params)
			p := ref.distVector(tc.logits) // the exact distribution the sampler draws from
			for _, x := range tc.xs {
				s := NewSampler(tc.params)
				counts := make([]float64, len(tc.logits))
				for range N {
					tok, _ := s.specStep(p, x)
					counts[tok]++
				}
				var maxDiff float64
				for i := range counts {
					d := math.Abs(counts[i]/N - p[i])
					if d > maxDiff {
						maxDiff = d
					}
				}
				if maxDiff > tol {
					t.Errorf("x=%d: emitted dist != p (maxDiff %.4f > %.4f)\n p=%v\n got=%v", x, maxDiff, tol, round(p), freq(counts, N))
				}
			}
		})
	}
}

func round(p []float64) []float64 {
	out := make([]float64, len(p))
	for i, v := range p {
		out[i] = math.Round(v*1000) / 1000
	}
	return out
}
func freq(c []float64, n int) []float64 {
	out := make([]float64, len(c))
	for i, v := range c {
		out[i] = math.Round(v/float64(n)*1000) / 1000
	}
	return out
}

// TestNgramSampledFirstTokenMatchesPlain checks the sampled path against plain
// Generate at the integration level: under pure temperature (no filters) with the
// same seed, the FIRST emitted token must be identical — both are drawFull over the
// same seed-position softmax with an identically seeded RNG. (Beyond the first token
// the two RNG streams diverge by construction, so only the first is bit-comparable;
// full-sequence equivalence is distributional, proven by TestSpecStepLossless.)
func TestNgramSampledFirstTokenMatchesPlain(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	ctx := context.Background()
	for _, seed := range []int64{1, 42, 1234} {
		sp := SamplingParams{Temperature: 0.8, Seed: seed} // no top-k/p/min-p → pure temperature
		for pi, prompt := range specPrompts {
			refCh, _ := m.Generate(ctx, prompt, 1, sp)
			ref := collectTokens(refCh)
			ch, g, err := m.GenerateNgramSpeculative(ctx, prompt, 1, &NgramDrafter{}, 8, sp)
			if err != nil {
				t.Fatalf("seed %d prompt %d: %v", seed, pi, err)
			}
			got := collectTokens(ch)
			if g.Err() != nil {
				t.Fatalf("seed %d prompt %d: stream err %v", seed, pi, g.Err())
			}
			if len(ref) != 1 || len(got) != 1 || ref[0] != got[0] {
				t.Fatalf("seed %d prompt %d: first token differs: plain %v vs sampled-spec %v", seed, pi, ref, got)
			}
		}
	}
}

// TestDistVectorHistMatchesSampler is the rigorous (model-free) gate for the
// penalty/bias threading: distVectorHist(logits, history) must equal the
// distribution the plain Sampler actually draws from after observing that history.
// If it does, the speculative path reproduces plain sampling exactly at every
// position — losslessly — even with repetition penalties and logit bias active.
func TestDistVectorHistMatchesSampler(t *testing.T) {
	const N = 300000
	const tol = 0.01
	sp := SamplingParams{
		Temperature: 0.9, TopK: 5,
		RepeatPenalty: 1.4, PresencePenalty: 0.3, FrequencyPenalty: 0.6,
		LogitBias: map[int]float32{1: 1.0, 4: -0.5},
	}
	logits := []float32{2.0, 1.2, 0.7, 0.1, -0.4, -1.2, -2.0}
	history := []int{0, 2, 2, 2, 5, 1} // repeats drive frequency/presence/repeat penalties

	p := NewSampler(sp).distVectorHist(logits, history)

	counts := make([]float64, len(logits))
	for i := range N {
		spi := sp
		spi.Seed = int64(i) // vary only the RNG; the distribution is seed-independent
		s := NewSampler(spi)
		s.Observe(history...)
		tok, _ := s.Sample(slices.Clone(logits))
		counts[tok]++
	}
	var maxDiff float64
	for i := range counts {
		if d := math.Abs(counts[i]/N - p[i]); d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff > tol {
		t.Errorf("distVectorHist != sampler draw distribution (maxDiff %.4f > %.4f)\n p   =%v\n draw=%v", maxDiff, tol, round(p), freq(counts, N))
	}
}

// TestNgramSampledRejectsUnsupported guards the remaining scope boundary: sampled
// speculation now threads penalties + logit bias (TestDistVectorHistMatchesSampler),
// so those are accepted; only a LogitProcessor (constrained/tool decoding) is still
// rejected rather than silently breaking losslessness.
func TestNgramSampledRejectsUnsupported(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	ctx := context.Background()
	// Penalties + bias are now supported — must NOT error.
	ok := []SamplingParams{
		{Temperature: 0.8, RepeatPenalty: 1.1},
		{Temperature: 0.8, PresencePenalty: 0.5, FrequencyPenalty: 0.5},
		{Temperature: 0.8, LogitBias: map[int]float32{1: 5}},
	}
	for i, sp := range ok {
		ch, g, err := m.GenerateNgramSpeculative(ctx, specPrompts[0], 4, &NgramDrafter{}, 8, sp)
		if err != nil {
			t.Errorf("supported case %d: unexpected error %v", i, err)
			continue
		}
		collectTokens(ch)
		if g.Err() != nil {
			t.Errorf("supported case %d: stream err %v", i, g.Err())
		}
	}
	// A LogitProcessor (constrained/tool decoding) is still rejected.
	bad := SamplingParams{Temperature: 0.8, LogitProcessor: func([]int, []float32) {}}
	if _, _, err := m.GenerateNgramSpeculative(ctx, specPrompts[0], 4, &NgramDrafter{}, 8, bad); err == nil {
		t.Error("LogitProcessor: expected rejection, got nil error")
	}
}
