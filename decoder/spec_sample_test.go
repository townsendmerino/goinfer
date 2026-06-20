package decoder

import (
	"context"
	"math"
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
				for i := 0; i < N; i++ {
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
		t.Skipf("no model (%v); set GINFER_PREQUANT_GGUF", err)
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

// TestNgramSampledRejectsUnsupported guards the losslessness scope: sampled
// speculation must refuse the history-dependent transforms it cannot yet thread,
// rather than silently producing a biased distribution.
func TestNgramSampledRejectsUnsupported(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GINFER_PREQUANT_GGUF", err)
	}
	ctx := context.Background()
	bad := []SamplingParams{
		{Temperature: 0.8, RepeatPenalty: 1.1},
		{Temperature: 0.8, PresencePenalty: 0.5},
		{Temperature: 0.8, FrequencyPenalty: 0.5},
		{Temperature: 0.8, LogitBias: map[int]float32{1: 5}},
	}
	for i, sp := range bad {
		if _, _, err := m.GenerateNgramSpeculative(ctx, specPrompts[0], 4, &NgramDrafter{}, 8, sp); err == nil {
			t.Errorf("case %d: expected rejection of unsupported sampler params, got nil error", i)
		}
	}
}
