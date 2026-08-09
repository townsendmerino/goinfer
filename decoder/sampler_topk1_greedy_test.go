package decoder

import (
	"math/rand"
	"testing"
)

// P1 gate (tier a) — `top_k=1` must emit BYTE-IDENTICAL token streams to `temperature=0` greedy.
//
// This is the correctness footing for routing top_k=1 to the on-device greedy fast path
// (GreedyEquivalent, decoder/model.go): the claim is that temperature scaling is monotone so it
// preserves ordering, and a one-token distribution is deterministic — therefore top_k=1 at ANY
// temperature picks the same token greedy would. If that is ever false, the fast path silently
// changes output, so it is asserted directly rather than argued.
//
// No GPU: this compares the two HOST sampling paths, which is what the device argmax is aligned to
// (audit C-14 gave the device the same lowest-index tie-break; gate cuda.TestArgmaxTieBreak).
func TestTopK1_MatchesGreedy(t *testing.T) {
	temps := []float64{0.01, 0.7, 1.5}
	vocabs := []int{152064, 262144} // qwen2.5 / gemma3 — the widths D6 measures
	const steps = 24                // a token STREAM, not a single draw: divergence compounds

	for _, V := range vocabs {
		for _, temp := range temps {
			for seed := int64(1); seed <= 6; seed++ {
				r := rand.New(rand.NewSource(seed))
				// Tie-heavy logits on half the seeds: the C-14 case shape, where the tie-break
				// decides the token and an ordering bug would actually surface.
				var logits []float32
				if seed%2 == 0 {
					logits = randLogitsWithTies(V, r)
				} else {
					logits = randLogits(V, r)
				}
				greedy := &Sampler{p: SamplingParams{Temperature: 0}, rng: rand.New(rand.NewSource(99))}
				topk1 := &Sampler{p: SamplingParams{Temperature: temp, TopK: 1}, rng: rand.New(rand.NewSource(7))}

				for step := range steps {
					gi, err := greedy.Sample(logits)
					if err != nil {
						t.Fatalf("greedy: %v", err)
					}
					ki, err := topk1.Sample(logits)
					if err != nil {
						t.Fatalf("top_k=1: %v", err)
					}
					if gi != ki {
						t.Fatalf("V=%d temp=%v seed=%d step=%d: top_k=1 chose %d, greedy chose %d — "+
							"routing top_k=1 to the greedy fast path would change emitted tokens",
							V, temp, seed, step, ki, gi)
					}
					// Perturb so successive steps are not the same argmax: a stream, not one draw.
					logits[gi] = float32(-1e9)
				}
			}
		}
	}
}

// TestTopK1_WithNucleusStillGreedy: top_p / min_p alongside top_k=1 must not change the pick. Both
// cuts clamp at >=1 retained token ("always keep the top token"), so the retained set is exactly
// top-1 — this pins that reasoning, since GreedyEquivalent deliberately does NOT exclude them.
func TestTopK1_WithNucleusStillGreedy(t *testing.T) {
	r := rand.New(rand.NewSource(11))
	logits := randLogitsWithTies(152064, r)
	greedy := &Sampler{p: SamplingParams{Temperature: 0}, rng: rand.New(rand.NewSource(1))}
	want, err := greedy.Sample(logits)
	if err != nil {
		t.Fatalf("greedy: %v", err)
	}
	for _, c := range []struct {
		name       string
		topP, minP float64
	}{
		{"top_p=0.01 (tightest nucleus)", 0.01, 0},
		{"top_p=0.95", 0.95, 0},
		{"min_p=0.99 (cuts all but the max)", 0, 0.99},
		{"top_p=0.5 + min_p=0.5", 0.5, 0.5},
	} {
		t.Run(c.name, func(t *testing.T) {
			s := &Sampler{p: SamplingParams{Temperature: 1.2, TopK: 1, TopP: c.topP, MinP: c.minP},
				rng: rand.New(rand.NewSource(5))}
			got, err := s.Sample(logits)
			if err != nil {
				t.Fatalf("sample: %v", err)
			}
			if got != want {
				t.Errorf("top_k=1 with %s chose %d, greedy chose %d", c.name, got, want)
			}
		})
	}
}

// TestGreedyEquivalent_predicate pins the predicate's exact shape — it is the routing decision, so
// a wrong TRUE silently changes output and a wrong FALSE silently costs the speedup.
func TestGreedyEquivalent_predicate(t *testing.T) {
	cases := []struct {
		name string
		p    SamplingParams
		want bool
	}{
		{"top_k=1, temp 0.8", SamplingParams{TopK: 1, Temperature: 0.8}, true},
		{"top_k=1, temp 1.5 + top_p", SamplingParams{TopK: 1, Temperature: 1.5, TopP: 0.9}, true},
		{"top_k=1, temp 0", SamplingParams{TopK: 1, Temperature: 0}, true},
		{"top_k=2 (not deterministic)", SamplingParams{TopK: 2, Temperature: 0.8}, false},
		{"top_k unset", SamplingParams{Temperature: 0.8}, false},
		{"top_k=1 + logprobs (needs the distribution)", SamplingParams{TopK: 1, Logprobs: true}, false},
		{"top_k=1 + logit bias (history-dependent)", SamplingParams{TopK: 1, LogitBias: map[int]float32{5: 1}}, false},
		{"top_k=1 + repeat penalty", SamplingParams{TopK: 1, RepeatPenalty: 1.1}, false},
		{"top_k=1 + presence penalty", SamplingParams{TopK: 1, PresencePenalty: 0.5}, false},
		{"top_k=1 + frequency penalty", SamplingParams{TopK: 1, FrequencyPenalty: 0.5}, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s := &Sampler{p: c.p, rng: rand.New(rand.NewSource(1))}
			if got := s.GreedyEquivalent(); got != c.want {
				t.Errorf("GreedyEquivalent() = %v, want %v", got, c.want)
			}
		})
	}
}

// TestGreedyEquivalent_stableAcrossHistory is the penaltiesActive trap, asserted.
//
// A backend consults this predicate ONCE per request, before any token is observed. penaltiesActive()
// is derived from observed history, so it is false before the first Observe and true after — a
// predicate built on it would answer differently at the moment of routing than mid-generation.
// HistoryDependent asks whether penalties are CONFIGURED, which cannot flip. This test would fail if
// someone "simplified" GreedyEquivalent to use penaltiesActive.
func TestGreedyEquivalent_stableAcrossHistory(t *testing.T) {
	s := &Sampler{p: SamplingParams{TopK: 1, Temperature: 0.8, RepeatPenalty: 1.1},
		rng: rand.New(rand.NewSource(1))}
	before := s.GreedyEquivalent()
	s.Observe(42)
	s.Observe(43)
	if after := s.GreedyEquivalent(); after != before {
		t.Fatalf("GreedyEquivalent flipped %v -> %v across Observe — a per-request routing decision "+
			"must not depend on history (use HistoryDependent, not penaltiesActive)", before, after)
	}
	if before {
		t.Error("configured penalties must make GreedyEquivalent false (the fast path cannot apply them)")
	}
}
