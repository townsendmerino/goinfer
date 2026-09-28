package decoder

import (
	"context"
	"testing"
)

// metalLikeCost is Metal's step-kernel verify curve (metal/batch.go stepVerifyCost): a fixed overhead, then nearly flat.
var metalLikeCost = []float64{0, 1, 1.79, 1.95, 2.11, 2.25, 2.38, 2.52, 2.65, 3.65}

// TestAdaptiveDepth_costCurve: with a cost curve, Depth picks the depth with the most expected tokens per unit cost.
// A copyable stream (alpha 0.95) drafts deep; chat-like acceptance (alpha 0.3-0.5), where the linear Theta rule would
// still draft one token and pay ~1.8 steps for ~1.4 tokens, drafts nothing; and the 9th row (a second, one-row piece)
// is not worth its extra step even at alpha 0.95.
func TestAdaptiveDepth_costCurve(t *testing.T) {
	depth := func(alpha float64) int {
		a := &AdaptiveDepth{MaxDraft: 8, Theta: 0.15, Cost: metalLikeCost}
		a.ensure()
		a.alpha = alpha
		return a.Depth(8)
	}
	if d := depth(0.95); d != 7 {
		t.Errorf("alpha 0.95: depth %d, want 7 (8 rows fill the fragment; a 9th row costs a whole extra step)", d)
	}
	for _, al := range []float64{0.3, 0.4, 0.5} {
		if d := depth(al); d != 0 {
			t.Errorf("alpha %.1f: depth %d, want 0 — no depth beats plain decode at this acceptance on this curve", al, d)
		}
	}
	// The linear rule on the same stream: floor(ln 0.15 / ln 0.4) = 2, i.e. it drafts.
	lin := &AdaptiveDepth{MaxDraft: 8, Theta: 0.15}
	lin.ensure()
	lin.alpha = 0.4
	if d := lin.Depth(8); d == 0 {
		t.Errorf("linear rule at alpha 0.4: depth 0 — the comparison this test draws is gone")
	}
	a := &AdaptiveDepth{Cost: metalLikeCost}
	a.ensure()
	if a.ProbeEvery != 64 {
		t.Errorf("ProbeEvery with a cost curve = %d, want 64", a.ProbeEvery)
	}
}

type costReportingResident struct {
	ResidentForward
	cost []float64
}

func (c costReportingResident) VerifyCost() []float64 { return c.cost }

// TestGenerateNgramSpeculativeAdaptive_costOnlyWhenGreedy: the entry point hands the resident's cost curve to a greedy
// request (whose verify is the argmax-only path the curve describes) and not to a sampled one (whose verify is the
// full-logits ForwardN).
func TestGenerateNgramSpeculativeAdaptive_costOnlyWhenGreedy(t *testing.T) {
	for _, temp := range []float64{0, 0.7} {
		m, _ := loadWithFakeResident(t)
		if !m.DecodeRunnerEligible() {
			t.Skip("fixture is not resident-decode-runner-eligible")
		}
		m.resident = costReportingResident{ResidentForward: m.resident, cost: metalLikeCost}
		ad := &AdaptiveDepth{}
		ch, g, err := m.GenerateNgramSpeculativeAdaptive(context.Background(), []int{1, 2, 3}, 4, &NgramDrafter{}, ad, SamplingParams{Temperature: temp, Seed: 1})
		if err != nil {
			t.Fatalf("temperature %.1f: %v", temp, err)
		}
		collectTokens(ch)
		if g.Err() != nil {
			t.Fatalf("temperature %.1f: %v", temp, g.Err())
		}
		if got := len(ad.Cost) > 0; got != (temp == 0) {
			t.Errorf("temperature %.1f: cost curve set = %v, want %v", temp, got, temp == 0)
		}
	}
}
