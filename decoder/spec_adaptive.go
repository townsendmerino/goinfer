package decoder

import (
	"context"
	"math"
)

// AdaptiveDepth chooses the per-round draft depth from a running estimate of
// per-position acceptance (an EMA of realized accepts), replacing a fixed K. It is
// the 04-adaptive-depth controller: depth grows on copy-heavy streams (α→1) and
// collapses toward plain decode on novel text (α low), which kills the fixed-K
// over-draft that makes low-acceptance workloads SLOWER than plain (a verify pass
// over draft tokens that mostly get rejected isn't paid back).
//
// The rule (00-core §2): extend depth while the expected marginal committed token
// — probability α^d that the chain reaches depth d — still beats the marginal cost
// of one more verify node, Theta:
//
//	D = floor( ln(Theta) / ln(α) ),  clamped to [0, min(MaxDraft, proposed)]
//
// Theta is the relative cost of one extra verify node on *this backend* — measure
// it. It is ~0.5 on the batched-CPU ForwardN path here (a node is half a target
// step), and →0 on a fully memory-bound GPU verify (extra nodes are ~free, so deep
// drafts always pay). D=0 means "don't speculate this round" (plain decode); a
// periodic probe forces D≥1 occasionally so a stream that becomes copyable again
// can climb back out of D=0.
type AdaptiveDepth struct {
	MaxDraft   int     // ceiling on tokens drafted per round (default 8)
	Theta      float64 // marginal verify-node cost in [0,1) (default 0.5)
	Lambda     float64 // EMA retention for the acceptance estimate (default 0.8)
	ProbeEvery int     // force D≥1 after this many idle (D=0) rounds (default 16)

	alpha  float64 // running per-position acceptance estimate
	idle   int     // consecutive rounds with no observation (D=0)
	inited bool
}

func (a *AdaptiveDepth) ensure() {
	if a.inited {
		return
	}
	if a.MaxDraft <= 0 {
		a.MaxDraft = 8
	}
	if a.Theta <= 0 || a.Theta >= 1 {
		a.Theta = 0.5
	}
	if a.Lambda <= 0 || a.Lambda >= 1 {
		a.Lambda = 0.8
	}
	if a.ProbeEvery <= 0 {
		a.ProbeEvery = 16
	}
	a.alpha = 0.9 // optimistic start: speculate, then adapt to the stream
	a.inited = true
}

// Depth returns how many of the proposedLen drafted tokens to verify this round.
func (a *AdaptiveDepth) Depth(proposedLen int) int {
	a.ensure()
	if proposedLen <= 0 {
		return 0
	}
	max := min(proposedLen, a.MaxDraft)
	if a.idle >= a.ProbeEvery { // refresh a stale estimate
		return 1
	}
	if a.alpha <= a.Theta { // even one node isn't worth it
		return 0
	}
	if a.alpha >= 0.999 { // avoid divide-by-~0; go as deep as allowed
		return max
	}
	d := int(math.Floor(math.Log(a.Theta) / math.Log(a.alpha)))
	if d < 0 {
		d = 0
	}
	return min(d, max)
}

// Observe folds this round's realized outcome into the acceptance estimate.
// observed is the number of draft positions actually checked (accepted plus the
// one rejected position, if any); accepted is how many matched. A round that
// drafted nothing (observed==0) advances the idle counter instead.
func (a *AdaptiveDepth) Observe(accepted, observed int) {
	a.ensure()
	if observed <= 0 {
		a.idle++
		return
	}
	a.idle = 0
	rate := float64(accepted) / float64(observed)
	a.alpha = a.Lambda*a.alpha + (1-a.Lambda)*rate
}

// Alpha is the current running per-position acceptance estimate (for telemetry).
func (a *AdaptiveDepth) Alpha() float64 { a.ensure(); return a.alpha }

// GenerateNgramSpeculativeAdaptive is GenerateNgramSpeculative with the fixed K
// replaced by the AdaptiveDepth controller ad (nil ⇒ a fresh default). Output is
// still token-identical to plain greedy (lossless); only the per-round depth — and
// thus the speed — changes. This is the path that should be preferred over fixed-K
// once acceptance varies across the stream.
func (target *Model) GenerateNgramSpeculativeAdaptive(ctx context.Context, prompt []int, maxTokens int, drafter Drafter, ad *AdaptiveDepth, sp SamplingParams) (<-chan int, *Generation, error) {
	if ad == nil {
		ad = &AdaptiveDepth{}
	}
	ad.ensure()
	return target.genNgram(ctx, prompt, maxTokens, drafter, ad.MaxDraft, sp, nil, ad)
}
