package decoder

import (
	"context"
	"math"
)

// AdaptiveDepth chooses the per-round draft depth from a running estimate of per-position acceptance (an EMA of
// realized accepts), replacing a fixed K. Depth grows on copy-heavy streams (alpha toward 1) and collapses toward
// plain decode on novel text, which avoids the fixed-K over-draft that makes low-acceptance workloads slower than plain
// (docs/spec/04-adaptive-depth.md).
//
// The rule (docs/spec/00-core.md §2): extend depth while the expected marginal committed token, probability alpha^d
// that the chain reaches depth d, still beats the marginal cost of one more verify node, Theta:
//
//	D = floor( ln(Theta) / ln(alpha) ),  clamped to [0, min(MaxDraft, proposed)]
//
// Theta is the relative cost of one extra verify node on this backend and must be measured (thetaFor). D=0 means
// "don't speculate this round" (plain decode); a periodic probe forces D>=1 occasionally so a stream that becomes
// copyable again can climb back out of D=0.
type AdaptiveDepth struct {
	MaxDraft int // ceiling on tokens drafted per round (default 8)
	// Theta is the marginal cost of one extra verify node, in units of one single-token target step. Its domain is
	// (0, +inf): >= 1 is legal and means an extra node costs at least a whole step, i.e. never draft. Zero or negative means
	// unset and takes the backend default (thetaFor). Do not narrow the domain to [0,1): that rejects the values Metal
	// actually measures (just above 1) and substitutes 0.5, the most over-drafting setting.
	Theta float64
	// Cost, when set, replaces the linear Theta rule with a measured cost curve: Cost[m] is the time to verify m rows
	// (m >= 1) in units of one single-token target step, so Cost[1] is 1 (a round that drafts nothing is a plain decode
	// step). A verify whose cost is not linear in m needs it (a fixed overhead, then a nearly flat slope): the linear rule,
	// which drafts while alpha^k > Theta, would draft single tokens at an acceptance that cannot pay for the overhead. With
	// Cost, Depth picks the depth that maximises expected emitted tokens per unit cost, and 0 when nothing beats plain
	// decode.
	Cost       []float64
	Lambda     float64 // EMA retention for the acceptance estimate (default 0.8)
	ProbeEvery int     // force D≥1 after this many idle (D=0) rounds (default 16; 64 with a Cost curve)

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
	if a.Theta <= 0 || math.IsNaN(a.Theta) {
		a.Theta = defaultTheta // unset; callers that know the backend set it first
	}
	if a.Lambda <= 0 || a.Lambda >= 1 {
		a.Lambda = 0.8
	}
	if a.ProbeEvery <= 0 {
		a.ProbeEvery = 16
		if len(a.Cost) > 1 {
			// A probe verifies one draft at Cost[2], a real overhead over a plain step on a step-kernel verify; probing every 64
			// idle rounds keeps that small on a stream that never drafts.
			a.ProbeEvery = 64
		}
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
	// Theta >= 1 means one extra verify node costs at least a whole target step, so no acceptance rate can pay for it: alpha
	// < 1 always, and the test below would return 0 for every alpha. Skip the periodic probe too: it exists to refresh a
	// stale alpha, and here the decision does not depend on alpha, so a probe is wasted draft work.
	if a.Theta >= 1 {
		return 0
	}
	if a.idle >= a.ProbeEvery { // refresh a stale estimate
		return 1
	}
	if len(a.Cost) > 1 {
		// Expected tokens a round emits when it verifies d drafts: the correction or bonus token, plus draft i when
		// the first i all land, E(d) = sum_{i=0..d} alpha^i. Pick the d with the best E(d) / Cost[d+1]; plain decode
		// (d = 0) scores exactly 1.
		best, bd, e, p := 1.0, 0, 1.0, 1.0
		for d := 1; d <= max && d+1 < len(a.Cost); d++ {
			p *= a.alpha
			e += p
			if r := e / a.Cost[d+1]; r > best {
				best, bd = r, d
			}
		}
		return bd
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
	if ad.Theta <= 0 { // unset by the caller: take this model's measured verify cost
		ad.Theta = target.verifyTheta()
	}
	// A resident whose argmax-only verify has a measured cost curve (VerifyCostReporter) gets it — for a greedy
	// request only, since only greedy verifies through the argmax path (genNgramInto's idsVerify); a sampled request
	// verifies through the full-logits ForwardN, which the curve does not describe.
	if ad.Cost == nil && sp.Temperature <= 0 && target.resident != nil {
		if cr, ok := target.resident.(VerifyCostReporter); ok {
			ad.Cost = cr.VerifyCost()
		}
	}
	ad.ensure()
	if ad.Theta >= 1 {
		// Theta >= 1 means Depth always returns 0 (see its comment): no acceptance rate can pay for even one verify node, on any
		// round. Every round would still pay for a history clone, an n-gram scan and a full-logits ForwardN to draft and verify
		// nothing. Validate first so an invalid drafter or sp still errors exactly as genNgram would, then decline straight to
		// the cheap path.
		if err := validateNgramSpec(target, drafter, sp); err != nil {
			return nil, nil, err
		}
		ch, gen := target.Generate(ctx, prompt, maxTokens, sp)
		return ch, gen, nil
	}
	return target.genNgram(ctx, prompt, maxTokens, drafter, ad.MaxDraft, sp, nil, ad)
}

// VerifyCostReporter is an OPTIONAL ResidentForward extension: a resident whose argmax-only batched verify
// (PrefillLastNArgmax) is not linear in its row count reports its measured cost curve, AdaptiveDepth.Cost's shape
// (index m = rows verified, in single-token steps; nil when that verify is unavailable). Exported so an implementation
// can pin itself to it at compile time: an optional interface that stops matching fails open, silently back to Theta.
type VerifyCostReporter interface {
	VerifyCost() []float64
}

// defaultTheta is the fallback when a caller sets no Theta and the backend is not in thetaFor's table: the CPU value.
const defaultTheta = 0.5

// thetaFor returns the measured marginal verify-node cost for a backend. Every value is measured by probes that share
// one definition, Theta = (least-squares slope of T(n)) / T(1), so they are comparable: decoder/theta_probe_test.go
// (CPU control), cuda/theta_probe_test.go, metal/theta_probe_test.go. Running the CPU constant on every backend is
// wrong in opposite directions: CUDA drafts too shallow, Metal drafts when it should not draft at all.
func thetaFor(backend string) float64 {
	switch backend {
	case "metal":
		// Measured on the layer-major batched ForwardBatch; every sweep cell stayed under 1.0, so speculation is enabled.
		// Understating Theta is the risky direction (as in the CUDA case below), so 0.96 covers the observed maximum.
		return 0.96
	case "cuda":
		// Measured with cuda/theta_probe_test.go. The conservative end of the measured range is used deliberately: Theta
		// appears inside floor(ln(Theta)/ln(alpha)), so understating it drafts deeper, and a too-deep draft on a
		// low-acceptance stream is the failure the adaptive controller exists to prevent. Taking the shallow end under-claims
		// the win rather than risking the regression.
		return 0.251
	default: // "", "cpu", and any backend with no measurement of its own
		return defaultTheta
	}
}

// verifyTheta returns the Theta for the path this model's speculative verify actually runs on. The distinction that
// matters is resident vs staged, not which backend was requested: genNgramInto verifies through
// target.resident.ForwardN when residency built and falls back to the CPU batched forwardN when it did not, so a
// "webgpu-staged" or "metal-staged" model verifies on CPU and must get the CPU constant. Keying off Options.Backend
// alone would hand a declined-residency model the GPU value and silently mis-tune the case where the decline is
// already costing the whole forward. VerifyPathReporter asks directly whether the verify forward (ForwardN) is
// batched, rather than guessing from PrefillPath, which prices a separate code path.
func (m *Model) verifyTheta() float64 {
	if m == nil || m.resident == nil {
		return defaultTheta // staged or CPU: the verify is the CPU batched ForwardN
	}
	if m.be == nil {
		return defaultTheta
	}
	// Prefer the explicit VerifyPathReporter if implemented: ask the resident directly
	// whether its ForwardN verify path is batched.
	if rep, ok := m.resident.(VerifyPathReporter); ok {
		if batched, _ := rep.VerifyPath(); !batched {
			return sequentialVerifyTheta
		}
		return thetaFor(m.be.Name())
	}
	// Fallback for residents that only implement PrefillPathReporter: ask whether their prefill path is batched.
	if pf, ok := m.resident.(Prefiller); ok {
		if rep, ok := pf.(PrefillPathReporter); ok {
			if batched, _ := rep.PrefillPath(); !batched {
				return sequentialVerifyTheta
			}
		}
	}
	return thetaFor(m.be.Name())
}

// sequentialVerifyTheta is Theta for a resident whose ForwardN is a loop of single-token forwards. T(n) is n*T(1) by
// construction, so the ratio is 1 plus whatever per-row overhead the loop adds; Metal measured exactly that shape
// and ships 1.02, and the same number is used here for the same reason.
const sequentialVerifyTheta = 1.02
