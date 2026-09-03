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
	MaxDraft int // ceiling on tokens drafted per round (default 8)
	// Theta is the marginal cost of one extra verify node, in units of one
	// single-token target step. Domain is (0, +inf): >= 1 is LEGAL and means
	// "an extra node costs at least a whole step", i.e. never draft. Zero or
	// negative means unset and takes the backend default (see thetaFor).
	//
	// It used to be documented and enforced as [0,1), which silently rejected
	// every value Metal actually measures -- 1.006 to 1.048 across two models
	// and two depths, 2026-09-01 -- and substituted 0.5, the most over-drafting
	// setting available. A parameter whose domain excludes the measurement is
	// not a default, it is a wrong answer that cannot be corrected.
	Theta      float64
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
	if a.Theta <= 0 || math.IsNaN(a.Theta) {
		a.Theta = defaultTheta // unset; callers that know the backend set it first
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
	// Theta >= 1 means one extra verify node costs at least a whole target step,
	// so no acceptance rate can pay for it: alpha < 1 always, and the test below
	// would return 0 for every alpha. Skip the periodic probe too -- it exists to
	// refresh a STALE ALPHA, and here the decision does not depend on alpha, so a
	// probe cannot change the answer and is pure wasted draft work. This is the
	// live case on Metal, whose ForwardN is a loop of single-token Forwards
	// (measured Theta 1.006-1.048).
	if a.Theta >= 1 {
		return 0
	}
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
	if ad.Theta <= 0 { // unset by the caller: take this model's measured verify cost
		ad.Theta = target.verifyTheta()
	}
	ad.ensure()
	if ad.Theta >= 1 {
		// P-16: Theta >= 1 means Depth() always returns 0 (see its own comment) --
		// no acceptance rate can pay for even one verify node, on ANY round,
		// regardless of the stream. Every round would still pay for a history
		// clone, an n-gram scan, and a full-logits ForwardN to draft and verify
		// nothing (measured 1.01x of plain Generate on Metal, where this is the
		// live case). Validate first so an invalid drafter/sp still errors exactly
		// as genNgram would, then decline straight to the cheap path.
		if err := validateNgramSpec(target, drafter, sp); err != nil {
			return nil, nil, err
		}
		ch, gen := target.Generate(ctx, prompt, maxTokens, sp)
		return ch, gen, nil
	}
	return target.genNgram(ctx, prompt, maxTokens, drafter, ad.MaxDraft, sp, nil, ad)
}

// defaultTheta is the fallback when a caller sets no Theta and the backend is
// not in the table below. It is the CPU value, which is where the constant came
// from originally -- re-measured 0.506 (depth 128) and 0.532 (depth 512) on
// 2026-09-01, so 0.5 remains right for the path it was named after.
const defaultTheta = 0.5

// thetaFor returns the measured marginal verify-node cost for a backend.
//
// Every value here is MEASURED by the probes that share one definition --
// Theta = (least-squares slope of T(n)) / T(1) -- so the three are directly
// comparable: decoder/theta_probe_test.go (CPU control),
// cuda/theta_probe_test.go, metal/theta_probe_test.go.
//
// Before this table every backend ran the CPU constant 0.5, which
// spec_adaptive.go's own doc comment had asked someone to measure since it was
// written. The consequences ran in opposite directions: CUDA drafts far too
// shallow (its real cost is ~1/3 of the constant, so deeper chains would pay
// and were never tried), while Metal drafts when it should not draft at all.
func thetaFor(backend string) float64 {
	switch backend {
	case "metal":
		// 1.006 / 1.019 / 1.020 / 1.048 across {0.5B, 1.5B} x {128, 512},
		// 2026-09-01, M1 Pro. T(n)/T(1) is linear to n=16 (16.07-16.81),
		// which is what a loop of single-token Forwards predicts exactly.
		// >= 1 by measurement, so this DISABLES speculation on Metal until
		// ForwardN becomes a real batch -- that is the finding, not a
		// workaround for it.
		return 1.02
	case "cuda":
		// 0.155-0.251 measured (cuda/theta_probe_test.go). The CONSERVATIVE
		// end of the measured range is used deliberately: Theta appears
		// inside floor(ln(Theta)/ln(alpha)), so understating it drafts
		// deeper, and a too-deep draft on a low-acceptance stream is the
		// exact failure the adaptive controller exists to prevent. Taking
		// the shallow end means the win is under-claimed rather than the
		// regression risked.
		return 0.251
	default: // "", "cpu", and any backend with no measurement of its own
		return defaultTheta
	}
}

// verifyTheta returns the Theta for the path this model's speculative VERIFY
// actually runs on.
//
// The distinction that matters is resident vs staged, not which backend was
// requested. genNgramInto verifies through target.resident.ForwardN when
// residency built, and falls back to the CPU batched forwardN when it did not
// (spec_ngram.go) -- so a "webgpu-staged" or "metal-staged" model verifies on
// CPU and must get the CPU constant, not its GPU one. Keying this off
// Options.Backend alone would hand a declined-residency model the GPU value and
// silently mis-tune the one case where the decline is already costing the user
// the whole forward.
func (m *Model) verifyTheta() float64 {
	if m == nil || m.resident == nil {
		return defaultTheta // staged or CPU: the verify is the CPU batched ForwardN
	}
	if m.be == nil {
		return defaultTheta
	}
	// M-14: ASK THE RESIDENT WHETHER ITS ForwardN IS ACTUALLY BATCHED, rather than trusting the
	// backend name. thetaFor("cuda") returns 0.251 — a number measured on DENSE 0.5B/1.5B, where
	// the batched pass runs. But cuda's ForwardN falls back to one `step` per row for every
	// MoE / K=V / non-uniform / non-int4-or-int8 model (prefillStaticDecline), and a loop of
	// single-token forwards has Theta ≈ 1 by construction — which is exactly what Metal
	// measured (1.006-1.048) and why Metal's constant disables speculation.
	//
	// On a resident MoE (Qwen3-30B-A3B, GLM-4.5-Air — no sliding window, so specRollbackSafe
	// admits them) the controller was told 0.251 and drafted 8, costing nine sequential steps
	// per round for ~6.7 committed tokens at high acceptance, and worse below it.
	//
	// >= 1 disables speculation, which is the honest answer for a sequential verify: the same
	// conclusion Metal reached, reached the same way.
	// Scoped to residents that EXPLICITLY report their path (PrefillPathReporter), not to
	// Model.PrefillPath(): that helper answers false for any resident which is not a Prefiller
	// at all, which says nothing about ForwardN and would disable speculation on backends whose
	// batched verify is fine. Only a resident that says "my batched pass declined" gets the
	// override — today that is cudaResident, whose ForwardN falls back to the per-row loop under
	// exactly the predicate it reports (prefillStaticDecline).
	if pf, ok := m.resident.(Prefiller); ok {
		if rep, ok := pf.(PrefillPathReporter); ok {
			if batched, _ := rep.PrefillPath(); !batched {
				return sequentialVerifyTheta
			}
		}
	}
	return thetaFor(m.be.Name())
}

// sequentialVerifyTheta is Theta for a resident whose ForwardN is a loop of single-token
// forwards. Not measured per backend because it does not need to be: T(n) is n·T(1) by
// construction, so the ratio is 1 plus whatever per-row overhead the loop adds. Metal MEASURED
// exactly that shape (1.006-1.048 across four configurations, linear to n=16) and ships 1.02;
// the same number is used here for the same reason.
const sequentialVerifyTheta = 1.02
