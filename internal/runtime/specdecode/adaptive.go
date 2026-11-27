package specdecode

import "math"

// AdaptiveDepth chooses the per-round draft depth from a running estimate of per-position acceptance.
type AdaptiveDepth struct {
	MaxDraft   int       // ceiling on tokens drafted per round (default 8)
	Theta      float64   // marginal cost of one extra verify node, in units of one single-token target step
	Cost       []float64 // optional measured cost curve: Cost[m] is time to verify m rows
	Lambda     float64   // EMA retention for the acceptance estimate (default 0.8)
	ProbeEvery int       // force D>=1 after this many idle (D=0) rounds (default 16; 64 with Cost curve)

	alpha  float64
	idle   int
	inited bool
}

// Ensure initializes default fields on a if not already initialized.
func EnsureAdaptive(a *AdaptiveDepth) { a.ensure() }

// SetAlpha overrides a's running acceptance estimate (used by unit tests).
func SetAlpha(a *AdaptiveDepth, alpha float64) { a.alpha = alpha }

// IdleCount returns a's consecutive idle-round count (used by unit tests).
func IdleCount(a *AdaptiveDepth) int { return a.idle }

func (a *AdaptiveDepth) ensure() {
	if a.inited {
		return
	}
	if a.MaxDraft <= 0 {
		a.MaxDraft = 8
	}
	if a.Theta <= 0 || math.IsNaN(a.Theta) {
		a.Theta = DefaultTheta
	}
	if a.Lambda <= 0 || a.Lambda >= 1 {
		a.Lambda = 0.8
	}
	if a.ProbeEvery <= 0 {
		a.ProbeEvery = 16
		if len(a.Cost) > 1 {
			a.ProbeEvery = 64
		}
	}
	a.alpha = 0.9
	a.inited = true
}

// Depth returns how many of the proposedLen drafted tokens to verify this round.
func (a *AdaptiveDepth) Depth(proposedLen int) int {
	a.ensure()
	if proposedLen <= 0 {
		return 0
	}
	maxD := min(proposedLen, a.MaxDraft)
	if a.Theta >= 1 {
		return 0
	}
	if a.idle >= a.ProbeEvery {
		return 1
	}
	if len(a.Cost) > 1 {
		best, bd, e, p := 1.0, 0, 1.0, 1.0
		for d := 1; d <= maxD && d+1 < len(a.Cost); d++ {
			p *= a.alpha
			e += p
			if r := e / a.Cost[d+1]; r > best {
				best, bd = r, d
			}
		}
		return bd
	}
	if a.alpha <= a.Theta {
		return 0
	}
	if a.alpha >= 0.999 {
		return maxD
	}
	d := int(math.Floor(math.Log(a.Theta) / math.Log(a.alpha)))
	if d < 0 {
		d = 0
	}
	return min(d, maxD)
}

// Observe folds this round's realized outcome into the acceptance estimate.
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

// VerifyCostReporter is an OPTIONAL ResidentForward extension reporting a measured cost curve.
type VerifyCostReporter interface {
	VerifyCost() []float64
}

// DefaultTheta is the fallback marginal verify-node cost (the CPU value).
const DefaultTheta = 0.5

// SequentialVerifyTheta is Theta for a resident whose ForwardN is a loop of single-token forwards.
const SequentialVerifyTheta = 1.02

// ThetaFor returns the measured marginal verify-node cost for a backend.
func ThetaFor(backend string) float64 {
	switch backend {
	case "metal":
		return 0.96
	case "cuda":
		return 0.251
	default:
		return DefaultTheta
	}
}
