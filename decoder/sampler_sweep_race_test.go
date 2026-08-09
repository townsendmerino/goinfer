//go:build race

package decoder

// Under -race the exactness sweep runs a STRIDED SUBSET of seeds rather than all of them.
//
// WHY. TestTopFilterLogits_MatchesReference is pure computation: no goroutines, no shared state,
// no channels. The race detector therefore has nothing to find in it, while costing ~34× (measured
// on this box: the sweep is 12.3 s sharded without -race; the decoder package went from a 600 s
// timeout panic to 414 s once sharded, and the sweep is the dominant term). On a 4-core CI runner —
// where the parallel sharding recovers much less than on a many-core box — the full sweep is
// plausibly 8–9 minutes of EVERY CI run, for a second execution of cases the non-race job already
// proved. Paying that indefinitely is the waste; the tax is not buying coverage.
//
// WHAT IS PRESERVED. This strides the SEED axis only. Every one of the 15 parameter configs and all
// 4 temperatures still run, for every seed selected — so the parameter space is spanned, not
// truncated to a prefix. Striding (rather than taking the first N) keeps the selected seeds spread
// across the whole range, so the tie-heavy and tie-free logit shapes (which alternate on seed
// parity) both stay represented.
//
// COVERAGE NOTE (corrected). An earlier version of this comment claimed the full sweep "still runs
// in the non-race job" — it did not. BOTH root CI jobs run `go test -race`, so gating on race mode
// alone would have meant the full 24,018-case sweep ran in NO CI job at all: a real weakening,
// asserted to be none. ci.yml now carries an explicit non-race step that runs the full sweep (and
// the throughput gate) on every push. If that step is ever removed, this gate silently shrinks
// again — TestSweepCoverage_fullSweepRunsSomewhere fails if the two drift apart.
// The stride is ODD on purpose. The sweep picks its logit shape by seed parity (`s%2 == 0` →
// tie-heavy, else tie-free), so an EVEN stride would select only even seeds and silently drop the
// tie-free shape entirely — a subset that no longer spans the space it claims to. 7 alternates
// parity on every step, keeping both shapes represented.
const (
	sweepSeedStride = 7
	// raceEnabled lets timing-based gates opt out under the detector, which distorts wall clock.
	raceEnabled = true
	sweepMode   = "strided subset (-race: the detector finds nothing in a pure-compute sweep)"
)
