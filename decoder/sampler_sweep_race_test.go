//go:build race

package decoder

// Under -race the exactness sweep runs a STRIDED SUBSET of seeds rather than all of them. TestTopFilterLogits_MatchesReference
// is pure computation (no goroutines, shared state or channels), so the detector has nothing to find in it while costing
// a large factor; the full sweep is a second execution of cases the non-race run already proves.
//
// This strides the SEED axis only: all 15 parameter configs and all 4 temperatures still run for every selected seed,
// and striding rather than taking the first N keeps the seeds spread across the range.
//
// Both root CI jobs run `go test -race`, so the full 24,018-case sweep runs only because ci.yml carries an explicit
// non-race step for it (with the throughput gate). TestSweepCoverage_fullSweepRunsSomewhere fails if that step is removed.
//
// The stride is ODD on purpose: the sweep picks its logit shape by seed parity (`s%2 == 0` → tie-heavy, else tie-free),
// so an EVEN stride would select only even seeds and silently drop the tie-free shape. 7 alternates parity on every step.
const (
	sweepSeedStride = 7
	// raceEnabled lets timing-based gates opt out under the detector, which distorts wall clock.
	raceEnabled = true
	sweepMode   = "strided subset (-race: the detector finds nothing in a pure-compute sweep)"
)
