package decoder

import "github.com/townsendmerino/goinfer/internal/cost"

// residentCostFor is the resident's CostAvailable for a generation that counts into c, nil when it does not count (c nil, a
// shared resident, or a resident without the interface).
func (m *Model) residentCostFor(c *cost.Counter, shared bool) ResidentRuntimeSubsystem {
	if c == nil || shared {
		return nil
	}
	rt := m.resRuntime()
	if !rt.CostAvailable() {
		return nil
	}
	return rt
}

// prefillComputed is how many prompt tokens a resident prefill from position from computes: residentPrefillSeedMarked's own
// clamp, which recomputes the whole prompt when from is out of range.
func prefillComputed(n, from int) int {
	if from < 0 || from >= n {
		return n
	}
	return n - from
}

// costSpan runs work in rc's span when rc is set, else runs it plainly.
func costSpan(rc ResidentRuntimeSubsystem, c *cost.Counter, p cost.Phase, work func() (int, error)) error {
	if rc == nil {
		_, err := work()
		return err
	}
	return rc.CostSpan(c, p, work)
}
