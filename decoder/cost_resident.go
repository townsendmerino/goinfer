package decoder

import "github.com/townsendmerino/goinfer/internal/cost"

// ResidentCost is a resident that counts SB11's device costs (internal/cost) over a span of its work: CostSpan runs work with c
// counting into phase p, adds the span's device-level counts and wall time, and counts the tokens work reports it covered.
// Optional. A generation that asks for its cost on a resident without it, or on one shared with other generations (MC3), gets
// no phase from the resident (docs/tasks/task-subsystems-2026-10.md §8, "SB11b").
type ResidentCost interface {
	CostSpan(c *cost.Counter, p cost.Phase, work func() (tokens int, err error)) error
}

// residentCostFor is the resident's ResidentCost for a generation that counts into c, nil when it does not count (c nil, a
// shared resident, or a resident without the interface).
func (m *Model) residentCostFor(c *cost.Counter, shared bool) ResidentCost {
	if c == nil || shared {
		return nil
	}
	rc, _ := m.resident.(ResidentCost)
	return rc
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
func costSpan(rc ResidentCost, c *cost.Counter, p cost.Phase, work func() (int, error)) error {
	if rc == nil {
		_, err := work()
		return err
	}
	return rc.CostSpan(c, p, work)
}
