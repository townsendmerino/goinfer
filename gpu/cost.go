//go:build gpu

package gpu

import (
	"time"

	"github.com/townsendmerino/goinfer/gpu/internal/kernels"
	"github.com/townsendmerino/goinfer/internal/cost"
)

// costPlantForTest plants SB11's extra-sync defect: 2 adds a blocking poll after the token.
var costPlantForTest int

// costToken runs one token's work with c counting into phase p (kernels' cost.go attributes the dispatches), and adds the token's
// Context-level counts: its blocking polls (sync), its command buffers (submit), its new leases on the Context's ledger (alloc),
// any recorded dispatch the runner did not attribute (dispatch_other) and its wall time (the phase's WallNs).
func (rd *residentDecoder) costToken(c *cost.Counter, p cost.Phase, token func() error) error {
	r := rd.runner
	r.Cost, r.CostPhase = c, p
	defer func() { r.Cost = nil }()
	d0, s0, w0 := rd.c.Activity()
	u0 := rd.c.Ledger().Usage()
	attributed0 := phaseLaunches(c, p)
	t0 := time.Now()
	err := token()
	if costPlantForTest == 2 {
		rd.c.Wait()
	}
	wall := time.Since(t0)
	d1, s1, w1 := rd.c.Activity()
	u1 := rd.c.Ledger().Usage()
	attributed := phaseLaunches(c, p) - attributed0
	c.Tokens(p, 1)
	c.Wall(p, int64(wall))
	c.Add(p, "sync", cost.Counts{Syncs: w1 - w0, Unmeasured: kernels.DispatchUnmeasured})
	c.Add(p, "submit", cost.Counts{Submits: s1 - s0, Unmeasured: kernels.DispatchUnmeasured})
	n, b := int64(u1.Reserved-u0.Reserved), u1.ReservedRequested-u0.ReservedRequested
	c.Add(p, "alloc", cost.Counts{Allocs: n, AllocBytes: b, Unmeasured: kernels.DispatchUnmeasured})
	if other := (d1 - d0) - attributed; other != 0 {
		c.Add(p, "dispatch_other", cost.Counts{Calls: other, Launches: other, Unmeasured: kernels.DispatchUnmeasured})
	}
	return err
}

// phaseLaunches is the launches counted in phase p so far.
func phaseLaunches(c *cost.Counter, p cost.Phase) int64 {
	r := c.Report()
	if p == cost.Prefill && r.Prefill != nil {
		return r.Prefill.Totals.Launches
	}
	if p == cost.Decode && r.Decode != nil {
		return r.Decode.Totals.Launches
	}
	return 0
}
