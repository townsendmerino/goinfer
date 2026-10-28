//go:build cuda

package cuda

import (
	"time"

	"github.com/townsendmerino/goinfer/internal/cost"
)

// The resident's cost counters (SB11, internal/cost). Until OP4's walker names the operation a kernel belongs to, a CUDA
// operation is a kernel, named by its field (pipeName, cached per kernel): each r.launch is one call and one launch of it. A
// counted token adds its host waits (sync), its new leases on the device's ledger (alloc), the launches not made through
// r.launch (launch_other), and its wall time (the phase's WallNs). The per-kernel and token-level launch counts are checked
// against the device's own (gpudev's Device.Activity). A kernel's bytes and time are not counted: a launch is asynchronous.

var kernelUnmeasured = []string{cost.FieldBytesRead, cost.FieldBytesWritten, cost.FieldTime}

// noLedgerUnmeasured is the alloc operation's, on a device without a ledger.
var noLedgerUnmeasured = []string{cost.FieldAllocBytes, cost.FieldAllocs, cost.FieldBytesRead, cost.FieldBytesWritten, cost.FieldTime}

// costPlantForTest plants SB11's CUDA defects: 1 launches one kernel a token twice, 2 adds a host wait after the token.
var costPlantForTest int

// countLaunch records one launch of kernel f. Executor goroutine only, while a token is counted.
func (r *cudaResident) countLaunch(f Pipeline) {
	name, ok := r.costNames[f]
	if !ok {
		name = r.pipeName(f)
		if r.costNames == nil {
			r.costNames = map[Pipeline]string{}
		}
		r.costNames[f] = name
	}
	r.costNamed++
	r.cost.Add(r.costPhase, name, cost.Counts{Calls: 1, Launches: 1, Unmeasured: kernelUnmeasured})
}

// costToken runs one token's work with c counting into phase p, and adds the token's device-level counts. The executor reads
// r.cost inside the token's jobs: the job's channel send orders this write before it.
func (r *cudaResident) costToken(c *cost.Counter, p cost.Phase, token func() error) error {
	r.cost, r.costPhase = c, p
	defer func() { r.cost = nil }()
	l0, w0 := r.dev.Activity()
	named0 := r.costNamed
	n0, b0, leased := r.ledgerTotals()
	t0 := time.Now()
	err := token()
	if costPlantForTest == 2 {
		_ = r.do(func() error { return r.stream.Sync() })
	}
	wall := time.Since(t0)
	l1, w1 := r.dev.Activity()
	n1, b1, _ := r.ledgerTotals()
	c.Tokens(p, 1)
	c.Wall(p, int64(wall))
	c.Add(p, "sync", cost.Counts{Syncs: w1 - w0, Unmeasured: kernelUnmeasured})
	alloc := cost.Counts{Allocs: n1 - n0, AllocBytes: b1 - b0, Unmeasured: kernelUnmeasured}
	if !leased {
		alloc.Unmeasured = noLedgerUnmeasured
	}
	c.Add(p, "alloc", alloc)
	if other := (l1 - l0) - (r.costNamed - named0); other != 0 {
		c.Add(p, "launch_other", cost.Counts{Calls: other, Launches: other, Unmeasured: kernelUnmeasured})
	}
	return err
}

// ledgerTotals is how many leases the device's ledger has ever reserved and the bytes they asked for; ok is false without one.
func (r *cudaResident) ledgerTotals() (leases, bytes int64, ok bool) {
	l, ok := r.dev.Leaser().(*deviceLedger)
	if !ok {
		return 0, 0, false
	}
	u := l.Usage()
	return int64(u.Reserved), u.ReservedRequested, true
}
