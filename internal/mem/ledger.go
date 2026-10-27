package mem

import "sync"

// LineID names the plan line an allocation is leased against: a term of the DevicePlan, or a share of the margin.
type LineID string

// The plan lines, untyped so a backend substrate that names lines as strings takes them unconverted. An allocation on any other
// line, or on none, is unplanned.
const (
	LineDense         = "dense"          // the dense weights (DevicePlan.DenseBytes)
	LineExperts       = "experts"        // routed experts on the device (DevicePlan.ExpertBytesUsed)
	LineKV            = "kv"             // the KV cache (DevicePlan.KVBytes)
	LineState         = "state"          // recurrent and conv state, and their checkpoints
	LineSlack         = "slack"          // the allocator's rounding the plan prices (DeviceLoad.FixedSlack)
	LineCompanion     = "companion"      // a drafter's or a tower's bytes (PlanRequest.ExtraBytes)
	LineScratch       = "scratch"        // per-pass activations: the margin's share
	LineStaging       = "staging"        // a build's host staging (BuildDemand.HostPackBytes)
	LinePinnedExperts = "pinned-experts" // the page-locked host copy of the routed experts
	LineTest          = "test"           // a test's own buffers
)

// Known reports whether line is a plan line.
func (line LineID) Known() bool {
	switch line {
	case LineDense, LineExperts, LineKV, LineState, LineSlack, LineCompanion, LineScratch, LineStaging, LinePinnedExperts, LineTest:
		return true
	}
	return false
}

// Ledger is the runtime account of one allocator: every allocation is a lease naming its plan line. It records what it is
// told and reads nothing; nothing reads it to decide. Safe for concurrent use.
type Ledger struct {
	mu    sync.Mutex
	open  map[*Lease]struct{}
	usage Usage
}

// Lease is one allocation's reservation. Release is idempotent and safe on a nil Lease.
type Lease struct {
	l                  *Ledger
	tier               Tier
	line               LineID
	requested, charged int64
}

// Usage is a ledger's open leases: their count and bytes, by line and by tier. Charged is what the allocator takes
// (its rounding included), Requested what was asked for. Unplanned is the charged bytes on no known line.
type Usage struct {
	Leases               int
	Requested, Charged   int64
	PeakCharged          int64
	Unplanned            int64
	ByLine               map[LineID]int64 // charged
	ByTier               map[Tier]int64   // charged
	Reserved, Released   int              // leases ever reserved and released
	PeakUnplannedCharged int64
}

// Reserve leases requested bytes, charged as the allocator takes them, on tier and line.
func (l *Ledger) Reserve(tier Tier, line LineID, requested, charged int64) *Lease {
	ls := &Lease{l: l, tier: tier, line: line, requested: requested, charged: charged}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.open == nil {
		l.open = map[*Lease]struct{}{}
		l.usage.ByLine, l.usage.ByTier = map[LineID]int64{}, map[Tier]int64{}
	}
	l.open[ls] = struct{}{}
	u := &l.usage
	u.Leases++
	u.Reserved++
	u.Requested += requested
	u.Charged += charged
	u.ByLine[line] += charged
	u.ByTier[tier] += charged
	if !line.Known() {
		u.Unplanned += charged
		u.PeakUnplannedCharged = max(u.PeakUnplannedCharged, u.Unplanned)
	}
	u.PeakCharged = max(u.PeakCharged, u.Charged)
	return ls
}

// Release returns the lease's bytes to the ledger. A second call, or a call on nil, does nothing.
func (ls *Lease) Release() {
	if ls == nil || ls.l == nil {
		return
	}
	l := ls.l
	l.mu.Lock()
	defer l.mu.Unlock()
	if _, ok := l.open[ls]; !ok {
		return
	}
	delete(l.open, ls)
	u := &l.usage
	u.Leases--
	u.Released++
	u.Requested -= ls.requested
	u.Charged -= ls.charged
	if u.ByLine[ls.line] -= ls.charged; u.ByLine[ls.line] == 0 {
		delete(u.ByLine, ls.line)
	}
	if u.ByTier[ls.tier] -= ls.charged; u.ByTier[ls.tier] == 0 {
		delete(u.ByTier, ls.tier)
	}
	if !ls.line.Known() {
		u.Unplanned -= ls.charged
	}
}

// Line is the lease's plan line.
func (ls *Lease) Line() LineID { return ls.line }

// Usage is a copy of the ledger's account now.
func (l *Ledger) Usage() Usage {
	l.mu.Lock()
	defer l.mu.Unlock()
	u := l.usage
	u.ByLine, u.ByTier = make(map[LineID]int64, len(l.usage.ByLine)), make(map[Tier]int64, len(l.usage.ByTier))
	for k, v := range l.usage.ByLine {
		u.ByLine[k] = v
	}
	for k, v := range l.usage.ByTier {
		u.ByTier[k] = v
	}
	return u
}
