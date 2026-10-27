package mem

import (
	"math/rand/v2"
	"sync"
	"testing"
)

// TestLedger_closesToZero (G-MM3a-3, I2 at tier 0): leases reserved and released in a random order close the ledger to zero,
// a second release changes nothing, and the peak, the lines and the tiers are as reserved.
func TestLedger_closesToZero(t *testing.T) {
	var l Ledger
	r := rand.New(rand.NewPCG(3, 9))
	lines := []LineID{LineDense, LineKV, LineScratch, LineExperts}
	var leases []*Lease
	var charged, peak int64
	for i := range 200 {
		n := r.Int64N(4<<20) + 1
		c := n + int64(i%3)
		leases = append(leases, l.Reserve([]Tier{VRAM, HostAnon}[i%2], lines[i%len(lines)], n, c))
		charged += c
		peak = max(peak, charged)
	}
	u := l.Usage()
	if u.Leases != 200 || u.Charged != charged || u.PeakCharged != peak || u.Unplanned != 0 {
		t.Fatalf("after 200 reserves: %+v, want 200 leases, %d charged", u, charged)
	}
	var byLine int64
	for _, v := range u.ByLine {
		byLine += v
	}
	if byLine != charged || u.ByTier[VRAM]+u.ByTier[HostAnon] != charged {
		t.Fatalf("by line %d, by tier %v, want %d", byLine, u.ByTier, charged)
	}
	r.Shuffle(len(leases), func(i, j int) { leases[i], leases[j] = leases[j], leases[i] })
	for _, ls := range leases {
		ls.Release()
		ls.Release()
	}
	var nilLease *Lease
	nilLease.Release()
	u = l.Usage()
	if u.Leases != 0 || u.Charged != 0 || u.Requested != 0 || len(u.ByLine) != 0 || len(u.ByTier) != 0 || u.PeakCharged != peak {
		t.Fatalf("after every release: %+v, want zero with the peak %d kept", u, peak)
	}
	if u.Reserved != 200 || u.Released != 200 {
		t.Fatalf("reserved %d released %d, want 200 each", u.Reserved, u.Released)
	}
}

// TestLedger_unplanned (G-MM3a-2's counter at tier 0): bytes on the empty line or an unknown one are unplanned until released;
// every named plan line is planned.
func TestLedger_unplanned(t *testing.T) {
	var l Ledger
	a := l.Reserve(VRAM, "", 10, 2<<20)
	b := l.Reserve(VRAM, "made-up", 5, 5)
	c := l.Reserve(VRAM, LineKV, 7, 7)
	if u := l.Usage(); u.Unplanned != 2<<20+5 {
		t.Fatalf("unplanned %d, want %d", u.Unplanned, 2<<20+5)
	}
	a.Release()
	b.Release()
	c.Release()
	if u := l.Usage(); u.Unplanned != 0 || u.PeakUnplannedCharged != 2<<20+5 {
		t.Fatalf("after release: unplanned %d (peak %d)", u.Unplanned, u.PeakUnplannedCharged)
	}
	for _, line := range []LineID{LineDense, LineExperts, LineKV, LineState, LineSlack, LineCompanion, LineScratch, LineStaging, LinePinnedExperts, LineTest} {
		if !line.Known() {
			t.Errorf("%q is a plan line", line)
		}
	}
}

// TestLedger_concurrent: reserves and releases from many goroutines close to zero (run under -race).
func TestLedger_concurrent(t *testing.T) {
	var l Ledger
	var wg sync.WaitGroup
	for g := range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range 500 {
				ls := l.Reserve(VRAM, LineScratch, int64(g+i), int64(g+i))
				_ = l.Usage()
				ls.Release()
			}
		}()
	}
	wg.Wait()
	if u := l.Usage(); u.Leases != 0 || u.Charged != 0 || u.Reserved != 16*500 {
		t.Fatalf("%+v", u)
	}
}
