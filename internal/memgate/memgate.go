// Package memgate is MM7's gate (docs/tasks/task-subsystems-2026-10.md): what a resident build leased on each plan line, against
// the plan's figure for that line. A lease over plan, a lease on an unknown line, or a lease open after Close is red. Each backend
// brings its rows and its allocator's named rounding; a known red lives in the backend's table of Gaps, which names the fix that
// removes it and may only shrink. It imports testing because tests are its only callers.
package memgate

import (
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/mem"
)

// Allowance is one allocator's rounding. A line may exceed its plan figure only by what the allocator actually rounded on it,
// and only up to Limit, so a generous allowance cannot hide a line that was requested over plan.
type Allowance struct {
	Name  string
	Limit func(leases int, requested int64) int64
}

// PerLease is an allocator that rounds each allocation up by less than quantum.
func PerLease(name string, quantum int64) Allowance {
	return Allowance{Name: name, Limit: func(leases int, _ int64) int64 { return int64(leases) * quantum }}
}

// PowerOfTwo is an allocator that rounds each allocation up to a power of two, so by less than the request.
func PowerOfTwo(name string) Allowance {
	return Allowance{Name: name, Limit: func(_ int, requested int64) int64 { return requested }}
}

// Snapshot is the ledger's open account at one named point in a row's life.
type Snapshot struct {
	Label string
	Usage mem.Usage
}

// Reading is one row: a build's ledger, and the plan for the same load.
type Reading struct {
	Row     string // fixture, quant, options: what Gap.Row matches
	Plan    decoder.Plan
	KVSlots int // the slots the build holds: the plan's KV and state lines are one slot's
	// Open is the account while the model is loaded: after the build, and after whatever the row ran.
	Open []Snapshot
	// Closed is the account after the model closed.
	Closed mem.Usage
	// Named is a figure for a line the plan has none for (the margin's share a backend gives scratch, a build's staging). A line
	// with neither a plan figure nor a named one is red.
	Named map[mem.LineID]int64
	// Unpriced names what a build holds that no line's total can show: a buffer set built without being charged.
	Unpriced []Unpriced
	// Facts is what a Gap's Bytes reads from the build, by a name the backend's table chooses.
	Facts map[string]int64
}

// Unpriced is one thing a backend knows it built without charging, on line.
type Unpriced struct {
	Line   mem.LineID
	What   string
	Excess int64
}

// Finding is one way a row is red.
type Finding struct {
	Row, What string
	Line      mem.LineID
	Excess    int64 // bytes over plan, past the allowance
}

func (f Finding) String() string {
	return fmt.Sprintf("%s: line %q: %s", f.Row, f.Line, f.What)
}

// Planned is the plan's figure for each line it prices, with this build's KV slots: KV and recurrent state are per slot.
func Planned(p decoder.Plan, kvSlots int) map[mem.LineID]int64 {
	n := int64(max(kvSlots, 1))
	return map[mem.LineID]int64{
		mem.LineDense:     p.DenseBytes,
		mem.LineExperts:   p.ExpertBytesUsed,
		mem.LineKV:        p.KVBytes * n,
		mem.LineState:     p.StateBytes * n,
		mem.LineSlack:     p.AllocSlackBytes,
		mem.LineCompanion: p.ExtraBytes,
	}
}

// Check is every way r is red under a.
func Check(r Reading, a Allowance) []Finding {
	var out []Finding
	add := func(line mem.LineID, excess int64, format string, args ...any) {
		out = append(out, Finding{Row: r.Row, Line: line, Excess: excess, What: fmt.Sprintf(format, args...)})
	}
	if r.Plan.Placement == decoder.PlacementDecline {
		add("", 0, "the build went resident, and the plan declines it: %s", r.Plan.Reason)
	}
	planned := Planned(r.Plan, r.KVSlots)
	for line, b := range r.Named {
		planned[line] = b
	}
	for _, s := range r.Open {
		for _, line := range sortedLines(s.Usage.ByLine) {
			charged := s.Usage.ByLine[line]
			want, ok := planned[line]
			switch {
			case !line.Known():
				add(line, charged, "%s: %d bytes leased on an unknown line", s.Label, charged)
				continue
			case !ok:
				add(line, charged, "%s: %d bytes leased on a line with no figure in the plan or from the backend", s.Label, charged)
				continue
			}
			requested, leases := s.Usage.RequestedByLine[line], s.Usage.LeasesByLine[line]
			rounded := charged - requested
			forgiven := min(max(rounded, 0), a.Limit(leases, requested))
			if excess := charged - want - forgiven; excess > 0 {
				add(line, excess, "%s: leased %d bytes over %d leases against a plan of %d: %d over, after %d of the %d rounded forgiven by %q",
					s.Label, charged, leases, want, excess, forgiven, rounded, a.Name)
			}
		}
	}
	for _, u := range r.Unpriced {
		add(u.Line, u.Excess, "built without being charged: %s", u.What)
	}
	if u := r.Closed; u.PeakUnplannedCharged > 0 {
		add("", u.PeakUnplannedCharged, "the ledger peaked at %d bytes on no known line", u.PeakUnplannedCharged)
	}
	if u := r.Closed; u.Leases > 0 {
		add("", u.Charged, "%d leases (%d bytes) open after Close, by line %v", u.Leases, u.Charged, u.ByLine)
	}
	return out
}

func sortedLines(m map[mem.LineID]int64) []mem.LineID {
	lines := make([]mem.LineID, 0, len(m))
	for l := range m {
		lines = append(lines, l)
	}
	slices.Sort(lines)
	return lines
}

// Gap is a known red: the rows it holds on, the line, the fix that removes it, and how many bytes it accounts for. A finding
// outside the bytes it names stays red, so a gap cannot hide a second one on the same line.
type Gap struct {
	Row  string // a Reading.Row, or a prefix of it ending in *
	Line mem.LineID
	What string // the gap, in a sentence
	Item string // the MM8 item, or the row of docs/tasks/task-subsystems-2026-10.md, that removes it
	// Bytes is the most this gap accounts for on a row; nil is any amount, for a finding that is no amount of a line.
	Bytes func(Reading) int64
}

// Gate runs a backend's rows against its table of known gaps.
type Gate struct {
	Allowance Allowance
	Gaps      []Gap
	// GapsPin is len(Gaps), written beside the table: the table may only shrink, and a fix lowers the pin with the row it removes.
	GapsPin int
	used    map[int]bool
	rows    int
}

// Check holds one row, naming each finding a Gap accounts for and failing t on every other.
func (g *Gate) Check(t testing.TB, r Reading) {
	t.Helper()
	if g.used == nil {
		g.used = map[int]bool{}
	}
	g.rows++
	for _, f := range Check(r, g.Allowance) {
		if i := g.match(f, r); i >= 0 {
			g.used[i] = true
			t.Logf("known gap (%s): %s", g.Gaps[i].Item, f)
			continue
		}
		t.Errorf("%s", f)
	}
}

func (g *Gate) match(f Finding, r Reading) int {
	for i, gap := range g.Gaps {
		if gap.Line != f.Line {
			continue
		}
		if p, ok := strings.CutSuffix(gap.Row, "*"); ok && !strings.HasPrefix(f.Row, p) || !ok && gap.Row != f.Row {
			continue
		}
		if gap.Bytes == nil || f.Excess <= gap.Bytes(r) {
			return i
		}
	}
	return -1
}

// Finish holds the table itself: every gap must still fire, and the table must be the pinned size. Call it once, after the last
// row, and only on a run of every row: a filtered run would see gaps go quiet.
func (g *Gate) Finish(t testing.TB) {
	t.Helper()
	if g.rows == 0 {
		t.Fatal("no row ran: the gate checked nothing")
	}
	if len(g.Gaps) != g.GapsPin {
		t.Errorf("%d known gaps against a pin of %d: the table may only shrink, and a fix lowers the pin with the row it removes", len(g.Gaps), g.GapsPin)
	}
	for i, gap := range g.Gaps {
		if !g.used[i] {
			t.Errorf("known gap on line %q (%s, %s) no longer fires on any row: remove it from the table and lower the pin", gap.Line, gap.Item, gap.What)
		}
	}
}
