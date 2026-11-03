// Package memgate is MM7's gate (docs/tasks/task-subsystems-2026-10.md): what a resident build leased on each plan line, against
// the plan's figure for that line. A lease over plan, a lease on an unknown line, or a lease open after Close is red. Each backend
// brings its rows and its allocator's named rounding; a known red lives in the backend's table of Gaps, which names the fix that
// removes it and may only shrink. It imports testing because tests are its only callers.
package memgate

import (
	"cmp"
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/mem"
)

// Allowance is one allocator's rounding: how much of one allocation's charge over its request it forgives. A line may exceed its
// plan figure only by the sum of that over its own leases, so an allocation that rounded nothing lends nothing to one that
// rounded past its limit, and an allowance cannot hide a line that was requested over plan.
type Allowance struct {
	Name    string
	Forgive func(requested, charged int64) int64
}

// PerLease is an allocator that rounds each allocation up by less than quantum.
func PerLease(name string, quantum int64) Allowance {
	return Allowance{Name: name, Forgive: func(requested, charged int64) int64 { return max(min(charged-requested, quantum), 0) }}
}

// PowerOfTwo is an allocator that rounds each allocation up to a power of two, so by less than its request.
func PowerOfTwo(name string) Allowance {
	return Allowance{Name: name, Forgive: func(requested, charged int64) int64 { return max(min(charged-requested, requested), 0) }}
}

// Snapshot is a ledger's open account at one named point in a row's life, and the facts a Gap's Bytes may read there.
type Snapshot struct {
	Label  string
	Usage  mem.Usage
	Leases []mem.OpenLease
	Facts  map[string]int64
}

// Snap reads l now.
func Snap(label string, l *mem.Ledger, facts map[string]int64) Snapshot {
	return Snapshot{Label: label, Usage: l.Usage(), Leases: l.OpenLeases(), Facts: facts}
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
	// Exact names lines whose plan figure must also not exceed what each Open snapshot holds, by more than ExactWithin: a line read
	// only one way passes a plan priced for the wrong precision.
	Exact       []mem.LineID
	ExactWithin int64
}

// Unpriced is one thing a backend knows it built without charging, on line.
type Unpriced struct {
	Line   mem.LineID
	What   string
	Excess int64
}

// Kind is what a finding is. Only an over-plan or an unpriced finding can be a known gap: the ledger's own invariants have none.
type Kind int

const (
	Invariant  Kind = iota // an unknown line, a line with no figure, a lease open after Close, a plan that declines the build
	OverPlan               // a line over its figure, past the allowance
	Built                  // something built without being charged (Reading.Unpriced)
	OverPriced             // a line of Reading.Exact holding less than its figure, past Reading.ExactWithin
)

// Finding is one way a row is red.
type Finding struct {
	Row, What string
	Kind      Kind
	Line      mem.LineID
	Excess    int64            // bytes over plan, past the allowance
	Facts     map[string]int64 // the snapshot's, for a Gap's Bytes
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
	add := func(kind Kind, line mem.LineID, excess int64, facts map[string]int64, format string, args ...any) {
		out = append(out, Finding{Row: r.Row, Kind: kind, Line: line, Excess: excess, Facts: facts, What: fmt.Sprintf(format, args...)})
	}
	if r.Plan.Placement == decoder.PlacementDecline {
		add(Invariant, "", 0, nil, "the build went resident, and the plan declines it: %s", r.Plan.Reason)
	}
	planned := Planned(r.Plan, r.KVSlots)
	for line, b := range r.Named {
		planned[line] = b
	}
	var reserved int
	for _, s := range r.Open {
		reserved = max(reserved, s.Usage.Reserved)
		forgiven := map[mem.LineID]int64{}
		for _, l := range s.Leases {
			forgiven[l.Line] += a.Forgive(l.Requested, l.Charged)
		}
		for _, line := range sortedLines(s.Usage.ByLine) {
			charged := s.Usage.ByLine[line]
			want, ok := planned[line]
			switch {
			case !line.Known():
				add(Invariant, line, charged, s.Facts, "%s: %d bytes leased on an unknown line", s.Label, charged)
				continue
			case !ok:
				add(Invariant, line, charged, s.Facts, "%s: %d bytes leased on a line with no figure in the plan or from the backend", s.Label, charged)
				continue
			}
			if excess := charged - want - forgiven[line]; excess > 0 {
				add(OverPlan, line, excess, s.Facts, "%s: leased %d bytes against a plan of %d: %d over, after %d forgiven by %q",
					s.Label, charged, want, excess, forgiven[line], a.Name)
			}
		}
	}
	for _, s := range r.Open {
		for _, line := range r.Exact {
			if excess := planned[line] - s.Usage.ByLine[line] - r.ExactWithin; excess > 0 {
				add(OverPriced, line, excess, s.Facts, "%s: leased %d bytes against a plan of %d: priced %d over, past the %d allowed",
					s.Label, s.Usage.ByLine[line], planned[line], excess, r.ExactWithin)
			}
		}
	}
	for _, u := range r.Unpriced {
		add(Built, u.Line, u.Excess, nil, "built without being charged: %s", u.What)
	}
	u := r.Closed
	if reserved > 0 && u.Reserved < reserved {
		add(Invariant, "", 0, nil, "the ledger was not read after Close: %d leases ever reserved there, %d while open", u.Reserved, reserved)
	}
	if u.PeakUnplannedCharged > 0 {
		add(Invariant, "", u.PeakUnplannedCharged, nil, "the ledger peaked at %d bytes on no known line", u.PeakUnplannedCharged)
	}
	if u.Leases > 0 {
		add(Invariant, "", u.Charged, nil, "%d leases (%d bytes) open after Close, by line %v", u.Leases, u.Charged, u.ByLine)
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

// Geometry is the figures a Gap's Bytes may read, each from a model's architecture and config alone, never from the plan or the ledger under
// test: a bound read from either would widen with the error it bounds.
type Geometry struct {
	// VectorUnit is one hidden-width f32 vector on each layer, and VectorBase one each for the final norm and its bias: the plan's dense figure,
	// a sum of weight matrices, counts neither. A fixture's gap names how many a layer it holds, so a family that holds more goes red.
	VectorUnit, VectorBase int64
	// DenseTables is what a resident build uploads on dense beside the vectors: a rotation table a layer and a few constants.
	DenseTables int64
	// KVFormula is the figure the plan's formula gives: K and V at ctx positions on each layer that owns a cache, 1/8 of a byte a scale at int8.
	KVFormula int64
	// KVHeld is what those layers hold: KVFormula less KVWindowOver and KVLatentOver, plus KVScaleUnder.
	KVHeld int64
	// KVWindowOver is what the formula prices past a sliding layer's window on a backend with rings (twice the window where an f32 ring keeps
	// its mirror), and KVLatentOver what it prices past an MLA layer's one latent row. KVScaleUnder is what one scale a head adds over the
	// formula's 1/8 of a byte an element at int8: negative on a wide head. They can net within one row, so each is its own figure.
	KVWindowOver, KVLatentOver, KVScaleUnder int64
	// MambaTail is what a Mamba-2 CPU window keeps past its rows: a view into the projection's output, whose last NHeads floats follow each row.
	MambaTail int64
}

// GeometryOf reads g for m at ctx positions and the KV precision kv, one slot's worth times slots. rings is whether the backend keeps a sliding
// layer's K/V in a ring.
func GeometryOf(m *decoder.Model, ctx, slots int, kv string, rings bool) Geometry {
	cfg := m.Config()
	_, layers, _, _, headDim, _, _ := m.Dims()
	hidden := int64(cfg.HiddenDim) * 4
	g := Geometry{VectorUnit: int64(cfg.NumLayers) * hidden, VectorBase: 2 * hidden, DenseTables: int64(layers) * int64(headDim*2+64)}
	elem := map[string]float64{mem.KVF16: 2, mem.KVI8: 1}[kv]
	if elem == 0 {
		elem = 4
	}
	formula := elem
	if kv == mem.KVI8 {
		formula = 1.125
	}
	var f, window, latent, scale float64
	for _, grp := range m.KVGeometry().Groups {
		for _, l := range grp.Layers {
			width := float64(grp.RowWidth)
			f += 2 * width * formula * float64(ctx)
			rows := float64(ctx)
			if rings && grp.Retention == decoder.KVRetainWindow && grp.Span < ctx {
				rows = float64(grp.Span)
				if kv == mem.KVF32 || kv == "" {
					rows *= 2
				}
			}
			window += 2 * width * formula * (float64(ctx) - rows)
			switch {
			case grp.Latent:
				latent += 2*width*formula*rows - rows*width*4
			case kv == mem.KVI8:
				scale += 2 * rows * (width*elem + 4*float64(m.KVHeadsAtResident(l)) - width*formula)
			}
		}
	}
	n := int64(max(slots, 1))
	g.KVFormula, g.KVWindowOver, g.KVLatentOver, g.KVScaleUnder = int64(f)*n, int64(window)*n, int64(latent)*n, int64(scale)*n
	g.KVHeld = g.KVFormula - g.KVWindowOver - g.KVLatentOver + g.KVScaleUnder
	if nh, _, _, _, k, _, _, ok := m.NemotronResidentParams(); ok {
		for l := range layers {
			if m.NemotronBlockKind(l) == 0 {
				g.MambaTail += int64((k-1)*nh) * 4
			}
		}
	}
	return g
}

// Facts is g as a Snapshot's facts.
func (g Geometry) Facts() map[string]int64 {
	return map[string]int64{"vectorUnit": g.VectorUnit, "vectorBase": g.VectorBase, "tables": g.DenseTables, "kvFormula": g.KVFormula, "kvHeld": g.KVHeld, "mambaTail": g.MambaTail}
}

// HoldRows fails t unless built is exactly the rows path names, one a line: a row that stopped building is as red as one over plan, and a new
// one is named in the change that adds it. With update it rewrites path instead, which only a full run of the rows may ask for.
func HoldRows(t testing.TB, path string, built []string, update bool) {
	t.Helper()
	got := slices.Sorted(slices.Values(built))
	if update {
		if err := os.WriteFile(path, []byte(strings.Join(got, "\n")+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	want := strings.Split(strings.TrimSpace(string(b)), "\n")
	for _, r := range want {
		if _, ok := slices.BinarySearch(got, r); !ok {
			t.Errorf("%s names %q, and it did not build", path, r)
		}
	}
	for _, r := range got {
		if !slices.Contains(want, r) {
			t.Errorf("%q built, and %s does not name it", r, path)
		}
	}
}

// KVPricedOver is a Gap's Bytes for a KV line the plan's formula prices over what the geometry holds: exactly that difference.
func KVPricedOver(f map[string]int64) int64 { return max(f["kvFormula"]-f["kvHeld"], 0) }

// KVPricedUnder is a Gap's Bytes for a KV line the plan's formula prices under what the geometry holds: exactly that difference.
func KVPricedUnder(f map[string]int64) int64 { return max(f["kvHeld"]-f["kvFormula"], 0) }

// Gap is a known red: the rows it holds on, the line, the fix that removes it, and how many bytes it accounts for. Gaps on one line
// add up; a finding past their sum stays red, so a gap cannot hide a second one on the same line.
type Gap struct {
	Row  string // a Reading.Row, where * matches any run of characters
	Line mem.LineID
	// Built is a gap on something built without being charged (Reading.Unpriced), and OverPriced one on a line of Reading.Exact priced
	// over what it holds; otherwise it is a line over its figure. None accounts for another.
	Built, OverPriced bool
	What              string // the gap, in a sentence
	Item              string // the MM8 item, or the row of docs/tasks/task-subsystems-2026-10.md, that removes it
	// Bytes is the most this gap accounts for at a snapshot, from that snapshot's facts. It must come from the build's shape, not from
	// the ledger it is checked against. A gap without one forgives nothing and fails Finish.
	Bytes func(facts map[string]int64) int64
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
		if gaps := g.match(f); gaps != nil {
			for _, i := range gaps {
				g.used[i] = true
				t.Logf("known gap (%s): %s", g.Gaps[i].Item, f)
			}
			continue
		}
		t.Errorf("%s", f)
	}
}

// Unknown is every finding on r the table does not account for: what Check would fail a test on.
func (g *Gate) Unknown(r Reading) []Finding {
	var out []Finding
	for _, f := range Check(r, g.Allowance) {
		if g.match(f) == nil {
			out = append(out, f)
		}
	}
	return out
}

// match is the gaps that together account for f, or nil if they do not. Gaps on one line add up, so each states only its own bytes,
// and fixing one leaves no slack for what another covers. It takes the most specific row patterns first, the smallest first among
// equals, so a wildcard's bound does not stand in for a row's own gap, and a gap is used only if f needed it: one the others cover
// without it is left out, the least specific first. So fixing one of two gaps on a row leaves the other's bytes alone, and the fixed
// one goes stale.
func (g *Gate) match(f Finding) []int {
	if f.Kind == Invariant {
		return nil
	}
	bytes := map[int]int64{}
	var fit []int
	for i, gap := range g.Gaps {
		if gap.Line != f.Line || gap.Built != (f.Kind == Built) || gap.OverPriced != (f.Kind == OverPriced) || gap.Bytes == nil || !globMatch(gap.Row, f.Row) {
			continue
		}
		if b := gap.Bytes(f.Facts); b > 0 {
			bytes[i] = b
			fit = append(fit, i)
		}
	}
	slices.SortStableFunc(fit, func(a, b int) int {
		return cmp.Or(cmp.Compare(len(g.Gaps[b].Row)-strings.Count(g.Gaps[b].Row, "*"), len(g.Gaps[a].Row)-strings.Count(g.Gaps[a].Row, "*")),
			cmp.Compare(bytes[a], bytes[b]))
	})
	var used []int
	var sum int64
	for _, i := range fit {
		used = append(used, i)
		if sum += bytes[i]; f.Excess <= sum {
			for k := len(used) - 1; k >= 0; k-- {
				if b := bytes[used[k]]; sum-b >= f.Excess {
					sum -= b
					used = slices.Delete(used, k, k+1)
				}
			}
			return used
		}
	}
	return nil
}

// globMatch reports whether s matches pattern, where * matches any run of characters, slashes included.
func globMatch(pattern, s string) bool {
	parts := strings.Split(pattern, "*")
	if len(parts) == 1 {
		return pattern == s
	}
	if !strings.HasPrefix(s, parts[0]) {
		return false
	}
	s = s[len(parts[0]):]
	for _, p := range parts[1 : len(parts)-1] {
		i := strings.Index(s, p)
		if i < 0 {
			return false
		}
		s = s[i+len(p):]
	}
	return strings.HasSuffix(s, parts[len(parts)-1])
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
		if gap.Bytes == nil {
			t.Errorf("known gap on line %q (%s) names no bytes: every gap is bounded", gap.Line, gap.Item)
		}
		if !g.used[i] {
			t.Errorf("known gap on line %q (%s, %s) no longer fires on any row: remove it from the table and lower the pin", gap.Line, gap.Item, gap.What)
		}
	}
}
