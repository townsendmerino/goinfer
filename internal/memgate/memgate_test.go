package memgate

import (
	"fmt"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/mem"
)

var rounding = PerLease("a 16 KiB test allocator", 16<<10)

// usage is a snapshot of a ledger after leases of {line, requested, charged}.
func usage(leases ...[3]any) Snapshot {
	var l mem.Ledger
	for _, x := range leases {
		l.Reserve(mem.Unified, mem.LineID(x[0].(string)), int64(x[1].(int)), int64(x[2].(int)))
	}
	return Snap("built", &l, nil)
}

// reading is a row of that one snapshot, closed cleanly.
func reading(open Snapshot) Reading {
	return Reading{Row: "fx int4", KVSlots: 1, Plan: decoder.Plan{DenseBytes: 1000, KVBytes: 500, StateBytes: 100},
		Open: []Snapshot{open}, Closed: mem.Usage{Reserved: open.Usage.Reserved}}
}

func lines(fs []Finding) string { return fmt.Sprint(fs) }

// TestCheck_lines holds what makes a row red, one way each. The over-plan cases are planted against a plan that the first case
// shows green.
func TestCheck_lines(t *testing.T) {
	for _, c := range []struct {
		name string
		r    func() Reading
		want string // in the finding, "" for green
	}{
		{"every line at its figure", func() Reading {
			return reading(usage([3]any{mem.LineDense, 1000, 1000}, [3]any{mem.LineKV, 500, 500}, [3]any{mem.LineState, 100, 100}))
		}, ""},
		{"a line under its figure", func() Reading { return reading(usage([3]any{mem.LineDense, 10, 10})) }, ""},
		{"one byte requested over plan", func() Reading { return reading(usage([3]any{mem.LineDense, 1001, 1001})) }, "1 over"},
		{"rounding forgiven up to what was rounded", func() Reading { return reading(usage([3]any{mem.LineDense, 1000, 1200})) }, ""},
		// The allowance would cover 1,001 bytes, but nothing was rounded: it must not hide a request over plan.
		{"an allowance does not cover a request over plan", func() Reading { return reading(usage([3]any{mem.LineState, 101, 101})) }, "1 over"},
		{"rounding past the allocator's limit", func() Reading {
			return reading(usage([3]any{mem.LineDense, 1000, 1000 + 16<<10 + 1}))
		}, "1 over"},
		{"KV is per slot", func() Reading {
			r := reading(usage([3]any{mem.LineKV, 1000, 1000}))
			r.KVSlots = 2
			return r
		}, ""},
		{"KV past its slots", func() Reading { return reading(usage([3]any{mem.LineKV, 1000, 1000})) }, "500 over"},
		{"state is per slot", func() Reading {
			r := reading(usage([3]any{mem.LineState, 300, 300}))
			r.KVSlots = 2
			return r
		}, "100 over"},
		{"an unknown line", func() Reading { return reading(usage([3]any{"not-a-line", 1, 1})) }, "unknown line"},
		{"a line with no figure", func() Reading { return reading(usage([3]any{mem.LineScratch, 1, 1})) }, "no figure"},
		{"a named figure", func() Reading {
			r := reading(usage([3]any{mem.LineScratch, 7, 7}))
			r.Named = map[mem.LineID]int64{mem.LineScratch: 7}
			return r
		}, ""},
		{"over a named figure", func() Reading {
			r := reading(usage([3]any{mem.LineScratch, 8, 8}))
			r.Named = map[mem.LineID]int64{mem.LineScratch: 7}
			return r
		}, "1 over"},
		{"open after Close", func() Reading {
			r := reading(usage())
			r.Closed = usage([3]any{mem.LineDense, 5, 5}).Usage
			return r
		}, "open after Close"},
		{"unplanned at any point", func() Reading {
			var l mem.Ledger
			l.Reserve(mem.Unified, "not-a-line", 9, 9).Release()
			r := reading(usage())
			r.Closed = l.Usage()
			return r
		}, "no known line"},
		{"the plan declines what was built", func() Reading {
			r := reading(usage())
			r.Plan.Placement, r.Plan.Reason = decoder.PlacementDecline, "no room"
			return r
		}, "no room"},
		{"something built without being charged", func() Reading {
			r := reading(usage())
			r.Unpriced = []Unpriced{{mem.LineScratch, "a step's scratch", 42}}
			return r
		}, "a step's scratch"},
		{"a later snapshot over plan", func() Reading {
			r := reading(usage())
			s := usage([3]any{mem.LineDense, 1001, 1001})
			s.Label = "generated"
			r.Open = append(r.Open, s)
			r.Closed.Reserved = 1
			return r
		}, "generated: leased"},
	} {
		t.Run(c.name, func(t *testing.T) {
			got := Check(c.r(), rounding)
			if c.want == "" {
				if len(got) != 0 {
					t.Errorf("want green, got %s", lines(got))
				}
				return
			}
			if len(got) != 1 || !strings.Contains(got[0].String(), c.want) {
				t.Errorf("want one finding naming %q, got %s", c.want, lines(got))
			}
		})
	}
}

// TestCheck_roundingIsPerLease: one lease's unused rounding is not another's. Two leases that rounded nothing lend no quota to a
// third charged 20,000 bytes over its request, whose own allowance is 16 KiB.
func TestCheck_roundingIsPerLease(t *testing.T) {
	r := reading(usage([3]any{mem.LineDense, 100, 100}, [3]any{mem.LineDense, 100, 100}, [3]any{mem.LineDense, 800, 800 + 20000}))
	got := Check(r, rounding)
	if len(got) != 1 || got[0].Excess != 20000-16<<10 {
		t.Errorf("want one finding of %d bytes over, got %s", 20000-16<<10, lines(got))
	}
}

// TestCheck_closedMustBeCaptured: a row that leased and never read the ledger after Close is red, rather than passing on a zero
// account.
func TestCheck_closedMustBeCaptured(t *testing.T) {
	r := reading(usage([3]any{mem.LineDense, 10, 10}))
	r.Closed = mem.Usage{}
	got := Check(r, rounding)
	if len(got) != 1 || !strings.Contains(got[0].String(), "after Close") {
		t.Errorf("want one finding that the account after Close was not read, got %s", lines(got))
	}
}

// TestGate_gapWithoutABoundForgivesNothing: a gap with no byte bound would silence its whole line, so it matches nothing and the
// table is red.
func TestGate_gapWithoutABoundForgivesNothing(t *testing.T) {
	r := reading(usage([3]any{mem.LineScratch, 8, 8}))
	r.Named = map[mem.LineID]int64{mem.LineScratch: 7}
	g := Gate{Allowance: rounding, Gaps: []Gap{{Row: "*", Line: mem.LineScratch, Item: "x"}}, GapsPin: 1}
	if len(g.Unknown(r)) == 0 {
		t.Error("a scratch line over its figure passed on a gap with no bound")
	}
}

// TestGlobMatch holds the table's row patterns.
func TestGlobMatch(t *testing.T) {
	for _, c := range []struct {
		p, s string
		want bool
	}{{"*", "testdata/x int4/slots2", true}, {"testdata/x *", "testdata/x int4/slots2", true}, {"testdata/x *", "testdata/y int4", false},
		{"* int4/slots2*", "testdata/x int4/slots2/blocks", true}, {"* int4/slots2*", "testdata/x int4", false},
		{"exact", "exact", true}, {"exact", "exact2", false}} {
		if got := globMatch(c.p, c.s); got != c.want {
			t.Errorf("globMatch(%q, %q) = %v, want %v", c.p, c.s, got, c.want)
		}
	}
}

// TestPowerOfTwo: the rounding it forgives is bounded by the request, so a block twice the request passes and a larger one does not.
func TestPowerOfTwo(t *testing.T) {
	a := PowerOfTwo("a power-of-two test allocator")
	if got := Check(reading(usage([3]any{mem.LineDense, 1000, 2000})), a); len(got) != 0 {
		t.Errorf("a block double the request: %s", lines(got))
	}
	if got := Check(reading(usage([3]any{mem.LineDense, 1000, 2001})), a); len(got) != 1 {
		t.Errorf("a block past double the request: %s", lines(got))
	}
}

// recorder is a testing.TB that records failures instead of failing, so the gate's own reds can be asserted.
type recorder struct {
	testing.TB
	errs []string
}

func (r *recorder) Helper()                   {}
func (r *recorder) Logf(string, ...any)       {}
func (r *recorder) Errorf(f string, a ...any) { r.errs = append(r.errs, fmt.Sprintf(f, a...)) }
func (r *recorder) Fatal(a ...any)            { r.errs = append(r.errs, fmt.Sprint(a...)) }

// TestGate_table holds the skip table's rules: a known gap passes only on its row, its line and within its bytes; a gap that no
// longer fires is red; a table that is not the pinned size is red.
func TestGate_table(t *testing.T) {
	over := func() Reading { return reading(usage([3]any{mem.LineState, 150, 150})) } // 50 over
	fifty := func(map[string]int64) int64 { return 50 }
	gap := Gap{Row: "fx *", Line: mem.LineState, What: "a test gap", Item: "MM8 test", Bytes: fifty}
	for _, c := range []struct {
		name string
		gaps []Gap
		pin  int
		rows func() []Reading
		want string // in the errors, "" for green
	}{
		{"a known gap passes", []Gap{gap}, 1, func() []Reading { return []Reading{over()} }, ""},
		{"no table, red", nil, 0, func() []Reading { return []Reading{over()} }, "50 over"},
		{"another line stays red", []Gap{gap}, 1,
			func() []Reading { return []Reading{over(), reading(usage([3]any{mem.LineKV, 501, 501}))} }, "1 over"},
		{"another row stays red", []Gap{{Row: "other *", Line: gap.Line, Item: "x", Bytes: fifty}}, 1,
			func() []Reading { return []Reading{over()} }, "50 over"},
		{"past the gap's bytes stays red", []Gap{{Row: gap.Row, Line: gap.Line, Item: "x", Bytes: func(map[string]int64) int64 { return 49 }}}, 1,
			func() []Reading { return []Reading{over()} }, "50 over"},
		{"a gap that no longer fires", []Gap{gap}, 1, func() []Reading { return []Reading{reading(usage())} }, "no longer fires"},
		{"the table above its pin", []Gap{gap}, 0, func() []Reading { return []Reading{over()} }, "may only shrink"},
		{"the table below its pin", []Gap{gap}, 2, func() []Reading { return []Reading{over()} }, "may only shrink"},
		{"a gap with no bound", []Gap{{Row: "fx *", Line: mem.LineState, Item: "x"}}, 1, func() []Reading { return []Reading{over()} }, "every gap is bounded"},
		{"an unpriced gap does not cover a line over plan", []Gap{{Row: "fx *", Line: mem.LineState, Built: true, Item: "x", Bytes: fifty}}, 1,
			func() []Reading { return []Reading{over()} }, "50 over"},
		{"a line's gap does not cover something unpriced", []Gap{gap}, 1, func() []Reading {
			r := over()
			r.Unpriced = []Unpriced{{mem.LineState, "a buffer", 1}}
			return []Reading{r}
		}, "built without being charged"},
		{"an invariant is never a known gap", []Gap{gap, {Row: "*", Line: "", Item: "x", Bytes: func(map[string]int64) int64 { return 1 << 40 }}}, 2, func() []Reading {
			r := over()
			r.Closed = usage([3]any{mem.LineDense, 5, 5}).Usage
			return []Reading{r}
		}, "open after Close"},
	} {
		t.Run(c.name, func(t *testing.T) {
			rec := &recorder{TB: t}
			g := Gate{Allowance: rounding, Gaps: c.gaps, GapsPin: c.pin}
			for _, r := range c.rows() {
				g.Check(rec, r)
			}
			g.Finish(rec)
			got := strings.Join(rec.errs, "\n")
			if c.want == "" && got != "" {
				t.Errorf("want green, got %s", got)
			}
			if c.want != "" && !strings.Contains(got, c.want) {
				t.Errorf("want an error naming %q, got %q", c.want, got)
			}
		})
	}
}

// TestGate_additive: with Additive, two gaps on one row and line account for a finding only up to their sum, and both fire; without it, neither
// alone accounts for what only their sum holds.
func TestGate_additive(t *testing.T) {
	bytes := func(n int64) func(map[string]int64) int64 { return func(map[string]int64) int64 { return n } }
	gaps := []Gap{
		{Row: "r", Line: mem.LineKV, What: "a", Item: "A", Bytes: bytes(10)},
		{Row: "r", Line: mem.LineKV, What: "b", Item: "B", Bytes: bytes(20)},
	}
	over := func(n int64) Finding { return Finding{Row: "r", Kind: OverPlan, Line: mem.LineKV, Excess: n} }
	sum := Gate{Gaps: gaps, Additive: true}
	if got := sum.match(over(30)); len(got) != 2 {
		t.Errorf("30 bytes under an additive 10 and 20: matched %v, want both", got)
	}
	if got := sum.match(over(31)); len(got) != 0 {
		t.Errorf("31 bytes under an additive 10 and 20: matched %v, want none", got)
	}
	first := Gate{Gaps: gaps}
	if got := first.match(over(30)); len(got) != 0 {
		t.Errorf("30 bytes under a 10 and a 20, not additive: matched %v, want none", got)
	}
	if got := first.match(over(15)); len(got) != 1 || got[0] != 1 {
		t.Errorf("15 bytes under a 10 and a 20, not additive: matched %v, want the 20", got)
	}
	if got := sum.match(Finding{Row: "r", Kind: Invariant, Line: mem.LineKV, Excess: 1}); len(got) != 0 {
		t.Errorf("an invariant finding matched %v", got)
	}
}
