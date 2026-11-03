package memgate

import (
	"fmt"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/mem"
)

var rounding = PerLease("a 16 KiB test allocator", 16<<10)

// usage is the ledger's account after leases of {line, requested, charged}.
func usage(leases ...[3]any) mem.Usage {
	var l mem.Ledger
	for _, x := range leases {
		l.Reserve(mem.Unified, mem.LineID(x[0].(string)), int64(x[1].(int)), int64(x[2].(int)))
	}
	return l.Usage()
}

func reading(open mem.Usage) Reading {
	return Reading{Row: "fx int4", KVSlots: 1, Plan: decoder.Plan{DenseBytes: 1000, KVBytes: 500, StateBytes: 100},
		Open: []Snapshot{{"built", open}}}
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
			r.Closed = usage([3]any{mem.LineDense, 5, 5})
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
			r.Open = append(r.Open, Snapshot{"generated", usage([3]any{mem.LineDense, 1001, 1001})})
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
	gap := Gap{Row: "fx *", Line: mem.LineState, What: "a test gap", Item: "MM8 test", Bytes: func(Reading) int64 { return 50 }}
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
		{"another row stays red", []Gap{{Row: "other *", Line: gap.Line, Item: "x", Bytes: gap.Bytes}}, 1,
			func() []Reading { return []Reading{over()} }, "50 over"},
		{"past the gap's bytes stays red", []Gap{{Row: gap.Row, Line: gap.Line, Item: "x", Bytes: func(Reading) int64 { return 49 }}}, 1,
			func() []Reading { return []Reading{over()} }, "50 over"},
		{"a gap that no longer fires", []Gap{gap}, 1, func() []Reading { return []Reading{reading(usage())} }, "no longer fires"},
		{"the table above its pin", []Gap{gap}, 0, func() []Reading { return []Reading{over()} }, "may only shrink"},
		{"the table below its pin", []Gap{gap}, 2, func() []Reading { return []Reading{over()} }, "may only shrink"},
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
