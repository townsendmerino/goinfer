//go:build goinfer_testhooks

package memgate

import (
	"flag"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/decoder/fit"
	"github.com/townsendmerino/goinfer/internal/diag/sb2census"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

// cpuGateCtx is the positions a CPU row writes: past every fixture's conv kernel, so each window is full.
const cpuGateCtx = 64

// cpuGateRowsPath names every row that must load and run, and cpuGateRowsUpdate has TestMemGate_cpu rewrite it.
const cpuGateRowsPath = "testdata/cpu_rows.txt"

var cpuGateRowsUpdate = flag.Bool("memgate.cpurows.update", false, "have TestMemGate_cpu rewrite testdata/cpu_rows.txt")

// cpuRounding is the CPU's allocator as a row sees it: the Go heap's size classes are not in what the walk leases.
var cpuRounding = PerLease("none: the walk leases each array's own bytes", 0)

// cpuMemGaps is the CPU's table of known reds. It may only shrink: the fix that removes a row lowers cpuMemGapsPin.
var cpuMemGaps []Gap

const cpuMemGapsPin = 0

// cpuGateRow loads fixture path on the CPU, writes cpuGateCtx positions, and returns the row. The CPU has no allocator to account, so its
// ledger is filled by walking what the model and the cache hold (Model.CPUHoldsForTest). Closed reads that ledger after the model closed and
// those leases are released, so it holds a plant's lease and not the model's own freeing, which no ledger sees here. plant, when set, runs on the ledger before it is read. ok is false when the load is refused, which has nothing to hold.
func cpuGateRow(t *testing.T, row, path string, o decoder.Options, plant func(*mem.Ledger)) (Reading, bool) {
	t.Helper()
	return cpuGateRowAt(t, row, path, o, plant, cpuGateCtx)
}

// cpuGateRowAt is cpuGateRow at ctx positions.
func cpuGateRowAt(t *testing.T, row, path string, o decoder.Options, plant func(*mem.Ledger), ctx int) (Reading, bool) {
	t.Helper()
	o.Backend, o.DirectLoad = "cpu", true
	m, err := decoder.Load(path, o)
	if err != nil {
		return Reading{}, false
	}
	defer m.Close()
	var l mem.Ledger
	c := m.NewCache(ctx)
	cfg := m.Config()
	for i := range ctx {
		if _, err := m.ForwardForTest(i%cfg.VocabSize, c); err != nil {
			t.Errorf("%s: forward %d: %v", row, i, err)
			return Reading{}, false
		}
	}
	release := m.CPUHoldsForTest(&l, c)
	if plant != nil {
		plant(&l)
	}
	kv := m.KVDeliveredFor("cpu") // what the cache really is: Options.KVPrecision is the GPU's, and a family can refuse what was asked
	p := m.Plan("cpu", math.MaxInt64/4, fit.PlanRequest{Ctx: ctx, CtxPinned: true, KVF16: kv == mem.KVF16, KVI8: kv == mem.KVI8})
	read := Reading{Row: row, KVSlots: 1, Plan: p, Exact: []mem.LineID{mem.LineKV, mem.LineState}, Named: map[mem.LineID]int64{mem.LineScratch: decoder.CPUScratchBudgetForTest}}
	facts := GeometryOf(m, ctx, 1, kv, true).Facts()
	read.Open = append(read.Open, Snap("generated", &l, facts))
	m.Close()
	release()
	read.Closed = l.Usage()
	return read, true
}

// TestMemGate_cpu is MM7's gate on the CPU (docs/tasks/task-subsystems-2026-10.md): every census fixture at every option cell that loads.
// It runs every row in one test, since a filtered run would see known gaps go quiet.
func TestMemGate_cpu(t *testing.T) {
	d, err := sb2census.ReadLayerD("../../..", "cuda") // the fixture list; which cells go resident on CUDA is not read
	if err != nil {
		t.Fatal(err)
	}
	g := Gate{Allowance: cpuRounding, Gaps: cpuMemGaps, GapsPin: cpuMemGapsPin}
	var built []string
	refused := 0
	for _, f := range d.Fixtures {
		path, err := sb2census.Path("../../..", f)
		if err != nil {
			t.Fatal(err)
		}
		for _, c := range sb2census.Cells {
			if c.SSM != "" || c.KVPrec != "" {
				continue // a GPU's setting: on the CPU the row would be int4's again
			}
			read, ok := cpuGateRow(t, f+" "+c.Name, path, decoder.Options{Quant: c.Quant, ActQuantGroup: c.ActGroup, KVPrecision: c.KVPrec, KVQuant: c.KVQuant}, nil)
			if !ok {
				refused++
				continue
			}
			built = append(built, f+" "+c.Name)
			g.Check(t, read)
		}
		// The census's cells ask the GPU's KVPrecision for f16 and the CPU's KVQuant for int8: this is the CPU's own f16.
		if read, ok := cpuGateRow(t, f+" int4/kvquant-f16", path, decoder.Options{Quant: "int4", KVQuant: "f16"}, nil); ok {
			built = append(built, f+" int4/kvquant-f16")
			g.Check(t, read)
		} else {
			refused++
		}
	}
	t.Logf("%d rows held, %d option cells refused at load", len(built), refused)
	HoldRows(t, cpuGateRowsPath, built, *cpuGateRowsUpdate)
	g.Finish(t)
}

// TestMemGate_cpuRingsAcrossTheWindow: on two ring families, at each K/V precision, the plan's KV is what the cache holds at a context
// below, at and one past the window, where TestMemGate_cpu's rows are all past it: a ring holds its W rows from its first write, and at
// f32 twice that once a read wraps.
func TestMemGate_cpuRingsAcrossTheWindow(t *testing.T) {
	g := Gate{Allowance: cpuRounding}
	for _, fx := range []string{"testdata/gemma2-tiny", "testdata/mistral-tiny-window"} {
		path, err := sb2census.Path("../../..", fx)
		if err != nil {
			t.Fatal(err)
		}
		m, err := decoder.Load(path, decoder.Options{Backend: "cpu", DirectLoad: true})
		if err != nil {
			t.Fatal(err)
		}
		w := m.SlidingWindowResident()
		m.Close()
		if w <= 1 || w >= cpuGateCtx {
			t.Fatalf("%s: window %d, want one between 1 and %d", fx, w, cpuGateCtx)
		}
		for _, kv := range []struct {
			name string
			o    decoder.Options
		}{{"f32", decoder.Options{Quant: "int4"}}, {"kvquant-f16", decoder.Options{Quant: "int4", KVQuant: "f16"}}, {"kv-i8", decoder.Options{Quant: "int4", KVQuant: "i8"}}} {
			for _, ctx := range []int{w - 1, w, w + 1} {
				read, ok := cpuGateRowAt(t, fmt.Sprintf("%s int4/%s ctx %d", fx, kv.name, ctx), path, kv.o, nil, ctx)
				if !ok {
					t.Fatalf("%s %s: the load was refused", fx, kv.name)
				}
				g.Check(t, read)
			}
		}
	}
	if g.rows != 18 {
		t.Fatalf("%d rows checked, want 18", g.rows)
	}
}

// TestMemGate_cpuPlantsThroughTheTable: bytes leased over plan on the row's ledger stay red through the table, on a line it forgives
// part of and on lines it forgives nothing of; a lease never released is red as open after Close; and an unknown line is red.
func TestMemGate_cpuPlantsThroughTheTable(t *testing.T) {
	const fx, row = "../../../testdata/llama-tiny", "testdata/llama-tiny int4"
	g := Gate{Allowance: cpuRounding, Gaps: cpuMemGaps, GapsPin: cpuMemGapsPin}
	o := decoder.Options{Quant: "int4"}
	clean, ok := cpuGateRow(t, row, fx, o, nil)
	if !ok {
		t.Fatal("llama-tiny did not load")
	}
	if u := g.Unknown(clean); len(u) != 0 {
		t.Fatalf("the unplanted row is red through the table: %v", u)
	}
	for _, line := range []mem.LineID{mem.LineDense, mem.LineKV, mem.LineState, mem.LineExperts} {
		read, ok := cpuGateRow(t, row, fx, o, func(l *mem.Ledger) {
			ls := l.Reserve(mem.HostAnon, line, 1<<20, 1<<20)
			t.Cleanup(ls.Release)
		})
		if !ok {
			t.Fatal("llama-tiny did not load")
		}
		red := false
		for _, f := range g.Unknown(read) {
			red = red || (f.Line == line && f.Kind == OverPlan)
		}
		if !red {
			t.Errorf("a mebibyte on %q is not red through the table: %v", line, g.Unknown(read))
		}
	}
	read, ok := cpuGateRow(t, row, fx, o, func(l *mem.Ledger) { l.Reserve(mem.HostAnon, mem.LineID("nowhere"), 8, 8) })
	if !ok {
		t.Fatal("llama-tiny did not load")
	}
	if s := fmt.Sprint(g.Unknown(read)); !strings.Contains(s, "unknown line") || !strings.Contains(s, "open after Close") {
		t.Errorf("a lease on no line, never released, reads: %s", s)
	}
}

// TestGeometry_namesEachCause: on a sliding int8 row the window's over-price and the scale's under-price are each their own figure, and what the
// layers hold is the formula less and plus them, so a row's netted figure does not stand for either.
func TestGeometry_namesEachCause(t *testing.T) {
	m, err := decoder.Load("../../../testdata/olmo3-tiny", decoder.Options{Backend: "cpu", Quant: "int4", DirectLoad: true})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	g := GeometryOf(m, cpuGateCtx, 1, mem.KVI8, true)
	if g.KVWindowOver <= 0 || g.KVScaleUnder == 0 {
		t.Fatalf("olmo3-tiny at int8: window %d, scale %d: want both", g.KVWindowOver, g.KVScaleUnder)
	}
	if got, want := g.KVHeld, g.KVFormula-g.KVWindowOver+g.KVScaleUnder; got != want {
		t.Errorf("held %d, the formula less and plus its causes %d", got, want)
	}
}
