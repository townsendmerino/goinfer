//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestHLChainAB is HL-G4 (docs/tasks/task-forward-architecture-2026-10.md §6.1): greedy Generate on Metal under a gated
// processor with the chain (HL) against greedyChainOff (today's path), one model, in one process, interleaved as
// TestCB01ChainAB is: one discarded warm-up of each arm, then reps alternating off-on and on-off. An arm's rate is its
// decode rate, generated tokens after the first over the time from the first token's arrival to the last's. Every
// rep's two token streams must be equal, or the comparison is void.
//
// GOINFER_METAL_HL_SHAPE picks the quantity:
//   - closed (a): the gate never opens, the prose part of a tool turn;
//   - toolcall (b): the gate opens for the last 48 tokens, a tool call, and the processor bans the row's argmax on every
//     8th open step, so a redo costs what it costs.
//
// Night-only: GOINFER_METAL_HL_AB=1. GOINFER_METAL_HL_MODEL (else the 1.5B's Metal .giw), GOINFER_METAL_HL_REPS (9),
// GOINFER_METAL_HL_TOKENS (256), GOINFER_METAL_HL_SHAPE (closed).
func TestHLChainAB(t *testing.T) {
	if os.Getenv("GOINFER_METAL_HL_AB") != "1" {
		t.Skip("set GOINFER_METAL_HL_AB=1: HL-G4's timed grade (night-only)")
	}
	path := os.Getenv("GOINFER_METAL_HL_MODEL")
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.v16.giw")
	}
	if strings.HasPrefix(path, "/Volumes/") || strings.HasPrefix(path, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md): a timing from it measures the disk", path)
	}
	envInt := func(k string, def int) int {
		var n int
		if _, err := fmt.Sscan(os.Getenv(k), &n); err == nil && n > 0 {
			return n
		}
		return def
	}
	reps, n := envInt("GOINFER_METAL_HL_REPS", 9), envInt("GOINFER_METAL_HL_TOKENS", 256)
	shape := os.Getenv("GOINFER_METAL_HL_SHAPE")
	if shape == "" {
		shape = "closed"
	}
	const callLen, banEvery = 48, 8
	var open func([]int) bool
	switch shape {
	case "closed":
		open = func([]int) bool { return false }
	case "toolcall":
		if n <= callLen {
			t.Fatalf("GOINFER_METAL_HL_TOKENS=%d: the toolcall shape needs more than %d", n, callLen)
		}
		open = func(gen []int) bool { return len(gen) >= n-callLen }
	default:
		t.Fatalf("GOINFER_METAL_HL_SHAPE=%q: want closed or toolcall", shape)
	}
	t0 := time.Now()
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 4096, ResidentKVSlots: 1})
	if err != nil {
		t.Fatalf("load %s: %v", path, err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("no metal resident: %s", m.ResidentDecline())
	}
	if !a.GreedyChainAvailable() {
		t.Fatalf("%s cannot run the greedy chain: %s", filepath.Base(path), a.r.greedyChainWhyNot())
	}
	auditHB("hl", t0, "loaded %s (tied head %v): shape %s, %d reps of %d tokens per arm", filepath.Base(path), a.r.lmTied, shape, reps, n)
	_, _, _, _, _, _, vocab := m.Dims()
	prompt := hlTextPrompt(64)
	for i := range prompt {
		prompt[i] %= vocab // the tiny fixture's vocabulary, for a harness check
	}
	defer func() { greedyChainOff = false }()
	type arm struct {
		ids                 []int
		rate                float64
		served, rows, redos int
	}
	run := func(chain bool) arm {
		greedyChainOff = !chain
		calls := 0
		sp := decoder.SamplingParams{
			LogitProcessor: func(_ []int, logits []float32) {
				if calls++; calls%banEvery == 0 {
					logits[argmaxF32(logits)] = float32(math.Inf(-1))
				}
			},
			LogitProcessorGate: open,
		}
		s0, r0, d0 := a.r.chainServed, a.r.chainRows, a.r.chainRedos
		ch, g := m.Generate(context.Background(), prompt, n, sp)
		var o arm
		var first, last time.Time
		for id := range ch {
			if last = time.Now(); len(o.ids) == 0 {
				first = last
			}
			o.ids = append(o.ids, id)
		}
		if err := g.Err(); err != nil {
			t.Fatalf("generate (chain %v): %v", chain, err)
		}
		if len(o.ids) != n {
			t.Fatalf("generate (chain %v): %d tokens, want %d (an EOS: change the prompt)", chain, len(o.ids), n)
		}
		o.rate = float64(n-1) / last.Sub(first).Seconds()
		o.served, o.rows, o.redos = a.r.chainServed-s0, a.r.chainRows-r0, a.r.chainRedos-d0
		return o
	}
	check := func(rep int, off, on arm) {
		if !slices.Equal(off.ids, on.ids) {
			t.Fatalf("rep %d: HL's tokens differ from today's", rep)
		}
		if off.served != 0 || on.served < n-1 {
			t.Fatalf("rep %d: the chain served %d tokens with it off and %d with it on (want 0 and >= %d)", rep, off.served, on.served, n-1)
		}
		if shape == "toolcall" && (on.rows < callLen-1 || on.redos == 0) {
			t.Fatalf("rep %d: %d rows and %d redos with the chain on: the tool call did not run through the chain", rep, on.rows, on.redos)
		}
	}
	// The first Generate prefills cold and every later one reuses the prompt (TestCB01ChainAB's note): discard one cold run.
	run(false)
	check(-1, run(false), run(true)) // warm-up, discarded
	var ratios, offR, onR []float64
	var rows, redos int
	for rep := range reps {
		var off, on arm
		if rep%2 == 0 {
			off, on = run(false), run(true)
		} else {
			on, off = run(true), run(false)
		}
		check(rep, off, on)
		ratios, offR, onR = append(ratios, on.rate/off.rate), append(offR, off.rate), append(onR, on.rate)
		rows, redos = on.rows, on.redos
		auditHB("hl", t0, "rep %d/%d: today %.1f tok/s, HL %.1f tok/s, HL/today %.4f", rep+1, reps, off.rate, on.rate, on.rate/off.rate)
	}
	above := 0
	for _, r := range ratios {
		if r > 1 {
			above++
		}
	}
	auditHB("hl", t0, "identity: %d reps x %d tokens, HL = today's path in every rep; per HL turn %d rows, %d redos", reps+1, n, rows, redos)
	auditHB("hl", t0, "HL-G4 METRIC hl/today %s %s: median %.4f (min %.4f, max %.4f), %d of %d pairs above 1; decode today %.1f, HL %.1f tok/s (medians)",
		shape, filepath.Base(path), auditMedian(ratios), slices.Min(ratios), slices.Max(ratios), above, reps, auditMedian(offR), auditMedian(onR))
}
