//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestCB01ChainAB is C-B01's grade (docs/tasks/task-metal-audit-2026-10.md, "C-B01: built, pending its grade"): greedy
// Generate on Metal with the greedy chain against the same request with it off (greedyChainOff, the path production
// ran before), one model, in one process, interleaved. Each rep runs both arms, in the order off-on on even reps and
// on-off on odd ones, after one discarded warm-up of each. An arm's rate is its decode rate: generated tokens after the
// first over the time from the first token's arrival to the last's, so the prompt's prefill is outside it. Every rep's
// two token streams must be equal, and the chain must serve every token after the first of its arm and none of the off
// arm's, or the comparison is void.
//
// Night-only: GOINFER_METAL_CB01_AB=1. GOINFER_METAL_CB01_MODEL (else the 1.5B), GOINFER_METAL_CB01_REPS (9),
// GOINFER_METAL_CB01_TOKENS (256).
func TestCB01ChainAB(t *testing.T) {
	if os.Getenv("GOINFER_METAL_CB01_AB") != "1" {
		t.Skip("set GOINFER_METAL_CB01_AB=1: C-B01's timed grade (night-only)")
	}
	path := os.Getenv("GOINFER_METAL_CB01_MODEL")
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
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
	reps, n := envInt("GOINFER_METAL_CB01_REPS", 9), envInt("GOINFER_METAL_CB01_TOKENS", 256)
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
	if why := a.r.greedyChainWhyNot(); why != "" {
		t.Fatalf("%s cannot run the greedy chain: %s", filepath.Base(path), why)
	}
	if _, _, why := a.r.chainEmbedTable(); why != "" {
		t.Fatalf("%s: no gather table for the greedy chain: %s", filepath.Base(path), why)
	}
	auditHB("cb01", t0, "loaded %s (tied head %v): %d reps of %d greedy tokens per arm", filepath.Base(path), a.r.lmTied, reps, n)
	prompt := make([]int, 64)
	for i := range prompt {
		prompt[i] = 1000 + (i*37+5)%15000
	}
	defer func() { greedyChainOff = false }()
	type arm struct {
		ids    []int
		rate   float64
		served int
	}
	run := func(chain bool) arm {
		greedyChainOff = !chain
		served0 := a.r.chainServed
		ch, g := m.Generate(context.Background(), prompt, n, decoder.SamplingParams{})
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
		o.rate, o.served = float64(n-1)/last.Sub(first).Seconds(), a.r.chainServed-served0
		return o
	}
	check := func(rep int, off, on arm) {
		if !slices.Equal(off.ids, on.ids) {
			t.Fatalf("rep %d: the chain's tokens differ from the full-logits path's", rep)
		}
		if off.served != 0 || on.served < n-1 {
			t.Fatalf("rep %d: the chain served %d tokens with it off and %d with it on (want 0 and >= %d)", rep, off.served, on.served, n-1)
		}
	}
	check(-1, run(false), run(true)) // warm-up, discarded
	var ratios, offR, onR []float64
	for rep := range reps {
		var off, on arm
		if rep%2 == 0 {
			off, on = run(false), run(true)
		} else {
			on, off = run(true), run(false)
		}
		check(rep, off, on)
		ratios, offR, onR = append(ratios, on.rate/off.rate), append(offR, off.rate), append(onR, on.rate)
		auditHB("cb01", t0, "rep %d/%d: off %.1f tok/s, chain %.1f tok/s, chain/off %.4f", rep+1, reps, off.rate, on.rate, on.rate/off.rate)
	}
	above := 0
	for _, r := range ratios {
		if r > 1 {
			above++
		}
	}
	auditHB("cb01", t0, "identity: %d reps x %d tokens, chain = full-logits path in every rep", reps+1, n)
	auditHB("cb01", t0, "C-B01 METRIC chain/off %s: median %.4f (min %.4f, max %.4f), %d of %d pairs above 1; decode off %.1f, chain %.1f tok/s (medians)",
		filepath.Base(path), auditMedian(ratios), slices.Min(ratios), slices.Max(ratios), above, reps, auditMedian(offR), auditMedian(onR))
}
