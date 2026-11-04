//go:build darwin

package probe

import (
	"os/exec"
	"regexp"
	"strconv"
	"testing"
)

// TestRead_matchesMemoryPressure reads both numbers: memory_pressure's "System-wide memory free percentage", read before and after
// this package's reading, and the reading. Available must be that percentage of physical memory less 500 MiB, computed here from
// memory_pressure's own figure, within the points the percentage moved between the two reads (and a point of rounding). Reclaimable
// must lie inside the percentage's figure (the percentage counts every page that is neither wired nor in the compressor), with 5
// points of slack: it comes from host_statistics64, which XNU may serve up to a second old, while the level is read live.
func TestRead_matchesMemoryPressure(t *testing.T) {
	pressure := func() int {
		out, err := exec.Command("/usr/bin/memory_pressure").Output()
		if err != nil {
			t.Fatalf("memory_pressure: %v", err)
		}
		m := regexp.MustCompile(`System-wide memory free percentage: (\d+)%`).FindSubmatch(out)
		if m == nil {
			t.Fatalf("no free percentage in memory_pressure's output:\n%s", out)
		}
		pct, _ := strconv.Atoi(string(m[1]))
		return pct
	}
	before := pressure()
	r, ok := Read()
	after := pressure()
	if !ok {
		t.Fatal("no reading")
	}
	g := func(b int64) float64 { return float64(b) / (1 << 30) }
	t.Logf("memory_pressure %d%% then %d%%; level %d%%, available %.2f GiB, reclaimable %.2f GiB, physical %.2f GiB", before, after,
		r.Level, g(r.Available), g(r.Reclaimable), g(r.Physical))
	lo, hi := min(before, after)-1, max(before, after)+1
	if r.Level < lo || r.Level > hi { // checked directly too: at a level of 3% or less the Available band below clamps to 0
		t.Errorf("level %d%% outside memory_pressure's %d%%..%d%% (±1)", r.Level, before, after)
	}
	want := func(pct int) int64 { return max(r.Physical*int64(pct)/100-524288000, 0) }
	if r.Available < want(lo) || r.Available > want(hi) {
		t.Errorf("available %d outside [%d, %d], memory_pressure's %d%%..%d%% of physical less 500 MiB", r.Available, want(lo), want(hi), before, after)
	}
	if r.Reclaimable <= 0 || r.Reclaimable > r.Physical*int64(r.Level+5)/100 {
		t.Errorf("reclaimable %d outside (0, level+5%% of physical]", r.Reclaimable)
	}
}
