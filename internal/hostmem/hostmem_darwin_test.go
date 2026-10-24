//go:build darwin

package hostmem

import (
	"os/exec"
	"regexp"
	"strconv"
	"testing"
)

// TestRead_matchesMemoryPressure reads both numbers: memory_pressure's "System-wide memory free percentage" and this package's
// reading. The reading's Level is that percentage, its Available is fromLevel of it, and the free + file-backed + purgeable pages it
// keeps as Reclaimable are inside the percentage's figure (the percentage counts every page that is not wired or in the compressor).
func TestRead_matchesMemoryPressure(t *testing.T) {
	out, err := exec.Command("/usr/bin/memory_pressure").Output()
	if err != nil {
		t.Skipf("memory_pressure: %v", err)
	}
	m := regexp.MustCompile(`System-wide memory free percentage: (\d+)%`).FindSubmatch(out)
	if m == nil {
		t.Fatalf("no free percentage in memory_pressure's output:\n%s", out)
	}
	pct, _ := strconv.Atoi(string(m[1]))
	r, ok := Read()
	if !ok {
		t.Fatal("no reading")
	}
	g := func(b int64) float64 { return float64(b) / (1 << 30) }
	t.Logf("memory_pressure %d%%; level %d%%, available %.2f GiB, reclaimable %.2f GiB, physical %.2f GiB", pct, r.Level, g(r.Available),
		g(r.Reclaimable), g(r.Physical))
	if d := r.Level - pct; d < -3 || d > 3 {
		t.Errorf("level %d%%, memory_pressure %d%%", r.Level, pct)
	}
	if r.Available != fromLevel(r.Level, r.Physical) {
		t.Errorf("available %d, fromLevel %d", r.Available, fromLevel(r.Level, r.Physical))
	}
	if r.Reclaimable <= 0 || r.Reclaimable > r.Physical/100*int64(r.Level+1) {
		t.Errorf("reclaimable %d outside (0, level+1%% of physical]", r.Reclaimable)
	}
}
