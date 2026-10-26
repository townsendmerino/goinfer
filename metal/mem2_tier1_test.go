//go:build darwin

package metal

import (
	"fmt"
	"math"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/internal/vision"
)

// TestMEM2_tier1Qwen3VL is MEM2-G4 (docs/tasks/task-memory-campaign-2026-10.md, MEM2 gates): the Qwen3-VL-2B tower on
// Metal, float32. Tower A runs a 1080p grid and then a 4K grid; tower B runs the 4K grid alone. A's scratch after the 4K
// forward (Metal's allocated size net of its reading after the build) is within 1% of B's, and after Close each is back
// within 16 MiB of its reading before the build.
func TestMEM2_tier1Qwen3VL(t *testing.T) {
	requireHeavyModel(t)
	dir := os.ExpandEnv("$HOME/models/qwen3-vl-2b-instruct")
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no checkpoint at %s", dir)
	}
	t0 := time.Now()
	logf := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[MEM2-G4 %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	enc, err := vision.LoadQwen3VisionEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	logf("tower loaded")
	probe, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer probe.ReleaseObjects()
	c := enc.Cfg
	pd := c.InChannels * c.TemporalPatchSize * c.PatchSize * c.PatchSize
	grids := map[string][3]int{"1080p": {1, 68, 120}, "4K": {1, 134, 240}} // patches, even for the 2x2 merge
	px := map[string][]float32{}
	for k, g := range grids {
		v := make([]float32, g[0]*g[1]*g[2]*pd)
		for i := range v {
			v[i] = float32(math.Sin(float64(i) * 0.001))
		}
		px[k] = v
	}
	const MiB = 1 << 20
	size := func() int64 { return int64(probe.CurrentAllocatedSize()) }
	// run builds a tower, runs the named grids in order and closes it; it returns the scratch after the last forward
	// (net of the reading after the build) and what remains after Close (net of the reading before the build).
	run := func(arm string, order ...string) (scratch, after int64) {
		base := size()
		a, err := newQwen3VAccel(enc)
		if err != nil {
			t.Fatal(err)
		}
		built := size()
		logf("%s: built, weights %+.1f MiB", arm, float64(built-base)/MiB)
		for _, k := range order {
			g := grids[k]
			if _, err := a.Hidden(px[k], [][3]int{g}); err != nil {
				t.Fatalf("%s %s: %v", arm, k, err)
			}
			logf("%s: %s (%d patches) done, scratch %+.1f MiB", arm, k, g[1]*g[2], float64(size()-built)/MiB)
		}
		scratch = size() - built
		_ = a.Close()
		after = size() - base
		logf("%s: closed, %+.1f MiB over the reading before the build", arm, float64(after)/MiB)
		return scratch, after
	}
	sa, aa := run("A", "1080p", "4K")
	sb, ab := run("B", "4K")
	logf("A scratch %.1f MiB, B scratch %.1f MiB (%+.2f%%); after Close A %+.1f MiB, B %+.1f MiB",
		float64(sa)/MiB, float64(sb)/MiB, 100*float64(sa-sb)/float64(sb), float64(aa)/MiB, float64(ab)/MiB)
	if math.Abs(float64(sa-sb)) > 0.01*float64(sb) {
		t.Errorf("G4: after 1080p then 4K the tower holds %.1f MiB of scratch, after 4K alone %.1f MiB", float64(sa)/MiB, float64(sb)/MiB)
	}
	if aa > 16*MiB || ab > 16*MiB {
		t.Errorf("G4: after Close Metal holds %+.1f MiB (A) and %+.1f MiB (B) over the reading before the build", float64(aa)/MiB, float64(ab)/MiB)
	}
}
