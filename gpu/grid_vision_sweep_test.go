//go:build gpu && goinfer_testhooks

package gpu

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/multimodal"
)

// TestGridTowersWebGPU_patchSweep is G-S12.3g (docs/tasks/task-multimodal-support-2026-10.md, "S12.3g"): for one grid
// tower, the WebGPU tower's encode time over the CPU tower's at a ladder of image sizes, to find the patch count above
// which the device tower is the faster one. The same image is preprocessed under a ladder of pixel ceilings, the last
// rung being the ceiling serve uses. A rung is a discarded warm-up pair and three timed pairs in alternating order; its
// reading is the median ratio. The threshold is the smallest rung whose ratio, and every larger rung's, is at most 0.98.
//
// GOINFER_TOWER_PATCH_SWEEP=1 runs it; GOINFER_GRID_SPEED_TOWER is qwen3, glm-ocr or qwen25. It fails only when it
// cannot run: the verdict line is a measurement, not a pass.
func TestGridTowersWebGPU_patchSweep(t *testing.T) {
	if os.Getenv("GOINFER_TOWER_PATCH_SWEEP") != "1" {
		t.Skip("a timed sweep: set GOINFER_TOWER_PATCH_SWEEP=1")
	}
	name := os.Getenv("GOINFER_GRID_SPEED_TOWER")
	var cpu, dev func([]float32, [][3]int) ([]float32, error)
	var pp multimodal.QwenPreprocessConfig
	switch name {
	case "qwen3", "glm-ocr":
		tw := gvLoadReal(t, name, t.Fatalf) // a queued sweep that cannot run is a failure, not a skip
		defer tw.close()
		cpu, dev, pp = tw.cpu, tw.dev, tw.pp
	case "qwen25":
		c, d, p := q25Real(t, t.Fatalf)
		cpu, dev, pp = c.Forward, d.Forward, p
	default:
		t.Fatalf("GOINFER_GRID_SPEED_TOWER=%q: want qwen3, glm-ocr or qwen25", name)
	}
	const img = "glm_ocr/table.png"
	data, err := os.ReadFile(filepath.Join("../testdata", img))
	if err != nil {
		t.Fatal(err)
	}
	t0 := time.Now()
	gate := "G-S12.3g " + name
	logf := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[%s %6.1fs] %s\n", gate, time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	type rung struct {
		patches int
		grid    [3]int
		ratio   float64
	}
	var rungs []rung
	seen := map[[3]int]bool{}
	sides := []int{112, 168, 224, 280, 336, 448, 672, 896, 0} // a pixel ceiling of side*side; 0 is the ceiling serve uses
	for ri, side := range sides {
		q := pp
		if side > 0 {
			if side*side >= pp.MaxPixels {
				continue // at or over serve's ceiling: the last rung covers it
			}
			q.MaxPixels = side * side
			q.MinPixels = min(q.MinPixels, q.MaxPixels)
		}
		px, grid, err := multimodal.QwenPreprocess(data, q)
		if err != nil {
			t.Fatal(err)
		}
		if seen[grid] {
			continue
		}
		seen[grid] = true
		g := [][3]int{grid}
		run := func(f func([]float32, [][3]int) ([]float32, error)) float64 {
			ts := time.Now()
			if _, err := f(px, g); err != nil {
				t.Fatal(err)
			}
			return time.Since(ts).Seconds()
		}
		wc, wd := run(cpu), run(dev)
		const pairs = 3
		var ratios, cs, ds []float64
		for i := range pairs {
			var c, d float64
			if i%2 == 0 {
				c = run(cpu)
				d = run(dev)
			} else {
				d = run(dev)
				c = run(cpu)
			}
			ratios, cs, ds = append(ratios, d/c), append(cs, c), append(ds, d)
		}
		sort.Float64s(ratios)
		sort.Float64s(cs)
		sort.Float64s(ds)
		patches := grid[1] * grid[2]
		rungs = append(rungs, rung{patches, grid, ratios[pairs/2]})
		logf("rung %d/%d: ceiling %s, grid %v, %d patches: CPU %.3f s, WebGPU %.3f s (medians of %d; warm-up %.3f, %.3f), ratios %.3f, median %.3f",
			ri+1, len(sides), map[bool]string{true: "as served", false: fmt.Sprintf("%dx%d px", side, side)}[side == 0], grid, patches,
			cs[pairs/2], ds[pairs/2], pairs, wc, wd, ratios, ratios[pairs/2])
	}
	if len(rungs) == 0 {
		t.Fatal("no rung ran")
	}
	sort.Slice(rungs, func(i, j int) bool { return rungs[i].patches < rungs[j].patches })
	threshold := -1 // index of the smallest rung from which every rung is at most 0.98
	for i := len(rungs) - 1; i >= 0 && rungs[i].ratio <= 0.98; i-- {
		threshold = i
	}
	switch {
	case threshold < 0:
		logf("THRESHOLD none: at the largest rung (%d patches) the device tower's ratio is %.3f, over 0.98", rungs[len(rungs)-1].patches, rungs[len(rungs)-1].ratio)
	case threshold == 0:
		logf("THRESHOLD all: the device tower is at most 0.98 of the CPU's at every rung, from %d patches up", rungs[0].patches)
	default:
		logf("THRESHOLD %d patches (grid %v): at most 0.98 from there up; the rung below it, %d patches, read %.3f",
			rungs[threshold].patches, rungs[threshold].grid, rungs[threshold-1].patches, rungs[threshold-1].ratio)
	}
}
