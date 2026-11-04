//go:build darwin

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
)

// TestKSD6_towerOutputsUnchanged is D6-G1 (docs/tasks/task-kernel-survey-2026-10.md, "KS-D6 gates"): MEM2-G1's seven tower
// fixtures, three inputs each, with the grid towers' scratch allocated only where read against main's allocation
// (gridScratchAllForTest): every output byte-equal. The grid towers also run with the oldAttn seam, so the unfused attention's
// lazily allocated scratch is exercised.
func TestKSD6_towerOutputsUnchanged(t *testing.T) {
	run := func(tw mem2Tower, all bool) [][]float32 {
		prev := gridScratchAllForTest
		gridScratchAllForTest = all
		defer func() { gridScratchAllForTest = prev }()
		in := tw.build(t)
		defer in.close()
		var outs [][]float32
		for i := range tw.n {
			out, err := in.fwd(i)
			if err != nil {
				t.Fatalf("%s forward %d: %v", tw.name, i, err)
			}
			outs = append(outs, append([]float32(nil), out...))
		}
		return outs
	}
	for _, tw := range mem2Towers(t) {
		want, got := run(tw, true), run(tw, false)
		d, n := 0, 0
		for i := range want {
			n += len(want[i])
			for j := range want[i] {
				if math.Float32bits(want[i][j]) != math.Float32bits(got[i][j]) {
					d++
				}
			}
		}
		fmt.Fprintf(os.Stderr, "[D6-G1] %-16s %d of %d outputs differ\n", tw.name, d, n)
		if d != 0 {
			t.Errorf("%s: %d outputs differ from main's allocation", tw.name, d)
		}
	}
	for _, gc := range []struct {
		name string
		mk   func(t *testing.T) gvCase
	}{{"qwen3.5", gvQwen3}, {"glm-ocr", gvGlm}} {
		c := gc.mk(t)
		px := gvPixels(rand.New(rand.NewSource(5)), gvGrids[1], c.patchDim)
		c.acc.oldAttn = true
		out, err := c.acc.Hidden(px, gvGrids[1])
		if err != nil {
			t.Fatalf("%s with the unfused attention: %v", gc.name, err)
		}
		c.acc.oldAttn = false
		fused, err := c.acc.Hidden(px, gvGrids[1])
		if err != nil {
			t.Fatal(err)
		}
		bad := 0
		for _, v := range out {
			if math.IsNaN(float64(v)) {
				bad++
			}
		}
		fmt.Fprintf(os.Stderr, "[D6-G1] %-16s the unfused attention runs on lazily allocated scratch: %d outputs, %d NaN (fused gives %d)\n", gc.name, len(out), bad, len(fused))
		if bad != 0 || c.acc.qh == (Buffer{}) {
			t.Errorf("%s: the unfused attention's scratch was not allocated for the oldAttn path (%d NaN)", gc.name, bad)
		}
		c.acc.Close()
	}
}
