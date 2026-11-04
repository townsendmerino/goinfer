//go:build darwin

package metal

import (
	"fmt"
	"math"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/internal/vision"
)

// TestKSH4_repeatCallTiming is H4-G3 (docs/tasks/task-kernel-survey-2026-10.md, "KS-H4 gates"): Qwen3-VL-2B's grid tower at 4K,
// the wall time of a repeat Hidden issued right after the previous one, with the release on (A) and off (B), A B B A over 5 pairs
// after a warm call. One tower serves both arms: a repeat call inside the idle window cancels the pending release, so the arms
// differ only in arming and cancelling it. Then, read and not graded, the first call after a release against a warm call. A timed
// run: hold the timing lock (GOINFER_KSH4_TIMING=1, GOINFER_HEAVY_TESTS=1).
func TestKSH4_repeatCallTiming(t *testing.T) {
	requireHeavyModel(t)
	if os.Getenv("GOINFER_KSH4_TIMING") == "" {
		t.Skip("set GOINFER_KSH4_TIMING=1 (a timed run under the timing lock)")
	}
	dir := os.ExpandEnv("$HOME/models/qwen3-vl-2b-instruct")
	enc, err := vision.LoadQwen3VisionEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	a, err := newQwen3VAccel(enc)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	c := enc.Cfg
	pd := c.InChannels * c.TemporalPatchSize * c.PatchSize * c.PatchSize
	g := [3]int{1, 134, 240}
	px := make([]float32, g[0]*g[1]*g[2]*pd)
	for i := range px {
		px[i] = float32(math.Sin(float64(i) * 0.001))
	}
	call := func(off bool) float64 {
		gridReleaseOffForTest = off
		defer func() { gridReleaseOffForTest = false }()
		t0 := time.Now()
		if _, err := a.Hidden(px, [][3]int{g}); err != nil {
			t.Fatal(err)
		}
		return time.Since(t0).Seconds()
	}
	call(true)
	call(false)
	var ratios []float64
	for i := range 5 {
		var on, off float64
		if i%2 == 0 {
			on, off = call(false), call(true)
		} else {
			off, on = call(true), call(false)
		}
		ratios = append(ratios, on/off)
		fmt.Fprintf(os.Stderr, "[H4-G3] pair %d: release on %.1f ms, off %.1f ms, on/off %.4f\n", i, on*1e3, off*1e3, on/off)
	}
	slices.Sort(ratios)
	fmt.Fprintf(os.Stderr, "[H4-G3] median on/off %.4f\n", ratios[2])

	warm := call(false)
	prev := gridIdleRelease
	gridIdleRelease = 100 * time.Millisecond
	call(false) // arms a 100 ms release
	time.Sleep(time.Second)
	a.mu.Lock()
	released := a.scratchBytes() == 0
	a.mu.Unlock()
	cold := call(false)
	gridIdleRelease = prev
	fmt.Fprintf(os.Stderr, "[H4-G3] read only: a warm call %.1f ms; the first call after a release (released: %v) %.1f ms, %+.1f ms\n", warm*1e3, released, cold*1e3, (cold-warm)*1e3)
}
