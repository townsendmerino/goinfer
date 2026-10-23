//go:build darwin

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
	"time"
)

// ksh4Released waits for a's pending release (gridIdleRelease shortened) and reports whether its scratch is empty.
func ksh4Released(a *gridVAccel) bool {
	for range 200 {
		a.mu.Lock()
		n := a.scratchBytes()
		a.mu.Unlock()
		if n == 0 {
			return true
		}
		time.Sleep(5 * time.Millisecond)
	}
	return false
}

// TestKSH4_releaseKeepsOutputs is H4-G1 (docs/tasks/task-kernel-survey-2026-10.md, "KS-H4 gates"): with the ceiling at 0 and a
// 1 ms idle wait, every grid tiny fixture releases its scratch after each forward, and the forwards after a release (a larger then a
// smaller grid, then the first again) give the same outputs, byte for byte, as a tower that never releases.
func TestKSH4_releaseKeepsOutputs(t *testing.T) {
	prevC, prevD := gridReleaseCeiling, gridIdleRelease
	gridReleaseCeiling, gridIdleRelease = 0, time.Millisecond
	defer func() { gridReleaseCeiling, gridIdleRelease = prevC, prevD }()
	order := []int{0, 2, 1, 0}
	for _, gc := range []struct {
		name string
		mk   func(t *testing.T) gvCase
	}{{"qwen3.5", gvQwen3}, {"glm-ocr", gvGlm}} {
		pd := gc.mk(t).patchDim
		rng := rand.New(rand.NewSource(9))
		var px [][]float32
		for _, g := range gvGrids[:3] {
			px = append(px, gvPixels(rng, g, pd))
		}
		run := func(off bool) [][]float32 {
			gridReleaseOffForTest = off
			defer func() { gridReleaseOffForTest = false }()
			a := gc.mk(t).acc
			defer a.Close()
			var outs [][]float32
			for _, i := range order {
				out, err := a.Hidden(px[i], gvGrids[i])
				if err != nil {
					t.Fatalf("%s: %v", gc.name, err)
				}
				outs = append(outs, out)
				if !off && !ksh4Released(a) {
					t.Fatalf("%s: the scratch was not released after forward %d", gc.name, i)
				}
			}
			return outs
		}
		want, got := run(true), run(false)
		d, n := 0, 0
		for s := range want {
			n += len(want[s])
			for j := range want[s] {
				if math.Float32bits(want[s][j]) != math.Float32bits(got[s][j]) {
					d++
				}
			}
		}
		fmt.Fprintf(os.Stderr, "[H4-G1] %-8s released after every forward; %d of %d outputs differ from the tower that never releases\n", gc.name, d, n)
		if d != 0 {
			t.Errorf("%s: %d outputs differ", gc.name, d)
		}
	}
	for _, tw := range mem2Towers(t) {
		if tw.name != "qwen2.5-vl" {
			continue
		}
		run := func(off bool) [][]float32 {
			gridReleaseOffForTest = off
			defer func() { gridReleaseOffForTest = false }()
			in := tw.build(t)
			defer in.close()
			var outs [][]float32
			for _, i := range []int{0, 2, 1, 0} {
				out, err := in.fwd(i)
				if err != nil {
					t.Fatal(err)
				}
				outs = append(outs, append([]float32(nil), out...))
				time.Sleep(20 * time.Millisecond)
			}
			return outs
		}
		want, got := run(true), run(false)
		d := 0
		for s := range want {
			for j := range want[s] {
				if math.Float32bits(want[s][j]) != math.Float32bits(got[s][j]) {
					d++
				}
			}
		}
		fmt.Fprintf(os.Stderr, "[H4-G1] qwen2.5-vl %d outputs differ from the tower that never releases\n", d)
		if d != 0 {
			t.Errorf("qwen2.5-vl: %d outputs differ", d)
		}
	}
}
