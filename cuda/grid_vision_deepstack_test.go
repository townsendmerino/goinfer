//go:build cuda

package cuda

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// G-S10f of docs/tasks/task-multimodal-support-2026-10.md ("S10 on CUDA", registered before this code): the Qwen3-VL tower on CUDA with DeepStack, CUDA twin of metal/grid_vision_deepstack_test.go. The grid tower taps the
// DeepStack blocks (HiddenTaps) and aikit's host tails make the merged rows and the sets (multimodal.Qwen3TowerFeaturesDeepstack), against aikit's CPU ForwardDeepstack: every set and the merged rows at worst-row
// cosine >= 0.9999 (0.999-0.9999 parked), on the tiny DeepStack tower (testdata/qwen3vl-vision-tiny, norms randomised) and on Qwen3-VL-2B's four F2a images (heavy). Two planted defects must go red, each alone: the
// taps one block late, and the taps in reverse order.

func dsCompare(t *testing.T, what string, enc *vision.Qwen3VisionEncoder, acc *gridTower, px []float32, grid [][3]int) (worst float64, err error) {
	t.Helper()
	wantM, wantD, err := enc.ForwardDeepstack(px, grid)
	if err != nil {
		t.Fatal(err)
	}
	gotM, gotD, err := multimodal.Qwen3TowerFeaturesDeepstack(enc, acc, px, grid)
	if err != nil {
		return 0, err
	}
	out := enc.Cfg.OutHiddenSize
	worst = gridWorst(t, gotM, wantM, out)
	line := fmt.Sprintf("merged %.9f", worst)
	for k := range wantD {
		w := gridWorst(t, gotD[k], wantD[k], out)
		line += fmt.Sprintf(", deep%d %.9f", k, w)
		worst = math.Min(worst, w)
	}
	fmt.Fprintf(os.Stderr, "[G-S10f] %s: %s\n", what, line)
	return worst, nil
}

func dsGrade(t *testing.T, what string, w float64) {
	t.Helper()
	switch {
	case w < 0.999:
		t.Errorf("%s: worst cosine %.9f under 0.999 (G-S10f FAIL)", what, w)
	case w < 0.9999:
		t.Errorf("%s: worst cosine %.9f in 0.999-0.9999 (G-S10f ambiguous, parked)", what, w)
	}
}

func TestGridVisionCUDA_deepstackTiny(t *testing.T) {
	enc, err := vision.LoadQwen3VisionEncoder("../testdata/qwen3vl-vision-tiny", false)
	if err != nil {
		t.Skipf("no tiny DeepStack tower: %v", err)
	}
	w, err := enc.Weights()
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(3))
	for _, b := range w.Blocks { // the tiny checkpoint's norms are all ones, which would hide a dropped or swapped norm
		gridRandomise(rng, b.Norm1W, b.Norm1B, b.Norm2W, b.Norm2B)
	}
	acc, err := newQwen3Tower(enc)
	if err != nil {
		t.Skipf("no CUDA device for the Qwen3-VL tower: %v", err)
	}
	t.Cleanup(func() { _ = acc.Close() })
	c := enc.Cfg
	pd := c.InChannels * c.TemporalPatchSize * c.PatchSize * c.PatchSize
	prng := rand.New(rand.NewSource(17))
	grids := [][][3]int{{{1, 4, 6}}, {{1, 6, 4}, {1, 4, 4}}}
	for _, grid := range grids {
		px := gridPixels(prng, grid, pd)
		worst, err := dsCompare(t, fmt.Sprintf("tiny %v", grid), enc, acc, px, grid)
		if err != nil {
			t.Fatal(err)
		}
		dsGrade(t, fmt.Sprintf("tiny %v", grid), worst)
	}
	// The planted defects, alone. The tiny tower has three blocks with DeepStack at 0 and 1, so taps a block late (1 and 2) stay inside it and the defect is measured as a number, not refused.
	for _, d := range []struct {
		name string
		set  func(on bool)
	}{
		{"taps one block late", func(on bool) { acc.planted.tapShift = map[bool]int{true: 1}[on] }},
		{"taps in reverse order", func(on bool) { acc.planted.tapReverse = on }},
	} {
		d.set(true)
		px := gridPixels(rand.New(rand.NewSource(17)), grids[0], pd)
		wd, err := dsCompare(t, "tiny, "+d.name, enc, acc, px, grids[0])
		d.set(false)
		switch {
		case err != nil:
			t.Errorf("planted defect (%s) was refused (%v), not measured: is the tiny tower deep enough?", d.name, err)
		case wd >= 0.9999:
			t.Errorf("planted defect (%s) left the bar green (worst %.9f): the fixture cannot see it", d.name, wd)
		default:
			fmt.Fprintf(os.Stderr, "[G-S10f] planted %s: worst %.6f, red\n", d.name, wd)
		}
	}
	// And the tower still refuses a tap past its last block, by name.
	if _, _, err := acc.HiddenTaps(gridPixels(prng, grids[0], pd), grids[0], []int{len(acc.blocks)}); err == nil {
		t.Error("a tap past the last block was not refused")
	}
}

func TestGridVisionCUDA_deepstackReal(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", "qwen3-vl-2b-instruct")
	if strings.HasPrefix(dir, "/Volumes/") || strings.HasPrefix(dir, "/srv/models") {
		t.Fatalf("%s is the archive", dir)
	}
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no %s", dir)
	}
	pp, err := multimodal.LoadQwen3PreprocessConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	if limit := 1024 * pp.MergeSize * pp.MergeSize * pp.PatchSize * pp.PatchSize; pp.MaxPixels > limit { // serve's cap
		pp.MaxPixels = limit
	}
	enc, err := vision.LoadQwen3VisionEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	acc, err := newQwen3Tower(enc)
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	t.Cleanup(func() { _ = acc.Close() })
	for _, img := range []string{"gemma3_preprocess_image.png", "qwen25vl_preprocess_image.png", "glm_ocr/formula.png", "glm_ocr/table.png"} {
		data, err := os.ReadFile(filepath.Join("../testdata", img))
		if err != nil {
			t.Fatal(err)
		}
		px, grid, err := multimodal.QwenPreprocess(data, pp)
		if err != nil {
			t.Fatal(err)
		}
		g := [][3]int{grid}
		// The speed split, exploratory: the CUDA path (device blocks plus aikit's host tails) alone, then the CPU reference alone. The night record is the interleaved one.
		t0 := time.Now()
		if _, _, err := multimodal.Qwen3TowerFeaturesDeepstack(enc, acc, px, g); err != nil {
			t.Fatal(err)
		}
		cuda := time.Since(t0)
		t1 := time.Now()
		if _, _, err := enc.ForwardDeepstack(px, g); err != nil {
			t.Fatal(err)
		}
		cpu := time.Since(t1)
		t2 := time.Now()
		if _, _, err := acc.HiddenTaps(px, g, enc.Cfg.DeepstackVisualIndexes); err != nil {
			t.Fatal(err)
		}
		dev := time.Since(t2)
		fmt.Fprintf(os.Stderr, "[G-S10f] %s grid %v: CUDA tower with host tails %s (device blocks alone %s), CPU tower %s (exploratory, single samples)\n",
			img, grid, cuda.Round(time.Millisecond), dev.Round(time.Millisecond), cpu.Round(time.Millisecond))
		w, err := dsCompare(t, fmt.Sprintf("%s grid %v", img, grid), enc, acc, px, g)
		if err != nil {
			t.Fatal(err)
		}
		dsGrade(t, img, w)
		if img == "qwen25vl_preprocess_image.png" { // the planted defects as numbers: 24 blocks leave room to shift every tap
			acc.planted.tapShift = 1
			wd, err := dsCompare(t, img+", taps one block late", enc, acc, px, g)
			acc.planted.tapShift = 0
			if err != nil || wd >= 0.9999 {
				t.Errorf("planted taps one block late: worst %.9f, err %v (must be red with a number)", wd, err)
			}
		}
	}
}
