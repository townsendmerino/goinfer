//go:build darwin

package metal

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

// G-S10e of docs/tasks/task-multimodal-support-2026-10.md (S10, registered before this ran): the Qwen3-VL tower on Metal
// with DeepStack. S2's grid tower taps the DeepStack blocks (HiddenTaps) and aikit's host tails make the merged rows and
// the sets (multimodal.Qwen3TowerFeaturesDeepstack), against aikit's CPU ForwardDeepstack: every set and the merged rows
// at worst-row cosine >= 0.9999 (0.999-0.9999 parked), on the tiny DeepStack tower (testdata/qwen3vl-vision-tiny, norms
// randomised) and on Qwen3-VL-2B's four F2a images (heavy). The planted defect, taps one block late, must go red.

func gvDeepstackCompare(t *testing.T, what string, enc *vision.Qwen3VisionEncoder, acc *gridVAccel, px []float32, grid [][3]int) (worst float64, err error) {
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
	worst = gvWorst(t, gotM, wantM, out)
	line := fmt.Sprintf("merged %.9f", worst)
	for k := range wantD {
		w := gvWorst(t, gotD[k], wantD[k], out)
		line += fmt.Sprintf(", deep%d %.9f", k, w)
		worst = math.Min(worst, w)
	}
	fmt.Fprintf(os.Stderr, "[G-S10e] %s: %s\n", what, line)
	return worst, nil
}

func gvDeepstackGrade(t *testing.T, what string, w float64) {
	t.Helper()
	switch {
	case w < 0.999:
		t.Errorf("%s: worst cosine %.9f under 0.999 (G-S10e FAIL)", what, w)
	case w < 0.9999:
		t.Errorf("%s: worst cosine %.9f in 0.999-0.9999 (G-S10e ambiguous, parked)", what, w)
	}
}

func TestGridVisionMetal_deepstackTiny(t *testing.T) {
	enc, err := vision.LoadQwen3VisionEncoder("../testdata/qwen3vl-vision-tiny", false)
	if err != nil {
		t.Skipf("no tiny DeepStack tower: %v", err)
	}
	acc, err := newQwen3VAccel(enc)
	if err != nil {
		t.Fatal(err)
	}
	c := enc.Cfg
	pd := c.InChannels * c.TemporalPatchSize * c.PatchSize * c.PatchSize
	rng := rand.New(rand.NewSource(17))
	for _, grid := range [][][3]int{{{1, 4, 6}}, {{1, 6, 4}, {1, 4, 4}}} {
		px := gvPixels(rng, grid, pd)
		w, err := gvDeepstackCompare(t, fmt.Sprintf("tiny %v", grid), enc, acc, px, grid)
		if err != nil {
			t.Fatal(err)
		}
		gvDeepstackGrade(t, fmt.Sprintf("tiny %v", grid), w)
	}
	// The planted defect: every tap a block late. On this two-block tower the last tap then falls past the end, which
	// HiddenTaps refuses: that is red too (the sets are not produced), and it is reported as such.
	acc.planted.tapShift = 1
	px := gvPixels(rng, [][3]int{{1, 4, 6}}, pd)
	w, err := gvDeepstackCompare(t, "tiny, taps one block late", enc, acc, px, [][3]int{{1, 4, 6}})
	acc.planted.tapShift = 0
	switch {
	case err != nil:
		fmt.Fprintf(os.Stderr, "[G-S10e] planted taps one block late: refused (%v), red\n", err)
	case w >= 0.9999:
		t.Errorf("planted defect (taps one block late) left the bar green (worst %.9f)", w)
	default:
		fmt.Fprintf(os.Stderr, "[G-S10e] planted taps one block late: worst %.6f, red\n", w)
	}
}

func TestGridVisionMetal_deepstackReal(t *testing.T) {
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
	acc, err := newQwen3VAccel(enc)
	if err != nil {
		t.Fatal(err)
	}
	for _, img := range []string{"gemma3_preprocess_image.png", "qwen25vl_preprocess_image.png", "glm_ocr/formula.png", "glm_ocr/table.png"} {
		data, err := os.ReadFile(filepath.Join("../testdata", img))
		if err != nil {
			t.Fatal(err)
		}
		px, grid, err := multimodal.QwenPreprocess(data, pp)
		if err != nil {
			t.Fatal(err)
		}
		t0 := time.Now()
		w, err := gvDeepstackCompare(t, fmt.Sprintf("%s grid %v", img, grid), enc, acc, px, [][3]int{grid})
		if err != nil {
			t.Fatal(err)
		}
		fmt.Fprintf(os.Stderr, "[G-S10e] %s: CPU and Metal together %s (exploratory)\n", img, time.Since(t0).Round(time.Millisecond))
		gvDeepstackGrade(t, img, w)
		if img == "qwen25vl_preprocess_image.png" { // the planted defect as a number: 24 blocks leave room to shift every tap
			acc.planted.tapShift = 1
			wd, err := gvDeepstackCompare(t, img+", taps one block late", enc, acc, px, [][3]int{grid})
			acc.planted.tapShift = 0
			if err != nil || wd >= 0.9999 {
				t.Errorf("planted taps one block late: worst %.9f, err %v (must be red with a number)", wd, err)
			}
		}
	}
}
