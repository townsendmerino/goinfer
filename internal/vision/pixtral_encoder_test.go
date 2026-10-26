package vision

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"
)

// The Pixtral tower (Mistral 3 / Ministral 3) against HF's PixtralVisionModel (transformers 5.15.0, float32, eager), on a
// tiny random tower of the real structure pinned by goinfer scripts/pin_pixtral_vision.py (S10, G-S10m-b's tiny half,
// goinfer docs/tasks/task-multimodal-support-2026-10.md): every norm randomised, two non-square images of different sizes
// in one call. Each stage at relative max|diff| <= 5e-6 (the GLM-OCR and Qwen3.5+ tiny towers' bar), and each image alone.
// Planted defects, each must miss: the column frequencies from the even list (Qwen's rule), the row and column halves
// swapped, and the block-diagonal mask dropped.

type pixtralGolden struct {
	Sizes  [][2]int             `json:"sizes"`
	Grids  [][2]int             `json:"grids"`
	Images [][]float32          `json:"images"`
	Hidden int                  `json:"hidden"`
	Stages map[string][]float32 `json:"stages"`
	Alone  [][]float32          `json:"alone"`
}

func loadPixtralTiny(t *testing.T) (*PixtralVisionEncoder, pixtralGolden, []float32) {
	t.Helper()
	raw, err := os.ReadFile("testdata/pixtral_vision_golden.json")
	if err != nil {
		t.Skipf("no golden: %v", err)
	}
	var g pixtralGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	enc, err := LoadPixtralVisionEncoder("testdata/pixtral-vision-tiny", false)
	if err != nil {
		t.Fatal(err)
	}
	var patches []float32
	for i, im := range g.Images {
		p, grid, err := PixtralPatchify(im, 3, g.Sizes[i][0], g.Sizes[i][1], enc.Cfg.PatchSize)
		if err != nil {
			t.Fatal(err)
		}
		if grid != g.Grids[i] {
			t.Fatalf("image %d: grid %v, HF %v", i, grid, g.Grids[i])
		}
		patches = append(patches, p...)
	}
	return enc, g, patches
}

func TestPixtralVisionEncoder_matchesHF(t *testing.T) {
	enc, g, patches := loadPixtralTiny(t)
	const relBar = 5e-6
	stages, err := enc.ForwardStages(patches, g.Grids)
	if err != nil {
		t.Fatal(err)
	}
	names := []string{"s_conv", "s_lnpre"}
	for i := range enc.Cfg.NumHiddenLayers {
		names = append(names, fmt.Sprintf("s_layer%d", i))
	}
	if len(stages) != len(names) {
		t.Fatalf("%d stages, want %d", len(stages), len(names))
	}
	for i, n := range names {
		gateStage(t, n, stages[i], g.Stages[n], g.Hidden, 0.999999, relBar)
	}
	// Each image alone: its own block, so the two-image call's rows must equal it too.
	for i := range g.Images {
		p, _, _ := PixtralPatchify(g.Images[i], 3, g.Sizes[i][0], g.Sizes[i][1], enc.Cfg.PatchSize)
		got, err := enc.Forward(p, g.Grids[i:i+1])
		if err != nil {
			t.Fatal(err)
		}
		gateStage(t, fmt.Sprintf("image %d alone", i), got, g.Alone[i], g.Hidden, 0.999999, relBar)
	}
}

func TestPixtralVisionEncoder_plantedDefects(t *testing.T) {
	enc, g, patches := loadPixtralTiny(t)
	want := g.Stages[fmt.Sprintf("s_layer%d", enc.Cfg.NumHiddenLayers-1)]
	for _, d := range []struct {
		name string
		set  func(on bool)
	}{
		{"column frequencies from the even list", func(on bool) {
			pixtralColFreqOffset = 1
			if on {
				pixtralColFreqOffset = 0
			}
		}},
		{"row and column halves swapped", func(on bool) { pixtralSwapRowCol = on }},
		{"the block mask dropped", func(on bool) { pixtralNoBlockMask = on }},
	} {
		d.set(true)
		got, err := enc.Forward(patches, g.Grids)
		d.set(false)
		if err != nil {
			t.Fatal(err)
		}
		worst, _, maxAbs := rowCosines(got, want, g.Hidden)
		t.Logf("planted %s: worst-row cosine %.6f, max|diff| %.3g", d.name, worst, maxAbs)
		if worst >= 0.9999 {
			t.Errorf("BLIND: %s still reads worst-row cosine %.6f", d.name, worst)
		}
	}
}
