package multimodal

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// S10, Ministral 3 (docs/tasks/task-multimodal-support-2026-10.md): the projector against HF's own
// Mistral3MultiModalProjector (transformers 5.15.0, float32), pinned by scripts/pin_pixtral_projector.py: random
// weights, a randomised norm, two images of different patch grids in one call. Relative max|diff| <= 5e-6, the tiny
// towers' bar. Planted: the merger laid out position-major instead of HF's channel-major unfold must miss.

type pixtralProjGolden struct {
	Grids        [][2]int             `json:"grids"`
	VisionHidden int                  `json:"vision_hidden"`
	TextHidden   int                  `json:"text_hidden"`
	Input        []float32            `json:"input"`
	Output       []float32            `json:"output"`
	Weights      map[string][]float32 `json:"weights"`
}

func loadPixtralProjGolden(t *testing.T) (pixtralProjGolden, *PixtralProjector) {
	t.Helper()
	raw, err := os.ReadFile("../testdata/pixtral_projector_golden.json")
	if err != nil {
		t.Skipf("no golden: %v", err)
	}
	var g pixtralProjGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	p := &PixtralProjector{visionHidden: g.VisionHidden, textHidden: g.TextHidden, merge: 2, eps: 1e-5,
		normW: g.Weights["norm.weight"], mergeW: g.Weights["patch_merger.merging_layer.weight"],
		l1W: g.Weights["linear_1.weight"], l2W: g.Weights["linear_2.weight"]}
	return g, p
}

func relMaxDiff(got, want []float32) float64 {
	var md, sc float64
	for i := range want {
		md = math.Max(md, math.Abs(float64(got[i]-want[i])))
		sc = math.Max(sc, math.Abs(float64(want[i])))
	}
	return md / sc
}

func TestPixtralProjector_matchesHF(t *testing.T) {
	g, p := loadPixtralProjGolden(t)
	got, err := p.Forward(g.Input, g.Grids)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(g.Output) {
		t.Fatalf("output %d values, HF %d", len(got), len(g.Output))
	}
	rel := relMaxDiff(got, g.Output)
	t.Logf("projector against HF: relative max|diff| %.3g (bar 5e-6)", rel)
	if rel > 5e-6 {
		t.Errorf("projector differs from HF: relative max|diff| %.3g > 5e-6", rel)
	}
	pixtralMergerPositionMajor = true
	bad, _ := p.Forward(g.Input, g.Grids)
	pixtralMergerPositionMajor = false
	t.Logf("planted: the merger position-major: relative max|diff| %.3g", relMaxDiff(bad, g.Output))
	if relMaxDiff(bad, g.Output) <= 5e-6 {
		t.Error("BLIND: a position-major merger still matches HF")
	}
	// The loader, where the (gitignored) tiny checkpoint is present: it must read the same weights.
	if _, err := os.Stat("../testdata/pixtral-projector-tiny/model.safetensors"); err == nil {
		lp, err := LoadPixtralProjector("../testdata/pixtral-projector-tiny")
		if err != nil {
			t.Fatal(err)
		}
		lg, _ := lp.Forward(g.Input, g.Grids)
		if r := relMaxDiff(lg, g.Output); r > 5e-6 {
			t.Errorf("the loaded projector differs from HF: %.3g", r)
		}
	}
}

func TestPixtralResizeTargetAndBlock(t *testing.T) {
	c := PixtralDefaultPreprocess()
	for _, tc := range []struct{ h, w, th, tw int }{
		{768, 1024, 784, 1036},   // the desk map's worked example, measured with HF's processor
		{2000, 3000, 1036, 1540}, // downscaled: ratio 3000/1540, then rounded up to 28
		{20, 30, 28, 56},         // upscaled to one merge unit tall
		{896, 896, 896, 896},
	} {
		if th, tw := PixtralResizeTarget(tc.h, tc.w, c); th != tc.th || tw != tc.tw {
			t.Errorf("%dx%d -> %dx%d, want %dx%d", tc.h, tc.w, th, tw, tc.th, tc.tw)
		}
	}
	if got := PixtralImageBlock(2, 3); got != "[IMG][IMG][IMG][IMG_BREAK][IMG][IMG][IMG][IMG_END]" {
		t.Errorf("block %q", got)
	}
	if got := PixtralImageBlock(1, 2); got != "[IMG][IMG][IMG_END]" {
		t.Errorf("one-row block %q", got)
	}
}
