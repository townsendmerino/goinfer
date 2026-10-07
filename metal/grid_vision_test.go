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

// G-S2b and G-S2c of docs/tasks/task-multimodal-support-2026-10.md (S2), on the committed tiny towers: the Metal tower plus
// aikit's FinishHidden against aikit's CPU Forward, every merged token at cosine >= 0.9999, at three grid sizes and a
// two-image batch (two attention segments); and each planted defect, alone, below the bar. The norms are randomised
// first through the export's aliasing slices: the tiny checkpoints' norms are all ones, which would hide a dropped or
// swapped norm.

var gvGrids = [][][3]int{{{1, 4, 4}}, {{1, 6, 8}}, {{1, 10, 6}}, {{1, 4, 4}, {1, 6, 4}}}

func gvRandomise(rng *rand.Rand, xs ...[]float32) {
	for _, x := range xs {
		for i := range x {
			x[i] = float32(1 + 0.3*rng.NormFloat64())
		}
	}
}

func gvPixels(rng *rand.Rand, grid [][3]int, patchDim int) []float32 {
	n := 0
	for _, g := range grid {
		n += g[0] * g[1] * g[2]
	}
	px := make([]float32, n*patchDim)
	for i := range px {
		px[i] = float32(rng.NormFloat64())
	}
	return px
}

// gvWorst is the worst per-token cosine between two [n, width] outputs.
func gvWorst(t *testing.T, got, want []float32, width int) float64 {
	t.Helper()
	if len(got) != len(want) || len(got)%width != 0 {
		t.Fatalf("%d values against %d (width %d)", len(got), len(want), width)
	}
	worst := 1.0
	for r := 0; r < len(got)/width; r++ {
		var dot, na, nb float64
		for j := range width {
			a, b := float64(got[r*width+j]), float64(want[r*width+j])
			dot, na, nb = dot+a*b, na+a*a, nb+b*b
		}
		c := dot / math.Sqrt(na*nb)
		if math.IsNaN(c) {
			c = -1
		}
		worst = math.Min(worst, c)
	}
	return worst
}

type gvCase struct {
	name     string
	patchDim int
	out      int
	cpu      func(px []float32, g [][3]int) ([]float32, error)
	dev      func(acc multimodal.GridTowerAccelerator, px []float32, g [][3]int) ([]float32, error)
	acc      *gridVAccel
	defects  []string
}

func gvQwen3(t *testing.T) gvCase {
	enc, err := vision.LoadQwen3VisionEncoder("../testdata/qwen35vl-tiny", false)
	if err != nil {
		t.Skipf("no Qwen3.5 tiny tower: %v", err)
	}
	w, err := enc.Weights()
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(3))
	for _, b := range w.Blocks {
		gvRandomise(rng, b.Norm1W, b.Norm1B, b.Norm2W, b.Norm2B)
	}
	acc, err := newQwen3VAccel(enc)
	if err != nil {
		t.Fatal(err)
	}
	c := enc.Cfg
	return gvCase{name: "Qwen3.5", patchDim: c.InChannels * c.TemporalPatchSize * c.PatchSize * c.PatchSize, out: c.OutHiddenSize,
		cpu: enc.Forward,
		dev: func(a multimodal.GridTowerAccelerator, px []float32, g [][3]int) ([]float32, error) {
			return multimodal.Qwen3TowerFeatures(enc, a, px, g)
		}, acc: acc, defects: []string{"noScale", "swapRope", "transposePos", "noPatchBias"}}
}

func gvGlm(t *testing.T) gvCase {
	enc, err := vision.LoadGlmOcrVisionEncoder("../testdata/glm-ocr-vision-tiny", false)
	if err != nil {
		t.Skipf("no GLM-OCR tiny tower: %v", err)
	}
	w, err := enc.Weights()
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(5))
	for _, b := range w.Blocks {
		gvRandomise(rng, b.Norm1W, b.Norm2W, b.QNorm, b.KNorm)
	}
	acc, err := newGlmOcrVAccel(enc)
	if err != nil {
		t.Fatal(err)
	}
	c := enc.Cfg
	return gvCase{name: "GLM-OCR", patchDim: c.InChannels * c.TemporalPatchSize * c.PatchSize * c.PatchSize, out: c.OutHiddenSize,
		cpu: enc.Forward,
		dev: func(a multimodal.GridTowerAccelerator, px []float32, g [][3]int) ([]float32, error) {
			return multimodal.GlmOcrTowerFeatures(enc, a, px, g)
		}, acc: acc, defects: []string{"noScale", "swapRope", "noQKNorm", "noPatchBias"}}
}

// worstOver runs every grid and returns the worst token cosine.
func (c gvCase) worstOver(t *testing.T, seed int64) float64 {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	worst := 1.0
	for _, g := range gvGrids {
		px := gvPixels(rng, g, c.patchDim)
		want, err := c.cpu(px, g)
		if err != nil {
			t.Fatal(err)
		}
		got, err := c.dev(c.acc, px, g)
		if err != nil {
			t.Fatal(err)
		}
		worst = math.Min(worst, gvWorst(t, got, want, c.out))
	}
	return worst
}

func setDefect(d *gvDefect, name string, on bool) {
	switch name {
	case "noScale":
		d.noScale = on
	case "swapRope":
		d.swapRope = on
	case "transposePos":
		d.transposePos = on
	case "noQKNorm":
		d.noQKNorm = on
	case "noPatchBias":
		d.noPatchBias = on
	}
}

func TestGridVisionMetal_matchesCPU(t *testing.T) {
	for name, mk := range map[string]func(*testing.T) gvCase{"qwen3": gvQwen3, "glm-ocr": gvGlm} {
		t.Run(name, func(t *testing.T) { gvCheck(t, mk(t)) })
	}
}

func gvCheck(t *testing.T, c gvCase) {
	{
		w := c.worstOver(t, 1)
		t.Logf("%s: worst merged-token cosine %.9f over %d grid sets", c.name, w, len(gvGrids))
		if w < 0.9999 {
			t.Errorf("%s: worst token cosine %.9f under 0.9999 (G-S2b)", c.name, w)
		}
		for _, d := range c.defects {
			setDefect(&c.acc.planted, d, true)
			wd := c.worstOver(t, 1)
			setDefect(&c.acc.planted, d, false)
			t.Logf("%s planted %s: worst %.6f", c.name, d, wd)
			if wd >= 0.9999 {
				t.Errorf("%s: planted defect %s left the bar green (worst %.9f): the fixture cannot see it (G-S2c)", c.name, d, wd)
			}
		}
	}
}

// TestGridVisionMetal_real is G-S2b on the real towers (~/models/qwen3.5-0.8b, ~/models/glm-ocr; never the archive),
// through serve's own preprocessing and caps: the four images F2a uses for both towers, plus GLM-OCR's three O3
// documents. Every merged token at cosine >= 0.9999 against aikit's CPU Forward; a worst token in 0.999-0.9999 is
// ambiguous (parked). Times are exploratory. Heavy: GOINFER_HEAVY_TESTS=1.
func TestGridVisionMetal_real(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1 (loads the real towers)")
	}
	home, _ := os.UserHomeDir()
	images := []string{"gemma3_preprocess_image.png", "qwen25vl_preprocess_image.png", "glm_ocr/formula.png", "glm_ocr/table.png"}
	type fam struct {
		name, dir string
		extra     []string
		run       func(t *testing.T, dir string) (pp multimodal.QwenPreprocessConfig, cpu, dev func([]float32, [][3]int) ([]float32, error), out int)
	}
	fams := []fam{
		{"qwen3", filepath.Join(home, "models", "qwen3.5-0.8b"), nil,
			func(t *testing.T, dir string) (multimodal.QwenPreprocessConfig, func([]float32, [][3]int) ([]float32, error), func([]float32, [][3]int) ([]float32, error), int) {
				pp, err := multimodal.LoadQwen3PreprocessConfig(dir)
				if err != nil {
					t.Fatal(err)
				}
				if limit := 1024 * pp.MergeSize * pp.MergeSize * pp.PatchSize * pp.PatchSize; pp.MaxPixels > limit { // serve's qwen3MaxImageTokens
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
				return pp, enc.Forward, func(px []float32, g [][3]int) ([]float32, error) {
					return multimodal.Qwen3TowerFeatures(enc, acc, px, g)
				}, enc.Cfg.OutHiddenSize
			}},
		{"glm-ocr", filepath.Join(home, "models", "glm-ocr"), []string{"glm_ocr/invoice.png"},
			func(t *testing.T, dir string) (multimodal.QwenPreprocessConfig, func([]float32, [][3]int) ([]float32, error), func([]float32, [][3]int) ([]float32, error), int) {
				pp, err := multimodal.LoadGlmOcrPreprocessConfig(dir) // serve's default: the model's own pixel ceiling
				if err != nil {
					t.Fatal(err)
				}
				enc, err := vision.LoadGlmOcrVisionEncoder(dir, false)
				if err != nil {
					t.Fatal(err)
				}
				acc, err := newGlmOcrVAccel(enc)
				if err != nil {
					t.Fatal(err)
				}
				return pp, enc.Forward, func(px []float32, g [][3]int) ([]float32, error) {
					return multimodal.GlmOcrTowerFeatures(enc, acc, px, g)
				}, enc.Cfg.OutHiddenSize
			}},
	}
	for _, f := range fams {
		t.Run(f.name, func(t *testing.T) {
			if strings.HasPrefix(f.dir, "/Volumes/") || strings.HasPrefix(f.dir, "/srv/models") {
				t.Fatalf("%s is on the archive (CLAUDE.md)", f.dir)
			}
			if _, err := os.Stat(f.dir); err != nil {
				t.Skipf("no %s: %v", f.dir, err)
			}
			pp, cpu, dev, out := f.run(t, f.dir)
			for _, img := range append(append([]string(nil), images...), f.extra...) {
				data, err := os.ReadFile(filepath.Join("../testdata", img))
				if err != nil {
					t.Fatal(err)
				}
				px, grid, err := multimodal.QwenPreprocess(data, pp)
				if err != nil {
					t.Fatal(err)
				}
				g := [][3]int{grid}
				t0 := time.Now()
				want, err := cpu(px, g)
				tc := time.Since(t0)
				if err != nil {
					t.Fatal(err)
				}
				t0 = time.Now()
				got, err := dev(px, g)
				tm := time.Since(t0)
				if err != nil {
					t.Fatal(err)
				}
				w := gvWorst(t, got, want, out)
				fmt.Fprintf(os.Stderr, "[S2 real %s] %-30s grid %v (%d merged): worst token cosine %.9f; tower Metal %s, CPU %s (exploratory)\n",
					f.name, img, grid, len(want)/out, w, tm.Round(time.Millisecond), tc.Round(time.Millisecond))
				switch {
				case w < 0.999:
					t.Errorf("%s %s: worst token cosine %.9f under 0.999 (G-S2b FAIL)", f.name, img, w)
				case w < 0.9999:
					t.Errorf("%s %s: worst token cosine %.9f in 0.999-0.9999 (G-S2b ambiguous, parked)", f.name, img, w)
				}
			}
		})
	}
}
