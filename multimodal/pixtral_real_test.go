package multimodal

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/vision"
)

// S10, Ministral 3 (Pixtral), G-S10m-a and G-S10m-b (docs/tasks/task-multimodal-support-2026-10.md), against transformers' references from scripts/pin_pixtral_real.py (nobara;
// GOINFER_PIXTRAL_REAL_ARTIFACTS, default ~/goinfer-logs/pixtral-real) and the real checkpoint (GOINFER_MINISTRAL3,
// default ~/models/ministral3-3b-bf16).
//
//   - a: PixtralPreprocess on the four F2a images and the two layout cases: the size equal, the pixels within one 8-bit
//     level (HF's fast processor resizes uint8 to uint8, then normalises), the [IMG]/[IMG_BREAK]/[IMG_END] counts equal.
//   - b: from HF's own pixels, every tower and projector stage at worst-row cosine >= 0.9999 on the four images
//     (0.999-0.9999 ambiguous, parked); and two images in one call equal to each image alone (and to HF's two-image
//     call) at the same bar. The planted defects are the tiny tests' (aikit's tower, TestPixtralProjector_*).
//
//	GOINFER_HEAVY_TESTS=1 go test ./multimodal/ -run TestPixtralReal -v -timeout 30m

type pixtralRealGolden struct {
	Transformers string `json:"transformers"`
	A            []struct {
		Image, Name   string
		Src, Size     [2]int
		Img, Brk, End int
	} `json:"a"`
	B []struct {
		Image, Name string
		Size        [2]int
		Stages      []string
		Units       int
	} `json:"b"`
	Two struct {
		Images []string
		Sizes  [][2]int
	} `json:"two"`
}

func pixtralRealArtifacts(t *testing.T) (art, model string, g pixtralRealGolden) {
	t.Helper()
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	art = os.Getenv("GOINFER_PIXTRAL_REAL_ARTIFACTS")
	if art == "" {
		art = filepath.Join(home, "goinfer-logs", "pixtral-real")
	}
	model = os.Getenv("GOINFER_MINISTRAL3")
	if model == "" {
		model = filepath.Join(home, "models", "ministral3-3b-bf16")
	}
	if strings.HasPrefix(model, "/Volumes/") || strings.HasPrefix(model, "/srv/models") {
		t.Fatalf("%s is the archive (CLAUDE.md)", model)
	}
	raw, err := os.ReadFile(filepath.Join(art, "golden.json"))
	if err != nil {
		t.Skipf("no artifacts: %v (scripts/pin_pixtral_real.py)", err)
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return art, model, g
}

func readF32File(t *testing.T, path string) []float32 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v
}

func TestPixtralReal_preprocess(t *testing.T) {
	art, model, g := pixtralRealArtifacts(t)
	cfg, err := LoadPixtralPreprocessConfig(model)
	if err != nil {
		t.Fatal(err)
	}
	level := 1.0 / 255 / float64(min(cfg.Std[0], cfg.Std[1], cfg.Std[2])) // one 8-bit level after normalising
	if len(g.A) != 6 {
		t.Fatalf("%d preprocessing cases, want the four images and the two layout cases", len(g.A))
	}
	for _, c := range g.A {
		path := filepath.Join("../testdata", c.Image)
		if _, err := os.Stat(path); err != nil { // the made-up layout cases live beside the references
			path = filepath.Join(art, c.Image)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		chw, h, w, err := PixtralPreprocess(data, cfg)
		if err != nil {
			t.Fatal(err)
		}
		if [2]int{h, w} != c.Size {
			t.Errorf("%s: size %dx%d, HF %dx%d", c.Image, h, w, c.Size[0], c.Size[1])
			continue
		}
		hf := readF32File(t, filepath.Join(art, c.Name+".pixels.f32"))
		var maxAbs, sum float64
		for i := range hf {
			d := math.Abs(float64(chw[i] - hf[i]))
			maxAbs = math.Max(maxAbs, d)
			sum += d
		}
		f := cfg.PatchSize * cfg.MergeSize
		rows, cols := h/f, w/f
		img, brk := strings.Count(PixtralImageBlock(rows, cols), PixtralImageToken), strings.Count(PixtralImageBlock(rows, cols), PixtralImageBreak)
		fmt.Fprintf(os.Stderr, "[G-S10m-a] %-34s %4dx%-4d -> %4dx%-4d  pixels max|diff| %.5f (%.2f levels), mean %.2e  [IMG] %d (HF %d), breaks %d (HF %d), end 1 (HF %d)\n",
			c.Image, c.Src[1], c.Src[0], w, h, maxAbs, maxAbs/level, sum/float64(len(hf)), img, c.Img, brk, c.Brk, c.End)
		if maxAbs > level*1.001 {
			t.Errorf("%s: pixels differ by %.5f, more than one 8-bit level (%.5f)", c.Image, maxAbs, level)
		}
		if img != c.Img || brk != c.Brk || c.End != 1 {
			t.Errorf("%s: block has %d [IMG] and %d breaks, HF's ids %d, %d (end %d)", c.Image, img, brk, c.Img, c.Brk, c.End)
		}
	}
}

func TestPixtralReal_towerAndProjector(t *testing.T) {
	art, model, g := pixtralRealArtifacts(t)
	enc, err := vision.LoadPixtralVisionEncoder(model, false)
	if err != nil {
		t.Fatal(err)
	}
	proj, err := LoadPixtralProjector(model)
	if err != nil {
		t.Fatal(err)
	}
	const bar, parked = 0.9999, 0.999
	patch := enc.Cfg.PatchSize
	single := map[string][]float32{}
	run := func(name string, size [2]int) ([][]float32, [][]float32, [2]int) {
		px := readF32File(t, filepath.Join(art, name+".pixels.f32"))
		patches, grid, err := vision.PixtralPatchify(px, 3, size[0], size[1], patch)
		if err != nil {
			t.Fatal(err)
		}
		ts, err := enc.ForwardStages(patches, [][2]int{grid})
		if err != nil {
			t.Fatal(err)
		}
		ps, err := proj.ForwardStages(ts[len(ts)-1], [][2]int{grid})
		if err != nil {
			t.Fatal(err)
		}
		return ts, ps, grid
	}
	if len(g.B) != 4 {
		t.Fatalf("%d images, want four", len(g.B))
	}
	for _, c := range g.B {
		ts, ps, grid := run(c.Name, c.Size)
		got := map[string][]float32{"conv": ts[0], "lnpre": ts[1], "proj_norm": ps[0], "proj_patch_merger": ps[1], "proj_linear_1": ps[2], "proj_linear_2": ps[3]}
		for L := range enc.Cfg.NumHiddenLayers {
			got[fmt.Sprintf("block%d", L)] = ts[2+L]
		}
		if len(got) != len(c.Stages) {
			t.Fatalf("%s: %d goinfer stages, HF %d", c.Image, len(got), len(c.Stages))
		}
		worst, worstAt := 1.0, ""
		for _, s := range c.Stages {
			hf := readF32File(t, filepath.Join(art, c.Name+"."+s+".f32"))
			v, ok := got[s]
			if !ok || len(v) != len(hf) {
				t.Fatalf("%s %s: %d values, HF %d", c.Image, s, len(v), len(hf))
			}
			rows := grid[0] * grid[1]
			if strings.HasPrefix(s, "proj_") && s != "proj_norm" {
				rows = c.Units
			}
			cos, maxAbs := cosineRows(v, hf, rows)
			fmt.Fprintf(os.Stderr, "[G-S10m-b] %-34s %-18s worst row cosine %.9f  max|diff| %.3e\n", c.Image, s, cos, maxAbs)
			if cos < worst {
				worst, worstAt = cos, s
			}
		}
		switch {
		case worst >= bar:
			fmt.Fprintf(os.Stderr, "[G-S10m-b] %s: PASS, worst %.9f (%s)\n", c.Image, worst, worstAt)
		case worst >= parked:
			t.Errorf("%s: worst %.9f at %s, in the ambiguous band [%.3f, %.4f): parked", c.Image, worst, worstAt, parked, bar)
		default:
			t.Errorf("%s: worst %.9f at %s, under %.3f: FAIL", c.Image, worst, worstAt, parked)
		}
		single[c.Image] = ps[3]
	}

	// Two images in one call: each equal to the image alone, and to HF's two-image call.
	var patches []float32
	var grids [][2]int
	for k, img := range g.Two.Images {
		name := strings.ReplaceAll(strings.TrimSuffix(img, ".png"), "/", "_")
		px := readF32File(t, filepath.Join(art, name+".pixels.f32"))
		p, grid, err := vision.PixtralPatchify(px, 3, g.Two.Sizes[k][0], g.Two.Sizes[k][1], patch)
		if err != nil {
			t.Fatal(err)
		}
		patches, grids = append(patches, p...), append(grids, grid)
	}
	hid, err := enc.Forward(patches, grids)
	if err != nil {
		t.Fatal(err)
	}
	feats, err := proj.Forward(hid, grids)
	if err != nil {
		t.Fatal(err)
	}
	th, off := proj.TextHidden(), 0
	for k, img := range g.Two.Images {
		units := grids[k][0] / 2 * grids[k][1] / 2
		mine := feats[off*th : (off+units)*th]
		off += units
		cs, ms := cosineRows(mine, single[img], units)
		ch, mh := cosineRows(mine, readF32File(t, filepath.Join(art, fmt.Sprintf("two.%d.f32", k))), units)
		fmt.Fprintf(os.Stderr, "[G-S10m-b] two images, %s: against itself alone %.9f (max|diff| %.3e), against HF's two-image call %.9f (%.3e)\n", img, cs, ms, ch, mh)
		if cs < bar || ch < bar {
			t.Errorf("two images, %s: %.9f against itself alone, %.9f against HF's: under %.4f", img, cs, ch, bar)
		}
	}
}
