package multimodal

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/vision"
)

// S10, LFM2.5-VL, G-S10l-a and G-S10l-b (docs/tasks/task-multimodal-support-2026-10.md, registered 2026-10-09 before any
// code), against transformers' references from scripts/pin_lfm2vl_real.py (nobara; GOINFER_LFM2VL_REAL_ARTIFACTS,
// default ~/goinfer-logs/lfm2vl-real) and the real checkpoint (GOINFER_LFM2VL, default ~/models/lfm25-vl-1.6b).
//
//   - a: Lfm2VLPreprocess on the four F2a images and three layout cases: the tiles and their patch grids, the <image>
//     runs, and each tile's patch rows within one 8-bit level (0.0078 after normalising).
//   - b: from HF's own patches, every tower and projector stage of every tile at worst-row cosine >= 0.9999 (0.999-0.9999
//     ambiguous, parked). The planted defects are the tiny tests' (aikit's tower, TestLfm2VLProjector_matchesHF).
//
//	GOINFER_HEAVY_TESTS=1 go test ./multimodal/ -run TestLfm2VLReal -v -timeout 30m

type lfm2vlRealGolden struct {
	A []struct {
		Image, Name string
		Src         [2]int
		Tiles       [][2]int
		Runs        []int
	} `json:"a"`
	B []struct {
		Image, Name string
		Tiles       [][2]int
		Stages      []string
	} `json:"b"`
}

func lfm2vlRealArtifacts(t *testing.T) (art, model string, g lfm2vlRealGolden) {
	t.Helper()
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	art = os.Getenv("GOINFER_LFM2VL_REAL_ARTIFACTS")
	if art == "" {
		art = filepath.Join(home, "goinfer-logs", "lfm2vl-real")
	}
	model = os.Getenv("GOINFER_LFM2VL")
	if model == "" {
		model = filepath.Join(home, "models", "lfm25-vl-1.6b")
	}
	if strings.HasPrefix(model, "/Volumes/") || strings.HasPrefix(model, "/srv/models") {
		t.Fatalf("%s is the archive (CLAUDE.md)", model)
	}
	raw, err := os.ReadFile(filepath.Join(art, "golden.json"))
	if err != nil {
		t.Skipf("no artifacts: %v (scripts/pin_lfm2vl_real.py)", err)
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return art, model, g
}

func TestLfm2VLReal_preprocess(t *testing.T) {
	art, model, g := lfm2vlRealArtifacts(t)
	cfg, err := LoadLfm2VLPreprocessConfig(model)
	if err != nil {
		t.Fatal(err)
	}
	level := 1.0 / 255 / float64(min(cfg.Std[0], cfg.Std[1], cfg.Std[2])) // one 8-bit level after normalising
	if len(g.A) != 7 {
		t.Fatalf("%d preprocessing cases, want the four images and the three layout cases", len(g.A))
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
		l, err := Lfm2VLPreprocess(data, cfg)
		if err != nil {
			t.Fatal(err)
		}
		var grids [][2]int
		for _, tile := range l.Tiles {
			grids = append(grids, [2]int{tile.H / cfg.PatchSize, tile.W / cfg.PatchSize})
		}
		if !slices.Equal(grids, c.Tiles) || !slices.Equal(l.Tokens(cfg), c.Runs) {
			t.Errorf("%s: tiles %v runs %v; HF tiles %v runs %v", c.Image, grids, l.Tokens(cfg), c.Tiles, c.Runs)
			continue
		}
		var maxAbs, sum float64
		var count int
		for k, tile := range l.Tiles {
			p, _, err := vision.PatchifyNaFlex(tile.CHW, 3, tile.H, tile.W, cfg.PatchSize)
			if err != nil {
				t.Fatal(err)
			}
			hf := readF32File(t, filepath.Join(art, fmt.Sprintf("%s.tile%d.patches.f32", c.Name, k)))
			if len(hf) != len(p) {
				t.Fatalf("%s tile %d: %d patch values, HF %d", c.Image, k, len(p), len(hf))
			}
			for i := range hf {
				d := math.Abs(float64(p[i] - hf[i]))
				maxAbs = math.Max(maxAbs, d)
				sum += d
			}
			count += len(hf)
		}
		fmt.Fprintf(os.Stderr, "[G-S10l-a] %-34s %4dx%-4d -> %d tile(s) %v (thumbnail/single %v), runs %v; pixels max|diff| %.5f (%.2f levels), mean %.2e\n",
			c.Image, c.Src[1], c.Src[0], len(l.Tiles), [2]int{l.Rows, l.Cols}, grids[len(grids)-1], c.Runs, maxAbs, maxAbs/level, sum/float64(count))
		if maxAbs > level*1.001 {
			t.Errorf("%s: pixels differ by %.5f, more than one 8-bit level (%.5f)", c.Image, maxAbs, level)
		}
	}
}

func TestLfm2VLReal_towerAndProjector(t *testing.T) {
	art, model, g := lfm2vlRealArtifacts(t)
	enc, err := vision.LoadSiglip2NaFlexEncoder(model, false)
	if err != nil {
		t.Fatal(err)
	}
	proj, err := LoadLfm2VLProjector(model)
	if err != nil {
		t.Fatal(err)
	}
	const bar, parked = 0.9999, 0.999
	if len(g.B) != 4 {
		t.Fatalf("%d images, want four", len(g.B))
	}
	for _, c := range g.B {
		worst, worstAt := 1.0, ""
		for k, grid := range c.Tiles {
			patches := readF32File(t, filepath.Join(art, fmt.Sprintf("%s.tile%d.patches.f32", c.Name, k)))
			ts, err := enc.ForwardStages(patches, grid)
			if err != nil {
				t.Fatal(err)
			}
			last := ts[len(ts)-1]
			got := map[string][]float32{"embeddings": ts[0], "post_layernorm": last}
			for L := range enc.Cfg.NumHiddenLayers {
				got[fmt.Sprintf("block%d", L)] = ts[1+L]
			}
			un, l1, l2, err := proj.forwardStages(last, grid)
			if err != nil {
				t.Fatal(err)
			}
			got["proj_unshuffle"], got["proj_linear_1"], got["proj_linear_2"] = un, l1, l2
			if len(got) != len(c.Stages) {
				t.Fatalf("%s tile %d: %d goinfer stages, HF %d", c.Image, k, len(got), len(c.Stages))
			}
			for _, s := range c.Stages {
				hf := readF32File(t, filepath.Join(art, fmt.Sprintf("%s.tile%d.%s.f32", c.Name, k, s)))
				v := got[s]
				if len(v) != len(hf) {
					t.Fatalf("%s tile %d %s: %d values, HF %d", c.Image, k, s, len(v), len(hf))
				}
				rows := grid[0] * grid[1]
				if strings.HasPrefix(s, "proj_") {
					rows /= 4
				}
				cos, maxAbs := cosineRows(v, hf, rows)
				if cos < worst {
					worst, worstAt = cos, fmt.Sprintf("tile %d %s", k, s)
				}
				if s == "embeddings" || s == "post_layernorm" || strings.HasPrefix(s, "proj_") {
					fmt.Fprintf(os.Stderr, "[G-S10l-b] %-30s tile %d %v %-15s worst row cosine %.9f  max|diff| %.3e\n", c.Image, k, grid, s, cos, maxAbs)
				}
			}
		}
		switch {
		case worst >= bar:
			fmt.Fprintf(os.Stderr, "[G-S10l-b] %s: PASS, worst %.9f (%s) over %d tile(s)\n", c.Image, worst, worstAt, len(c.Tiles))
		case worst >= parked:
			t.Errorf("%s: worst %.9f at %s, in the ambiguous band [%.3f, %.4f): parked", c.Image, worst, worstAt, parked, bar)
		default:
			t.Errorf("%s: worst %.9f at %s, under %.3f: FAIL", c.Image, worst, worstAt, parked)
		}
	}
}
