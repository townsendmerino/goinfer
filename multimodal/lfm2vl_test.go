package multimodal

import (
	"encoding/json"
	"math"
	"os"
	"regexp"
	"slices"
	"testing"
)

// S10, LFM2.5-VL (docs/tasks/task-multimodal-support-2026-10.md, "S10, LFM2.5-VL"): the preprocessing layout against HF's
// Lfm2VlProcessor on 21 image sizes (scripts/pin_lfm2vl_tiny.py), and the projector against HF's
// Lfm2VlMultiModalProjector at toy size. The goldens carry everything (the projector's weights too), so these run in CI.

type lfm2vlLayoutGolden struct {
	Cases []struct {
		Size        [2]int   `json:"size"`
		Rows        int      `json:"rows"`
		Cols        int      `json:"cols"`
		TilePatches [][2]int `json:"tile_patches"`
		Runs        []int    `json:"runs"`
		Markers     []string `json:"markers"`
	} `json:"cases"`
}

func readLfm2vlLayout(t *testing.T) lfm2vlLayoutGolden {
	t.Helper()
	raw, err := os.ReadFile("../testdata/lfm2vl_layout_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g lfm2vlLayoutGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

var lfm2vlMarker = regexp.MustCompile(`<\|[^|]+\|>`)

// lfm2vlLayoutMismatches is how many golden cases the plan (grid, tile sizes, token runs, markers) gets wrong.
func lfm2vlLayoutMismatches(t *testing.T, g lfm2vlLayoutGolden, report bool) int {
	c := Lfm2VLDefaultPreprocess()
	bad := 0
	for _, k := range g.Cases {
		rows, cols, thumb, sizes := lfm2VLPlan(k.Size[0], k.Size[1], c)
		l := Lfm2VLLayout{Rows: rows, Cols: cols, Thumbnail: thumb}
		var patches [][2]int
		for _, sz := range sizes {
			l.Tiles = append(l.Tiles, Lfm2VLTile{H: sz[0], W: sz[1]})
			patches = append(patches, [2]int{sz[0] / c.PatchSize, sz[1] / c.PatchSize})
		}
		markers := lfm2vlMarker.FindAllString(Lfm2VLImageBlock(l, c), -1)
		ok := rows == k.Rows && cols == k.Cols && slices.Equal(patches, k.TilePatches) && slices.Equal(l.Tokens(c), k.Runs) &&
			slices.Equal(markers, k.Markers)
		if !ok {
			bad++
			if report {
				t.Errorf("%dx%d: grid %dx%d tiles %v runs %v markers %d; HF grid %dx%d tiles %v runs %v markers %d",
					k.Size[0], k.Size[1], rows, cols, patches, l.Tokens(c), len(markers), k.Rows, k.Cols, k.TilePatches, k.Runs, len(k.Markers))
			}
		}
	}
	return bad
}

func TestLfm2VLLayout_matchesHF(t *testing.T) {
	g := readLfm2vlLayout(t)
	if len(g.Cases) < 20 {
		t.Fatalf("%d layout cases, want the 21 pinned", len(g.Cases))
	}
	if bad := lfm2vlLayoutMismatches(t, g, true); bad != 0 {
		t.Errorf("%d of %d layouts differ from HF's", bad, len(g.Cases))
	}
	// Planted: the tie-break's area rule dropped (the first of equally close grids kept). The 1200x1000 and 1000x1250
	// cases are exact aspect ties that the rule decides (3x3 over 2x3), so the check must see it.
	lfm2VLNoAreaRule = true
	bad := lfm2vlLayoutMismatches(t, g, false)
	lfm2VLNoAreaRule = false
	t.Logf("planted (the area rule dropped): %d layouts differ", bad)
	if bad < 2 {
		t.Errorf("BLIND: dropping the area rule changes %d pinned layouts, want both tie cases", bad)
	}
}

type lfm2vlProjGolden struct {
	VisionHidden int `json:"vision_hidden"`
	TextHidden   int `json:"text_hidden"`
	Mid          int `json:"mid"`
	Factor       int `json:"factor"`
	Tiles        []struct {
		Grid   [2]int    `json:"grid"`
		Input  []float32 `json:"input"`
		Output []float32 `json:"output"`
	} `json:"tiles"`
	Weights map[string][]float32 `json:"weights"`
}

func TestLfm2VLProjector_matchesHF(t *testing.T) {
	raw, err := os.ReadFile("../testdata/lfm2vl_projector_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g lfm2vlProjGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	p := &Lfm2VLProjector{visionHidden: g.VisionHidden, textHidden: g.TextHidden, mid: g.Mid, factor: g.Factor,
		l1W: g.Weights["linear_1.weight"], l1B: g.Weights["linear_1.bias"], l2W: g.Weights["linear_2.weight"], l2B: g.Weights["linear_2.bias"]}
	if len(p.l1W) != g.Mid*g.VisionHidden*g.Factor*g.Factor || len(p.l2B) != g.TextHidden {
		t.Fatalf("golden weights have the wrong shapes: %v", func() []string {
			var k []string
			for n := range g.Weights {
				k = append(k, n)
			}
			return k
		}())
	}
	worst := func() float64 {
		var w float64
		for _, tile := range g.Tiles {
			got, err := p.Forward(tile.Input, tile.Grid)
			if err != nil {
				t.Fatal(err)
			}
			for i := range got {
				w = math.Max(w, math.Abs(float64(got[i]-tile.Output[i])))
			}
		}
		return w
	}
	if d := worst(); d > 1e-5 {
		t.Errorf("projector max|diff| %.3g against HF", d)
	} else {
		t.Logf("projector max|diff| %.3g against HF on grids %v and %v", d, g.Tiles[0].Grid, g.Tiles[1].Grid)
	}
	lfm2VLUnshuffleChannelMajor = true
	d := worst()
	lfm2VLUnshuffleChannelMajor = false
	t.Logf("planted (the unshuffle channel-major, Pixtral's order): max|diff| %.3g", d)
	if d < 1e-3 {
		t.Errorf("BLIND: the channel-major unshuffle reads max|diff| %.3g", d)
	}
}
