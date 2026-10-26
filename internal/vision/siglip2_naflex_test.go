package vision

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"
)

// SigLIP2's NaFlex tower (LFM2-VL's) against HF's Siglip2VisionModel (transformers 5.12.0, float32, eager), on a tiny
// random tower pinned by goinfer scripts/pin_siglip2_naflex_tiny.py (S10, G-S10l-b's tiny half, goinfer
// docs/tasks/task-multimodal-support-2026-10.md): every LayerNorm randomised, a 4x4 position table, three tiles that grow,
// shrink and keep it. HF's own patchify is checked byte for byte, each stage at the Pixtral tiny tower's bar. Planted
// defects, each must miss: the table resized without antialias, the patches flattened channel-major, the post-layernorm
// dropped. The resizes are checked too: torchvision's uint8 bilinear exactly, F.interpolate's float bilinear to 1e-6.

type naflexGolden struct {
	Hidden int `json:"hidden"`
	Patch  int `json:"patch"`
	Tiles  map[string]struct {
		Grid    [2]int               `json:"grid"`
		Image   []float32            `json:"image"`
		Patches []float32            `json:"patches"`
		Stages  map[string][]float32 `json:"stages"`
	} `json:"tiles"`
	ResizeSrc     []int  `json:"resize_src"` // 8-bit values (a JSON array, not []uint8's base64)
	ResizeSrcSize [2]int `json:"resize_src_size"`
	Resizes       []struct {
		Size [2]int `json:"size"`
		Out  []int  `json:"out"`
	} `json:"resizes"`
	FloatSrc     []float32 `json:"float_src"`
	FloatSrcSize [3]int    `json:"float_src_size"`
	Floats       []struct {
		Size [2]int    `json:"size"`
		Out  []float32 `json:"out"`
	} `json:"floats"`
}

func loadNaflexTiny(t *testing.T) (*Siglip2NaFlexEncoder, naflexGolden) {
	t.Helper()
	raw, err := os.ReadFile("testdata/siglip2_naflex_golden.json")
	if err != nil {
		t.Skipf("no golden: %v", err)
	}
	var g naflexGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	enc, err := LoadSiglip2NaFlexEncoder("testdata/siglip2-naflex-tiny", false)
	if err != nil {
		t.Fatal(err)
	}
	return enc, g
}

func naflexPatches(t *testing.T, enc *Siglip2NaFlexEncoder, grid [2]int, img []float32) []float32 {
	t.Helper()
	p := enc.Cfg.PatchSize
	patches, got, err := PatchifyNaFlex(img, 3, grid[0]*p, grid[1]*p, p)
	if err != nil {
		t.Fatal(err)
	}
	if got != grid {
		t.Fatalf("patchify grid %v, want %v", got, grid)
	}
	return patches
}

func TestSiglip2NaFlexEncoder_matchesHF(t *testing.T) {
	enc, g := loadNaflexTiny(t)
	const relBar = 5e-6
	if len(g.Tiles) != 3 {
		t.Fatalf("%d tiles, want grow, shrink and same", len(g.Tiles))
	}
	for name, tile := range g.Tiles {
		patches := naflexPatches(t, enc, tile.Grid, tile.Image)
		for i := range patches {
			if patches[i] != tile.Patches[i] {
				t.Fatalf("%s: patch value %d is %v, HF's convert_image_to_patches %v", name, i, patches[i], tile.Patches[i])
			}
		}
		stages, err := enc.ForwardStages(patches, tile.Grid)
		if err != nil {
			t.Fatal(err)
		}
		names := []string{"embeddings"}
		for i := range enc.Cfg.NumHiddenLayers {
			names = append(names, fmt.Sprintf("layer%d", i))
		}
		names = append(names, "last")
		if len(stages) != len(names) {
			t.Fatalf("%s: %d stages, want %d", name, len(stages), len(names))
		}
		for i, n := range names {
			gateStage(t, name+" "+n, stages[i], tile.Stages[n], g.Hidden, 0.999999, relBar)
		}
	}
}

func TestSiglip2NaFlexEncoder_plantedDefects(t *testing.T) {
	enc, g := loadNaflexTiny(t)
	for _, d := range []struct {
		name  string
		tiles []string // the tiles the defect can show on (antialias changes nothing when the table grows)
		set   func(on bool)
	}{
		{"the position table resized without antialias", []string{"shrink"}, func(on bool) { naflexPlainBilinear = on }},
		{"the patches flattened channel-major", []string{"grow", "shrink", "same"}, func(on bool) { naflexChannelMajor = on }},
		{"the post-layernorm dropped", []string{"grow", "shrink", "same"}, func(on bool) { naflexNoPostLayerNorm = on }},
	} {
		for _, name := range d.tiles {
			tile := g.Tiles[name]
			d.set(true)
			patches := naflexPatches(t, enc, tile.Grid, tile.Image)
			got, err := enc.Forward(patches, tile.Grid)
			d.set(false)
			if err != nil {
				t.Fatal(err)
			}
			worst, _, maxAbs := rowCosines(got, tile.Stages["last"], g.Hidden)
			t.Logf("planted %s, tile %s: worst-row cosine %.6f, max|diff| %.3g", d.name, name, worst, maxAbs)
			if worst >= 0.9999 {
				t.Errorf("BLIND: %s still reads worst-row cosine %.6f on tile %s", d.name, worst, name)
			}
		}
	}
	// The control for the first: on the growing tile antialias is a no-op, so the plain resize must agree there.
	tile := g.Tiles["grow"]
	naflexPlainBilinear = true
	got, err := enc.Forward(naflexPatches(t, enc, tile.Grid, tile.Image), tile.Grid)
	naflexPlainBilinear = false
	if err != nil {
		t.Fatal(err)
	}
	if worst, _, _ := rowCosines(got, tile.Stages["last"], g.Hidden); worst < 0.999999 {
		t.Errorf("the plain resize differs on the growing tile (worst %.6f): the antialias planted defect is not what it claims", worst)
	}
}

func TestResizeBilinearAA_matchesTorchvision(t *testing.T) {
	_, g := loadNaflexTiny(t)
	h, w := g.ResizeSrcSize[0], g.ResizeSrcSize[1]
	src := make([]uint8, len(g.ResizeSrc))
	for i, v := range g.ResizeSrc {
		src[i] = uint8(v)
	}
	for _, r := range g.Resizes {
		got := ResizeBilinearAA(src, h, w, r.Size[0], r.Size[1])
		diff := 0
		for i := range got {
			if int(got[i]) != r.Out[i] {
				diff++
			}
		}
		t.Logf("%dx%d -> %dx%d: %d of %d values differ from torchvision", h, w, r.Size[0], r.Size[1], diff, len(got))
		if diff != 0 {
			t.Errorf("%dx%d -> %dx%d: %d values differ from torchvision's uint8 bilinear", h, w, r.Size[0], r.Size[1], diff)
		}
	}
	sh, sw, c := g.FloatSrcSize[0], g.FloatSrcSize[1], g.FloatSrcSize[2]
	for _, r := range g.Floats {
		got := ResizeBilinearAAFloat(g.FloatSrc, sh, sw, c, r.Size[0], r.Size[1])
		var maxAbs float64
		for i := range got {
			maxAbs = math.Max(maxAbs, math.Abs(float64(got[i]-r.Out[i])))
		}
		t.Logf("float %dx%d -> %dx%d: max|diff| %.3g", sh, sw, r.Size[0], r.Size[1], maxAbs)
		if maxAbs > 1e-6 {
			t.Errorf("float %dx%d -> %dx%d: max|diff| %.3g against F.interpolate", sh, sw, r.Size[0], r.Size[1], maxAbs)
		}
	}
}
