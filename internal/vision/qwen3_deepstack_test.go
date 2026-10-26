package vision

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
)

// TestQwen3Deepstack_tiny is G-S10b's tiny half (goinfer's docs/tasks/task-multimodal-support-2026-10.md, S10): a tiny
// Qwen3-VL tower WITH DeepStack (blocks 0 and 1), every LayerNorm randomised, against transformers' Qwen3VLVisionModel
// on two images (testdata/qwen3vl-vision-tiny, pinned by goinfer's scripts/pin_qwen3vl_vision_tiny.py). ForwardDeepstack's
// main rows and each DeepStack set at worst-row cosine >= 0.9999, and the planted defect, each DeepStack merger's
// post-shuffle LayerNorm dropped, must fall under it.
func TestQwen3Deepstack_tiny(t *testing.T) {
	const dir = "testdata/qwen3vl-vision-tiny"
	f, err := os.Open(dir + "/golden.json.gz")
	if err != nil {
		t.Skipf("no fixture: %v", err)
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Grids       [][3]int    `json:"grids"`
		PixelValues []float32   `json:"pixel_values"`
		Merged      []float32   `json:"merged"`
		Deepstack   [][]float32 `json:"deepstack"`
		OutHidden   int         `json:"out_hidden"`
	}
	if err := json.NewDecoder(zr).Decode(&g); err != nil {
		t.Fatal(err)
	}
	e, err := LoadQwen3VisionEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	worst := func(got, want []float32) float64 {
		if len(got) != len(want) {
			t.Fatalf("%d values, want %d", len(got), len(want))
		}
		w := 1.0
		d := g.OutHidden
		for r := range len(want) / d {
			var dot, na, nb float64
			for j := range d {
				x, y := float64(got[r*d+j]), float64(want[r*d+j])
				dot, na, nb = dot+x*y, na+x*x, nb+y*y
			}
			w = math.Min(w, dot/math.Sqrt(na*nb))
		}
		return w
	}
	merged, deep, err := e.ForwardDeepstack(g.PixelValues, g.Grids)
	if err != nil {
		t.Fatal(err)
	}
	if len(deep) != len(g.Deepstack) {
		t.Fatalf("%d DeepStack sets, HF %d", len(deep), len(g.Deepstack))
	}
	if w := worst(merged, g.Merged); w < 0.9999 {
		t.Errorf("main merged rows: worst cosine %.9f under 0.9999", w)
	} else {
		t.Logf("main merged rows: worst cosine %.9f", w)
	}
	plain, err := e.Forward(g.PixelValues, g.Grids)
	if err != nil {
		t.Fatal(err)
	}
	bitEqual(t, "Forward against ForwardDeepstack's main rows", plain, merged)
	for k := range deep {
		w := worst(deep[k], g.Deepstack[k])
		t.Logf("DeepStack set %d (block %d): worst cosine %.9f", k, e.Cfg.DeepstackVisualIndexes[k], w)
		if w < 0.9999 {
			t.Errorf("DeepStack set %d: worst cosine %.9f under 0.9999", k, w)
		}
	}
	// The planted defect: each DeepStack merger without its post-shuffle LayerNorm (fc1 -> GELU -> fc2 straight on the
	// shuffled rows), from the same tapped block outputs.
	h := map[int][]float32{}
	if _, err := e.forwardBlocks(g.PixelValues, g.Grids, func(li int, x []float32) { h[li] = append([]float32(nil), x...) }); err != nil {
		t.Fatal(err)
	}
	c := e.Cfg
	mh := c.HiddenSize * c.SpatialMergeSize * c.SpatialMergeSize
	for k, idx := range c.DeepstackVisualIndexes {
		m := &e.deepstack[k]
		groups := len(h[idx]) / mh
		mid := make([]float32, groups*mh)
		m.fc1w.MatmulBT(h[idx], mid, groups)
		addBias(mid, m.fc1b, groups, mh)
		geluErf(mid)
		out := make([]float32, groups*c.OutHiddenSize)
		m.fc2w.MatmulBT(mid, out, groups)
		addBias(out, m.fc2b, groups, c.OutHiddenSize)
		w := worst(out, g.Deepstack[k])
		t.Logf("planted: DeepStack set %d without its post-shuffle norm: worst cosine %.6f", k, w)
		if w >= 0.9999 {
			t.Errorf("planted defect left DeepStack set %d green (worst %.9f): the fixture cannot see a dropped norm", k, w)
		}
	}
}

// TestQwen3VisionEncoder_realDeepstack is G-S10b's real half (goinfer's task-multimodal-support-2026-10.md, S10): the
// Qwen3-VL-2B tower (AIKIT_QWEN3VL_2B, default ~/models/qwen3-vl-2b-instruct) against transformers on the four F2a
// images, from HF's own pixel values (AIKIT_QWEN3VL_TOWER_ARTIFACTS, written by goinfer's
// scripts/pin_qwen3vl_tower_real.py). Every stage, the embed (patch embed plus position rows, block 0's input), each
// block, the main merger and each DeepStack set, at worst-row cosine >= 0.9999; the first stage under it is named, and
// a worst stage in 0.999-0.9999 is ambiguous (parked).
func TestQwen3VisionEncoder_realDeepstack(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skipf("no home directory: %v", err)
	}
	ckpt := os.Getenv("AIKIT_QWEN3VL_2B")
	if ckpt == "" {
		ckpt = home + "/models/qwen3-vl-2b-instruct"
	}
	art := os.Getenv("AIKIT_QWEN3VL_TOWER_ARTIFACTS")
	if art == "" {
		art = home + "/goinfer-logs/qwen3vl-tower"
	}
	raw, err := os.ReadFile(art + "/golden.json")
	if err != nil {
		t.Skipf("no artifacts at %s: %v", art, err)
	}
	var g struct {
		Hidden    int   `json:"hidden"`
		OutHidden int   `json:"out_hidden"`
		Depth     int   `json:"depth"`
		Deepstack []int `json:"deepstack"`
		Images    []struct {
			Image string `json:"image"`
			Name  string `json:"name"`
			Grid  [3]int `json:"grid"`
		} `json:"images"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if g.Hidden <= 0 || g.OutHidden <= 0 || g.Depth <= 0 || len(g.Images) == 0 {
		t.Fatalf("golden.json did not parse into usable shapes: %+v", g)
	}
	if _, err := os.Stat(art + "/" + g.Images[0].Name + ".merged.f32"); err != nil {
		t.Skipf("artifacts at %s hold no stage files (a copy taken for something else?): %v", art, err)
	}
	e, err := LoadQwen3VisionEncoder(ckpt, false)
	if err != nil {
		t.Fatal(err)
	}
	read := func(name string) []float32 {
		b, err := os.ReadFile(art + "/" + name)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]float32, len(b)/4)
		for i := range out {
			out[i] = math.Float32frombits(uint32(b[4*i]) | uint32(b[4*i+1])<<8 | uint32(b[4*i+2])<<16 | uint32(b[4*i+3])<<24)
		}
		return out
	}
	worst := func(got, want []float32, d int) float64 {
		if len(got) != len(want) {
			t.Fatalf("%d values, HF %d", len(got), len(want))
		}
		w := 1.0
		for r := range len(want) / d {
			var dot, na, nb float64
			for j := range d {
				x, y := float64(got[r*d+j]), float64(want[r*d+j])
				dot, na, nb = dot+x*y, na+x*x, nb+y*y
			}
			w = math.Min(w, dot/math.Sqrt(na*nb))
		}
		return w
	}
	for _, im := range g.Images {
		px := read(im.Name + ".pixels.f32")
		grid := [][3]int{im.Grid}
		type stage struct {
			name string
			got  []float32
			d    int
		}
		var stages []stage
		emb, err := e.Embed(px, grid)
		if err != nil {
			t.Fatal(err)
		}
		stages = append(stages, stage{"embed", emb, g.Hidden})
		blocks := map[int][]float32{}
		if _, err := e.forwardBlocks(px, grid, func(li int, h []float32) { blocks[li] = append([]float32(nil), h...) }); err != nil {
			t.Fatal(err)
		}
		for L := range g.Depth {
			stages = append(stages, stage{fmt.Sprintf("block%d", L), blocks[L], g.Hidden})
		}
		merged, deep, err := e.ForwardDeepstack(px, grid)
		if err != nil {
			t.Fatal(err)
		}
		stages = append(stages, stage{"merged", merged, g.OutHidden})
		for k := range deep {
			stages = append(stages, stage{fmt.Sprintf("deep%d", k), deep[k], g.OutHidden})
		}
		first, minW, line := "", 1.0, ""
		for _, s := range stages {
			w := worst(s.got, read(im.Name+"."+s.name+".f32"), s.d)
			minW = math.Min(minW, w)
			if w < 0.9999 && first == "" {
				first = s.name
			}
			if s.name == "embed" || s.name == "merged" || strings.HasPrefix(s.name, "deep") || s.name == fmt.Sprintf("block%d", g.Depth-1) {
				line += fmt.Sprintf(" %s %.9f", s.name, w)
			}
		}
		fmt.Fprintf(os.Stderr, "[G-S10b] %-30s grid %v: worst %.9f;%s\n", im.Image, im.Grid, minW, line)
		switch {
		case minW < 0.999:
			t.Errorf("%s: G-S10b FAIL, first stage under 0.9999: %s (worst %.9f)", im.Image, first, minW)
		case first != "":
			t.Errorf("%s: G-S10b ambiguous (parked), first stage under 0.9999: %s (worst %.9f)", im.Image, first, minW)
		}
	}
}

// TestQwen3Deepstack_exportRecomposes: a device tower's recomposition, the tapped block outputs through
// DeepstackFromHidden and the last through FinishHidden, is ForwardDeepstack bit for bit (the same host code, called
// from outside), on the tiny DeepStack tower.
func TestQwen3Deepstack_exportRecomposes(t *testing.T) {
	const dir = "testdata/qwen3vl-vision-tiny"
	e, err := LoadQwen3VisionEncoder(dir, false)
	if err != nil {
		t.Skipf("no fixture: %v", err)
	}
	if _, err := e.Weights(); err != nil {
		t.Fatalf("a DeepStack tower's export: %v", err)
	}
	grid := [][3]int{{1, 4, 6}, {1, 6, 4}}
	c := e.Cfg
	pd := c.InChannels * c.TemporalPatchSize * c.PatchSize * c.PatchSize
	px := make([]float32, 48*pd)
	for i := range px {
		px[i] = float32(math.Sin(float64(i) * 0.37))
	}
	merged, deep, err := e.ForwardDeepstack(px, grid)
	if err != nil {
		t.Fatal(err)
	}
	taps := map[int][]float32{}
	last, err := e.forwardBlocks(px, grid, func(li int, h []float32) { taps[li] = append([]float32(nil), h...) })
	if err != nil {
		t.Fatal(err)
	}
	m2, err := e.FinishHidden(last, grid)
	if err != nil {
		t.Fatal(err)
	}
	bitEqual(t, "merged", m2, merged)
	for k, idx := range c.DeepstackVisualIndexes {
		d, err := e.DeepstackFromHidden(k, taps[idx], grid)
		if err != nil {
			t.Fatal(err)
		}
		bitEqual(t, fmt.Sprintf("DeepStack set %d", k), d, deep[k])
	}
	if _, err := e.DeepstackFromHidden(len(deep), last, grid); err == nil {
		t.Error("an out-of-range set must be refused")
	}
}
