//go:build cuda

package cuda

import (
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/vision"
)

// S4's Gemma 4 vision tower on CUDA (docs/tasks/task-multimodal-support-2026-10.md), the twin of metal/gemma4_vision_test.go: the tower followed by aikit's tail
// (FinishHidden) against aikit's CPU Forward, every soft token at cosine >= 0.9999, on the tiny tower with finite clip bounds and random position tables and
// norms, over raster and shuffled grids; each CUDA-specific planted defect alone must break it; the real E2B on the four F2a images under GOINFER_HEAVY_TESTS=1.

func g4TinyTower(t *testing.T) *vision.Gemma4Encoder {
	t.Helper()
	enc, err := vision.LoadGemma4Encoder("../testdata/gemma4-vision-tiny", false)
	if err != nil {
		t.Fatal(err)
	}
	w, err := enc.Weights()
	if err != nil {
		t.Fatal(err)
	}
	// Random position tables and norm weights through the export's aliasing slices: the fixture's own are all ones, under which a swapped table or a norm
	// given the wrong weight is invisible.
	r := rand.New(rand.NewPCG(7, 8))
	for _, tb := range [][]float32{w.PosEmbX, w.PosEmbY} {
		for i := range tb {
			tb[i] = r.Float32()*2 - 1
		}
	}
	for _, l := range w.Layers {
		for _, nw := range [][]float32{l.InputNorm, l.PostAttnNorm, l.PreFFNNorm, l.PostFFNNorm, l.QNorm, l.KNorm} {
			for i := range nw {
				nw[i] = 0.5 + r.Float32()
			}
		}
	}
	clipped := 0
	for _, l := range w.Layers {
		for _, p := range []vision.Gemma4Proj{l.Q, l.K, l.V, l.O, l.Gate, l.Up, l.Down} {
			if p.Clipped() {
				clipped++
			}
		}
	}
	if clipped == 0 {
		t.Fatal("the tiny tower has no finite clip bound; the clamps would go unchecked")
	}
	return enc
}

func newG4TowerForTest(t *testing.T, enc *vision.Gemma4Encoder) *g4Tower {
	t.Helper()
	a, err := newG4Tower(enc)
	if err != nil {
		t.Skipf("no CUDA device for the Gemma 4 tower: %v", err)
	}
	t.Cleanup(func() { _ = a.Close() })
	return a
}

// g4Case is a patch grid of gw x gh pooled cells (each k x k patches), its patches random, its positions in raster order or shuffled.
func g4Case(enc *vision.Gemma4Encoder, gw, gh int, shuffle bool, seed uint64) ([]float32, [][2]int) {
	c := enc.Cfg
	k := c.PoolingKernelSize
	var pos [][2]int
	for y := range gh * k {
		for x := range gw * k {
			pos = append(pos, [2]int{x, y})
		}
	}
	r := rand.New(rand.NewPCG(seed, 1))
	if shuffle {
		r.Shuffle(len(pos), func(i, j int) { pos[i], pos[j] = pos[j], pos[i] })
	}
	pd := 3 * c.PatchSize * c.PatchSize
	p := make([]float32, len(pos)*pd)
	for i := range p {
		p[i] = r.Float32()
	}
	return p, pos
}

// g4Worst is the worst per-soft-token cosine of got against want, [n, d] each.
func g4Worst(got, want []float32, d int) float64 {
	worst := 1.0
	for r := range len(want) / d {
		var dot, na, nb float64
		for j := range d {
			x, y := float64(got[r*d+j]), float64(want[r*d+j])
			dot, na, nb = dot+x*y, na+x*x, nb+y*y
		}
		c := dot / math.Sqrt(na*nb)
		if math.IsNaN(c) {
			c = -1
		}
		worst = math.Min(worst, c)
	}
	return worst
}

// g4Run is the tower's result through aikit's tail for one grid, and the CPU reference.
func g4Run(t *testing.T, enc *vision.Gemma4Encoder, a *g4Tower, p []float32, pos [][2]int) (got, want []float32) {
	t.Helper()
	want, err := enc.Forward(p, pos)
	if err != nil {
		t.Fatal(err)
	}
	h, err := a.Hidden(p, pos)
	if err != nil {
		t.Fatal(err)
	}
	got, err = enc.FinishHidden(h, pos)
	if err != nil {
		t.Fatal(err)
	}
	return got, want
}

type g4Grid struct {
	gw, gh  int
	shuffle bool
}

var g4Grids = []g4Grid{{2, 1, false}, {3, 2, true}, {5, 3, false}, {5, 3, true}}

func TestG4VCUDA_matchesCPU(t *testing.T) {
	enc := g4TinyTower(t)
	a := newG4TowerForTest(t, enc)
	c := enc.Cfg
	t.Logf("tiny tower: hidden %d, %d layers, %d heads of %d, MLP %d, pool %d", c.HiddenSize, c.NumHiddenLayers, c.NumAttentionHeads, c.HeadDim, c.IntermediateSize, c.PoolingKernelSize)
	for gi, g := range g4Grids {
		p, pos := g4Case(enc, g.gw, g.gh, g.shuffle, uint64(10+gi))
		got, want := g4Run(t, enc, a, p, pos)
		w := g4Worst(got, want, enc.TextHiddenSize)
		t.Logf("%3d patches (shuffled %v): worst soft-token cosine %.9f over %d", len(pos), g.shuffle, w, len(want)/enc.TextHiddenSize)
		if w < 0.9999 {
			t.Errorf("%d patches: worst soft-token cosine %.9f, under 0.9999", len(pos), w)
		}
	}
}

// Large, small, large on one tower: scratch is per call, so a stale buffer or a sized-once bug shows as the second large grid disagreeing with the first, or
// with the CPU. Two identical runs must also be bit-identical (the kernels are deterministic).
func TestG4VCUDA_scratchAcrossSizesAndDeterminism(t *testing.T) {
	enc := g4TinyTower(t)
	a := newG4TowerForTest(t, enc)
	big, small := g4Grids[2], g4Grids[0]
	pBig, posBig := g4Case(enc, big.gw, big.gh, false, 31)
	pSmall, posSmall := g4Case(enc, small.gw, small.gh, false, 32)
	h1, err := a.Hidden(pBig, posBig)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.Hidden(pSmall, posSmall); err != nil {
		t.Fatal(err)
	}
	h2, err := a.Hidden(pBig, posBig)
	if err != nil {
		t.Fatal(err)
	}
	for i := range h1 {
		if h1[i] != h2[i] {
			t.Fatalf("the same grid after a small one differs at %d: %v vs %v", i, h1[i], h2[i])
		}
	}
	got, want := g4Run(t, enc, a, pBig, posBig)
	if w := g4Worst(got, want, enc.TextHiddenSize); w < 0.9999 {
		t.Errorf("after the size change the tower no longer matches the CPU: %.9f", w)
	}
}

// Each planted defect alone must take the comparison below 0.9999 on at least one grid (G-S2c's rule): a green means the fixture cannot see it.
func TestG4VCUDA_plantedDefects(t *testing.T) {
	enc := g4TinyTower(t)
	a := newG4TowerForTest(t, enc)
	defects := []struct {
		name string
		set  func(*g4Defect)
	}{
		{"clamps skipped", func(d *g4Defect) { d.noClamp = true }},
		{"position x and y swapped", func(d *g4Defect) { d.swapPosXY = true }},
		{"v norm weighted", func(d *g4Defect) { d.weightedVNorm = true }},
		{"attention scale 1/sqrt(hd)", func(d *g4Defect) { d.scaleRoot = true }},
		{"position tables skipped", func(d *g4Defect) { d.noPosAdd = true }},
		{"RoPE skipped", func(d *g4Defect) { d.noRope = true }},
	}
	for _, d := range defects {
		a.defect = g4Defect{}
		d.set(&a.defect)
		worst := 1.0
		for gi, g := range g4Grids {
			p, pos := g4Case(enc, g.gw, g.gh, g.shuffle, uint64(50+gi))
			got, want := g4Run(t, enc, a, p, pos)
			worst = math.Min(worst, g4Worst(got, want, enc.TextHiddenSize))
		}
		t.Logf("planted defect %-28s worst soft-token cosine %.9f", d.name, worst)
		if worst >= 0.9999 {
			t.Errorf("planted defect %q left the comparison green (%.9f): the fixture cannot see it", d.name, worst)
		}
	}
	a.defect = g4Defect{}
}

// Close releases everything the device holds, and an unusable request returns an error (never a panic).
func TestG4VCUDA_closeAndErrors(t *testing.T) {
	enc := g4TinyTower(t)
	a, err := newG4Tower(enc)
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	p, pos := g4Case(enc, 2, 1, false, 5)
	if _, err := a.Hidden(p, pos); err != nil {
		t.Fatal(err)
	}
	if _, err := a.Hidden(p[:len(p)-1], pos); err == nil {
		t.Error("a short patch buffer must be refused")
	}
	bad := append([][2]int(nil), pos...)
	bad[0] = [2]int{enc.Cfg.PosEmbTableSize, 0}
	if _, err := a.Hidden(p, bad); err == nil {
		t.Error("a position past the table must be refused")
	}
	if allocs, _ := a.ops.dev.LedgerLen(); allocs == 0 {
		t.Fatal("the tower holds no device allocations before Close: the leak check below would prove nothing")
	}
	dev := a.ops.dev
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if allocs, objs := dev.LedgerLen(); allocs != 0 || objs != 0 {
		t.Errorf("after Close the device still holds %d allocations and %d objects", allocs, objs)
	}
}

// TestG4VCUDA_gemma4E2B is F2a on CUDA: Gemma 4 E2B's own vision tower (GOINFER_GEMMA4_E2B_VISION, default ~/models/gemma-4-E2B-unq, never the archive) against
// aikit's CPU tower, every soft token at cosine >= 0.9999, on the four repo images through aikit's own preprocessing. Times are exploratory.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags cuda -count=1 -run '^TestG4VCUDA_gemma4E2B$' -v ./cuda/
func TestG4VCUDA_gemma4E2B(t *testing.T) {
	requireHeavyModel(t)
	home, _ := os.UserHomeDir()
	dir := os.Getenv("GOINFER_GEMMA4_E2B_VISION")
	if dir == "" {
		dir = filepath.Join(home, "models", "gemma-4-E2B-unq")
	}
	if strings.HasPrefix(dir, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md)", dir)
	}
	enc, err := vision.LoadGemma4Encoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	a := newG4TowerForTest(t, enc)
	t.Logf("E2B tower: hidden %d, %d layers, clipped %v, standardize %v, text width %d", enc.Cfg.HiddenSize, enc.Cfg.NumHiddenLayers, enc.Cfg.UseClippedLinears, enc.Cfg.Standardize, enc.TextHiddenSize)
	for _, img := range []string{"gemma3_preprocess_image.png", "qwen25vl_preprocess_image.png", "glm_ocr/formula.png", "glm_ocr/table.png"} {
		data, err := os.ReadFile(filepath.Join("../testdata", img))
		if err != nil {
			t.Fatal(err)
		}
		p, pos, err := vision.Gemma4Preprocess(data, 280)
		if err != nil {
			t.Fatal(err)
		}
		t0 := time.Now()
		cpu, err := enc.Forward(p, pos)
		tc := time.Since(t0)
		if err != nil {
			t.Fatal(err)
		}
		t0 = time.Now()
		h, err := a.Hidden(p, pos)
		if err != nil {
			t.Fatal(err)
		}
		got, err := enc.FinishHidden(h, pos)
		tm := time.Since(t0)
		if err != nil {
			t.Fatal(err)
		}
		w := g4Worst(got, cpu, enc.TextHiddenSize)
		t.Logf("%-32s %4d patches: worst soft-token cosine %.9f; tower CUDA %s, CPU %s (exploratory)", img, len(pos), w, tm.Round(time.Millisecond), tc.Round(time.Millisecond))
		if w < 0.9999 {
			t.Errorf("%s: worst soft-token cosine %.9f, under 0.9999", img, w)
		}
	}
}
