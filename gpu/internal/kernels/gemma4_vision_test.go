//go:build gpu && goinfer_testhooks

package kernels

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"slices"
	"sort"
	"strconv"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/internal/vision"

	"github.com/townsendmerino/goinfer/multimodal"
)

// g4vTinyTower loads the committed tiny Gemma 4 tower (finite ClippableLinear bounds) with random position tables and
// norm weights written through the export's aliasing slices: the fixture's own are all ones, under which a swapped
// table or a norm given the wrong weight is invisible.
func g4vTinyTower(t *testing.T) *vision.Gemma4Encoder {
	t.Helper()
	enc, err := vision.LoadGemma4Encoder("../../../testdata/gemma4-vision-tiny", false)
	if err != nil {
		t.Fatal(err)
	}
	w, err := enc.Weights()
	if err != nil {
		t.Fatal(err)
	}
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

// g4vCase is a patch grid of gw x gh pooled cells (each k x k patches), its patches random, its positions in raster
// order or shuffled (the tower must not depend on the order patches arrive in).
func g4vCase(enc *vision.Gemma4Encoder, gw, gh int, shuffle bool, seed uint64) ([]float32, [][2]int) {
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

// g4vLowest runs the device tower and aikit's tail on one case and returns the lowest soft-token cosine against the
// CPU Forward (NaN if the device returned one).
func g4vLowest(t *testing.T, enc *vision.Gemma4Encoder, a *g4vTower, p []float32, pos [][2]int) float64 {
	t.Helper()
	want, err := enc.Forward(p, pos)
	if err != nil {
		t.Fatal(err)
	}
	got, err := multimodal.Gemma4TowerFeatures(enc, a, p, pos)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("the device tower gave %d values, the CPU %d", len(got), len(want))
	}
	return minTokenCos(got, want, enc.TextHiddenSize)
}

// TestG4VWebGPU_matchesCPU is G-S12.3d-1 (docs/tasks/task-multimodal-support-2026-10.md, S12.3d): on the tiny tower,
// with its finite clip bounds and random position tables and norm weights, the WebGPU tower followed by aikit's tail
// matches aikit's CPU Forward to cosine 0.9999 on every soft token: grids of 18, 54 and 135 patches, raster and
// shuffled, at the attention block the tower picks and at blocks of 6 (divides 18 and 54) and 5 (divides neither).
// Each planted defect, alone, must put the same comparison under the bar.
func TestG4VWebGPU_matchesCPU(t *testing.T) {
	if c, err := New(); err != nil {
		t.Skipf("no gpu: %v", err)
	} else {
		c.Close()
	}
	enc := g4vTinyTower(t)
	a, err := newG4VTower(enc)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	if got := multimodal.Gemma4Towers(); !slices.Contains(got, "webgpu") {
		t.Errorf("the registered Gemma 4 towers are %v, without webgpu", got)
	}
	c := enc.Cfg
	t.Logf("tiny tower: hidden %d, %d layers, %d heads of %d, MLP %d, pool %d", c.HiddenSize, c.NumHiddenLayers, c.NumAttentionHeads, c.HeadDim, c.IntermediateSize, c.PoolingKernelSize)
	const bar = 0.9999
	type grid struct {
		gw, gh  int
		shuffle bool
	}
	grids := []grid{{2, 1, false}, {3, 2, true}, {5, 3, false}, {5, 3, true}}
	defer func() { towerBlockForTest, g4vDefectForTest = 0, 0 }()
	for _, B := range []int{0, 6, 5} {
		towerBlockForTest = B
		for gi, g := range grids {
			p, pos := g4vCase(enc, g.gw, g.gh, g.shuffle, uint64(10+gi))
			low := g4vLowest(t, enc, a, p, pos)
			t.Logf("block %d, %3d patches (shuffled %v): lowest soft-token cosine %.9f", B, len(pos), g.shuffle, low)
			if !(low >= bar) { // written this way so a NaN fails
				t.Errorf("block %d, %d patches: lowest soft-token cosine %.9f, under %v", B, len(pos), low, bar)
			}
		}
	}
	towerBlockForTest = 0
	p, pos := g4vCase(enc, 5, 3, false, 99)
	for d, name := range []string{1: "the clamps skipped", 2: "the x and y position tables swapped", 3: "RoPE not applied to k",
		4: "the v norm skipped", 5: "k normed with q's weight"} {
		if d == 0 {
			continue
		}
		g4vDefectForTest = d
		low := g4vLowest(t, enc, a, p, pos)
		g4vDefectForTest = 0
		t.Logf("planted (%d) %s: lowest soft-token cosine %.9f", d, name, low)
		if low >= bar {
			t.Errorf("planted defect (%d) %s left the gate green (%.9f)", d, name, low)
		}
	}
}

// g4vRealImages are the five images of the real-size tower gates.
var g4vRealImages = []string{
	"../../../testdata/glm_ocr/table.png", "../../../testdata/glm_ocr/invoice.png", "../../../testdata/glm_ocr/formula.png",
	"../../../testdata/gemma3_preprocess_image.png", "../../../testdata/glm_ocr/invoices/inv01.png",
}

// g4vRealTower loads Gemma 4 E2B's vision tower in float32 from the local bench set, and the checkpoint's image budget.
// missing reports a checkpoint that is not there (a skip by day, a failure for a queued gate).
func g4vRealTower(t *testing.T, missing func(format string, a ...any)) (*vision.Gemma4Encoder, int) {
	t.Helper()
	dir := os.Getenv("HOME") + "/models/gemma-4-E2B-unq"
	if _, err := os.Stat(dir); err != nil {
		missing("no gemma-4-E2B-unq: %v", err)
	}
	enc, err := vision.LoadGemma4Encoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	pp, err := multimodal.LoadGemma4PreprocessConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	return enc, pp.MaxSoftTokens
}

// g4vBudgetOverride is GOINFER_G4V_BUDGET, a soft-token budget (70 to 1120) that replaces the checkpoint's in the
// real-size gate and limits it to the first image: at 1120 an image is about 10,000 patches and a head's query rows
// are scored in several blocks, which the checkpoint's own budget never reaches. 0 when unset.
func g4vBudgetOverride(t *testing.T) int {
	v := os.Getenv("GOINFER_G4V_BUDGET")
	if v == "" {
		return 0
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		t.Fatalf("GOINFER_G4V_BUDGET=%q: want a soft-token budget", v)
	}
	return n
}

// TestG4VWebGPU_realE2B is G-S12.3d-2: Gemma 4 E2B's vision tower on five images at the checkpoint's image budget, the
// WebGPU tower against the CPU tower. Every soft token at cosine >= 0.9999 is a PASS, a lowest token in 0.999-0.9999
// is parked, lower or a NaN fails. The times it prints are a by-day smoke, not the speed gate.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 30m -tags 'gpu goinfer_testhooks' -run '^TestG4VWebGPU_realE2B$' -v ./gpu/
func TestG4VWebGPU_realE2B(t *testing.T) {
	requireHeavyModel(t)
	enc, budget := g4vRealTower(t, t.Skipf)
	t0 := time.Now()
	logf := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[G-S12.3d-2 %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	a, err := newG4VTower(enc)
	if err != nil {
		t.Skipf("no gpu tower: %v", err)
	}
	defer a.Close()
	c := enc.Cfg
	logf("tower loaded: %d layers, hidden %d, %d heads of %d, MLP %d; image budget %d soft tokens", c.NumHiddenLayers, c.HiddenSize, c.NumAttentionHeads, c.HeadDim, c.IntermediateSize, budget)
	images := g4vRealImages
	if b := g4vBudgetOverride(t); b > 0 {
		budget, images = b, g4vRealImages[:1]
		logf("GOINFER_G4V_BUDGET: %d soft tokens, the first image only", b)
	}
	worst := 1.0
	for i, path := range images {
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Skipf("no image %s: %v", path, err)
		}
		patches, pos, err := vision.Gemma4Preprocess(raw, budget)
		if err != nil {
			t.Fatal(err)
		}
		ts := time.Now()
		want, err := enc.Forward(patches, pos)
		if err != nil {
			t.Fatal(err)
		}
		cpuS := time.Since(ts).Seconds()
		ts = time.Now()
		got, err := multimodal.Gemma4TowerFeatures(enc, a, patches, pos)
		if err != nil {
			t.Fatal(err)
		}
		devS := time.Since(ts).Seconds()
		if len(got) != len(want) {
			t.Fatalf("image %d: the device tower gave %d values, the CPU %d", i+1, len(got), len(want))
		}
		low := minTokenCos(got, want, enc.TextHiddenSize)
		worst = math.Min(worst, low)
		if low != low {
			worst = math.NaN()
		}
		logf("image %d/%d %s: %d patches, %d soft tokens; CPU %.2f s, WebGPU %.2f s; lowest soft-token cosine %.7f, whole-output cosine %.7f",
			i+1, len(images), path, len(pos), len(want)/enc.TextHiddenSize, cpuS, devS, low, cosG(got, want))
	}
	logf("lowest soft-token cosine over %d image(s) %.7f", len(images), worst)
	switch {
	case worst != worst || worst < 0.999:
		t.Errorf("G-S12.3d-2 FAIL: lowest soft-token cosine %.7f", worst)
	case worst < 0.9999:
		t.Errorf("G-S12.3d-2 PARKED (for the owner): lowest soft-token cosine %.7f, between 0.999 and 0.9999", worst)
	default:
		t.Logf("G-S12.3d-2 PASS: lowest soft-token cosine %.7f", worst)
	}
}

// TestG4VWebGPU_speedGate is G-S12.3d-4: Gemma 4 E2B's tower on one image at the checkpoint's image budget, the WebGPU
// tower's encode time over the CPU tower's, five pairs in alternating order after one discarded pass of each. The
// device tower becomes WebGPU's default at a median ratio of at most 0.98; 0.98-1.00 is parked; above 1.00 it stays
// opt-in. A night gate: it runs only with GOINFER_TOWER_SPEED_GATE=1, under the timing lock.
func TestG4VWebGPU_speedGate(t *testing.T) {
	if os.Getenv("GOINFER_TOWER_SPEED_GATE") != "1" {
		t.Skip("a timed gate: set GOINFER_TOWER_SPEED_GATE=1")
	}
	enc, budget := g4vRealTower(t, t.Fatalf) // a queued gate that cannot run is a failure, not a skip
	a, err := newG4VTower(enc)
	if err != nil {
		t.Fatalf("no gpu tower: %v", err)
	}
	defer a.Close()
	raw, err := os.ReadFile(g4vRealImages[0])
	if err != nil {
		t.Fatal(err)
	}
	patches, pos, err := vision.Gemma4Preprocess(raw, budget)
	if err != nil {
		t.Fatal(err)
	}
	t0 := time.Now()
	logf := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[G-S12.3d-4 %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	run := func(acc multimodal.Gemma4TowerAccelerator) float64 { // the host tail is part of both towers' encode
		ts := time.Now()
		if _, err := multimodal.Gemma4TowerFeatures(enc, acc, patches, pos); err != nil {
			t.Fatal(err)
		}
		return time.Since(ts).Seconds()
	}
	cpu := func() float64 { return run(nil) }
	dev := func() float64 { return run(a) }
	logf("%s, %d patches; warm-up (discarded): CPU %.2f s, WebGPU %.2f s", g4vRealImages[0], len(pos), cpu(), dev())
	const pairs = 5
	var ratios []float64
	for i := range pairs {
		var cs, ds float64
		if i%2 == 0 {
			cs = cpu()
			ds = dev()
		} else {
			ds = dev()
			cs = cpu()
		}
		ratios = append(ratios, ds/cs)
		logf("pair %d/%d (%s first): CPU %.2f s, WebGPU %.2f s, ratio %.3f", i+1, pairs, map[bool]string{true: "CPU", false: "WebGPU"}[i%2 == 0], cs, ds, ds/cs)
	}
	sorted := append([]float64(nil), ratios...)
	sort.Float64s(sorted)
	med := sorted[len(sorted)/2]
	logf("ratios %.3f; median %.3f (ship at most 0.98, park 0.98-1.00, off above 1.00)", ratios, med)
	switch {
	case med <= 0.98:
		t.Logf("G-S12.3d-4 SHIP: median ratio %.3f", med)
	case med <= 1.00:
		t.Errorf("G-S12.3d-4 PARKED (for the owner): median ratio %.3f", med)
	default:
		t.Errorf("G-S12.3d-4 OFF: median ratio %.3f", med)
	}
}
