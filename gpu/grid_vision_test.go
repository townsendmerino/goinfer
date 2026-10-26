//go:build gpu && goinfer_testhooks

package gpu

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/internal/vision"

	"github.com/townsendmerino/goinfer/multimodal"
)

// The grid towers' gates (docs/tasks/task-multimodal-support-2026-10.md, S12.3e): the WebGPU tower plus aikit's tail
// against aikit's CPU Forward. Tiny norms are randomised first through the export's aliasing slices: the tiny
// checkpoints' norms are all ones, which would hide a dropped or swapped norm.

var gvGrids = [][][3]int{{{1, 4, 4}}, {{1, 6, 8}}, {{1, 10, 6}}, {{1, 4, 4}, {1, 6, 4}}}

func gvRandomise(rng *rand.Rand, xs ...[]float32) {
	makeWritable(xs...)
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

// gvLowest is the lowest per-row cosine between two [n, width] outputs, NaN if got holds one.
func gvLowest(t *testing.T, got, want []float32, width int) float64 {
	t.Helper()
	if len(got) != len(want) || len(got)%width != 0 {
		t.Fatalf("%d values against %d (width %d)", len(got), len(want), width)
	}
	return minTokenCos(got, want, width)
}

type gvCase struct {
	name     string
	patchDim int
	out      int
	cpu      func(px []float32, g [][3]int) ([]float32, error)
	dev      func(px []float32, g [][3]int) ([]float32, error)
	acc      *gridVTower
	defects  map[string]func(d *gvDefect, on bool)
}

// gvNoGPU skips a test on a machine with no adapter.
func gvNoGPU(t *testing.T) {
	t.Helper()
	c, err := New()
	if err != nil {
		t.Skipf("no gpu: %v", err)
	}
	c.Close()
}

func gvQwen3(t *testing.T) gvCase {
	enc, err := vision.LoadQwen3VisionEncoder("../testdata/qwen35vl-tiny", false)
	if err != nil {
		t.Fatal(err)
	}
	w, err := enc.Weights()
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(3))
	for _, b := range w.Blocks {
		gvRandomise(rng, b.Norm1W, b.Norm1B, b.Norm2W, b.Norm2B)
	}
	acc, err := newQwen3VTower(enc)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { acc.Close() })
	c := enc.Cfg
	return gvCase{name: "Qwen3.5", patchDim: c.InChannels * c.TemporalPatchSize * c.PatchSize * c.PatchSize, out: c.OutHiddenSize,
		cpu: enc.Forward,
		dev: func(px []float32, g [][3]int) ([]float32, error) {
			return multimodal.Qwen3TowerFeatures(enc, acc, px, g)
		},
		acc: acc, defects: map[string]func(*gvDefect, bool){
			"the attention scale dropped":               func(d *gvDefect, on bool) { d.noScale = on },
			"the RoPE halves swapped":                   func(d *gvDefect, on bool) { d.swapRope = on },
			"the position rows for the transposed grid": func(d *gvDefect, on bool) { d.transposePos = on },
			"the patch bias dropped":                    func(d *gvDefect, on bool) { d.noPatchBias = on },
		}}
}

func gvGlm(t *testing.T) gvCase {
	enc, err := vision.LoadGlmOcrVisionEncoder("../testdata/glm-ocr-vision-tiny", false)
	if err != nil {
		t.Fatal(err)
	}
	w, err := enc.Weights()
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(5))
	for _, b := range w.Blocks {
		gvRandomise(rng, b.Norm1W, b.Norm2W, b.QNorm, b.KNorm)
	}
	acc, err := newGlmOcrVTower(enc)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { acc.Close() })
	c := enc.Cfg
	return gvCase{name: "GLM-OCR", patchDim: c.InChannels * c.TemporalPatchSize * c.PatchSize * c.PatchSize, out: c.OutHiddenSize,
		cpu: enc.Forward,
		dev: func(px []float32, g [][3]int) ([]float32, error) {
			return multimodal.GlmOcrTowerFeatures(enc, acc, px, g)
		},
		acc: acc, defects: map[string]func(*gvDefect, bool){
			"the attention scale dropped": func(d *gvDefect, on bool) { d.noScale = on },
			"the RoPE halves swapped":     func(d *gvDefect, on bool) { d.swapRope = on },
			"the q and k norms skipped":   func(d *gvDefect, on bool) { d.noQKNorm = on },
			"the patch bias dropped":      func(d *gvDefect, on bool) { d.noPatchBias = on },
		}}
}

// lowestOver runs every grid and returns the lowest row cosine.
func (c gvCase) lowestOver(t *testing.T, seed int64) float64 {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	lowest := 1.0
	for _, g := range gvGrids {
		px := gvPixels(rng, g, c.patchDim)
		want, err := c.cpu(px, g)
		if err != nil {
			t.Fatal(err)
		}
		got, err := c.dev(px, g)
		if err != nil {
			t.Fatal(err)
		}
		w := gvLowest(t, got, want, c.out)
		if w != w {
			return w
		}
		lowest = math.Min(lowest, w)
	}
	return lowest
}

// TestGridVisionWebGPU_matchesCPU is G-S12.3e-1: the tiny Qwen3.5 and GLM-OCR towers at three grid sizes and a
// two-image batch (two attention segments), every merged token at cosine >= 0.9999 against the CPU Forward, at the
// attention block the tower picks and at a block of 5 query rows; and each planted defect, alone, under the bar.
func TestGridVisionWebGPU_matchesCPU(t *testing.T) {
	gvNoGPU(t)
	const bar = 0.9999
	for name, mk := range map[string]func(*testing.T) gvCase{"qwen3": gvQwen3, "glm-ocr": gvGlm} {
		t.Run(name, func(t *testing.T) {
			c := mk(t)
			defer func() { towerBlockForTest = 0 }()
			for _, B := range []int{0, 5} {
				towerBlockForTest = B
				w := c.lowestOver(t, 1)
				t.Logf("%s, block %d: lowest merged-token cosine %.9f over %d grid sets", c.name, B, w, len(gvGrids))
				if !(w >= bar) { // written this way so a NaN fails
					t.Errorf("%s, block %d: lowest token cosine %.9f under %v", c.name, B, w, bar)
				}
			}
			towerBlockForTest = 0
			for d, set := range c.defects {
				set(&c.acc.planted, true)
				wd := c.lowestOver(t, 1)
				set(&c.acc.planted, false)
				t.Logf("%s planted, %s: lowest %.6f", c.name, d, wd)
				if wd >= bar {
					t.Errorf("%s: planted defect (%s) left the gate green (%.9f)", c.name, d, wd)
				}
			}
		})
	}
}

// gvDeepstackCompare returns the lowest row cosine over the merged rows and every DeepStack set, the device tower's
// taps through aikit's host tails against aikit's CPU ForwardDeepstack.
func gvDeepstackCompare(t *testing.T, tag, what string, enc *vision.Qwen3VisionEncoder, acc *gridVTower, px []float32, grid [][3]int) (float64, error) {
	t.Helper()
	wantM, wantD, err := enc.ForwardDeepstack(px, grid)
	if err != nil {
		t.Fatal(err)
	}
	gotM, gotD, err := multimodal.Qwen3TowerFeaturesDeepstack(enc, acc, px, grid)
	if err != nil {
		return 0, err
	}
	if len(gotD) != len(wantD) {
		t.Fatalf("%d DeepStack sets from the device tower, %d from the CPU", len(gotD), len(wantD))
	}
	out := enc.Cfg.OutHiddenSize
	lowest := gvLowest(t, gotM, wantM, out)
	line := fmt.Sprintf("merged %.9f", lowest)
	for k := range wantD {
		w := gvLowest(t, gotD[k], wantD[k], out)
		line += fmt.Sprintf(", deep%d %.9f", k, w)
		if w != w || lowest != lowest {
			lowest = math.NaN()
		} else {
			lowest = math.Min(lowest, w)
		}
	}
	fmt.Fprintf(os.Stderr, "[%s] %s: %s\n", tag, what, line)
	return lowest, nil
}

// gvGrade applies a real-size gate's three bands to a lowest cosine.
func gvGrade(t *testing.T, gate, what string, w float64) {
	t.Helper()
	switch {
	case w != w || w < 0.999:
		t.Errorf("%s FAIL: %s: lowest cosine %.9f", gate, what, w)
	case w < 0.9999:
		t.Errorf("%s PARKED (for the owner): %s: lowest cosine %.9f, between 0.999 and 0.9999", gate, what, w)
	}
}

// TestGridVisionWebGPU_deepstackTiny is G-S12.3e-1b: the tiny DeepStack tower (Qwen3-VL's shape), the merged rows and
// every DeepStack set at cosine >= 0.9999 against the CPU ForwardDeepstack; taps read one block late must be red.
func TestGridVisionWebGPU_deepstackTiny(t *testing.T) {
	gvNoGPU(t)
	enc, err := vision.LoadQwen3VisionEncoder("../testdata/qwen3vl-vision-tiny", false)
	if err != nil {
		t.Fatal(err)
	}
	acc, err := newQwen3VTower(enc)
	if err != nil {
		t.Fatal(err)
	}
	defer acc.Close()
	c := enc.Cfg
	pd := c.InChannels * c.TemporalPatchSize * c.PatchSize * c.PatchSize
	rng := rand.New(rand.NewSource(17))
	for _, grid := range [][][3]int{{{1, 4, 6}}, {{1, 6, 4}, {1, 4, 4}}} {
		px := gvPixels(rng, grid, pd)
		w, err := gvDeepstackCompare(t, "G-S12.3e-1b", fmt.Sprintf("tiny %v", grid), enc, acc, px, grid)
		if err != nil {
			t.Fatal(err)
		}
		if !(w >= 0.9999) {
			t.Errorf("tiny %v: lowest cosine %.9f under 0.9999", grid, w)
		}
	}
	// The planted defect: every tap a block late. The tiny tower has three blocks with DeepStack at 0 and 1, so the
	// late taps, 1 and 2, are inside the tower and the defect comes back as sets, under the bar.
	acc.planted.tapShift = 1
	px := gvPixels(rng, [][3]int{{1, 4, 6}}, pd)
	w, err := gvDeepstackCompare(t, "G-S12.3e-1b", "tiny, taps one block late", enc, acc, px, [][3]int{{1, 4, 6}})
	acc.planted.tapShift = 0
	switch {
	case err != nil:
		t.Errorf("planted defect (taps one block late) was refused (%v), not measured", err)
	case w >= 0.9999:
		t.Errorf("planted defect (taps one block late) left the gate green (lowest %.9f)", w)
	}
}

// gvRealImages are the images of the real-size grid tower gates.
var gvRealImages = []string{"gemma3_preprocess_image.png", "qwen25vl_preprocess_image.png", "glm_ocr/formula.png", "glm_ocr/table.png"}

// gvRealTower is one real tower for the real-size and speed gates: its preprocessing as serve applies it, the CPU and
// device forwards, and the output width.
type gvRealTower struct {
	name     string
	pp       multimodal.QwenPreprocessConfig
	cpu, dev func([]float32, [][3]int) ([]float32, error)
	out      int
	close    func()
}

// gvLoadReal loads a real tower from the local bench set ("qwen3" is qwen3.5-0.8b, "glm-ocr" is glm-ocr). missing
// reports a checkpoint that is not there (a skip by day, a failure for a queued gate).
func gvLoadReal(t *testing.T, name string, missing func(format string, a ...any)) gvRealTower {
	t.Helper()
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", map[string]string{"qwen3": "qwen3.5-0.8b", "glm-ocr": "glm-ocr"}[name])
	if strings.HasPrefix(dir, "/Volumes/") || strings.HasPrefix(dir, "/srv/models") {
		t.Fatalf("%s is on the archive", dir)
	}
	if _, err := os.Stat(dir); err != nil {
		missing("no %s: %v", dir, err)
	}
	if name == "glm-ocr" {
		pp, err := multimodal.LoadGlmOcrPreprocessConfig(dir) // serve's default: the model's own pixel ceiling
		if err != nil {
			t.Fatal(err)
		}
		enc, err := vision.LoadGlmOcrVisionEncoder(dir, false)
		if err != nil {
			t.Fatal(err)
		}
		acc, err := newGlmOcrVTower(enc)
		if err != nil {
			missing("no gpu tower: %v", err)
		}
		return gvRealTower{name: name, pp: pp, cpu: enc.Forward, out: enc.Cfg.OutHiddenSize, close: func() { acc.Close() },
			dev: func(px []float32, g [][3]int) ([]float32, error) {
				return multimodal.GlmOcrTowerFeatures(enc, acc, px, g)
			}}
	}
	pp, err := multimodal.LoadQwen3PreprocessConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	if limit := 1024 * pp.MergeSize * pp.MergeSize * pp.PatchSize * pp.PatchSize; pp.MaxPixels > limit { // serve's cap
		pp.MaxPixels = limit
	}
	enc, err := vision.LoadQwen3VisionEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	acc, err := newQwen3VTower(enc)
	if err != nil {
		missing("no gpu tower: %v", err)
	}
	return gvRealTower{name: name, pp: pp, cpu: enc.Forward, out: enc.Cfg.OutHiddenSize, close: func() { acc.Close() },
		dev: func(px []float32, g [][3]int) ([]float32, error) {
			return multimodal.Qwen3TowerFeatures(enc, acc, px, g)
		}}
}

// TestGridVisionWebGPU_real is G-S12.3e-2 for the two plain grid towers: qwen3.5-0.8b's and glm-ocr's, through serve's
// preprocessing and caps, on four images (plus the invoice for GLM-OCR). Every merged token at cosine >= 0.9999
// against the CPU Forward is a PASS; a lowest token in 0.999-0.9999 is parked; lower or a NaN fails. The times it
// prints are a by-day smoke, not the speed gate.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 40m -tags 'gpu goinfer_testhooks' -run '^TestGridVisionWebGPU_real$' -v ./gpu/
func TestGridVisionWebGPU_real(t *testing.T) {
	requireHeavyModel(t)
	for _, name := range []string{"qwen3", "glm-ocr"} {
		t.Run(name, func(t *testing.T) {
			tw := gvLoadReal(t, name, t.Skipf)
			defer tw.close()
			images := gvRealImages
			if name == "glm-ocr" {
				images = append(append([]string(nil), images...), "glm_ocr/invoice.png")
			}
			for _, img := range images {
				data, err := os.ReadFile(filepath.Join("../testdata", img))
				if err != nil {
					t.Fatal(err)
				}
				px, grid, err := multimodal.QwenPreprocess(data, tw.pp)
				if err != nil {
					t.Fatal(err)
				}
				g := [][3]int{grid}
				t0 := time.Now()
				want, err := tw.cpu(px, g)
				tc := time.Since(t0)
				if err != nil {
					t.Fatal(err)
				}
				t0 = time.Now()
				got, err := tw.dev(px, g)
				td := time.Since(t0)
				if err != nil {
					t.Fatal(err)
				}
				w := gvLowest(t, got, want, tw.out)
				fmt.Fprintf(os.Stderr, "[G-S12.3e-2 %s] %-30s grid %v (%d merged): lowest token cosine %.9f; tower WebGPU %s, CPU %s (a smoke)\n",
					name, img, grid, len(want)/tw.out, w, td.Round(time.Millisecond), tc.Round(time.Millisecond))
				gvGrade(t, "G-S12.3e-2", name+" "+img, w)
			}
		})
	}
}

// TestGridVisionWebGPU_deepstackReal is G-S12.3e-2's DeepStack half: qwen3-vl-2b-instruct's tower on the four images,
// the merged rows and every DeepStack set against the CPU ForwardDeepstack, with the same three bands; on one image
// the taps read one block late must be red.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 40m -tags 'gpu goinfer_testhooks' -run '^TestGridVisionWebGPU_deepstackReal$' -v ./gpu/
func TestGridVisionWebGPU_deepstackReal(t *testing.T) {
	requireHeavyModel(t)
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", "qwen3-vl-2b-instruct")
	if strings.HasPrefix(dir, "/Volumes/") || strings.HasPrefix(dir, "/srv/models") {
		t.Fatalf("%s is the archive", dir)
	}
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no %s", dir)
	}
	pp, err := multimodal.LoadQwen3PreprocessConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	if limit := 1024 * pp.MergeSize * pp.MergeSize * pp.PatchSize * pp.PatchSize; pp.MaxPixels > limit { // serve's cap
		pp.MaxPixels = limit
	}
	enc, err := vision.LoadQwen3VisionEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	acc, err := newQwen3VTower(enc)
	if err != nil {
		t.Skipf("no gpu tower: %v", err)
	}
	defer acc.Close()
	for _, img := range gvRealImages {
		data, err := os.ReadFile(filepath.Join("../testdata", img))
		if err != nil {
			t.Fatal(err)
		}
		px, grid, err := multimodal.QwenPreprocess(data, pp)
		if err != nil {
			t.Fatal(err)
		}
		t0 := time.Now()
		w, err := gvDeepstackCompare(t, "G-S12.3e-2 qwen3-vl", fmt.Sprintf("%s grid %v", img, grid), enc, acc, px, [][3]int{grid})
		if err != nil {
			t.Fatal(err)
		}
		fmt.Fprintf(os.Stderr, "[G-S12.3e-2 qwen3-vl] %s: CPU and WebGPU together %s (a smoke)\n", img, time.Since(t0).Round(time.Millisecond))
		gvGrade(t, "G-S12.3e-2", "qwen3-vl "+img, w)
		if img == "qwen25vl_preprocess_image.png" { // the planted defect as a number: the tower is deep enough to shift every tap
			acc.planted.tapShift = 1
			wd, err := gvDeepstackCompare(t, "G-S12.3e-2 qwen3-vl", img+", taps one block late", enc, acc, px, [][3]int{grid})
			acc.planted.tapShift = 0
			if err != nil || wd >= 0.9999 {
				t.Errorf("planted taps one block late: lowest %.9f, err %v (must be red with a number)", wd, err)
			}
		}
	}
}

// TestGridVisionWebGPU_speedGate is G-S12.3e-4: for the tower named by GOINFER_GRID_SPEED_TOWER (qwen3 or glm-ocr), the
// encode time on one image, WebGPU over CPU, five pairs in alternating order after one discarded pass of each. That
// tower becomes WebGPU's default at a median ratio of at most 0.98; 0.98-1.00 is parked; above 1.00 it stays opt-in.
// A night gate: it runs only with GOINFER_TOWER_SPEED_GATE=1, under the timing lock.
func TestGridVisionWebGPU_speedGate(t *testing.T) {
	if os.Getenv("GOINFER_TOWER_SPEED_GATE") != "1" {
		t.Skip("a timed gate: set GOINFER_TOWER_SPEED_GATE=1")
	}
	name := os.Getenv("GOINFER_GRID_SPEED_TOWER")
	if name != "qwen3" && name != "glm-ocr" {
		t.Fatalf("GOINFER_GRID_SPEED_TOWER=%q: want qwen3 or glm-ocr", name)
	}
	tw := gvLoadReal(t, name, t.Fatalf) // a queued gate that cannot run is a failure, not a skip
	defer tw.close()
	const img = "glm_ocr/table.png"
	data, err := os.ReadFile(filepath.Join("../testdata", img))
	if err != nil {
		t.Fatal(err)
	}
	px, grid, err := multimodal.QwenPreprocess(data, tw.pp)
	if err != nil {
		t.Fatal(err)
	}
	g := [][3]int{grid}
	t0 := time.Now()
	gate := "G-S12.3e-4 " + name
	logf := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[%s %6.1fs] %s\n", gate, time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	run := func(f func([]float32, [][3]int) ([]float32, error)) float64 { // the host tail is part of both towers' encode
		ts := time.Now()
		if _, err := f(px, g); err != nil {
			t.Fatal(err)
		}
		return time.Since(ts).Seconds()
	}
	logf("%s, grid %v; warm-up (discarded): CPU %.2f s, WebGPU %.2f s", img, grid, run(tw.cpu), run(tw.dev))
	const pairs = 5
	var ratios []float64
	for i := range pairs {
		var cs, ds float64
		if i%2 == 0 {
			cs = run(tw.cpu)
			ds = run(tw.dev)
		} else {
			ds = run(tw.dev)
			cs = run(tw.cpu)
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
		t.Logf("%s SHIP: median ratio %.3f", gate, med)
	case med <= 1.00:
		t.Errorf("%s PARKED (for the owner): median ratio %.3f", gate, med)
	default:
		t.Errorf("%s OFF: median ratio %.3f", gate, med)
	}
}
