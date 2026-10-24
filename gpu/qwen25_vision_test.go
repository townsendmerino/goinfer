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

	"github.com/townsendmerino/aikit/vision"

	"github.com/townsendmerino/goinfer/multimodal"
)

// q25Tiny loads the tiny Qwen2.5-VL tower sharpened through GPUWeights' aliasing slices, which the CPU tower reads too:
// random norms and qkv biases, and the q, k, v and output projections scaled up. The fixture's q and k weights are at
// init scale, so its attention is nearly uniform and the scale, RoPE and windowing would barely show; v and the output
// projection are scaled so attention is not a negligible share of the residual.
func q25Tiny(t *testing.T) *vision.QwenVisionEncoder {
	t.Helper()
	enc, err := vision.LoadQwenVisionEncoder("../testdata/qwen25vl-tiny", false)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(10))
	gw := enc.GPUWeights()
	gvRandomise(rng, gw.MergerLNw)
	H := gw.Hidden
	scale := func(f float32, xs ...[]float32) {
		makeWritable(xs...)
		for _, x := range xs {
			for i := range x {
				x[i] *= f
			}
		}
	}
	for _, b := range gw.Blocks {
		gvRandomise(rng, b.Norm1w, b.Norm2w)
		makeWritable(b.QKVb)
		for i := range b.QKVb[:2*H] {
			b.QKVb[i] = float32(0.5 * rng.NormFloat64())
		}
		scale(12, b.QKVw.F32[:2*H*H])             // the q and k rows
		scale(8, b.QKVw.F32[2*H*H:], b.Projw.F32) // v and the output projection
	}
	return enc
}

// TestQwen25VisionWebGPU_matchesCPU is G-S12.3f-1 (docs/tasks/task-multimodal-support-2026-10.md, S12.3f): the tiny
// Qwen2.5-VL tower on WebGPU then aikit's merger against the CPU Forward, every merged token at cosine >= 0.9999, on
// two grids and a two-image batch; once with the windowed attention kernels and once with every block on the matmul
// path. Each planted defect, alone, must put the comparison under the bar.
func TestQwen25VisionWebGPU_matchesCPU(t *testing.T) {
	gvNoGPU(t)
	enc := q25Tiny(t)
	r, err := newQwen25VResident(enc)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	c := enc.Cfg
	pd := c.InChans * c.TemporalPatchSize * c.PatchSize * c.PatchSize
	grids := [][][3]int{{{1, 8, 8}}, {{1, 12, 16}}, {{1, 16, 12}, {1, 8, 8}}}
	type tc struct {
		g    [][3]int
		px   []float32
		want []float32
	}
	rng := rand.New(rand.NewSource(11))
	var cases []tc
	for _, g := range grids {
		px := gvPixels(rng, g, pd)
		want, err := enc.Forward(px, g)
		if err != nil {
			t.Fatal(err)
		}
		cases = append(cases, tc{g, px, want})
	}
	lowest := func() float64 {
		w := 1.0
		for _, k := range cases {
			h, err := r.ForwardViT(k.px, k.g)
			if err != nil {
				t.Fatal(err)
			}
			l := gvLowest(t, enc.MergeHidden(h, k.g), k.want, c.OutHiddenSize)
			if l != l {
				return l
			}
			w = math.Min(w, l)
		}
		return w
	}
	const bar = 0.9999
	defer func() { gvNoWindowKernelsForTest = false }()
	for _, off := range []bool{false, true} {
		gvNoWindowKernelsForTest = off
		before := r.a.winBlocks
		w := lowest()
		ran := r.a.winBlocks - before
		t.Logf("windowed kernels off %v: lowest merged-token cosine %.9f over %d grid sets (%d blocks ran on the windowed kernels)", off, w, len(grids), ran)
		if !(w >= bar) { // written this way so a NaN fails
			t.Errorf("windowed kernels off %v: lowest token cosine %.9f under %v", off, w, bar)
		}
		if off == (ran > 0) {
			t.Errorf("windowed kernels off %v, yet %d blocks ran on them: the comparison is not of the two paths", off, ran)
		}
	}
	gvNoWindowKernelsForTest = false
	for _, d := range []struct {
		name string
		set  func(*gvDefect)
	}{
		{"(1) the attention scale dropped", func(d *gvDefect) { d.noScale = true }},
		{"(2) the RoPE halves swapped", func(d *gvDefect) { d.swapRope = true }},
		{"(3) every block attends its whole frame", func(d *gvDefect) { d.noWindows = true }},
		{"(4) the window reordering skipped", func(d *gvDefect) { d.noWindowOrder = true }},
		{"(5) the windowed kernels read each key one row late", func(d *gvDefect) { d.lateKey = true }},
	} {
		d.set(&r.a.planted)
		wd := lowest()
		r.a.planted = gvDefect{}
		t.Logf("planted %s: lowest %.6f", d.name, wd)
		if wd >= bar {
			t.Errorf("planted defect %s left the gate green (%.9f)", d.name, wd)
		}
	}
}

// q25Real loads qwen25vl-3b-instruct's tower twice from the local bench set: one encoder for the CPU and one with the
// WebGPU tower attached. missing reports a checkpoint or device that is not there.
func q25Real(t *testing.T, missing func(format string, a ...any)) (cpu, dev *vision.QwenVisionEncoder, pp multimodal.QwenPreprocessConfig) {
	t.Helper()
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", "qwen25vl-3b-instruct")
	if strings.HasPrefix(dir, "/Volumes/") || strings.HasPrefix(dir, "/srv/models") {
		t.Fatalf("%s is on the archive", dir)
	}
	if _, err := os.Stat(dir); err != nil {
		missing("no %s: %v", dir, err)
	}
	var err error
	if cpu, err = vision.LoadQwenVisionEncoder(dir, false); err != nil {
		t.Fatal(err)
	}
	if dev, err = vision.LoadQwenVisionEncoder(dir, false); err != nil {
		t.Fatal(err)
	}
	if err := dev.EnableResident(); err != nil || !dev.ResidentEnabled() {
		missing("the WebGPU Qwen2.5-VL tower did not attach: %v", err)
	}
	t.Cleanup(dev.Close)
	if pp, err = multimodal.LoadQwenPreprocessConfig(dir); err != nil {
		t.Fatal(err)
	}
	return cpu, dev, pp
}

// TestQwen25VisionWebGPU_real is G-S12.3f-2: qwen25vl-3b-instruct's tower through serve's preprocessing on four
// images, the encoder with the WebGPU tower attached against a CPU encoder. Every merged token at cosine >= 0.9999 is
// a PASS; a lowest token in 0.999-0.9999 is parked; lower or a NaN fails. The times it prints are a by-day smoke.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 30m -tags 'gpu goinfer_testhooks' -run '^TestQwen25VisionWebGPU_real$' -v ./gpu/
func TestQwen25VisionWebGPU_real(t *testing.T) {
	requireHeavyModel(t)
	cpu, dev, pp := q25Real(t, t.Skipf)
	for _, img := range gvRealImages {
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
		want, err := cpu.Forward(px, g)
		tc := time.Since(t0)
		if err != nil {
			t.Fatal(err)
		}
		t0 = time.Now()
		got, err := dev.Forward(px, g)
		td := time.Since(t0)
		if err != nil {
			t.Fatal(err)
		}
		w := gvLowest(t, got, want, cpu.Cfg.OutHiddenSize)
		fmt.Fprintf(os.Stderr, "[G-S12.3f-2] %-30s grid %v (%d merged): lowest token cosine %.9f; tower WebGPU %s, CPU %s (a smoke)\n",
			img, grid, len(want)/cpu.Cfg.OutHiddenSize, w, td.Round(time.Millisecond), tc.Round(time.Millisecond))
		gvGrade(t, "G-S12.3f-2", img, w)
	}
}

// TestQwen25VisionWebGPU_speedGate is G-S12.3f-4: qwen25vl-3b-instruct's tower on one image, the WebGPU tower's encode
// time over the CPU tower's, five pairs in alternating order after one discarded pass of each. The device tower becomes
// WebGPU's default at a median ratio of at most 0.98; 0.98-1.00 is parked; above 1.00 it stays opt-in. A night gate:
// it runs only with GOINFER_TOWER_SPEED_GATE=1, under the timing lock.
func TestQwen25VisionWebGPU_speedGate(t *testing.T) {
	if os.Getenv("GOINFER_TOWER_SPEED_GATE") != "1" {
		t.Skip("a timed gate: set GOINFER_TOWER_SPEED_GATE=1")
	}
	cpu, dev, pp := q25Real(t, t.Fatalf) // a queued gate that cannot run is a failure, not a skip
	const img = "glm_ocr/table.png"
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
	logf := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[G-S12.3f-4 %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	run := func(e *vision.QwenVisionEncoder) float64 { // the host merger is part of both towers' encode
		ts := time.Now()
		if _, err := e.Forward(px, g); err != nil {
			t.Fatal(err)
		}
		return time.Since(ts).Seconds()
	}
	logf("%s, grid %v; warm-up (discarded): CPU %.2f s, WebGPU %.2f s", img, grid, run(cpu), run(dev))
	const pairs = 5
	var ratios []float64
	for i := range pairs {
		var cs, ds float64
		if i%2 == 0 {
			cs = run(cpu)
			ds = run(dev)
		} else {
			ds = run(dev)
			cs = run(cpu)
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
		t.Logf("G-S12.3f-4 SHIP: median ratio %.3f", med)
	case med <= 1.00:
		t.Errorf("G-S12.3f-4 PARKED (for the owner): median ratio %.3f", med)
	default:
		t.Errorf("G-S12.3f-4 OFF: median ratio %.3f", med)
	}
}
