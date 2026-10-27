//go:build gpu && goinfer_testhooks

package kernels

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/internal/vision"
)

// siglipTinySharpened loads the tiny SigLIP tower in float32 and randomises its norms, biases and position table
// through Weights' aliasing slices: the fixture's are near-constant, which would let a dropped bias or a swapped
// projection pass. The same encoder then serves as the CPU reference and as the source of the device tower's weights.
func siglipTinySharpened(t *testing.T) (*vision.Encoder, vision.SiglipWeights, []float32) {
	t.Helper()
	const ckpt = "../../../testdata/siglip-tiny"
	if _, err := os.Stat(ckpt); err != nil {
		t.Skip("no siglip-tiny")
	}
	raw, err := os.ReadFile("../../../testdata/siglip_vision_golden.json")
	if err != nil {
		t.Skip("no golden")
	}
	var g struct {
		PixelValues []float32 `json:"pixel_values"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	enc, err := vision.LoadEncoder(ckpt, false)
	if err != nil {
		t.Fatal(err)
	}
	w, err := enc.Weights()
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(11))
	scale := func(x []float32) {
		makeWritable(x)
		for i := range x {
			x[i] *= 0.5 + rng.Float32()
		}
	}
	shift := func(x []float32, by float32) {
		makeWritable(x)
		for i := range x {
			x[i] += by * float32(rng.NormFloat64())
		}
	}
	shift(w.PosEmb, 0.5)
	for i := range w.Blocks {
		b := &w.Blocks[i]
		scale(b.LN1W)
		scale(b.LN2W)
		shift(b.LN1B, 0.1)
		shift(b.LN2B, 0.1)
		for _, p := range []*vision.VisionProj{&b.Q, &b.K, &b.V, &b.O, &b.FC1, &b.FC2} {
			shift(p.B, 0.1)
		}
	}
	return enc, w, g.PixelValues
}

// minTokenCos is the lowest per-token cosine between two [tokens, hidden] outputs, NaN if any value of a is NaN (cosG
// alone would not say: a token of NaNs has a zero norm test that fails open).
func minTokenCos(a, b []float32, hidden int) float64 {
	for _, x := range a {
		if x != x {
			return math.NaN()
		}
	}
	lowest := 1.0
	for o := 0; o+hidden <= len(a); o += hidden {
		lowest = math.Min(lowest, cosG(a[o:o+hidden], b[o:o+hidden]))
	}
	return lowest
}

// siglipF32Device runs the float32 device tower on the encoder's weights and finishes it on the host.
func siglipF32Device(t *testing.T, c *Context, enc *vision.Encoder, w vision.SiglipWeights, pixels []float32) []float32 {
	t.Helper()
	ve, err := c.NewVisionEncoderF32(w)
	if err != nil {
		t.Fatal(err)
	}
	defer ve.Close()
	patches, err := enc.GridPatches(pixels)
	if err != nil {
		t.Fatal(err)
	}
	h, err := ve.ForwardPatches(patches)
	if err != nil {
		t.Fatal(err)
	}
	out, err := enc.FinishHidden(h)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// TestVisionEncoderF32_parity is G-S12.3a-1 (docs/tasks/task-multimodal-support-2026-10.md, S12.3a): the float32
// SigLIP tower on WebGPU against the CPU float32 tower on the sharpened tiny fixture, every output token at cosine
// >= 0.9999, and each planted defect, alone, under that bar.
func TestVisionEncoderF32_parity(t *testing.T) {
	c, err := New()
	if err != nil {
		t.Skipf("no gpu: %v", err)
	}
	defer c.Close()
	enc, w, pixels := siglipTinySharpened(t)
	cpu, err := enc.Forward(pixels)
	if err != nil {
		t.Fatal(err)
	}
	hidden := w.Cfg.HiddenSize
	const bar = 0.9999
	clean := minTokenCos(siglipF32Device(t, c, enc, w, pixels), cpu, hidden)
	t.Logf("float32 WebGPU tower against the CPU float32 tower: lowest token cosine %.7f over %d tokens", clean, len(cpu)/hidden)
	if !(clean >= bar) { // written this way so a NaN fails
		t.Errorf("lowest token cosine %.7f, want at least %v", clean, bar)
	}
	for d, name := range map[int]string{
		1: "the position table not added",
		2: "the q bias dropped in the first layer",
		3: "q and k weights swapped in the first layer",
		4: "the GELU skipped in the first layer",
		5: "the MLP residual dropped in the first layer",
	} {
		visionDefectForTest = d
		got := minTokenCos(siglipF32Device(t, c, enc, w, pixels), cpu, hidden)
		visionDefectForTest = 0
		t.Logf("planted (%d) %s: lowest token cosine %.7f", d, name, got)
		if got >= bar {
			t.Errorf("planted defect (%d) %s left the gate green (%.7f)", d, name, got)
		}
	}
}

// TestVisionEncoderF32_largeActivations: a GELU input far past where a device tanh stays finite. The first layer's fc1
// bias is pushed to about ±40, as a real float32 SigLIP reaches; the CPU tower handles it and the device tower must
// match it, not return NaN.
func TestVisionEncoderF32_largeActivations(t *testing.T) {
	c, err := New()
	if err != nil {
		t.Skipf("no gpu: %v", err)
	}
	defer c.Close()
	enc, w, pixels := siglipTinySharpened(t)
	for i := range w.Blocks[0].FC1.B {
		w.Blocks[0].FC1.B[i] += float32(40 * (1 - 2*(i%2)))
	}
	cpu, err := enc.Forward(pixels)
	if err != nil {
		t.Fatal(err)
	}
	got := minTokenCos(siglipF32Device(t, c, enc, w, pixels), cpu, w.Cfg.HiddenSize)
	t.Logf("with GELU inputs near ±40: lowest token cosine %.7f", got)
	if !(got >= 0.9999) {
		t.Errorf("lowest token cosine %.7f, want at least 0.9999 (NaN is a tanh that overflowed)", got)
	}
}

// TestVisionEnableResidentF32_parity: the serve seam for a float32 encoder. EnableResident attaches the float32 device
// tower and Encoder.Forward, delegating to it, matches a separate CPU encoder of the same checkpoint.
func TestVisionEnableResidentF32_parity(t *testing.T) {
	if _, err := New(); err != nil {
		t.Skipf("no gpu: %v", err)
	}
	const ckpt = "../../../testdata/siglip-tiny"
	if _, err := os.Stat(ckpt); err != nil {
		t.Skip("no siglip-tiny")
	}
	raw, err := os.ReadFile("../../../testdata/siglip_vision_golden.json")
	if err != nil {
		t.Skip("no golden")
	}
	var g struct {
		PixelValues []float32 `json:"pixel_values"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	cpuEnc, err := vision.LoadEncoder(ckpt, false)
	if err != nil {
		t.Fatal(err)
	}
	cpu, err := cpuEnc.Forward(g.PixelValues)
	if err != nil {
		t.Fatal(err)
	}
	enc, err := vision.LoadEncoder(ckpt, false)
	if err != nil {
		t.Fatal(err)
	}
	if err := enc.EnableResident(); err != nil {
		t.Fatalf("EnableResident on a float32 encoder: %v", err)
	}
	defer enc.Close()
	got, err := enc.Forward(g.PixelValues)
	if err != nil {
		t.Fatal(err)
	}
	if lowest := minTokenCos(got, cpu, cpuEnc.Cfg.HiddenSize); !(lowest >= 0.9999) {
		t.Errorf("Encoder.Forward through the float32 device tower: lowest token cosine %.7f, want at least 0.9999", lowest)
	}
}

// siglipRealImages are the five images of the default-tower comparison (docs/tasks/task-multimodal-support-2026-10.md,
// "Gemma 3's default tower on WebGPU").
var siglipRealImages = []string{
	"../../../testdata/glm_ocr/table.png", "../../../testdata/glm_ocr/invoice.png", "../../../testdata/glm_ocr/formula.png",
	"../../../testdata/gemma3_preprocess_image.png", "../../../testdata/glm_ocr/invoices/inv01.png",
}

// TestVisionEncoderF32_realGemma3 is G-S12.3a-2: gemma-3-4b-it's SigLIP tower at 896x896 on five images, the float32
// WebGPU tower against the CPU float32 tower. Every token at cosine >= 0.9999 is a PASS, a lowest token in
// 0.999-0.9999 is parked, lower fails. The int8 device tower is run beside it as a record. The times it prints are a
// by-day smoke, not the speed gate.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 40m -tags 'gpu goinfer_testhooks' -run '^TestVisionEncoderF32_realGemma3$' -v ./gpu/
func TestVisionEncoderF32_realGemma3(t *testing.T) {
	requireHeavyModel(t)
	dir := os.Getenv("HOME") + "/models/gemma-3-4b-it"
	if _, err := os.Stat(dir); err != nil {
		t.Skip("no gemma-3-4b-it")
	}
	c, err := New()
	if err != nil {
		t.Skipf("no gpu: %v", err)
	}
	defer c.Close()
	t0 := time.Now()
	logf := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[G-S12.3a-2 %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	var pixels [][]float32
	for _, p := range siglipRealImages {
		raw, err := os.ReadFile(p)
		if err != nil {
			t.Skipf("no image %s: %v", p, err)
		}
		pv, err := vision.Preprocess(raw, vision.Gemma3())
		if err != nil {
			t.Fatal(err)
		}
		pixels = append(pixels, pv.Data)
	}
	enc, err := vision.LoadEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	w, err := enc.Weights()
	if err != nil {
		t.Fatal(err)
	}
	hidden := w.Cfg.HiddenSize
	logf("float32 tower loaded: %d layers, hidden %d, %d patches", len(w.Blocks), hidden, w.NumPatches)
	cpu := make([][]float32, len(pixels))
	for i, px := range pixels {
		ts := time.Now()
		if cpu[i], err = enc.Forward(px); err != nil {
			t.Fatal(err)
		}
		logf("image %d/%d %s: CPU float32 tower %.1f s", i+1, len(pixels), siglipRealImages[i], time.Since(ts).Seconds())
	}
	ve, err := c.NewVisionEncoderF32(w)
	if err != nil {
		t.Fatal(err)
	}
	worst := 1.0
	for i, px := range pixels {
		patches, err := enc.GridPatches(px)
		if err != nil {
			t.Fatal(err)
		}
		ts := time.Now()
		h, err := ve.ForwardPatches(patches)
		if err != nil {
			t.Fatal(err)
		}
		out, err := enc.FinishHidden(h)
		if err != nil {
			t.Fatal(err)
		}
		lowest := minTokenCos(out, cpu[i], hidden)
		worst = math.Min(worst, lowest)
		logf("image %d/%d: float32 WebGPU tower %.1f s, lowest token cosine %.7f, whole-output cosine %.7f", i+1, len(pixels), time.Since(ts).Seconds(), lowest, cosG(out, cpu[i]))
	}
	ve.Close()
	// The record: the int8 device tower on the same images.
	enc8, err := vision.LoadEncoder(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	w8, err := enc8.GPUWeights()
	if err != nil {
		t.Fatal(err)
	}
	ve8, err := c.NewVisionEncoder(w8)
	if err != nil {
		t.Fatal(err)
	}
	defer ve8.Close()
	for i, px := range pixels {
		patches, err := enc8.GridPatches(px)
		if err != nil {
			t.Fatal(err)
		}
		ts := time.Now()
		out, err := ve8.ForwardPatches(patches)
		if err != nil {
			t.Fatal(err)
		}
		logf("image %d/%d: int8 WebGPU tower (record) %.1f s, lowest token cosine %.4f, whole-output cosine %.4f", i+1, len(pixels), time.Since(ts).Seconds(), minTokenCos(out, cpu[i], hidden), cosG(out, cpu[i]))
	}
	logf("float32 WebGPU tower: lowest token cosine over the five images %.7f", worst)
	switch {
	case math.IsNaN(worst) || worst < 0.999:
		t.Errorf("G-S12.3a-2 FAIL: lowest token cosine %.7f", worst)
	case worst < 0.9999:
		t.Errorf("G-S12.3a-2 PARKED (for the owner): lowest token cosine %.7f, between 0.999 and 0.9999", worst)
	default:
		t.Logf("G-S12.3a-2 PASS: lowest token cosine %.7f", worst)
	}
}

// TestVisionEncoderF32_tiledSmoke is G-S12.3b-3 and G-S12.3c-3, a by-day smoke with no bar: the real float32 and int8
// towers' time for one image on the register-blocked matmul and on the 16x16 one, interleaved in one process, the
// first pass of each tower discarded (GOINFER_TOWER_SMOKE_NAIVE adds a pass on the naive kernel).
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 30m -tags 'gpu goinfer_testhooks' -run '^TestVisionEncoderF32_tiledSmoke$' -v ./gpu/
func TestVisionEncoderF32_tiledSmoke(t *testing.T) {
	requireHeavyModel(t)
	dir := os.Getenv("HOME") + "/models/gemma-3-4b-it"
	if _, err := os.Stat(dir); err != nil {
		t.Skip("no gemma-3-4b-it")
	}
	c, err := New()
	if err != nil {
		t.Skipf("no gpu: %v", err)
	}
	defer c.Close()
	raw, err := os.ReadFile(siglipRealImages[0])
	if err != nil {
		t.Skip(err)
	}
	pv, err := vision.Preprocess(raw, vision.Gemma3())
	if err != nil {
		t.Fatal(err)
	}
	t0 := time.Now()
	logf := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[G-S12.3b-3 %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	run := func(label string, ve *VisionEncoder, patches []float32) {
		names := map[int]string{visionMatmulRB64: "register-blocked", visionMatmulTiled16: "16x16", visionMatmulNaive: "naive"}
		kernels := []int{visionMatmulRB64, visionMatmulRB64, visionMatmulTiled16, visionMatmulRB64, visionMatmulTiled16} // the first is a warm-up
		if os.Getenv("GOINFER_TOWER_SMOKE_NAIVE") != "" {
			kernels = append(kernels, visionMatmulNaive)
		}
		for i, k := range kernels {
			visionMatmulForTest = k
			ts := time.Now()
			if _, err := ve.ForwardPatches(patches); err != nil {
				t.Fatal(err)
			}
			logf("%s tower, pass %d, %s matmul: %.1f s", label, i, names[k], time.Since(ts).Seconds())
		}
		visionMatmulForTest = visionMatmulRB64
	}
	enc, err := vision.LoadEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	w, err := enc.Weights()
	if err != nil {
		t.Fatal(err)
	}
	patches, err := enc.GridPatches(pv.Data)
	if err != nil {
		t.Fatal(err)
	}
	ve, err := c.NewVisionEncoderF32(w)
	if err != nil {
		t.Fatal(err)
	}
	run("float32", ve, patches)
	ve.Close()
	enc8, err := vision.LoadEncoder(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	w8, err := enc8.GPUWeights()
	if err != nil {
		t.Fatal(err)
	}
	ve8, err := c.NewVisionEncoder(w8)
	if err != nil {
		t.Fatal(err)
	}
	defer ve8.Close()
	run("int8", ve8, patches)
}

// TestVisionEncoder_profile is a by-day reading with no bar: where the real float32 and int8 WebGPU towers spend an
// image's time, by step, from a pass that waits for the device after every step. The unprofiled pass beside it is the
// time the tower actually takes.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 30m -tags 'gpu goinfer_testhooks' -run '^TestVisionEncoder_profile$' -v ./gpu/
func TestVisionEncoder_profile(t *testing.T) {
	requireHeavyModel(t)
	dir := os.Getenv("HOME") + "/models/gemma-3-4b-it"
	if _, err := os.Stat(dir); err != nil {
		t.Skip("no gemma-3-4b-it")
	}
	c, err := New()
	if err != nil {
		t.Skipf("no gpu: %v", err)
	}
	defer c.Close()
	raw, err := os.ReadFile(siglipRealImages[0])
	if err != nil {
		t.Skip(err)
	}
	pv, err := vision.Preprocess(raw, vision.Gemma3())
	if err != nil {
		t.Fatal(err)
	}
	profile := func(label string, ve *VisionEncoder, patches []float32) {
		for range 2 { // a warm-up, then the time the tower takes
			ts := time.Now()
			if _, err := ve.ForwardPatches(patches); err != nil {
				t.Fatal(err)
			}
			fmt.Fprintf(os.Stderr, "[profile] %s tower, unprofiled pass: %.1f s\n", label, time.Since(ts).Seconds())
		}
		visionProfileForTest = map[string]time.Duration{}
		ts := time.Now()
		if _, err := ve.ForwardPatches(patches); err != nil {
			t.Fatal(err)
		}
		total := time.Since(ts)
		prof := visionProfileForTest
		visionProfileForTest = nil
		type row struct {
			step string
			d    time.Duration
		}
		var rows []row
		var sum time.Duration
		for k, d := range prof {
			rows = append(rows, row{k, d})
			sum += d
		}
		sort.Slice(rows, func(i, j int) bool { return rows[i].d > rows[j].d })
		fmt.Fprintf(os.Stderr, "[profile] %s tower, profiled pass: %.1f s (steps sum %.1f s)\n", label, total.Seconds(), sum.Seconds())
		for _, r := range rows {
			fmt.Fprintf(os.Stderr, "[profile]   %-36s %6.2f s  %5.1f%%\n", r.step, r.d.Seconds(), 100*r.d.Seconds()/sum.Seconds())
		}
	}
	enc, err := vision.LoadEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	w, err := enc.Weights()
	if err != nil {
		t.Fatal(err)
	}
	patches, err := enc.GridPatches(pv.Data)
	if err != nil {
		t.Fatal(err)
	}
	ve, err := c.NewVisionEncoderF32(w)
	if err != nil {
		t.Fatal(err)
	}
	profile("float32", ve, patches)
	ve.Close()
	enc8, err := vision.LoadEncoder(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	w8, err := enc8.GPUWeights()
	if err != nil {
		t.Fatal(err)
	}
	ve8, err := c.NewVisionEncoder(w8)
	if err != nil {
		t.Fatal(err)
	}
	defer ve8.Close()
	profile("int8", ve8, patches)
}

// TestVisionEncoderF32_speedGate is G-S12.3a-4 (docs/tasks/task-multimodal-support-2026-10.md, S12.3a): gemma-3-4b-it's
// tower at 896x896 on one image, the float32 WebGPU tower's encode time over the CPU float32 tower's, five pairs in
// alternating order after one discarded pass of each. The float32 device tower becomes WebGPU's default at a median
// ratio of at most 0.98; 0.98-1.00 is parked; above 1.00 it stays opt-in. A night gate: it runs only with
// GOINFER_TOWER_SPEED_GATE=1, under the timing lock.
func TestVisionEncoderF32_speedGate(t *testing.T) {
	if os.Getenv("GOINFER_TOWER_SPEED_GATE") != "1" {
		t.Skip("a timed gate: set GOINFER_TOWER_SPEED_GATE=1")
	}
	dir := os.Getenv("HOME") + "/models/gemma-3-4b-it"
	if _, err := os.Stat(dir); err != nil {
		t.Fatalf("no gemma-3-4b-it: %v", err) // a queued gate that cannot run is a failure, not a skip
	}
	c, err := New()
	if err != nil {
		t.Fatalf("no gpu: %v", err)
	}
	defer c.Close()
	raw, err := os.ReadFile(siglipRealImages[0])
	if err != nil {
		t.Fatal(err)
	}
	pv, err := vision.Preprocess(raw, vision.Gemma3())
	if err != nil {
		t.Fatal(err)
	}
	t0 := time.Now()
	logf := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[G-S12.3a-4 %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	enc, err := vision.LoadEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	w, err := enc.Weights()
	if err != nil {
		t.Fatal(err)
	}
	ve, err := c.NewVisionEncoderF32(w)
	if err != nil {
		t.Fatal(err)
	}
	defer ve.Close()
	cpu := func() float64 {
		ts := time.Now()
		if _, err := enc.Forward(pv.Data); err != nil {
			t.Fatal(err)
		}
		return time.Since(ts).Seconds()
	}
	dev := func() float64 { // GridPatches and the host post-layernorm are part of the device tower's encode
		ts := time.Now()
		patches, err := enc.GridPatches(pv.Data)
		if err != nil {
			t.Fatal(err)
		}
		h, err := ve.ForwardPatches(patches)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := enc.FinishHidden(h); err != nil {
			t.Fatal(err)
		}
		return time.Since(ts).Seconds()
	}
	logf("warm-up (discarded): CPU %.1f s, WebGPU %.1f s", cpu(), dev())
	const pairs = 5
	var ratios []float64
	for i := range pairs {
		var cs, ds float64
		if i%2 == 0 {
			cs, ds = cpu(), 0
			ds = dev()
		} else {
			ds = dev()
			cs = cpu()
		}
		ratios = append(ratios, ds/cs)
		logf("pair %d/%d (%s first): CPU %.2f s, WebGPU float32 %.2f s, ratio %.3f", i+1, pairs, map[bool]string{true: "CPU", false: "WebGPU"}[i%2 == 0], cs, ds, ds/cs)
	}
	sorted := append([]float64(nil), ratios...)
	sort.Float64s(sorted)
	med := sorted[len(sorted)/2]
	logf("ratios %.3f; median %.3f (ship at most 0.98, park 0.98-1.00, off above 1.00)", ratios, med)
	switch {
	case med <= 0.98:
		t.Logf("G-S12.3a-4 SHIP: median ratio %.3f", med)
	case med <= 1.00:
		t.Errorf("G-S12.3a-4 PARKED (for the owner): median ratio %.3f", med)
	default:
		t.Errorf("G-S12.3a-4 OFF: median ratio %.3f", med)
	}
}
