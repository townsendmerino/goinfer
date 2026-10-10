//go:build cuda

package cuda

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
	"unsafe"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// G-S3a for the Qwen2.5-VL tower on CUDA (docs/tasks/task-multimodal-support-2026-10.md, "S7 on CUDA, the fix"): every merged token at cosine >= 0.9999 against aikit's CPU tower, both
// float32; 0.999-0.9999 ambiguous (parked). Tiny first, sharpened exactly as metal/s3_towers_test.go does (norms randomised; q/k biases and weights, then v and the output projection, scaled up, because the tiny tower's
// init-scale weights left attention nearly uniform and most defects invisible), over three grids including a two-image batch, with each registered planted defect alone required red: (1) the attention scale dropped, (2) RoPE's
// halves swapped, (3) every block attending its whole frame, (4) the window reordering skipped. Then the real tower on the four F2a images. Needs a CUDA device; skips without one.

func TestQwen25CUDA_tiny(t *testing.T) {
	newTestTower(t, 64)
	enc, err := vision.LoadQwenVisionEncoder("../testdata/qwen25vl-tiny", false)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(10))
	gw := enc.GPUWeights()
	gridRandomise(rng, gw.MergerLNw)
	H := gw.Hidden
	for _, b := range gw.Blocks {
		gridRandomise(rng, b.Norm1w, b.Norm2w)
		sigSharpen(rng, b.QKVb[:2*H])
		sigScale(12, b.QKVw.F32[:2*H*H])             // the q and k rows
		sigScale(8, b.QKVw.F32[2*H*H:], b.Projw.F32) // v and the output projection: attention's share of the residual
	}
	tw, err := newQwen25Tower(enc)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(tw.Close)
	c := enc.Cfg
	pd := c.InChans * c.TemporalPatchSize * c.PatchSize * c.PatchSize
	grids := [][][3]int{{{1, 8, 8}}, {{1, 12, 16}}, {{1, 16, 12}, {1, 8, 8}}}
	type tc struct {
		g    [][3]int
		px   []float32
		want []float32
	}
	var cases []tc
	for _, g := range grids {
		n := 0
		for _, x := range g {
			n += x[0] * x[1] * x[2]
		}
		px := make([]float32, n*pd)
		for i := range px {
			px[i] = float32(rng.NormFloat64())
		}
		want, err := enc.Forward(px, g) // no resident attached: aikit's CPU tower
		if err != nil {
			t.Fatal(err)
		}
		cases = append(cases, tc{g, px, want})
	}
	worst := func() float64 {
		w := 1.0
		for _, k := range cases {
			h, err := tw.ForwardViT(k.px, k.g)
			if err != nil {
				t.Fatal(err)
			}
			w = math.Min(w, gridWorst(t, enc.MergeHidden(h, k.g), k.want, c.OutHiddenSize))
		}
		return w
	}
	w := worst()
	fmt.Fprintf(os.Stderr, "[S7 CUDA G-S3a] qwen25vl-tiny over grids %v: worst token cosine %.9f\n", grids, w)
	if w < 0.9999 {
		t.Errorf("qwen25vl-tiny: worst token cosine %.9f under 0.9999 (G-S3a)", w)
	}
	for _, d := range []struct {
		n    int
		name string
		set  func(d *gridDefect)
	}{
		{1, "attention scale dropped", func(d *gridDefect) { d.noScale = true }},
		{2, "RoPE halves swapped", func(d *gridDefect) { d.swapRope = true }},
		{3, "every block attends its whole frame", func(d *gridDefect) { d.noWindows = true }},
		{4, "window reordering skipped", func(d *gridDefect) { d.noWindowOrder = true }},
	} {
		tw.g.planted = gridDefect{}
		d.set(&tw.g.planted)
		wd := worst()
		tw.g.planted = gridDefect{}
		fmt.Fprintf(os.Stderr, "[S7 CUDA G-S3a] qwen25vl-tiny planted (%d) %s: worst %.6f\n", d.n, d.name, wd)
		if wd >= 0.9999 {
			t.Errorf("qwen25vl-tiny: planted defect (%d) %s left the bar green (worst %.9f): the fixture cannot see it", d.n, d.name, wd)
		}
	}
}

// TestQwen25CUDA_int8Declines: an int8-loaded encoder does not get this tower (it runs float32); EnableResident reports it instead of attaching something wrong.
func TestQwen25CUDA_int8Declines(t *testing.T) {
	newTestTower(t, 64)
	enc, err := vision.LoadQwenVisionEncoder("../testdata/qwen25vl-tiny", true)
	if err != nil {
		t.Skipf("no int8 tiny Qwen2.5-VL tower: %v", err)
	}
	if err := enc.EnableResident(); err == nil {
		enc.Close()
		t.Fatal("an int8 Qwen2.5-VL encoder attached the float32 CUDA tower")
	} else if !strings.Contains(err.Error(), "float32") {
		t.Errorf("the refusal does not say why: %v", err)
	}
}

func TestQwen25CUDA_real(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1 (loads the real tower)")
	}
	newTestTower(t, 64)
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", "qwen25vl-3b-instruct")
	if strings.HasPrefix(dir, "/srv/models") {
		t.Fatalf("%s is on the archive (CLAUDE.md)", dir)
	}
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no %s: %v", dir, err)
	}
	cpu, err := vision.LoadQwenVisionEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	dev, err := vision.LoadQwenVisionEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	if err := dev.EnableResident(); err != nil || !dev.ResidentEnabled() {
		t.Fatalf("the CUDA Qwen2.5-VL tower did not attach: %v", err)
	}
	t.Cleanup(dev.Close)
	rv := reflect.ValueOf(dev).Elem().FieldByName("resident")
	if _, ok := reflect.NewAt(rv.Type(), unsafe.Pointer(rv.UnsafeAddr())).Elem().Interface().(vision.QwenResidentEncoder).(*qwen25Tower); !ok {
		t.Fatalf("the attached tower is %s, not goinfer's *cuda.qwen25Tower", rv.Elem().Type())
	}
	pp, err := multimodal.LoadQwenPreprocessConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, img := range sigImages {
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
		w := gridWorst(t, got, want, cpu.Cfg.OutHiddenSize)
		fmt.Fprintf(os.Stderr, "[S7 CUDA G-S3a real] %-30s grid %v: worst token cosine %.9f; tower CUDA %s, CPU %s (exploratory)\n", img, grid, w, td.Round(time.Millisecond), tc.Round(time.Millisecond))
		switch {
		case w < 0.999:
			t.Errorf("%s: worst token cosine %.9f under 0.999 (G-S3a FAIL)", img, w)
		case w < 0.9999:
			t.Errorf("%s: worst token cosine %.9f in 0.999-0.9999 (G-S3a ambiguous, parked)", img, w)
		}
	}
}
