package multimodal

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// TestQwenPreprocess_exact gates the Qwen2.5-VL preprocessing (normalize +
// patchify) bit-exactly against the HF Qwen2VLImageProcessor on a PRE-SIZED image
// (dims already multiples of patch*merge), where smart_resize / PIL.resize is a
// no-op — so the only sources of difference are float rounding. This isolates the
// exact path; PIL-bicubic resize parity is a separate follow-on (a tolerance gate
// on a non-aligned image).
func TestQwenPreprocess_exact(t *testing.T) {
	const golden = "../testdata/qwen25vl_preprocess_golden.json"
	const imgPath = "../testdata/qwen25vl_preprocess_image.png"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_qwen25vl_preprocess.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g struct {
		ImageMean        [3]float32 `json:"image_mean"`
		ImageStd         [3]float32 `json:"image_std"`
		GridTHW          [][3]int   `json:"grid_thw"`
		PixelValuesShape []int      `json:"pixel_values_shape"`
		PixelValues      []float32  `json:"pixel_values"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	imgData, err := os.ReadFile(imgPath)
	if err != nil {
		t.Skipf("no image — run scripts/pin_qwen25vl_preprocess.py")
	}

	cfg := QwenDefaultPreprocess()
	cfg.Mean, cfg.Std = g.ImageMean, g.ImageStd
	cfg.MinPixels, cfg.MaxPixels = 4*28*28, 64*28*28 // match the pin script: 56×84 is a no-op

	pv, grid, err := QwenPreprocess(imgData, cfg)
	if err != nil {
		t.Fatalf("QwenPreprocess: %v", err)
	}
	if grid != [3]int{g.GridTHW[0][0], g.GridTHW[0][1], g.GridTHW[0][2]} {
		t.Fatalf("grid_thw = %v, want %v", grid, g.GridTHW[0])
	}
	if len(pv) != len(g.PixelValues) {
		t.Fatalf("pixel_values len %d, want %d (shape %v)", len(pv), len(g.PixelValues), g.PixelValuesShape)
	}
	var maxAbs float64
	for i := range pv {
		if d := math.Abs(float64(pv[i] - g.PixelValues[i])); d > maxAbs {
			maxAbs = d
		}
	}
	t.Logf("qwen preprocess exact: %d values, maxAbs diff %.3g", len(pv), maxAbs)
	if maxAbs > 1e-4 {
		t.Errorf("pixel_values maxAbs diff %.3g > 1e-4 (not bit-exact)", maxAbs)
	}
}
