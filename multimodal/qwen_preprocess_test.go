package multimodal

import (
	"encoding/json"
	"errors"
	"image"
	"io/fs"
	"math"
	"math/rand"
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

// TestQwenPreprocess_resize gates the PIL-bicubic resize port on a NON-aligned
// image (90×150 → smart_resize 84×140, so PIL BICUBIC actually runs) at a tolerance
// — float coefficients vs PIL's fixed-point aren't bit-exact, but pixel_values
// cosine must be ~1 (the downstream ViT is robust to the last-ULP resize diff).
func TestQwenPreprocess_resize(t *testing.T) {
	const golden = "../testdata/qwen25vl_preprocess_resize_golden.json"
	const imgPath = "../testdata/qwen25vl_preprocess_image_resize.png"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_qwen25vl_preprocess.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g struct {
		GridTHW     [][3]int  `json:"grid_thw"`
		PixelValues []float32 `json:"pixel_values"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	imgData, err := os.ReadFile(imgPath)
	if err != nil {
		t.Skipf("no image — run scripts/pin_qwen25vl_preprocess.py")
	}
	cfg := QwenDefaultPreprocess() // standard min/max so 90×150 → 84×140 (matches the pin)

	pv, grid, err := QwenPreprocess(imgData, cfg)
	if err != nil {
		t.Fatalf("QwenPreprocess: %v", err)
	}
	if grid != [3]int{g.GridTHW[0][0], g.GridTHW[0][1], g.GridTHW[0][2]} {
		t.Fatalf("grid_thw = %v, want %v (smart_resize rounding mismatch?)", grid, g.GridTHW[0])
	}
	if len(pv) != len(g.PixelValues) {
		t.Fatalf("pixel_values len %d, want %d", len(pv), len(g.PixelValues))
	}
	var dot, na, nb, maxAbs float64
	for i := range pv {
		a, b := float64(pv[i]), float64(g.PixelValues[i])
		dot += a * b
		na += a * a
		nb += b * b
		if d := math.Abs(a - b); d > maxAbs {
			maxAbs = d
		}
	}
	cos := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-12)
	t.Logf("qwen preprocess resize (bicubic): cosine %.6f, maxAbs %.4g", cos, maxAbs)
	if cos < 0.999 {
		t.Errorf("resized pixel_values cosine %.6f < 0.999 (bicubic parity off)", cos)
	}
}

// genericImage wraps an image.Image and forwards nothing but the interface
// methods, so qwenExtractRGB's type switch can never match a concrete fast-path
// case — forcing the generic img.At(x,y).RGBA() loop regardless of what src
// actually is. This is the reference oracle for TestQwenExtractRGB_fastPathsMatchGeneric.
type genericImage struct{ image.Image }

// TestQwenExtractRGB_fastPathsMatchGeneric is the P-19 gate: the *image.YCbCr
// and *image.RGBA fast paths must produce EXACTLY the same bytes as the generic
// img.At(x,y).RGBA() path they replace — proven, not just close, by the >>8
// round-trip identity in qwenExtractRGB's own doc comment. Also checks
// *image.NRGBA WITH a non-255 alpha channel, which is deliberately NOT
// fast-pathed (RGBA() premultiplies by alpha for NRGBA, so a naive raw-buffer
// read would silently diverge from the generic path on any transparent pixel);
// confirming it still goes through the (correct) generic loop is itself a proof
// that leaving it out was the right call, not an oversight.
func TestQwenExtractRGB_fastPathsMatchGeneric(t *testing.T) {
	const h, w = 5, 7
	rng := rand.New(rand.NewSource(1))

	mkYCbCr := func() *image.YCbCr {
		im := image.NewYCbCr(image.Rect(0, 0, w, h), image.YCbCrSubsampleRatio420)
		for i := range im.Y {
			im.Y[i] = uint8(rng.Intn(256))
		}
		for i := range im.Cb {
			im.Cb[i] = uint8(rng.Intn(256))
			im.Cr[i] = uint8(rng.Intn(256))
		}
		return im
	}
	mkRGBA := func() *image.RGBA {
		im := image.NewRGBA(image.Rect(0, 0, w, h))
		rng.Read(im.Pix)
		return im
	}
	mkNRGBAWithAlpha := func() *image.NRGBA {
		im := image.NewNRGBA(image.Rect(0, 0, w, h))
		rng.Read(im.Pix)
		for y := range h { // force a genuinely non-255 alpha so premultiplication actually matters
			for x := range w {
				im.Pix[im.PixOffset(x, y)+3] = uint8(rng.Intn(255))
			}
		}
		return im
	}

	cases := []struct {
		name string
		img  image.Image
	}{
		{"YCbCr", mkYCbCr()},
		{"RGBA", mkRGBA()},
		{"NRGBA-with-alpha", mkNRGBAWithAlpha()},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := qwenExtractRGB(c.img, h, w)
			want := qwenExtractRGB(genericImage{c.img}, h, w)
			if len(got) != len(want) {
				t.Fatalf("len(got)=%d, len(want)=%d", len(got), len(want))
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("mismatch at %d: got %v, want %v (fast path diverged from the generic path)", i, got[i], want[i])
				}
			}
		})
	}
}
