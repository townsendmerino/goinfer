package vision

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/png"
	"os"
	"testing"
)

// TestResizeBicubicAA_matchesTorchvision holds ResizeBicubicAA to torchvision's antialiased bicubic on uint8
// (tvF.resize(..., BICUBIC, antialias=True)) bit for bit, on seeded noise and gradients over downscales, upscales,
// mixed and single-axis resizes (testdata/bicubic-resize/golden.json, scripts/oracle/pin_bicubic_resize.py). The float64
// variant, one rounding at the end, must differ somewhere: the control that shows the comparison sees a
// rounding-level change.
func TestResizeBicubicAA_matchesTorchvision(t *testing.T) {
	raw, err := os.ReadFile("testdata/bicubic-resize/golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Torchvision string `json:"torchvision"`
		Cases       []struct {
			Name         string
			H, W, TH, TW int
			Src, Dst     string
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if len(g.Cases) == 0 {
		t.Fatal("no cases")
	}
	floatDiffers := 0
	for _, c := range g.Cases {
		src, err := base64.StdEncoding.DecodeString(c.Src)
		if err != nil {
			t.Fatal(err)
		}
		want, err := base64.StdEncoding.DecodeString(c.Dst)
		if err != nil {
			t.Fatal(err)
		}
		got := ResizeBicubicAA(src, c.H, c.W, c.TH, c.TW)
		if len(got) != len(want) {
			t.Fatalf("%s: %d values, torchvision %d", c.Name, len(got), len(want))
		}
		bad := 0
		for i := range got {
			if got[i] != want[i] {
				bad++
			}
		}
		bicubicFloatForTest = true
		gotF := ResizeBicubicAA(src, c.H, c.W, c.TH, c.TW)
		bicubicFloatForTest = false
		for i := range gotF {
			if gotF[i] != want[i] {
				floatDiffers++
			}
		}
		if bad != 0 {
			t.Errorf("%s %dx%d -> %dx%d: %d of %d values differ from torchvision %s", c.Name, c.H, c.W, c.TH, c.TW, bad, len(got), g.Torchvision)
		}
	}
	if floatDiffers == 0 {
		t.Error("control: the float64 variant matched torchvision everywhere, so the comparison cannot see a rounding-level change")
	}
	t.Logf("%d cases equal torchvision %s; the float64 control differs on %d values", len(g.Cases), g.Torchvision, floatDiffers)
}

// TestGemma4PreprocessResize_layout: on an image already at its target size neither resampler resizes, so bicubic and
// bilinear give the same patches and positions; and an unknown resampler is refused.
func TestGemma4PreprocessResize_layout(t *testing.T) {
	h, w := 0, 0
	for a := 48; a <= 1200 && h == 0; a += 48 {
		for b := 48; b <= 1200; b += 48 {
			if th, tw := Gemma4AspectRatioSize(a, b, 280, 16, 3); th == a && tw == b {
				h, w = a, b
				break
			}
		}
	}
	if h == 0 {
		t.Fatal("no size is its own resize target")
	}
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for i := range img.Pix {
		img.Pix[i] = uint8(i*37 + i/7)
	}
	for i := 3; i < len(img.Pix); i += 4 {
		img.Pix[i] = 255
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	pb, posB, err := Gemma4PreprocessResize(buf.Bytes(), 280, Gemma4Bicubic)
	if err != nil {
		t.Fatal(err)
	}
	pa, posA, err := Gemma4PreprocessResize(buf.Bytes(), 280, Gemma4Bilinear)
	if err != nil {
		t.Fatal(err)
	}
	if len(pb) != len(pa) || len(posB) != len(posA) {
		t.Fatalf("bicubic %d values / %d positions, bilinear %d / %d", len(pb), len(posB), len(pa), len(posA))
	}
	for i := range posA {
		if posA[i] != posB[i] {
			t.Fatalf("position %d: bicubic %v, bilinear %v", i, posB[i], posA[i])
		}
	}
	for i := range pa {
		if d := pa[i] - pb[i]; d > 1e-6 || d < -1e-6 {
			t.Fatalf("%dx%d (no resize): value %d bicubic %g, bilinear %g", h, w, i, pb[i], pa[i])
		}
	}
	if _, _, err := Gemma4PreprocessResize(buf.Bytes(), 280, Gemma4Resize(9)); err == nil {
		t.Fatal("an unknown resampler was accepted")
	}
}
