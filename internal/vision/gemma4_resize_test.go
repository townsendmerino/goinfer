package vision

import (
	"bytes"
	"image"
	"image/png"
	"testing"
)

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
