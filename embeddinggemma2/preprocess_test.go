package embeddinggemma2

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/png"
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/vision"
)

// TestResizeBicubic_matchesTorchvision holds resizeBicubicRGB to torchvision's antialiased bicubic on uint8
// (tvF.resize(..., BICUBIC, antialias=True), the reference processor's resize) bit for bit, on seeded noise and
// gradients over downscales, upscales, mixed and single-axis resizes (testdata/embeddinggemma2-resize/golden.json,
// scripts/pin_embeddinggemma2_resize.py). The float64 variant, one rounding at the end, must differ somewhere on the
// noise: that is the control showing the comparison sees a rounding-level change.
func TestResizeBicubic_matchesTorchvision(t *testing.T) {
	raw, err := os.ReadFile("../testdata/embeddinggemma2-resize/golden.json")
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
		src, _ := base64.StdEncoding.DecodeString(c.Src)
		want, _ := base64.StdEncoding.DecodeString(c.Dst)
		got := resizeBicubicRGB(src, c.H, c.W, c.TH, c.TW)
		if len(got) != len(want) {
			t.Fatalf("%s: %d values, torchvision %d", c.Name, len(got), len(want))
		}
		bad, worst := 0, 0
		for i := range got {
			if d := absInt(int(got[i]) - int(want[i])); d != 0 {
				bad++
				worst = max(worst, d)
			}
		}
		eg2ResizeFloat = true
		gotF := resizeBicubicRGB(src, c.H, c.W, c.TH, c.TW)
		eg2ResizeFloat = false
		fd := 0
		for i := range gotF {
			if gotF[i] != want[i] {
				fd++
			}
		}
		floatDiffers += fd
		t.Logf("%-20s %dx%d -> %dx%d: %d of %d values differ from torchvision %s (worst %d); float64 variant %d", c.Name, c.H, c.W, c.TH, c.TW, bad, len(got), g.Torchvision, worst, fd)
		if bad != 0 {
			t.Errorf("%s: %d values differ from torchvision (worst %d uint8 steps)", c.Name, bad, worst)
		}
	}
	if floatDiffers == 0 {
		t.Error("control: the float64 variant matched torchvision everywhere, so the comparison cannot see a rounding-level change")
	}
}

func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// TestPreprocessBicubic_layout checks PreprocessBicubic against aikit's Gemma4Preprocess on what the two share: the
// patch count, the positions, and (on an image already at its target size, where neither resizes) every value.
func TestPreprocessBicubic_layout(t *testing.T) {
	// Find a size its own resize target leaves unchanged (multiples of the 48-pixel pooled patch).
	h, w := 0, 0
	for a := 48; a <= 1200 && h == 0; a += 48 {
		for b := 48; b <= 1200; b += 48 {
			if th, tw := vision.Gemma4AspectRatioSize(a, b, MaxImageSoftTokens, 16, 3); th == a && tw == b {
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
	pb, posB, err := PreprocessBicubic(buf.Bytes(), MaxImageSoftTokens)
	if err != nil {
		t.Fatal(err)
	}
	pa, posA, err := vision.Gemma4Preprocess(buf.Bytes(), MaxImageSoftTokens)
	if err != nil {
		t.Fatal(err)
	}
	if len(pb) != len(pa) || len(posB) != len(posA) {
		t.Fatalf("bicubic %d values / %d positions, aikit %d / %d", len(pb), len(posB), len(pa), len(posA))
	}
	for i := range posA {
		if posA[i] != posB[i] {
			t.Fatalf("position %d: bicubic %v, aikit %v", i, posB[i], posA[i])
		}
	}
	for i := range pa {
		if d := pa[i] - pb[i]; d > 1e-6 || d < -1e-6 {
			t.Fatalf("%dx%d (no resize either way): value %d bicubic %g, aikit %g", h, w, i, pb[i], pa[i])
		}
	}
	t.Logf("%dx%d: %d patches, every value and position equal to aikit's", h, w, len(posA))
}

func TestSetImageResize(t *testing.T) {
	var e Encoder
	if e.ImageResizeMode() != defaultImageResize {
		t.Fatalf("default %q, want %q", e.ImageResizeMode(), defaultImageResize)
	}
	for _, r := range []ImageResize{ResizeBilinear, ResizeBicubic, "BICUBIC"} {
		if err := e.SetImageResize(r); err != nil || e.ImageResizeMode() != ImageResize(strings.ToLower(string(r))) {
			t.Fatalf("SetImageResize(%q): %v, mode %q", r, err, e.ImageResizeMode())
		}
	}
	if err := e.SetImageResize("lanczos"); err == nil {
		t.Fatal("SetImageResize(lanczos) accepted")
	}
}
