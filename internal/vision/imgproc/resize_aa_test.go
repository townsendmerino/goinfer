package imgproc

import (
	"encoding/base64"
	"encoding/json"
	"os"
	"testing"
)

// TestResizeBicubicAA_matchesTorchvision holds ResizeBicubicAA to torchvision's antialiased bicubic on uint8
// (tvF.resize(..., BICUBIC, antialias=True)) bit for bit, on seeded noise and gradients over downscales, upscales,
// mixed and single-axis resizes (testdata/bicubic-resize/golden.json, pinned by internal/vision/testdata/oracle/pin_bicubic_resize.py). The float64
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
		gotF := resizeRGB(src, c.H, c.W, c.TH, c.TW, aaBicubic, true)
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
