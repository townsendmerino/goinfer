package imgproc

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math"
	"math/rand/v2"
	"testing"
)

// benchJPEG encodes a synthetic photo-like image so the decode has realistic
// work (smooth gradients plus noise, not a flat field a JPEG would compress to
// nothing).
func benchImage(w, h int, asPNG bool) []byte {
	rng := rand.New(rand.NewPCG(uint64(w*h), 0))
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			img.SetRGBA(x, y, color.RGBA{
				R: uint8((x*255/w + rng.IntN(24)) % 256),
				G: uint8((y*255/h + rng.IntN(24)) % 256),
				B: uint8(((x+y)*255/(w+h) + rng.IntN(24)) % 256),
				A: 255,
			})
		}
	}
	var buf bytes.Buffer
	if asPNG {
		_ = png.Encode(&buf, img)
	} else {
		_ = jpeg.Encode(&buf, img, &jpeg.Options{Quality: 85})
	}
	return buf.Bytes()
}

// TestResizeNormalize_matchesGenericPath is the gate for ResizeNormalize's type dispatch.
// Every specialized path must produce EXACTLY what the original
// ToNRGBA+resizeNormalize produced — the change is meant to skip a copy, not to
// resample differently. Exact equality, because the specialized path applies the
// same color.YCbCrToRGB conversion image/draw does and then the same bilinear
// arithmetic in the same order; any difference is a bug, not rounding.
func TestResizeNormalize_matchesGenericPath(t *testing.T) {
	n := Norm{Mean: [3]float32{0.5, 0.5, 0.5}, Std: [3]float32{0.5, 0.5, 0.5}}
	for _, tc := range []struct {
		name string
		w, h int
		png  bool
		size int
	}{
		{"jpeg_640x480_to384", 640, 480, false, 384},
		{"jpeg_1920x1080_to224", 1920, 1080, false, 224},
		{"jpeg_65x33_to64", 65, 33, false, 64},     // odd dims: chroma subsampling edges
		{"jpeg_17x17_to32", 17, 17, false, 32},     // UPSCALE, not downscale
		{"jpeg_8x8_to8", 8, 8, false, 8},           // identity-ish
		{"png_640x480_to384", 640, 480, true, 384}, // NRGBA path
		{"png_33x65_to128", 33, 65, true, 128},
	} {
		data := benchImage(tc.w, tc.h, tc.png)
		img, _, err := image.Decode(bytes.NewReader(data))
		if err != nil {
			t.Fatalf("%s: decode: %v", tc.name, err)
		}

		want := make([]float32, 3*tc.size*tc.size)
		resizeNormalize(ToNRGBA(img), want, tc.size, n)

		got := make([]float32, 3*tc.size*tc.size)
		ResizeNormalize(img, got, tc.size, n)

		var diff int
		var worst float64
		for i := range want {
			if got[i] != want[i] {
				diff++
				if d := math.Abs(float64(got[i]) - float64(want[i])); d > worst {
					worst = d
				}
			}
		}
		if diff != 0 {
			t.Errorf("%s (%T): %d/%d values differ from the generic path, worst |Δ|=%g",
				tc.name, img, diff, len(want), worst)
		}
	}
}

// TestXMap_matchesInlineComputation pins the hoist: precomputing the x plan once
// must give the same indices and fractions the per-row inline code produced.
func TestXMap_matchesInlineComputation(t *testing.T) {
	for _, sw := range []int{1, 2, 7, 33, 640, 1920} {
		for _, size := range []int{1, 8, 64, 384} {
			m := newXMap(size, sw)
			for dx := range size {
				sx := (float64(dx)+0.5)*float64(sw)/float64(size) - 0.5
				x0, fx := splitCoord(sx, sw)
				x1 := clampInt(x0+1, 0, sw-1)
				x0 = clampInt(x0, 0, sw-1)
				if m.x0[dx] != x0 || m.x1[dx] != x1 || m.fx[dx] != fx {
					t.Fatalf("sw=%d size=%d dx=%d: hoisted (%d,%d,%v) vs inline (%d,%d,%v)",
						sw, size, dx, m.x0[dx], m.x1[dx], m.fx[dx], x0, x1, fx)
				}
			}
		}
	}
}
