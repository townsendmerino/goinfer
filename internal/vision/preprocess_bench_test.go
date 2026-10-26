package vision

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math"
	"math/rand"
	"testing"
)

// benchJPEG encodes a synthetic photo-like image so the decode has realistic
// work (smooth gradients plus noise, not a flat field a JPEG would compress to
// nothing).
func benchImage(w, h int, asPNG bool) []byte {
	rng := rand.New(rand.NewSource(int64(w * h)))
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			img.SetRGBA(x, y, color.RGBA{
				R: uint8((x*255/w + rng.Intn(24)) % 256),
				G: uint8((y*255/h + rng.Intn(24)) % 256),
				B: uint8(((x+y)*255/(w+h) + rng.Intn(24)) % 256),
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

// BenchmarkPreprocess is the arbiter for perf-campaign item 32. It also splits
// the two halves the item names — the decode+convert and the resize — so the
// claim "draw.Draw costs more than the JPEG decode" can be checked rather than
// assumed.
func BenchmarkPreprocess(b *testing.B) {
	cfg := Gemma3()
	cfg.Size = 384
	for _, tc := range []struct {
		name string
		w, h int
		png  bool
	}{
		{"jpeg_1920x1080", 1920, 1080, false},
		{"jpeg_640x480", 640, 480, false},
		{"png_1920x1080", 1920, 1080, true},
	} {
		data := benchImage(tc.w, tc.h, tc.png)
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				pv, err := Preprocess(data, cfg)
				if err != nil {
					b.Fatal(err)
				}
				sinkVision = pv.Data
			}
		})
	}
}

// BenchmarkPreprocessStages attributes the cost between decode, the NRGBA
// conversion (draw.Draw), and resizeNormalize.
func BenchmarkPreprocessStages(b *testing.B) {
	cfg := Gemma3()
	cfg.Size = 384
	data := benchImage(1920, 1080, false)
	out := make([]float32, 3*cfg.Size*cfg.Size)

	b.Run("decode", func(b *testing.B) {
		for b.Loop() {
			img, _, err := image.Decode(bytes.NewReader(data))
			if err != nil {
				b.Fatal(err)
			}
			sinkImg = img
		}
	})
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		b.Fatal(err)
	}
	b.Run("toNRGBA", func(b *testing.B) {
		for b.Loop() {
			sinkNR = toNRGBA(img)
		}
	})
	nr := toNRGBA(img)
	b.Run("resizeNormalize", func(b *testing.B) {
		for b.Loop() {
			resizeNormalize(nr, out, cfg)
		}
	})
}

var (
	sinkImg image.Image
	sinkNR  *image.NRGBA
)

// TestResizeNormalize_matchesGenericPath is the gate for item 32's dispatch.
// Every specialized path must produce EXACTLY what the original
// toNRGBA+resizeNormalize produced — the change is meant to skip a copy, not to
// resample differently. Exact equality, because the specialized path applies the
// same color.YCbCrToRGB conversion image/draw does and then the same bilinear
// arithmetic in the same order; any difference is a bug, not rounding.
func TestResizeNormalize_matchesGenericPath(t *testing.T) {
	cfg := Gemma3()
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
		cfg.Size = tc.size
		data := benchImage(tc.w, tc.h, tc.png)
		img, _, err := image.Decode(bytes.NewReader(data))
		if err != nil {
			t.Fatalf("%s: decode: %v", tc.name, err)
		}

		want := make([]float32, 3*cfg.Size*cfg.Size)
		resizeNormalize(toNRGBA(img), want, cfg)

		got := make([]float32, 3*cfg.Size*cfg.Size)
		resizeNormalizeImage(img, got, cfg)

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
