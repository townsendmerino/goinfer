package vision

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math/rand/v2"
	"testing"

	"github.com/townsendmerino/goinfer/internal/vision/imgproc"
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
			sinkNR = imgproc.ToNRGBA(img)
		}
	})
	nr := imgproc.ToNRGBA(img)
	b.Run("resizeNormalize", func(b *testing.B) {
		for b.Loop() {
			imgproc.ResizeNormalize(nr, out, cfg.Size, imgproc.Norm{Mean: cfg.Mean, Std: cfg.Std})
		}
	})
}

var (
	sinkImg image.Image
	sinkNR  *image.NRGBA
)
