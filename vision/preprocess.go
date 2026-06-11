// Package vision is goinfer's pure-Go image preprocessing for vision-language
// models: decode → resize → normalize → pixel_values, the tensor a vision
// encoder (SigLIP / ViT) consumes. Stdlib-only (no cgo), and it bounds the
// decoded pixel count BEFORE decoding so a hostile/corrupt image yields a typed
// error, never an OOM (the campaign Track-2 posture, extended to image bytes).
//
// Parity note: the resize here is bilinear (half-pixel centers). HF/PIL Gemma 3
// uses BICUBIC, so this is NOT pixel-exact yet — per docs/multimodal.md §2 the
// end-to-end gate runs on precomputed pixel_values, and a PIL-exact separable
// resampler is a follow-on. This file is the structure + the security guard.
package vision

import (
	"bytes"
	"fmt"
	"image"
	"image/draw"

	_ "image/jpeg" // register JPEG decoder
	_ "image/png"  // register PNG decoder
)

// Config parameterizes preprocessing per model family.
type Config struct {
	Size      int        // target square side (Gemma 3 SigLIP: 896)
	Mean      [3]float32 // per-channel normalization mean (applied after /255)
	Std       [3]float32 // per-channel normalization std
	MaxPixels int        // reject a decoded image with more than this many pixels (W*H)
}

// Gemma3 is the SigLIP preprocessing Gemma 3 / 4 use: 896×896, mean=std=0.5 on
// all channels (so the output lands in [-1, 1]).
func Gemma3() Config {
	return Config{
		Size:      896,
		Mean:      [3]float32{0.5, 0.5, 0.5},
		Std:       [3]float32{0.5, 0.5, 0.5},
		MaxPixels: 16 << 20, // 16 MP — far above any real photo, caps the decode alloc
	}
}

// PixelValues is normalized image data in CHW order ([3*Size*Size]) plus its
// spatial size, ready for a vision encoder's patch-embed conv.
type PixelValues struct {
	Data []float32 // channel-major: Data[c*Size*Size + y*Size + x]
	Size int
}

// Preprocess decodes image bytes and produces normalized pixel_values. It reads
// the header first (image.DecodeConfig) and rejects an oversized image before
// decoding — a decompression bomb (tiny file, huge declared dimensions) errors
// here, it never allocates the full bitmap.
func Preprocess(data []byte, cfg Config) (*PixelValues, error) {
	if cfg.Size <= 0 {
		return nil, fmt.Errorf("vision: invalid target size %d", cfg.Size)
	}
	ic, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("vision: decode header: %w", err)
	}
	if ic.Width <= 0 || ic.Height <= 0 {
		return nil, fmt.Errorf("vision: non-positive image dims %dx%d", ic.Width, ic.Height)
	}
	if cfg.MaxPixels > 0 && ic.Width*ic.Height > cfg.MaxPixels {
		return nil, fmt.Errorf("vision: image %dx%d exceeds %d-pixel limit (decompression bomb?)", ic.Width, ic.Height, cfg.MaxPixels)
	}

	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("vision: decode: %w", err)
	}
	// Normalize to straight-alpha 8-bit RGBA so channel reads are trivial and
	// premultiplication doesn't skew an image with transparency.
	b := img.Bounds()
	nr := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(nr, nr.Bounds(), img, b.Min, draw.Src)

	size := cfg.Size
	out := make([]float32, 3*size*size)
	resizeNormalize(nr, out, cfg)
	return &PixelValues{Data: out, Size: size}, nil
}

// resizeNormalize bilinearly resizes nr to cfg.Size² and writes normalized CHW
// floats into out. Half-pixel sample centers (align_corners=false), matching the
// torch/PIL convention (the interpolation kernel — bilinear vs bicubic — is the
// remaining parity gap, see the package note).
func resizeNormalize(nr *image.NRGBA, out []float32, cfg Config) {
	size := cfg.Size
	sw, sh := nr.Rect.Dx(), nr.Rect.Dy()
	stride := nr.Stride
	plane := size * size
	for dy := range size {
		sy := (float64(dy)+0.5)*float64(sh)/float64(size) - 0.5
		y0, fy := splitCoord(sy, sh)
		y1 := clampInt(y0+1, 0, sh-1)
		y0 = clampInt(y0, 0, sh-1)
		for dx := range size {
			sx := (float64(dx)+0.5)*float64(sw)/float64(size) - 0.5
			x0, fx := splitCoord(sx, sw)
			x1 := clampInt(x0+1, 0, sw-1)
			x0 = clampInt(x0, 0, sw-1)
			for c := range 3 {
				p00 := float64(nr.Pix[y0*stride+x0*4+c])
				p01 := float64(nr.Pix[y0*stride+x1*4+c])
				p10 := float64(nr.Pix[y1*stride+x0*4+c])
				p11 := float64(nr.Pix[y1*stride+x1*4+c])
				top := p00 + (p01-p00)*fx
				bot := p10 + (p11-p10)*fx
				v := float32((top + (bot-top)*fy) / 255.0) // → [0,1]
				out[c*plane+dy*size+dx] = (v - cfg.Mean[c]) / cfg.Std[c]
			}
		}
	}
}

// splitCoord returns the floor index and fractional part of a source coordinate,
// with the fraction zeroed once the coordinate is past the edge (so edge clamping
// doesn't blend in a wrapped neighbor).
func splitCoord(s float64, n int) (int, float64) {
	if s <= 0 {
		return 0, 0
	}
	if s >= float64(n-1) {
		return n - 1, 0
	}
	i := int(s)
	return i, s - float64(i)
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
