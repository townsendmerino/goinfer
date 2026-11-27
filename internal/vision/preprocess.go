// Package vision is goinfer's pure-Go image preprocessing for vision-language
// models: decode → resize → normalize → pixel_values, the tensor a vision
// encoder (SigLIP / ViT) consumes. Stdlib-only (no cgo), and it bounds the
// decoded pixel count BEFORE decoding so a hostile/corrupt image yields a typed
// error, never an OOM (the campaign Track-2 posture, extended to image bytes).
//
// Parity note: the resize here is bilinear (half-pixel centers). Gemma 3's own
// processor is bilinear too (google/gemma-3-4b-it preprocessor_config.json:
// "resample": 2, read 2026-10-06; this note used to say BICUBIC), but PIL's
// bilinear antialiases on a downscale and this one does not, so it is NOT
// pixel-exact on a downscaled image (unmeasured). Gemma 4's reference resize is
// bicubic and is exact: ResizeBicubicAA. The resize math lives in
// internal/vision/imgproc; this file is the structure + the security guard.
//
// "multimodal.md" refers to
// https://github.com/townsendmerino/goinfer/blob/main/docs/multimodal.md.
package vision

import (
	"bytes"
	"fmt"
	"image"

	_ "image/jpeg" // register JPEG decoder
	_ "image/png"  // register PNG decoder

	"github.com/townsendmerino/goinfer/internal/vision/imgproc"
)

// Config parameterizes preprocessing per model family.
type Config struct {
	Size      int        // target square side (Gemma 3 SigLIP: 896)
	Mean      [3]float32 // per-channel normalization mean (applied after /255)
	Std       [3]float32 // per-channel normalization std
	MaxPixels int        // reject a decoded image with more than this many pixels (W*H); <=0 uses DefaultMaxPixels
}

// DefaultMaxPixels is the decompression-bomb cap applied when Config.MaxPixels
// is unset. 16 MP is far above any real photo and caps the decode allocation.
//
// A ZERO MaxPixels means this default, not "no limit": otherwise any caller who
// built a Config literal without naming the field — the natural thing to do, and
// what a zero value gives you — would silently turn the guard OFF. That is the wrong
// direction for a fail-safe: the default for an unset security limit should be
// the limit, not its absence. A caller who genuinely wants a larger bound sets
// a larger number.
const DefaultMaxPixels = 16 << 20

// maxPixels resolves the effective decompression-bomb cap: the configured value
// when set, DefaultMaxPixels when not. Factored out so the resolution can be
// tested without synthesising a 16-megapixel image.
func (c Config) maxPixels() int {
	if c.MaxPixels <= 0 {
		return DefaultMaxPixels
	}
	return c.MaxPixels
}

// Gemma3 is the SigLIP preprocessing Gemma 3 / 4 use: 896×896, mean=std=0.5 on
// all channels (so the output lands in [-1, 1]).
func Gemma3() Config {
	return Config{
		Size:      896,
		Mean:      [3]float32{0.5, 0.5, 0.5},
		Std:       [3]float32{0.5, 0.5, 0.5},
		MaxPixels: DefaultMaxPixels,
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
	// A zero Std yields +Inf (v != Mean) or NaN (v == Mean) pixels that propagate
	// silently through softmax to an all-NaN hidden state. Config is an exported
	// struct with no mandatory constructor, so reject it like the other bad-field
	// checks here.
	for c := range cfg.Std {
		if cfg.Std[c] == 0 {
			return nil, fmt.Errorf("vision: Std[%d] is 0 (would produce Inf/NaN pixels)", c)
		}
	}
	ic, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("vision: decode header: %w", err)
	}
	if ic.Width <= 0 || ic.Height <= 0 {
		return nil, fmt.Errorf("vision: non-positive image dims %dx%d", ic.Width, ic.Height)
	}
	// int64: Width and Height are attacker-controlled header ints, so on a
	// 32-bit build (386/arm) Width*Height in int can wrap negative and bypass
	// this decompression-bomb guard.
	maxPix := cfg.maxPixels()
	if int64(ic.Width)*int64(ic.Height) > int64(maxPix) {
		return nil, fmt.Errorf("vision: image %dx%d exceeds %d-pixel limit (decompression bomb?)", ic.Width, ic.Height, maxPix)
	}

	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("vision: decode: %w", err)
	}
	size := cfg.Size
	out := make([]float32, 3*size*size)
	imgproc.ResizeNormalize(img, out, size, imgproc.Norm{Mean: cfg.Mean, Std: cfg.Std})
	return &PixelValues{Data: out, Size: size}, nil
}
