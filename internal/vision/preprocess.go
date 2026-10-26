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
// bicubic and is exact: ResizeBicubicAA, bicubic.go. This file is the structure
// + the security guard.
//
// "multimodal.md" refers to
// https://github.com/townsendmerino/goinfer/blob/main/docs/multimodal.md.
package vision

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"

	_ "image/jpeg" // register JPEG decoder
	_ "image/png"  // register PNG decoder
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
// A ZERO MaxPixels used to mean "no limit" (audit C-06), so any caller who built
// a Config literal without naming the field — the natural thing to do, and what
// a zero value gives you — silently turned the guard OFF. That is the wrong
// direction for a fail-safe: the default for an unset security limit should be
// the limit, not its absence. A caller who genuinely wants a larger bound sets
// a larger number.
const DefaultMaxPixels = 16 << 20

// maxPixels resolves the effective decompression-bomb cap: the configured value
// when set, DefaultMaxPixels when not. Factored out so the resolution can be
// tested without synthesising a 16-megapixel image (audit C-06).
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
	// checks here (audit #23).
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
	resizeNormalizeImage(img, out, cfg)
	return &PixelValues{Data: out, Size: size}, nil
}

// resizeNormalizeImage dispatches on the decoded image's concrete type.
//
// The generic path has to materialize the WHOLE image as NRGBA first
// (draw.Draw), which for a 1920×1080 JPEG converts 2.07 M pixels — measured 28.2
// ms, 42% of Preprocess — to serve a bilinear downscale that only ever reads
// size²·4 ≈ 590 K taps (perf-campaign item 32). The specialized paths sample the
// decoded image directly and skip the copy entirely.
//
// *image.YCbCr is the JPEG case and the one that matters. Bit-identity is not
// asserted by argument here but tested: TestResizeNormalize_matchesGenericPath
// compares every path against toNRGBA+resizeNormalize on real encoded images.
func resizeNormalizeImage(img image.Image, out []float32, cfg Config) {
	switch src := img.(type) {
	case *image.YCbCr:
		resizeNormalizeYCbCr(src, out, cfg)
	case *image.NRGBA:
		if src.Rect.Min == (image.Point{}) {
			resizeNormalize(src, out, cfg) // already the right form; no copy
			return
		}
		resizeNormalize(toNRGBA(src), out, cfg)
	default:
		resizeNormalize(toNRGBA(img), out, cfg)
	}
}

// xMap is the per-output-column sampling plan. It depends only on dx, sw and
// size, so the generic path used to recompute all of it once per ROW — size
// times more often than needed.
type xMap struct {
	x0, x1 []int
	fx     []float64
}

func newXMap(size, sw int) xMap {
	m := xMap{x0: make([]int, size), x1: make([]int, size), fx: make([]float64, size)}
	for dx := range size {
		sx := (float64(dx)+0.5)*float64(sw)/float64(size) - 0.5
		i, f := splitCoord(sx, sw)
		m.x1[dx] = clampInt(i+1, 0, sw-1)
		m.x0[dx] = clampInt(i, 0, sw-1)
		m.fx[dx] = f
	}
	return m
}

// yTap returns the two source rows and the vertical fraction for output row dy.
func yTap(dy, size, sh int) (y0, y1 int, fy float64) {
	sy := (float64(dy)+0.5)*float64(sh)/float64(size) - 0.5
	i, f := splitCoord(sy, sh)
	y1 = clampInt(i+1, 0, sh-1)
	y0 = clampInt(i, 0, sh-1)
	return y0, y1, f
}

// resizeNormalizeYCbCr bilinearly resizes a JPEG-decoded image without
// materializing an NRGBA copy, converting only the four taps each output pixel
// actually reads. color.YCbCrToRGB is the same conversion image/draw applies, so
// the sampled values match.
func resizeNormalizeYCbCr(src *image.YCbCr, out []float32, cfg Config) {
	size := cfg.Size
	sw, sh := src.Rect.Dx(), src.Rect.Dy()
	plane := size * size
	m := newXMap(size, sw)

	// Tap coordinates are relative to Rect.Min, matching what draw.Draw copies.
	minX, minY := src.Rect.Min.X, src.Rect.Min.Y
	rgb := func(x, y int) (float64, float64, float64) {
		yi := src.YOffset(minX+x, minY+y)
		ci := src.COffset(minX+x, minY+y)
		r, g, b := color.YCbCrToRGB(src.Y[yi], src.Cb[ci], src.Cr[ci])
		return float64(r), float64(g), float64(b)
	}

	for dy := range size {
		y0, y1, fy := yTap(dy, size, sh)
		for dx := range size {
			x0, x1, fx := m.x0[dx], m.x1[dx], m.fx[dx]
			r00, g00, b00 := rgb(x0, y0)
			r01, g01, b01 := rgb(x1, y0)
			r10, g10, b10 := rgb(x0, y1)
			r11, g11, b11 := rgb(x1, y1)
			for c, p := range [3][4]float64{
				{r00, r01, r10, r11},
				{g00, g01, g10, g11},
				{b00, b01, b10, b11},
			} {
				top := p[0] + (p[1]-p[0])*fx
				bot := p[2] + (p[3]-p[2])*fx
				v := float32((top + (bot-top)*fy) / 255.0)
				out[c*plane+dy*size+dx] = (v - cfg.Mean[c]) / cfg.Std[c]
			}
		}
	}
}

// toNRGBA normalizes any decoded image to straight-alpha 8-bit RGBA so channel
// reads are trivial and premultiplication doesn't skew an image with
// transparency.
func toNRGBA(img image.Image) *image.NRGBA {
	b := img.Bounds()
	nr := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(nr, nr.Bounds(), img, b.Min, draw.Src)
	return nr
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
	// The x plan depends only on dx, sw and size — it used to be recomputed for
	// every row, i.e. `size` times more often than necessary (item 32).
	m := newXMap(size, sw)
	for dy := range size {
		y0, y1, fy := yTap(dy, size, sh)
		row0 := y0 * stride
		row1 := y1 * stride
		outRowBase := dy * size
		for dx := range size {
			x0, x1, fx := m.x0[dx], m.x1[dx], m.fx[dx]
			off0 := row0 + x0*4
			off1 := row0 + x1*4
			off2 := row1 + x0*4
			off3 := row1 + x1*4
			outBase := outRowBase + dx
			for c := range 3 {
				p00 := float64(nr.Pix[off0+c])
				p01 := float64(nr.Pix[off1+c])
				p10 := float64(nr.Pix[off2+c])
				p11 := float64(nr.Pix[off3+c])
				top := p00 + (p01-p00)*fx
				bot := p10 + (p11-p10)*fx
				v := float32((top + (bot-top)*fy) / 255.0) // → [0,1]
				out[c*plane+outBase] = (v - cfg.Mean[c]) / cfg.Std[c]
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
