package embeddinggemma2

import (
	"bytes"
	"fmt"
	"image"
	"image/draw"
	_ "image/jpeg" // decoders for image.Decode
	_ "image/png"
	"math"
	"strings"

	"github.com/townsendmerino/aikit/vision"
)

// Image preprocessing with the reference's resize (Phase V, docs/tasks/task-embeddinggemma2.md). The checkpoint's
// Gemma4ImageProcessor resizes the decoded uint8 image with torchvision's antialiased bicubic
// (tvF.resize(..., BICUBIC, antialias=True)) before rescaling to [0, 1]; aikit's vision.Gemma4Preprocess resizes
// bilinearly. ResizeBicubic is that bicubic, as torchvision's uint8 path computes it: separable, the horizontal pass
// first, Keys weights with a = -0.5 over a support widened by the scale on downscale, normalised per output pixel,
// quantised to int16 at the precision the pass's largest weight allows, accumulated in integers with a rounding offset,
// and clamped to uint8 after each pass. Everything else (the target size, the patch layout, the positions) is aikit's.

// ImageResize chooses an encoder's image resize.
type ImageResize string

const (
	ResizeBilinear ImageResize = "bilinear" // aikit's vision.Gemma4Preprocess
	ResizeBicubic  ImageResize = "bicubic"  // the reference's torchvision antialiased bicubic (this file)
)

// eg2ResizeFloat (tests only) runs the bicubic in float64, rounding to uint8 once at the end, instead of
// torchvision's fixed-point passes.
var eg2ResizeFloat = false

// defaultImageResize is the resize an encoder uses until SetImageResize.
var defaultImageResize = ResizeBicubic

// ParseImageResize reads a resize name, case-insensitively.
func ParseImageResize(s string) (ImageResize, error) {
	switch r := ImageResize(strings.ToLower(s)); r {
	case ResizeBilinear, ResizeBicubic:
		return r, nil
	}
	return "", fmt.Errorf("embeddinggemma2: image resize %q (want %s or %s)", s, ResizeBilinear, ResizeBicubic)
}

// SetImageResize chooses how images are resized before the tower (default: defaultImageResize).
func (e *Encoder) SetImageResize(r ImageResize) error {
	r, err := ParseImageResize(string(r))
	if err != nil {
		return err
	}
	e.resize = r
	return nil
}

// ImageResizeMode is the encoder's image resize.
func (e *Encoder) ImageResizeMode() ImageResize {
	if e.resize == "" {
		return defaultImageResize
	}
	return e.resize
}

// preprocessImage returns the tower's patches and positions for encoded image bytes under the encoder's resize.
func (e *Encoder) preprocessImage(data []byte) ([]float32, [][2]int, error) {
	if e.ImageResizeMode() == ResizeBilinear {
		return vision.Gemma4Preprocess(data, MaxImageSoftTokens)
	}
	return PreprocessBicubic(data, MaxImageSoftTokens)
}

// PreprocessBicubic is vision.Gemma4Preprocess with the reference's resize: [0, 1] patches
// [numPatches, 3*16*16] in row-major patch order, (row, col, channel) within a patch, and (x=col, y=row) positions.
func PreprocessBicubic(data []byte, maxSoftTokens int) ([]float32, [][2]int, error) {
	const patch, pool = 16, 3
	const maxPixels = 16 << 20 // the same decompression-bomb bound as aikit's
	ic, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return nil, nil, fmt.Errorf("embeddinggemma2: image header: %w", err)
	}
	if ic.Width <= 0 || ic.Height <= 0 || int64(ic.Width)*int64(ic.Height) > maxPixels {
		return nil, nil, fmt.Errorf("embeddinggemma2: image %dx%d outside (0, %d] pixels", ic.Width, ic.Height, maxPixels)
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, nil, fmt.Errorf("embeddinggemma2: image: %w", err)
	}
	tH, tW := vision.Gemma4AspectRatioSize(ic.Height, ic.Width, maxSoftTokens, patch, pool)
	if tH < patch || tW < patch {
		return nil, nil, fmt.Errorf("embeddinggemma2: resize target %dx%d is under one patch", tH, tW)
	}
	rgb := toRGB(img)
	px := resizeBicubicRGB(rgb, ic.Height, ic.Width, tH, tW)
	gh, gw := tH/patch, tW/patch
	pd := 3 * patch * patch
	patches := make([]float32, gh*gw*pd)
	pos := make([][2]int, 0, gh*gw)
	for r := range gh {
		for c := range gw {
			base := (r*gw + c) * pd
			k := 0
			for py := range patch {
				for pxl := range patch {
					o := ((r*patch+py)*tW + c*patch + pxl) * 3
					for ch := range 3 {
						patches[base+k] = float32(px[o+ch]) / 255
						k++
					}
				}
			}
			pos = append(pos, [2]int{c, r})
		}
	}
	return patches, pos, nil
}

// toRGB returns the image's pixels as packed 8-bit RGB, HWC, dropping alpha (the processor's do_convert_rgb on an
// opaque image).
func toRGB(img image.Image) []uint8 {
	b := img.Bounds()
	nr := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(nr, nr.Rect, img, b.Min, draw.Src)
	out := make([]uint8, b.Dx()*b.Dy()*3)
	for i := range b.Dx() * b.Dy() {
		copy(out[i*3:i*3+3], nr.Pix[i*4:i*4+3])
	}
	return out
}

// aaWeights are one axis's antialiased bicubic taps: for output i, inputs [start[i], start[i]+len(w[i])).
type aaWeights struct {
	start []int
	w     [][]float64
}

// bicubicAA is the Keys cubic with a = -0.5, torchvision's antialiased bicubic filter.
func bicubicAA(x float64) float64 {
	const a = -0.5
	x = math.Abs(x)
	if x < 1 {
		return ((a+2)*x-(a+3))*x*x + 1
	}
	if x < 2 {
		return (((x-5)*x+8)*x - 4) * a
	}
	return 0
}

// aaTaps computes torchvision's antialiased taps from in samples to out (align_corners false).
func aaTaps(in, out int) aaWeights {
	scale := float64(in) / float64(out)
	support, invscale := 2.0, 1.0
	if scale >= 1 {
		support, invscale = 2*scale, 1/scale
	}
	t := aaWeights{start: make([]int, out), w: make([][]float64, out)}
	for i := range out {
		center := scale * (float64(i) + 0.5)
		xmin := max(int(center-support+0.5), 0)
		xmax := min(int(center+support+0.5), in)
		w := make([]float64, xmax-xmin)
		var total float64
		for j := range w {
			w[j] = bicubicAA((float64(j+xmin) - center + 0.5) * invscale)
			total += w[j]
		}
		if total != 0 {
			for j := range w {
				w[j] /= total
			}
		}
		t.start[i], t.w[i] = xmin, w
	}
	return t
}

// resizeBicubicRGB resizes HWC RGB from (h, w) to (th, tw), the horizontal pass first, each pass skipped when its
// size does not change.
func resizeBicubicRGB(src []uint8, h, w, th, tw int) []uint8 {
	cur, cw := src, w
	if tw != w {
		cur = resizePass(cur, h, w, tw, true)
		cw = tw
	}
	if th != h {
		cur = resizePass(cur, h, cw, th, false)
	}
	return cur
}

// resizePass resizes along one axis: horizontal (w -> n) or vertical (h -> n), in torchvision's uint8 fixed point
// (or in float64 under eg2ResizeFloat).
func resizePass(src []uint8, h, w, n int, horizontal bool) []uint8 {
	in := w
	if !horizontal {
		in = h
	}
	taps := aaTaps(in, n)
	oh, ow := h, n
	if !horizontal {
		oh, ow = n, w
	}
	dst := make([]uint8, oh*ow*3)
	// torchvision's precision: the largest that keeps the largest weight under 2^15 at one more bit.
	var maxW float64
	for _, ws := range taps.w {
		for _, x := range ws {
			maxW = max(maxW, x)
		}
	}
	prec := uint(0)
	for prec = 0; prec < 22; prec++ {
		if int(0.5+maxW*float64(int(1)<<(prec+1))) >= 1<<15 {
			break
		}
	}
	iw := make([][]int32, len(taps.w))
	for i, ws := range taps.w {
		iw[i] = make([]int32, len(ws))
		for j, x := range ws {
			r := 0.5
			if x < 0 {
				r = -0.5
			}
			iw[i][j] = int32(int16(r + x*float64(int(1)<<prec)))
		}
	}
	at := func(y, x, c int) int { return (y*w+x)*3 + c }
	for oy := range oh {
		for ox := range ow {
			i := ox
			if !horizontal {
				i = oy
			}
			s := taps.start[i]
			for c := range 3 {
				if eg2ResizeFloat {
					var acc float64
					for j, wt := range taps.w[i] {
						if horizontal {
							acc += wt * float64(src[at(oy, s+j, c)])
						} else {
							acc += wt * float64(src[at(s+j, ox, c)])
						}
					}
					dst[(oy*ow+ox)*3+c] = uint8(math.Max(0, math.Min(255, math.Round(acc))))
					continue
				}
				acc := int32(1) << (prec - 1)
				for j, wt := range iw[i] {
					if horizontal {
						acc += wt * int32(src[at(oy, s+j, c)])
					} else {
						acc += wt * int32(src[at(s+j, ox, c)])
					}
				}
				v := acc >> prec
				dst[(oy*ow+ox)*3+c] = uint8(max(0, min(255, v)))
			}
		}
	}
	return dst
}
