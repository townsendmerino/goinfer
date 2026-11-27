package imgproc

import (
	"image"
	"image/color"
	"image/draw"
)

// Norm is a per-channel normalization applied after the /255 rescale: (v - Mean[c]) / Std[c]. Std must have no zero
// entry (the caller validates).
type Norm struct {
	Mean, Std [3]float32
}

// ResizeNormalize bilinearly resizes img to size×size and writes normalized CHW floats into out
// (len 3*size*size): out[c*size*size + y*size + x]. Half-pixel sample centers (align_corners=false), the torch/PIL
// convention, without PIL's downscale antialias.
//
// It dispatches on the decoded image's concrete type. The generic path has to materialize the WHOLE image as NRGBA
// first (draw.Draw), which costs more than the bilinear downscale that reads only size²·4 taps; the specialized paths
// sample the decoded image directly and skip the copy. *image.YCbCr is the JPEG case and the one that matters.
// Bit-identity across paths is tested, not argued: TestResizeNormalize_matchesGenericPath compares every path against
// ToNRGBA+resizeNormalize on real encoded images.
func ResizeNormalize(img image.Image, out []float32, size int, n Norm) {
	switch src := img.(type) {
	case *image.YCbCr:
		resizeNormalizeYCbCr(src, out, size, n)
	case *image.NRGBA:
		if src.Rect.Min == (image.Point{}) {
			resizeNormalize(src, out, size, n) // already the right form; no copy
			return
		}
		resizeNormalize(ToNRGBA(src), out, size, n)
	default:
		resizeNormalize(ToNRGBA(img), out, size, n)
	}
}

// xMap is the per-output-column sampling plan. It depends only on dx, sw and size, so it is computed once per image,
// not once per row.
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
func resizeNormalizeYCbCr(src *image.YCbCr, out []float32, size int, n Norm) {
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
				out[c*plane+dy*size+dx] = (v - n.Mean[c]) / n.Std[c]
			}
		}
	}
}

// ToNRGBA normalizes any decoded image to straight-alpha 8-bit RGBA with its origin at (0,0), so channel reads are
// trivial and premultiplication doesn't skew an image with transparency.
func ToNRGBA(img image.Image) *image.NRGBA {
	b := img.Bounds()
	nr := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(nr, nr.Bounds(), img, b.Min, draw.Src)
	return nr
}

// resizeNormalize bilinearly resizes nr to size² and writes normalized CHW
// floats into out. Half-pixel sample centers (align_corners=false), matching the
// torch/PIL convention.
func resizeNormalize(nr *image.NRGBA, out []float32, size int, n Norm) {
	sw, sh := nr.Rect.Dx(), nr.Rect.Dy()
	stride := nr.Stride
	plane := size * size
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
				out[c*plane+outBase] = (v - n.Mean[c]) / n.Std[c]
			}
		}
	}
}

// BilinearHWC01 bilinearly resizes img to targetH×targetW (independent axes, the same per-axis plan as
// ResizeNormalize) and returns [0,1]-scaled HWC floats with no mean/std subtraction.
func BilinearHWC01(img image.Image, targetH, targetW int) []float32 {
	nr := ToNRGBA(img)
	sw, sh := nr.Rect.Dx(), nr.Rect.Dy()
	stride := nr.Stride
	out := make([]float32, targetH*targetW*3)
	m := newXMap(targetW, sw)
	for dy := range targetH {
		y0, y1, fy := yTap(dy, targetH, sh)
		row0 := y0 * stride
		row1 := y1 * stride
		outRowBase := dy * targetW * 3
		for dx := range targetW {
			x0, x1, fx := m.x0[dx], m.x1[dx], m.fx[dx]
			off0 := row0 + x0*4
			off1 := row0 + x1*4
			off2 := row1 + x0*4
			off3 := row1 + x1*4
			outBase := outRowBase + dx*3
			for c := range 3 {
				p00 := float64(nr.Pix[off0+c])
				p01 := float64(nr.Pix[off1+c])
				p10 := float64(nr.Pix[off2+c])
				p11 := float64(nr.Pix[off3+c])
				top := p00 + (p01-p00)*fx
				bot := p10 + (p11-p10)*fx
				v := float32((top + (bot-top)*fy) / 255.0)
				out[outBase+c] = v
			}
		}
	}
	return out
}

// BicubicHWC01 is the torchvision reference resize (ResizeBicubicAA on the uint8 RGB image, alpha dropped as PIL's
// convert("RGB") drops it) followed by the HF processors' [0,1] rescale, bit for bit (Rescale01). h and w are img's
// dimensions.
func BicubicHWC01(img image.Image, h, w, targetH, targetW int) []float32 {
	nr := ToNRGBA(img)
	rgb := make([]uint8, h*w*3)
	for y := range h {
		row := nr.Pix[y*nr.Stride:]
		for x := range w {
			copy(rgb[(y*w+x)*3:(y*w+x)*3+3], row[x*4:x*4+3])
		}
	}
	px := ResizeBicubicAA(rgb, h, w, targetH, targetW)
	out := make([]float32, len(px))
	for i, v := range px {
		out[i] = float32(v) * Rescale01
	}
	return out
}

// Rescale01 is the HF processors' rescale_factor (1/255 as the float64 0.00392156862745098) as torch applies it to a
// float32 image: a multiply by the factor rounded to float32. A division by 255 differs from it by one ulp on 126 of
// the 256 byte values.
const Rescale01 = float32(0.00392156862745098)

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
