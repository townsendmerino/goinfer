package multimodal

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg" // decode JPEG inputs
	_ "image/png"  // decode PNG inputs
	"math"
	"os"
	"path/filepath"
)

// Qwen2.5-VL image preprocessing: image bytes -> pre-flattened pixel_values
// [n_patches, channels*temporal*patch*patch] + grid_thw (t,h,w in patch units),
// matching HF Qwen2VLImageProcessor. The aikit Qwen vision encoder consumes this
// directly (it does no preprocessing — see docs/prompts/aikit-qwen25vl-vit.md).
//
// Three stages: smart_resize (round H,W to a multiple of patch*merge within the
// pixel budget), resize+rescale+CLIP-normalize, and the spatial-merge patchify
// rearrange. The patchify order is HF-exact: patches sequence (block-row,
// block-col, merge-row, merge-col), each patch's values (channel, temporal,
// patch-row, patch-col). The resize is BICUBIC (qwenBicubicU8, called at the resize site) and is
// a no-op when the image is already grid-aligned, so a pre-sized image preprocesses bit-exactly
// (TestQwenPreprocess_exact). It is tolerance-matched to PIL rather than bit-exact, because the
// coefficients here are float where PIL's are fixed-point.
//
// N-34: this said "the resize here is bilinear" and described PIL-bicubic parity as a future
// refinement, after the bicubic path had already landed.

// QwenPreprocessConfig holds the Qwen2.5-VL image-processor parameters.
type QwenPreprocessConfig struct {
	PatchSize, MergeSize, TemporalPatchSize int
	MinPixels, MaxPixels                    int
	Mean, Std                               [3]float32
}

// QwenDefaultPreprocess returns the standard Qwen2.5-VL processor config (patch 14,
// merge 2, temporal 2, OpenAI-CLIP normalization).
func QwenDefaultPreprocess() QwenPreprocessConfig {
	return QwenPreprocessConfig{
		PatchSize: 14, MergeSize: 2, TemporalPatchSize: 2,
		MinPixels: 4 * 28 * 28, MaxPixels: 16384 * 28 * 28,
		Mean: [3]float32{0.48145466, 0.4578275, 0.40821073},
		Std:  [3]float32{0.26862954, 0.26130258, 0.27577711},
	}
}

// LoadQwenPreprocessConfig reads dir/preprocessor_config.json into a
// QwenPreprocessConfig, falling back to QwenDefaultPreprocess for an absent file or
// field — so a model dir with a tuned min/max or non-CLIP normalization is honored,
// while a stripped checkpoint still works on the standard defaults.
func LoadQwenPreprocessConfig(dir string) (QwenPreprocessConfig, error) {
	cfg := QwenDefaultPreprocess()
	raw, err := os.ReadFile(filepath.Join(dir, "preprocessor_config.json"))
	if err != nil {
		return cfg, nil // no file: standard Qwen2.5-VL defaults
	}
	var pc struct {
		MinPixels         int       `json:"min_pixels"`
		MaxPixels         int       `json:"max_pixels"`
		PatchSize         int       `json:"patch_size"`
		MergeSize         int       `json:"merge_size"`
		TemporalPatchSize int       `json:"temporal_patch_size"`
		ImageMean         []float32 `json:"image_mean"`
		ImageStd          []float32 `json:"image_std"`
	}
	if err := json.Unmarshal(raw, &pc); err != nil {
		return cfg, fmt.Errorf("multimodal(qwen): parse preprocessor_config: %w", err)
	}
	if pc.MinPixels > 0 {
		cfg.MinPixels = pc.MinPixels
	}
	if pc.MaxPixels > 0 {
		cfg.MaxPixels = pc.MaxPixels
	}
	if pc.PatchSize > 0 {
		cfg.PatchSize = pc.PatchSize
	}
	if pc.MergeSize > 0 {
		cfg.MergeSize = pc.MergeSize
	}
	if pc.TemporalPatchSize > 0 {
		cfg.TemporalPatchSize = pc.TemporalPatchSize
	}
	if len(pc.ImageMean) == 3 {
		cfg.Mean = [3]float32{pc.ImageMean[0], pc.ImageMean[1], pc.ImageMean[2]}
	}
	if len(pc.ImageStd) == 3 {
		cfg.Std = [3]float32{pc.ImageStd[0], pc.ImageStd[1], pc.ImageStd[2]}
	}
	return cfg, nil
}

// qwenMaxInputPixels bounds the raw (pre-resize) image area QwenPreprocess will allocate for,
// independent of cfg.MaxPixels (which caps only the resized output). ~33.5 MP: above any real
// camera image, below the memory a decompression bomb would demand (audit M-15).
const qwenMaxInputPixels = 32 * 1024 * 1024

// QwenPreprocess turns image bytes into the Qwen2.5-VL vision input. Returns the
// flattened pixel_values and grid_thw = (1, grid_h, grid_w) for a single image.
func QwenPreprocess(data []byte, cfg QwenPreprocessConfig) ([]float32, [3]int, error) {
	// Peek the declared dimensions from the header BEFORE decoding pixels.
	// image.DecodeConfig only parses the header (PNG's IHDR chunk / JPEG's SOF
	// marker) — unlike image.Decode, it never allocates or fills a pixel buffer.
	// A decompression bomb (a few KB of highly-compressible data declaring a huge
	// canvas) must be rejected here: image.Decode itself allocates the full
	// decoded pixel buffer (e.g. ~1.2 GB for a 10000×10000 PNG) as part of
	// decoding, before any check on img.Bounds() could run (audit M-15 originally
	// only guarded the later qwenExtractRGB/qwenBicubicU8 allocations, missing
	// this earlier and larger one).
	ic, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return nil, [3]int{}, fmt.Errorf("multimodal(qwen): decode image header: %w", err)
	}
	if ic.Width <= 0 || ic.Height <= 0 {
		return nil, [3]int{}, fmt.Errorf("multimodal(qwen): empty image")
	}
	if int64(ic.Height)*int64(ic.Width) > qwenMaxInputPixels {
		return nil, [3]int{}, fmt.Errorf("multimodal(qwen): image %dx%d exceeds %d-pixel input limit", ic.Width, ic.Height, qwenMaxInputPixels)
	}

	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, [3]int{}, fmt.Errorf("multimodal(qwen): decode image: %w", err)
	}
	b := img.Bounds()
	h, w := b.Dy(), b.Dx()
	if h == 0 || w == 0 {
		return nil, [3]int{}, fmt.Errorf("multimodal(qwen): empty image")
	}
	// Re-check the decoded bounds too: cheap, and guards against a decoder whose
	// DecodeConfig and Decode paths ever disagree.
	if int64(h)*int64(w) > qwenMaxInputPixels {
		return nil, [3]int{}, fmt.Errorf("multimodal(qwen): image %dx%d exceeds %d-pixel input limit", w, h, qwenMaxInputPixels)
	}
	factor := cfg.PatchSize * cfg.MergeSize
	hb, wb := qwenSmartResize(h, w, factor, cfg.MinPixels, cfg.MaxPixels)

	// norm[(y*wb+x)*3+c] = resized, rescaled, CLIP-normalized channel value.
	norm := qwenResizeNormalize(img, h, w, hb, wb, cfg)

	patch, merge, tp := cfg.PatchSize, cfg.MergeSize, cfg.TemporalPatchSize
	gridH, gridW := hb/patch, wb/patch
	bH, bW := gridH/merge, gridW/merge
	patchDim := 3 * tp * patch * patch
	out := make([]float32, gridH*gridW*patchDim)
	pi := 0
	for bh := range bH {
		for bw := range bW {
			for mh := range merge {
				for mw := range merge {
					gh, gw := bh*merge+mh, bw*merge+mw
					di := pi * patchDim
					for c := range 3 {
						for t := range tp { // single image: every temporal frame identical
							_ = t
							for ph := range patch {
								row := ((gh*patch + ph) * wb) * 3
								for pw := range patch {
									out[di] = norm[row+(gw*patch+pw)*3+c]
									di++
								}
							}
						}
					}
					pi++
				}
			}
		}
	}
	return out, [3]int{1, gridH, gridW}, nil
}

// qwenSmartResize rounds (h,w) to multiples of factor within [minPixels,maxPixels],
// preserving aspect ratio — the HF smart_resize. Uses round-half-to-even
// (math.RoundToEven) to match Python's round(), so the chosen grid is identical
// (half-away-from-zero would pick a different size at x.5/factor).
func qwenSmartResize(h, w, factor, minPixels, maxPixels int) (int, int) {
	roundF := func(x int) int { return int(math.RoundToEven(float64(x)/float64(factor))) * factor }
	hb := max(factor, roundF(h))
	wb := max(factor, roundF(w))
	if hb*wb > maxPixels {
		beta := math.Sqrt(float64(h*w) / float64(maxPixels))
		// max(factor, …): an extreme aspect ratio floors a dimension to 0 → empty
		// pixel_values with no error (HF raises past ratio 200). Keep at least one cell.
		hb = max(factor, int(math.Floor(float64(h)/beta/float64(factor)))*factor)
		wb = max(factor, int(math.Floor(float64(w)/beta/float64(factor)))*factor)
	} else if hb*wb < minPixels {
		beta := math.Sqrt(float64(minPixels) / float64(h*w))
		hb = int(math.Ceil(float64(h)*beta/float64(factor))) * factor
		wb = int(math.Ceil(float64(w)*beta/float64(factor))) * factor
	}
	return hb, wb
}

// qwenResizeNormalize resizes the image to (hb,wb) — identity when already that
// size, else PIL bicubic — then rescales /255 and CLIP-normalizes, returning HWC
// float [hb*wb*3]. Grid-aligned inputs skip the resize for bit-exact output.
func qwenResizeNormalize(img image.Image, h, w, hb, wb int, cfg QwenPreprocessConfig) []float32 {
	src := qwenExtractRGB(img, h, w) // [h*w*3], 0..255
	resized := src
	if hb != h || wb != w {
		resized = qwenBicubicU8(src, h, w, hb, wb)
	}
	out := make([]float32, hb*wb*3)
	for i := range out {
		c := i % 3
		out[i] = (resized[i]/255 - cfg.Mean[c]) / cfg.Std[c]
	}
	return out
}

// qwenExtractRGB reads img into a [h*w*3] 0..255 channel-last float buffer. The
// >>8 recovers the 8-bit channel value exactly for an 8-bit (PNG/JPEG) source.
//
// P-19: the generic img.At(x,y).RGBA() path dispatches through the image.Image
// and color.Color interfaces per pixel — measured ~1-2s per image at the vision
// preprocessing cap. *image.YCbCr (Go's decoded-JPEG format — the common case
// for real photos) and *image.RGBA get a direct fast path reading the backing
// buffer instead. Both are PROVABLY bit-identical to the generic path, not just
// close: image.RGBA.At() computes uint32(pix)*0x101, and (pix*0x101)>>8 == pix
// exactly for any pix in [0,255] (pix*257 = pix*256+pix, and the >>8 term
// vanishes since pix<256) — so reading Pix directly and skipping the >>8
// round-trip gives the same result. image.YCbCr.At() calls color.YCbCrToRGB
// then wraps it in color.RGBA, so calling color.YCbCrToRGB directly and
// skipping the same round-trip is exactly the same identity. Every other
// concrete type (NRGBA's alpha premultiplication makes a naive buffer read
// WRONG for a transparent pixel, so it is deliberately not fast-pathed) falls
// through to the generic loop unchanged.
func qwenExtractRGB(img image.Image, h, w int) []float32 {
	b := img.Bounds()
	out := make([]float32, h*w*3)
	switch src := img.(type) {
	case *image.YCbCr:
		for y := range h {
			for x := range w {
				yi := src.YOffset(b.Min.X+x, b.Min.Y+y)
				ci := src.COffset(b.Min.X+x, b.Min.Y+y)
				r, g, bl := color.YCbCrToRGB(src.Y[yi], src.Cb[ci], src.Cr[ci])
				i := (y*w + x) * 3
				out[i+0], out[i+1], out[i+2] = float32(r), float32(g), float32(bl)
			}
		}
		return out
	case *image.RGBA:
		for y := range h {
			for x := range w {
				pi := src.PixOffset(b.Min.X+x, b.Min.Y+y)
				i := (y*w + x) * 3
				out[i+0], out[i+1], out[i+2] = float32(src.Pix[pi+0]), float32(src.Pix[pi+1]), float32(src.Pix[pi+2])
			}
		}
		return out
	}
	for y := range h {
		for x := range w {
			cr, cg, cb, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
			i := (y*w + x) * 3
			out[i+0], out[i+1], out[i+2] = float32(cr>>8), float32(cg>>8), float32(cb>>8)
		}
	}
	return out
}

// qwenBicubicU8 resizes src ([h*w*3] 0..255 channel-last) to [hb*wb*3] with PIL's
// bicubic filter (Keys cubic a=-0.5, antialiased on downscale), separable
// horizontal then vertical with a uint8 clamp+round between/after passes (matching
// PIL's uint8 intermediates). Tolerance-matched to PIL — float coefficients, not
// PIL's fixed-point, so not bit-exact.
func qwenBicubicU8(src []float32, h, w, hb, wb int) []float32 {
	hxmin, hk := qwenCubicCoeffs(w, wb) // horizontal: w -> wb
	tmp := make([]float32, h*wb*3)
	for y := range h {
		for xx := range wb {
			k := hk[xx]
			var s [3]float64
			for x := range k {
				si := (y*w + hxmin[xx] + x) * 3
				s[0] += k[x] * float64(src[si])
				s[1] += k[x] * float64(src[si+1])
				s[2] += k[x] * float64(src[si+2])
			}
			di := (y*wb + xx) * 3
			tmp[di], tmp[di+1], tmp[di+2] = qwenClip8(s[0]), qwenClip8(s[1]), qwenClip8(s[2])
		}
	}
	vymin, vk := qwenCubicCoeffs(h, hb) // vertical: h -> hb
	dst := make([]float32, hb*wb*3)
	for yy := range hb {
		k := vk[yy]
		for x := range wb {
			var s [3]float64
			for y := range k {
				si := ((vymin[yy]+y)*wb + x) * 3
				s[0] += k[y] * float64(tmp[si])
				s[1] += k[y] * float64(tmp[si+1])
				s[2] += k[y] * float64(tmp[si+2])
			}
			di := (yy*wb + x) * 3
			dst[di], dst[di+1], dst[di+2] = qwenClip8(s[0]), qwenClip8(s[1]), qwenClip8(s[2])
		}
	}
	return dst
}

func qwenClip8(v float64) float32 {
	v = math.Round(v) // PIL: (int)(v+0.5) — round-half-up for the non-negative resampled range
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return float32(v)
}

// qwenCubicCoeffs precomputes PIL's per-output-pixel sample start (xmin) and
// normalized cubic weights for a 1-D resize inSize->outSize (precompute_coeffs).
func qwenCubicCoeffs(inSize, outSize int) (xmins []int, kk [][]float64) {
	scale := float64(inSize) / float64(outSize)
	filterscale := scale
	if filterscale < 1 {
		filterscale = 1
	}
	support := 2.0 * filterscale // cubic support = 2.0
	xmins = make([]int, outSize)
	kk = make([][]float64, outSize)
	for xx := range outSize {
		center := (float64(xx) + 0.5) * scale
		ss := 1.0 / filterscale
		xmin := max(int(center-support+0.5), 0)
		xmax := min(int(center+support+0.5), inSize)
		n := xmax - xmin
		k := make([]float64, n)
		var ww float64
		for x := range n {
			w := qwenCubicFilter((float64(x+xmin) - center + 0.5) * ss)
			k[x] = w
			ww += w
		}
		if ww != 0 {
			for x := range k {
				k[x] /= ww
			}
		}
		xmins[xx], kk[xx] = xmin, k
	}
	return
}

func qwenCubicFilter(x float64) float64 {
	const a = -0.5
	if x < 0 {
		x = -x
	}
	switch {
	case x < 1:
		return ((a+2)*x-(a+3))*x*x + 1
	case x < 2:
		return (((x-5)*x+8)*x - 4) * a
	default:
		return 0
	}
}
