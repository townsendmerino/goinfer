package multimodal

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
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
// patch-row, patch-col). PIL-bicubic resize parity is a separate refinement — the
// resize here is bilinear (a no-op when the image is already grid-aligned), so a
// pre-sized image preprocesses bit-exactly (TestQwenPreprocess_exact).

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

// QwenPreprocess turns image bytes into the Qwen2.5-VL vision input. Returns the
// flattened pixel_values and grid_thw = (1, grid_h, grid_w) for a single image.
func QwenPreprocess(data []byte, cfg QwenPreprocessConfig) ([]float32, [3]int, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, [3]int{}, fmt.Errorf("multimodal(qwen): decode image: %w", err)
	}
	b := img.Bounds()
	h, w := b.Dy(), b.Dx()
	if h == 0 || w == 0 {
		return nil, [3]int{}, fmt.Errorf("multimodal(qwen): empty image")
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
func qwenExtractRGB(img image.Image, h, w int) []float32 {
	b := img.Bounds()
	out := make([]float32, h*w*3)
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
