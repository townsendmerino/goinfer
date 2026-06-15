package multimodal

import (
	"bytes"
	"fmt"
	"image"
	_ "image/jpeg" // decode JPEG inputs
	_ "image/png"  // decode PNG inputs
	"math"
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
	for bh := 0; bh < bH; bh++ {
		for bw := 0; bw < bW; bw++ {
			for mh := 0; mh < merge; mh++ {
				for mw := 0; mw < merge; mw++ {
					gh, gw := bh*merge+mh, bw*merge+mw
					di := pi * patchDim
					for c := 0; c < 3; c++ {
						for t := 0; t < tp; t++ { // single image: every temporal frame identical
							_ = t
							for ph := 0; ph < patch; ph++ {
								row := ((gh*patch + ph) * wb) * 3
								for pw := 0; pw < patch; pw++ {
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
// preserving aspect ratio — the HF smart_resize. (Banker's-rounding edge cases vs
// Python round() are sub-factor and don't affect grid-aligned inputs.)
func qwenSmartResize(h, w, factor, minPixels, maxPixels int) (int, int) {
	roundF := func(x int) int { return int(math.Round(float64(x)/float64(factor))) * factor }
	hb := max(factor, roundF(h))
	wb := max(factor, roundF(w))
	if hb*wb > maxPixels {
		beta := math.Sqrt(float64(h*w) / float64(maxPixels))
		hb = int(math.Floor(float64(h)/beta/float64(factor))) * factor
		wb = int(math.Floor(float64(w)/beta/float64(factor))) * factor
	} else if hb*wb < minPixels {
		beta := math.Sqrt(float64(minPixels) / float64(h*w))
		hb = int(math.Ceil(float64(h)*beta/float64(factor))) * factor
		wb = int(math.Ceil(float64(w)*beta/float64(factor))) * factor
	}
	return hb, wb
}

// qwenResizeNormalize resizes the image to (hb,wb) — identity when already that
// size, else bilinear — then rescales /255 and CLIP-normalizes, returning HWC
// float [hb*wb*3]. Bilinear is a placeholder for PIL-bicubic (resize-parity is a
// follow-on); grid-aligned inputs skip it entirely for bit-exact output.
func qwenResizeNormalize(img image.Image, h, w, hb, wb int, cfg QwenPreprocessConfig) []float32 {
	b := img.Bounds()
	at := func(y, x int) (r, g, bl float32) {
		cr, cg, cb, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA() // 16-bit; >>8 recovers the 8-bit value exactly
		return float32(cr >> 8), float32(cg >> 8), float32(cb >> 8)
	}
	out := make([]float32, hb*wb*3)
	for y := 0; y < hb; y++ {
		for x := 0; x < wb; x++ {
			var r, g, bl float32
			if hb == h && wb == w {
				r, g, bl = at(y, x)
			} else {
				fy := (float64(y)+0.5)*float64(h)/float64(hb) - 0.5
				fx := (float64(x)+0.5)*float64(w)/float64(wb) - 0.5
				r, g, bl = qwenBilinear(at, fy, fx, h, w)
			}
			i := (y*wb + x) * 3
			out[i+0] = (r/255 - cfg.Mean[0]) / cfg.Std[0]
			out[i+1] = (g/255 - cfg.Mean[1]) / cfg.Std[1]
			out[i+2] = (bl/255 - cfg.Mean[2]) / cfg.Std[2]
		}
	}
	return out
}

func qwenBilinear(at func(y, x int) (float32, float32, float32), fy, fx float64, h, w int) (float32, float32, float32) {
	clamp := func(v, hi int) int {
		if v < 0 {
			return 0
		}
		if v > hi {
			return hi
		}
		return v
	}
	y0, x0 := int(math.Floor(fy)), int(math.Floor(fx))
	dy, dx := float32(fy-float64(y0)), float32(fx-float64(x0))
	y0c, y1c := clamp(y0, h-1), clamp(y0+1, h-1)
	x0c, x1c := clamp(x0, w-1), clamp(x0+1, w-1)
	r00, g00, b00 := at(y0c, x0c)
	r01, g01, b01 := at(y0c, x1c)
	r10, g10, b10 := at(y1c, x0c)
	r11, g11, b11 := at(y1c, x1c)
	lerp := func(a, b, t float32) float32 { return a + (b-a)*t }
	r := lerp(lerp(r00, r01, dx), lerp(r10, r11, dx), dy)
	g := lerp(lerp(g00, g01, dx), lerp(g10, g11, dx), dy)
	bl := lerp(lerp(b00, b01, dx), lerp(b10, b11, dx), dy)
	return r, g, bl
}
