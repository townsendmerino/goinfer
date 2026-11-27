package vision

import (
	"bytes"
	"fmt"
	"image"

	"github.com/townsendmerino/goinfer/internal/vision/imgproc"
)

// Gemma 4 vision preprocessing: image bytes → pre-unfolded patches + per-patch
// (x,y) grid position ids, following the real image_processing_gemma4.py
// pipeline. Deliberately NOT sharing Preprocess/Config (preprocess.go) — that
// path is a fixed SQUARE resize with mean/std normalize for SigLIP; Gemma 4
// resizes to an ASPECT-RATIO-PRESERVING, variable (multiple-of-48) target size
// with NO mean/std normalize (image_mean=[0,0,0], image_std=[1,1,1] — a plain
// [0,1] rescale; the [-1,1] remap happens model-side, in Gemma4Encoder.Forward).
//
// gemma4SupportedSoftTokens are the only legal max_soft_tokens budgets
// (image_processing_gemma4.py's _SUPPORTED_SOFT_TOKENS) — the caller picks one;
// the real per-image soft-token count is data-dependent and can be anything
// ≤ the budget (numPatches/9 after resize).
var gemma4SupportedSoftTokens = map[int]bool{70: true, 140: true, 280: true, 560: true, 1120: true}

// Gemma4AspectRatioSize computes the resize target (a multiple of
// poolingKernelSize*patchSize on each axis, aspect-ratio preserved by a single
// uniform scale factor, each axis independently rounded DOWN — never up) for a
// source image of the given height/width, bounded by maxSoftTokens patches worth
// of pixels. Mirrors get_aspect_ratio_preserving_size (image_processing_gemma4.py).
func Gemma4AspectRatioSize(height, width, maxSoftTokens, patchSize, poolingKernelSize int) (targetH, targetW int) {
	return imgproc.AspectRatioSize(height, width, maxSoftTokens, patchSize, poolingKernelSize)
}

// Gemma4Resize chooses Gemma4PreprocessResize's resampler.
type Gemma4Resize int

const (
	// Gemma4Bicubic is the reference processor's resize (torchvision's antialiased bicubic on uint8, resample=3,
	// imgproc.ResizeBicubicAA), skipped when the image is already at its target size, as the processor skips it. The default.
	Gemma4Bicubic Gemma4Resize = iota
	// Gemma4Bilinear is aikit's bilinear resize, Gemma4Preprocess's only one before v1.58.0.
	Gemma4Bilinear
)

// Gemma4Preprocess decodes image bytes and returns pre-unfolded patches
// [numPatches, 3*patchSize*patchSize] ([0,1]-scaled, no mean/std) plus per-patch
// (x,y) grid position ids, both in row-major (row,col) patch order — the input
// Gemma4Encoder.Forward expects. maxSoftTokens must be one of the five legal
// budgets {70,140,280,560,1120} (0 defaults to HF's own default, 280). It resizes
// as the reference processor does (Gemma4Bicubic; bilinear before v1.58.0).
func Gemma4Preprocess(data []byte, maxSoftTokens int) ([]float32, [][2]int, error) {
	return Gemma4PreprocessResize(data, maxSoftTokens, Gemma4Bicubic)
}

// Gemma4PreprocessResize is Gemma4Preprocess with the resampler chosen.
func Gemma4PreprocessResize(data []byte, maxSoftTokens int, resize Gemma4Resize) ([]float32, [][2]int, error) {
	const patchSize = 16
	const poolingKernelSize = 3
	const maxPixels = 16 << 20 // decompression-bomb guard, same bound as Gemma3()
	if maxSoftTokens == 0 {
		maxSoftTokens = 280
	}
	if !gemma4SupportedSoftTokens[maxSoftTokens] {
		return nil, nil, fmt.Errorf("vision: max_soft_tokens %d not one of {70,140,280,560,1120}", maxSoftTokens)
	}
	ic, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return nil, nil, fmt.Errorf("vision: decode header: %w", err)
	}
	if ic.Width <= 0 || ic.Height <= 0 {
		return nil, nil, fmt.Errorf("vision: non-positive image dims %dx%d", ic.Width, ic.Height)
	}
	if int64(ic.Width)*int64(ic.Height) > maxPixels {
		return nil, nil, fmt.Errorf("vision: image %dx%d exceeds %d-pixel limit (decompression bomb?)", ic.Width, ic.Height, maxPixels)
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, nil, fmt.Errorf("vision: decode: %w", err)
	}
	targetH, targetW := Gemma4AspectRatioSize(ic.Height, ic.Width, maxSoftTokens, patchSize, poolingKernelSize)
	if targetH < patchSize || targetW < patchSize {
		return nil, nil, fmt.Errorf("vision: resize target %dx%d smaller than one patch (%d)", targetH, targetW, patchSize)
	}
	var hwc []float32
	switch resize {
	case Gemma4Bicubic:
		hwc = imgproc.BicubicHWC01(img, ic.Height, ic.Width, targetH, targetW)
	case Gemma4Bilinear:
		hwc = imgproc.BilinearHWC01(img, targetH, targetW)
	default:
		return nil, nil, fmt.Errorf("vision: unknown Gemma4Resize %d", resize)
	}
	patches, positionIDs := imgproc.UnfoldHWC(hwc, targetH, targetW, patchSize)
	return patches, positionIDs, nil
}
