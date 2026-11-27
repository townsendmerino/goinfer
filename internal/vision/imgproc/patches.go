package imgproc

import "math"

// AspectRatioSize computes a resize target that is a multiple of poolingKernelSize*patchSize on each axis, aspect ratio
// preserved by a single uniform scale factor and each axis independently rounded DOWN (never up), for a source image of
// the given height/width, bounded by maxSoftTokens pooled patches worth of pixels. It mirrors
// get_aspect_ratio_preserving_size in transformers' image_processing_gemma4.py.
func AspectRatioSize(height, width, maxSoftTokens, patchSize, poolingKernelSize int) (targetH, targetW int) {
	sideMult := poolingKernelSize * patchSize
	maxPatches := maxSoftTokens * poolingKernelSize * poolingKernelSize
	targetPx := float64(maxPatches) * float64(patchSize*patchSize)
	totalPx := float64(height) * float64(width)
	factor := math.Sqrt(targetPx / totalPx)
	idealH, idealW := factor*float64(height), factor*float64(width)
	targetH = int(idealH/float64(sideMult)) * sideMult
	targetW = int(idealW/float64(sideMult)) * sideMult
	// Degenerate-dimension clamp (extreme aspect ratios rounding one axis to 0):
	// floor to one side_mult unit and cap the other so the total patch budget
	// isn't exceeded.
	maxSide := (maxPatches / (poolingKernelSize * poolingKernelSize)) * sideMult
	if targetH <= 0 {
		targetH = sideMult
	}
	if targetW <= 0 {
		targetW = sideMult
	}
	if targetH > maxSide {
		targetH = (maxSide / sideMult) * sideMult
	}
	if targetW > maxSide {
		targetW = (maxSide / sideMult) * sideMult
	}
	return targetH, targetW
}

// UnfoldHWC reshapes an HWC image (targetH and targetW multiples of patchSize) into per-patch flat vectors, matching
// HF's convert_image_to_patches exactly: image.reshape(C,H/p,p,W/p,p).permute(1,3,2,4,0).reshape(-1,p*p*C) — within
// each patch the flat order is (row-within-patch, col-within-patch, channel), NOT channel-major. Patches are enumerated
// row-major over the (gridH, gridW) grid, with position ids (x=col, y=row).
func UnfoldHWC(hwc []float32, targetH, targetW, patchSize int) ([]float32, [][2]int) {
	gridH, gridW := targetH/patchSize, targetW/patchSize
	n := gridH * gridW
	patchDim := patchSize * patchSize * 3
	rowLen := patchSize * 3
	patches := make([]float32, n*patchDim)
	positionIDs := make([][2]int, n)
	for py := range gridH {
		for px := range gridW {
			idx := py*gridW + px
			positionIDs[idx] = [2]int{px, py}
			dst := patches[idx*patchDim : (idx+1)*patchDim]
			for ry := range patchSize {
				srcY := py*patchSize + ry
				srcBase := (srcY*targetW + px*patchSize) * 3
				dstBase := ry * rowLen
				copy(dst[dstBase:dstBase+rowLen], hwc[srcBase:srcBase+rowLen])
			}
		}
	}
	return patches, positionIDs
}

// PatchesChannelsLast cuts a channels-first image (c x h x w, h and w multiples of patch; the caller validates) into
// patch rows, row-major over the (h/patch, w/patch) grid, each flattened (row, column, channel) with the channel
// innermost, as HF's convert_image_to_patches lays them out (SigLIP2 NaFlex).
func PatchesChannelsLast(img []float32, c, h, w, patch int) []float32 {
	gh, gw := h/patch, w/patch
	cpp := c * patch * patch
	out := make([]float32, gh*gw*cpp)
	for py := range gh {
		for px := range gw {
			row := out[(py*gw+px)*cpp:]
			for y := range patch {
				for x := range patch {
					for ch := range c {
						row[(y*patch+x)*c+ch] = img[ch*h*w+(py*patch+y)*w+px*patch+x]
					}
				}
			}
		}
	}
	return out
}

// PatchesChannelsFirst cuts a channels-first image (c x h x w, h and w multiples of patch; the caller validates) into
// patch rows, row-major over the (h/patch, w/patch) grid, each patch [c, patch, patch] flattened as a conv patch
// embedding's weight is (Pixtral).
func PatchesChannelsFirst(img []float32, c, h, w, patch int) []float32 {
	rows, cols := h/patch, w/patch
	pd := c * patch * patch
	out := make([]float32, rows*cols*pd)
	for r := range rows {
		for col := range cols {
			dst := out[(r*cols+col)*pd:]
			n := 0
			for ch := range c {
				for dy := range patch {
					src := img[ch*h*w+(r*patch+dy)*w+col*patch:]
					copy(dst[n:n+patch], src[:patch])
					n += patch
				}
			}
		}
	}
	return out
}
