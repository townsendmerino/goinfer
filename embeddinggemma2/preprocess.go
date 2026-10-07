package embeddinggemma2

import (
	"fmt"
	"strings"

	"github.com/townsendmerino/aikit/vision"
)

// The encoder's image resize (Phase V, docs/tasks/task-embeddinggemma2.md). The checkpoint's Gemma4ImageProcessor
// resizes the decoded uint8 image with torchvision's antialiased bicubic before rescaling to [0, 1]. That resize lived
// here until 2026-10-06 and is aikit's now (vision.ResizeBicubicAA, the default of vision.Gemma4Preprocess, held to
// torchvision bit for bit by aikit's own test), so EmbeddingGemma 2 and Gemma 4's image input share it. aikit's
// bilinear stays selectable.

// ImageResize chooses an encoder's image resize.
type ImageResize string

const (
	ResizeBilinear ImageResize = "bilinear" // aikit's bilinear (vision.Gemma4Bilinear)
	ResizeBicubic  ImageResize = "bicubic"  // the reference processor's (vision.Gemma4Bicubic)
)

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
	r := vision.Gemma4Bicubic
	if e.ImageResizeMode() == ResizeBilinear {
		r = vision.Gemma4Bilinear
	}
	return vision.Gemma4PreprocessResize(data, MaxImageSoftTokens, r)
}
