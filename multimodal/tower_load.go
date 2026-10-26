package multimodal

import (
	"fmt"

	"github.com/townsendmerino/goinfer/internal/vision"
)

// SiglipTower is Gemma 3's SigLIP vision tower, for programs outside goinfer's own packages (the vision package is
// internal). Its output is the tower's last hidden state, the input Projector.Forward takes.
type SiglipTower struct{ enc *vision.Encoder }

// LoadSiglipTower loads the tower from a Gemma 3 checkpoint directory. quant selects int8 (W8A8) matmul weights. resident
// attaches the device tower a linked backend registered, and fails if none did; the WebGPU tower needs quant.
func LoadSiglipTower(dir string, quant, resident bool) (*SiglipTower, error) {
	enc, err := vision.LoadEncoder(dir, quant)
	if err != nil {
		return nil, err
	}
	if resident {
		if err := enc.EnableResident(); err != nil {
			enc.Close()
			return nil, fmt.Errorf("multimodal: resident SigLIP tower: %w", err)
		}
	}
	return &SiglipTower{enc: enc}, nil
}

// Preprocess decodes an encoded image and returns the tower's input pixels, as Gemma 3's processor prepares them.
func (t *SiglipTower) Preprocess(image []byte) ([]float32, error) {
	pv, err := vision.Preprocess(image, vision.Gemma3())
	if err != nil {
		return nil, err
	}
	return pv.Data, nil
}

// Forward runs the tower over Preprocess's pixels.
func (t *SiglipTower) Forward(pixels []float32) ([]float32, error) { return t.enc.Forward(pixels) }

// Close releases the tower and its device copy; Forward errors afterward.
func (t *SiglipTower) Close() { t.enc.Close() }

// GlmOcrTower is GLM-OCR's vision tower on the CPU in float32, for programs outside goinfer's own packages. Its output is
// the merged image features the language model splices in.
type GlmOcrTower struct{ enc *vision.GlmOcrVisionEncoder }

// LoadGlmOcrTower loads the tower from a GLM-OCR checkpoint directory.
func LoadGlmOcrTower(dir string) (*GlmOcrTower, error) {
	enc, err := vision.LoadGlmOcrVisionEncoder(dir, false)
	if err != nil {
		return nil, err
	}
	return &GlmOcrTower{enc: enc}, nil
}

// Forward runs the tower over one image's pixels in patch order (QwenPreprocess's output) with its grid.
func (t *GlmOcrTower) Forward(pixels []float32, grid [3]int) ([]float32, error) {
	return GlmOcrTowerFeatures(t.enc, nil, pixels, [][3]int{grid})
}

// Close releases the tower; Forward errors afterward.
func (t *GlmOcrTower) Close() { t.enc.Close() }
