//go:build cuda

package cuda

import "github.com/townsendmerino/goinfer/internal/vision"

// newSiglipResident builds the CUDA SigLIP vision tower for vision.RegisterResident (wired in register.go):
// float32 weights take the tower base (siglip_vision.go), and int8 weights take NewVisionEncoder.
func newSiglipResident(e *vision.Encoder) (vision.ResidentEncoder, error) {
	if _, err := e.Weights(); err == nil {
		return newSiglipTower(e)
	}
	w, err := e.GPUWeights() // requires int8 weights (LoadEncoder quant=true)
	if err != nil {
		return nil, err
	}
	return NewVisionEncoder(w)
}
