//go:build cuda

package cuda

import (
	"github.com/townsendmerino/aikit/vision"

	"github.com/townsendmerino/goinfer/multimodal"
)

// init plugs the resident CUDA SigLIP encoder into the vision package, so a `-tags cuda` build that blank-imports cuda
// lets vision.Encoder.EnableResident() run the tower on the device. It mirrors goinfer/gpu's vision_register.go (the
// WebGPU registration) and is the entire integration point: nothing else in the tree needs to know CUDA vision exists
// (docs/multimodal.md). The core vision package never imports cuda.
func init() {
	vision.RegisterResident(func(e *vision.Encoder) (vision.ResidentEncoder, error) {
		// A float32 encoder (LoadEncoder quant=false: Weights succeeds) gets the float32 tower on the tower base
		// (siglip_vision.go). An int8 one (quant=true: Weights errors) gets the W8A8 encoder below, so serve's choice of how to
		// load the tower picks between them.
		if _, err := e.Weights(); err == nil {
			return newSiglipTower(e)
		}
		w, err := e.GPUWeights() // requires int8 weights (LoadEncoder quant=true)
		if err != nil {
			return nil, err
		}
		return NewVisionEncoder(w)
	})
	multimodal.MarkResidentTower(multimodal.TowerSigLIP, "cuda") // S8: aikit's slot cannot be asked what is registered
}
