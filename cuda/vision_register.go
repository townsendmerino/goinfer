//go:build cuda

package cuda

import "github.com/townsendmerino/aikit/vision"

// init plugs the resident CUDA SigLIP encoder into the vision package, so a `-tags cuda` build
// that blank-imports cuda lets vision.Encoder.EnableResident() run the tower on the device.
// Mirrors goinfer/gpu's own vision_register.go (the WebGPU registration) — the ENTIRE integration
// point; nothing else in the tree needs to know CUDA vision exists (P6, docs/multimodal.md's "P6's
// other half"). The core vision package never imports cuda.
func init() {
	vision.RegisterResident(func(e *vision.Encoder) (vision.ResidentEncoder, error) {
		// A float32 encoder (LoadEncoder quant=false: Weights succeeds) gets the float32 tower on the tower base (siglip_vision.go, the S4 addendum). An int8 one
		// (quant=true: Weights errors) gets the W8A8 encoder below, exactly as before, so serve's choice of how to load the tower is what picks between them.
		if _, err := e.Weights(); err == nil {
			return newSiglipTower(e)
		}
		w, err := e.GPUWeights() // requires int8 weights (LoadEncoder quant=true)
		if err != nil {
			return nil, err
		}
		return NewVisionEncoder(w)
	})
}
