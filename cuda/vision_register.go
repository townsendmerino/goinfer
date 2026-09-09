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
		w, err := e.GPUWeights() // requires int8 weights (LoadEncoder quant=true)
		if err != nil {
			return nil, err
		}
		return NewVisionEncoder(w)
	})
}
