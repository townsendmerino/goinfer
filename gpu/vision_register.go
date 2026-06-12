//go:build gpu

package gpu

import "github.com/townsendmerino/goinfer/vision"

// init plugs the resident WebGPU SigLIP encoder into the vision package, so a
// `-tags gpu` build that blank-imports gpu lets vision.Encoder.EnableResident()
// run the tower on the device. The core vision package never imports webgpu.
func init() {
	vision.RegisterResident(func(e *vision.Encoder) (vision.ResidentEncoder, error) {
		w, err := e.GPUWeights() // requires int8 weights (LoadEncoder quant=true)
		if err != nil {
			return nil, err
		}
		c, err := New()
		if err != nil {
			return nil, err
		}
		ve, err := c.NewVisionEncoder(w)
		if err != nil {
			c.Close()
			return nil, err
		}
		return &residentVision{c: c, ve: ve}, nil
	})
}

// residentVision owns the encoder's WebGPU Context so Close tears down both the
// uploaded tower and the device.
type residentVision struct {
	c  *Context
	ve *VisionEncoder
}

func (r *residentVision) ForwardPatches(patches []float32) ([]float32, error) {
	return r.ve.ForwardPatches(patches)
}

func (r *residentVision) Close() {
	r.ve.Close()
	r.c.Close()
}
