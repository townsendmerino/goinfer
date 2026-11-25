//go:build gpu

package kernels

import "github.com/townsendmerino/goinfer/internal/vision"

// newSiglipResident builds the WebGPU SigLIP vision tower for vision.RegisterResident (wired in register.go).
func newSiglipResident(e *vision.Encoder) (vision.ResidentEncoder, error) {
	c, err := New()
	if err != nil {
		return nil, err
	}
	var ve *VisionEncoder
	if e.Quantized() { // int8 matmul weights (LoadEncoder quant=true)
		var w vision.GPUWeights
		if w, err = e.GPUWeights(); err == nil {
			ve, err = c.NewVisionEncoder(w)
		}
	} else {
		var w vision.SiglipWeights
		if w, err = e.Weights(); err == nil {
			ve, err = c.NewVisionEncoderF32(w)
		}
	}
	if err != nil {
		c.Close()
		return nil, err
	}
	return &residentVision{c: c, ve: ve, enc: e}, nil
}

// residentVision owns the encoder's WebGPU Context so Close tears down both the
// uploaded tower and the device.
type residentVision struct {
	c   *Context
	ve  *VisionEncoder
	enc *vision.Encoder
}

// ForwardPatches returns last_hidden_state. The float32 tower stops at the last block and the encoder's own
// post-layernorm finishes it on the host.
func (r *residentVision) ForwardPatches(patches []float32) ([]float32, error) {
	h, err := r.ve.ForwardPatches(patches)
	if err != nil || !r.ve.f32 {
		return h, err
	}
	return r.enc.FinishHidden(h)
}

func (r *residentVision) Close() {
	r.ve.Close()
	r.c.Close()
}
