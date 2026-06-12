package vision

import "fmt"

// ResidentEncoder is a device-resident SigLIP forward — the GPU path in
// goinfer/gpu (vision_encoder.go), which uploads the tower once and runs every
// op on-device. vision stays pure-Go: it delegates Forward here when a resident
// backend is attached (EnableResident), and the gpu module plugs the factory in
// from its init() under -tags gpu. ForwardPatches consumes GridPatches output.
type ResidentEncoder interface {
	ForwardPatches(patches []float32) ([]float32, error)
	Close()
}

// residentFactory builds a ResidentEncoder from a loaded tower. nil in a pure-Go
// build; set by the gpu module's init() via RegisterResident.
var residentFactory func(*Encoder) (ResidentEncoder, error)

// RegisterResident plugs in a device-resident encoder backend. Called by the gpu
// module's init() under -tags gpu — the core never imports the GPU package.
func RegisterResident(f func(*Encoder) (ResidentEncoder, error)) { residentFactory = f }

// EnableResident attaches a device-resident encoder (the WebGPU tower). Requires
// int8 matmul weights (LoadEncoder quant=true) and a -tags gpu build; Forward
// then runs on the device. Returns an error (and leaves the CPU path intact) if
// no backend is registered or the upload fails.
func (e *Encoder) EnableResident() error {
	if residentFactory == nil {
		return fmt.Errorf("vision: no resident encoder backend (rebuild with -tags gpu)")
	}
	r, err := residentFactory(e)
	if err != nil {
		return err
	}
	e.resident = r
	return nil
}

// Close releases a resident backend if one is attached (no-op otherwise).
func (e *Encoder) Close() {
	if e.resident != nil {
		e.resident.Close()
		e.resident = nil
	}
}
