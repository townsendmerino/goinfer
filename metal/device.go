//go:build darwin

// Metal device layer — now aikit's native-GPU substrate (github.com/townsendmerino/aikit/gpu),
// lifted verbatim from what used to be this package's metal.go. goinfer keeps its tuned kernels
// here and builds them on these device types — the GPU analogue of the linalg relationship. Only
// the device TYPES moved; nothing about the decode path changed, so it must stay bit-identical
// (the Metal device-parity suite is the tripwire).
package metal

import gpu "github.com/townsendmerino/aikit/gpu"

type (
	Device       = gpu.Device
	Buffer       = gpu.Buffer
	Queue        = gpu.Queue
	Pipeline     = gpu.Pipeline
	Encoder      = gpu.Encoder
	ARPool       = gpu.ARPool
	ResidencySet = gpu.ResidencySet
)

const MSL3_1 = gpu.MSL3_1

var (
	CreateSystemDefaultDevice = gpu.CreateSystemDefaultDevice
	NewARPool                 = gpu.NewARPool
	ResidencySetsSupported    = gpu.ResidencySetsSupported
)

// Thin re-wraps of aikit gpu v0.29.0's type-suffixed-Buffer-API collapse (NewBufferFloats/
// NewBufferInt8/NewBufferU32/NewBufferUint32s/NewBufferU16s deleted in favor of the generic
// NewBufferOf[T]). Go has no generic methods, so the aikit replacement is a free function
// (gpu.NewBufferOf(d, data)); these keep every one of this package's ~500 existing call sites at
// their original method-call shape (now a free function taking d first) instead of touching each
// one's argument list.
func NewBufferFloats(d *Device, data []float32) Buffer { return gpu.NewBufferOf(d, data) }
func NewBufferInt8(d *Device, data []int8) Buffer      { return gpu.NewBufferOf(d, data) }
func NewBufferUint32s(d *Device, data []uint32) Buffer { return gpu.NewBufferOf(d, data) }
func NewBufferU16s(d *Device, data []uint16) Buffer    { return gpu.NewBufferOf(d, data) }
func NewBufferU32(d *Device, v uint32) Buffer          { return gpu.NewBufferOf(d, []uint32{v}) }
