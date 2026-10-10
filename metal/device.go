//go:build darwin

// Metal device layer: aikit's native-GPU substrate (github.com/townsendmerino/aikit/gpu). goinfer keeps its tuned kernels
// here and builds them on these device types, the GPU analogue of the linalg relationship. The decode path must stay
// bit-identical across it (the Metal device-parity suite is the tripwire).
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

// Thin re-wraps of aikit gpu's generic NewBufferOf[T]. Go has no generic methods, so it is a free function taking the
// device first; these keep this package's many call sites at their original shape. History:
// docs/code-notes/metal.md#NewBufferFloats.
func NewBufferFloats(d *Device, data []float32) Buffer { return gpu.NewBufferOf(d, data) }
func NewBufferInt8(d *Device, data []int8) Buffer      { return gpu.NewBufferOf(d, data) }
func NewBufferUint32s(d *Device, data []uint32) Buffer { return gpu.NewBufferOf(d, data) }
func NewBufferU16s(d *Device, data []uint16) Buffer    { return gpu.NewBufferOf(d, data) }
func NewBufferU32(d *Device, v uint32) Buffer          { return gpu.NewBufferOf(d, []uint32{v}) }
