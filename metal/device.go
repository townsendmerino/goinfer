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
