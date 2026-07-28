//go:build cuda

// CUDA device layer — now aikit's native-GPU substrate (github.com/townsendmerino/aikit/gpu),
// the CUDA analogue of what metal/device.go did for Metal (goinfer 5ec20ff). goinfer keeps its
// tuned decode kernels here and builds them on these device types — the GPU analogue of the
// linalg relationship. Only the device TYPES moved; nothing about the decode path changed, so
// it must stay bit-identical (the CUDA device-parity suite is the tripwire).
//
// The type + non-generic-func aliases below let the tuned code read unqualified (Buffer, Arg,
// Grid1D, …), exactly as it did against gocudrv. Go has no generic-method/var aliases, so the
// generic verbs (ArgValue, NewBufferOf/LenOf, Upload/Download, NewHostBuffer, ReadToHost) are
// called as gpu.X[T](…) at the sites; everything else is unqualified here.
package cuda

import gpu "github.com/townsendmerino/aikit/gpu"

type (
	Device       = gpu.Device
	Buffer       = gpu.Buffer
	Queue        = gpu.Queue
	Pipeline     = gpu.Pipeline
	Library      = gpu.Library
	KernelArg    = gpu.KernelArg
	LaunchConfig = gpu.LaunchConfig
	Scalar       = gpu.Scalar
)

// HostBuffer is a generic type alias (Go 1.24+): page-locked host memory for the
// logits readback.
type HostBuffer[T Scalar] = gpu.HostBuffer[T]

var (
	CreateSystemDefaultDevice = gpu.CreateSystemDefaultDevice
	Arg                       = gpu.Arg
	ArgNull                   = gpu.ArgNull // null device-ptr arg for an absent optional buffer (was gc.ArgDevicePtr(0))
	Grid1D                    = gpu.Grid1D
	GridOne                   = gpu.GridOne
)
