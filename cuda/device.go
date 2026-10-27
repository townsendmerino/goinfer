//go:build cuda

// CUDA device layer: aikit's native-GPU substrate (github.com/townsendmerino/aikit/gpu), the CUDA analogue of
// metal/device.go. goinfer keeps its tuned decode kernels here and builds them on these device types, as it builds on
// linalg. Only the device types moved; the decode path is unchanged and must stay bit-identical (the CUDA device-parity
// suite is the tripwire).
//
// The type and non-generic-func aliases below let the tuned code read unqualified (Buffer, Arg, Grid1D, ...). Go has no
// generic-method/var aliases, so the generic verbs (ArgValue, NewBufferOf/LenOf, Upload/Download, NewHostBuffer,
// ReadToHost) are called as gpu.X[T](...) at the sites.
package cuda

import (
	gpu "github.com/townsendmerino/goinfer/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/internal/mem"
)

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
	CreateSystemDefaultDevice = createDevice
	Arg                       = gpu.Arg
	ArgNull                   = gpu.ArgNull // null device-ptr arg for an absent optional buffer (was gc.ArgDevicePtr(0))
	Grid1D                    = gpu.Grid1D
	GridOne                   = gpu.GridOne
)

// createDevice creates the device with a mem.Ledger attached, so every allocation it makes is a lease naming its plan line (MM3).
func createDevice() (*Device, error) {
	d, err := gpu.CreateSystemDefaultDevice()
	if err == nil {
		d.SetLeaser(&deviceLedger{})
	}
	return d, err
}

// deviceLedger is a device's mem.Ledger: device buffers on VRAM, pinned host memory on HostAnon.
type deviceLedger struct{ mem.Ledger }

func (l *deviceLedger) Lease(onDevice bool, line string, requested, charged int64) func() {
	tier := mem.HostAnon
	if onDevice {
		tier = mem.VRAM
	}
	return l.Reserve(tier, mem.LineID(line), requested, charged).Release
}
