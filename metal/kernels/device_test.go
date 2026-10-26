//go:build darwin

package kernels

import (
	"math"
	"testing"

	gpu "github.com/townsendmerino/goinfer/metal/internal/gpudev"
)

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

func byteBuf(d *Device, n int) Buffer { return d.NewBufferBytes(n) }

// mustFinite fails on a NaN or Inf metric: the threshold check after it cannot catch one, since NaN compares false to
// every bound. A copy of metal's helper of the same name.
func mustFinite(t *testing.T, label string, metric float64) {
	t.Helper()
	if math.IsNaN(metric) || math.IsInf(metric, 0) {
		t.Fatalf("%s is %v — degenerate (NaN/Inf) output; the threshold check below cannot catch this "+
			"(NaN compares false to every bound)", label, metric)
	}
}
