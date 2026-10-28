//go:build darwin

// Metal device layer: aikit's native-GPU substrate (github.com/townsendmerino/aikit/gpu). goinfer keeps its tuned kernels
// here and builds them on these device types, the GPU analogue of the linalg relationship. The decode path must stay
// bit-identical across it (the Metal device-parity suite is the tripwire).
package metal

import (
	"github.com/townsendmerino/goinfer/internal/mem"
	gpu "github.com/townsendmerino/goinfer/metal/internal/gpudev"
)

// Every device created in a process that links this package leases its MTLBuffers to a mem.Ledger (MM3), whichever package
// creates it.
func init() { gpu.NewLeaser = func() gpu.Leaser { return &deviceLedger{} } }

// deviceLedger is a device's mem.Ledger: MTLBuffers on Unified, a no-copy alias of caller memory on FileMapped.
type deviceLedger struct {
	mem.Ledger
	planted bool // ledgerChargeDeltaForTest has been applied on this ledger
}

// ledgerChargeDeltaForTest, when not zero, is added to the charge this package's ledger records for each device's first
// Unified lease of 16 KB or more on plan line ledgerChargeLineForTest: G-MM3c-4's plants for a ledger that records more, or
// less, than it was asked to. Zero in production.
var (
	ledgerChargeDeltaForTest int64
	ledgerChargeLineForTest  string
)

func (l *deviceLedger) Lease(fileMapped bool, line string, requested, charged int64) func() {
	tier := mem.Unified
	if fileMapped {
		tier = mem.FileMapped
	}
	if ledgerChargeDeltaForTest != 0 && !l.planted && !fileMapped && requested >= 16<<10 && line == ledgerChargeLineForTest {
		charged += ledgerChargeDeltaForTest
		l.planted = true
	}
	return l.Reserve(tier, mem.LineID(line), requested, charged).Release
}

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
func NewBufferFloatsOn(line string, d *Device, data []float32) Buffer {
	return gpu.NewBufferOfOn(d, line, data)
}

// NewBufferFloats is NewBufferFloatsOn on gpu.UnlinedLine: tests call it; production code names its plan line.
func NewBufferFloats(d *Device, data []float32) Buffer {
	return NewBufferFloatsOn(gpu.UnlinedLine, d, data)
}
func NewBufferInt8On(line string, d *Device, data []int8) Buffer {
	return gpu.NewBufferOfOn(d, line, data)
}

// NewBufferInt8 is NewBufferInt8On on gpu.UnlinedLine: tests call it; production code names its plan line.
func NewBufferInt8(d *Device, data []int8) Buffer {
	return NewBufferInt8On(gpu.UnlinedLine, d, data)
}
func NewBufferUint32sOn(line string, d *Device, data []uint32) Buffer {
	return gpu.NewBufferOfOn(d, line, data)
}

// NewBufferUint32s is NewBufferUint32sOn on gpu.UnlinedLine: tests call it; production code names its plan line.
func NewBufferUint32s(d *Device, data []uint32) Buffer {
	return NewBufferUint32sOn(gpu.UnlinedLine, d, data)
}
func NewBufferU16sOn(line string, d *Device, data []uint16) Buffer {
	return gpu.NewBufferOfOn(d, line, data)
}

// NewBufferU16s is NewBufferU16sOn on gpu.UnlinedLine: tests call it; production code names its plan line.
func NewBufferU16s(d *Device, data []uint16) Buffer {
	return NewBufferU16sOn(gpu.UnlinedLine, d, data)
}
func NewBufferU32On(line string, d *Device, v uint32) Buffer {
	return gpu.NewBufferOfOn(d, line, []uint32{v})
}

// NewBufferU32 is NewBufferU32On on gpu.UnlinedLine: tests call it; production code names its plan line.
func NewBufferU32(d *Device, v uint32) Buffer {
	return NewBufferU32On(gpu.UnlinedLine, d, v)
}
