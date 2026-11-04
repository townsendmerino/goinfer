//go:build cuda

package cuda

import (
	"fmt"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// ptxTargetMajor/Minor: every PTX file in this package targets sm_75 (Turing), JIT-compiled forward by the driver on a newer card (docs/tasks/task-hardware-coverage-2026-10.md section 1.A). A card
// OLDER than that cannot load the kernels at all, and the report says so.
const ptxTargetMajor, ptxTargetMinor = 7, 5

func init() { decoder.RegisterHardwareInfo("cuda", cudaHardwareInfo) }

// cudaHardwareInfo is the device facts `check --hardware` prints for the CUDA backend. It opens the device only when asked, and says so plainly when there is none.
func cudaHardwareInfo() []string {
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		return []string{"no usable device: " + err.Error()}
	}
	defer dev.ReleaseObjects()
	ctx := dev.Context()
	d := ctx.Device()
	out := []string{"device: " + dev.Name()}
	maj, e1 := d.Attribute(gc.DeviceAttributeComputeCapabilityMajor)
	min, e2 := d.Attribute(gc.DeviceAttributeComputeCapabilityMinor)
	if e1 == nil && e2 == nil {
		how := "PTX targets sm_75 and is JIT-compiled forward by the driver for this card"
		switch {
		case maj == ptxTargetMajor && min == ptxTargetMinor:
			how = "PTX targets sm_75: this card's own architecture"
		case maj < ptxTargetMajor || (maj == ptxTargetMajor && min < ptxTargetMinor):
			how = "OLDER than the sm_75 these kernels are built for: they cannot load on it"
		}
		out = append(out, fmt.Sprintf("compute capability: %d.%d (%s)", maj, min, how))
	}
	if n := dev.SMCount(); n > 0 {
		out = append(out, fmt.Sprintf("multiprocessors: %d", n))
	}
	if free, total, err := ctx.MemInfo(); err == nil {
		out = append(out, fmt.Sprintf("memory: %.1f GiB total, %.1f GiB free", float64(total)/(1<<30), float64(free)/(1<<30)))
	}
	if v, err := gc.DriverVersion(); err == nil {
		out = append(out, fmt.Sprintf("driver supports CUDA %d.%d", v/1000, (v%1000)/10))
	}
	return out
}
