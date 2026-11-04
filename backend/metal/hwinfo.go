//go:build darwin

package metal

import (
	"fmt"
	"syscall"

	"github.com/townsendmerino/goinfer/decoder"
)

func init() { decoder.RegisterHardwareInfo("metal", metalHardwareInfo) }

// metalHardwareInfo is the device facts `check --hardware` prints for the Metal backend (docs/tasks/task-hardware-coverage-2026-10.md,
// H3). It opens the device only when asked, and says so plainly when there is none. The GPU's name is its model, not a
// machine identifier. aikit/gpu exposes no GPU family or recommended working-set size, so the report says it cannot report
// them rather than leaving the gap unexplained. The memory is the machine's: on Apple silicon the GPU shares it.
func metalHardwareInfo() []string {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		return []string{"no usable device: " + err.Error()}
	}
	defer d.ReleaseObjects()
	out := []string{"device: " + d.Name()}
	if ram := decoder.HostRAMBytes(); ram > 0 {
		line := fmt.Sprintf("unified memory: %.1f GiB total", float64(ram)/(1<<30))
		if avail := decoder.HostRAMAvailableBytes(); avail > 0 {
			line += fmt.Sprintf(", %.1f GiB available now", float64(avail)/(1<<30))
		}
		out = append(out, line)
	}
	out = append(out, fmt.Sprintf("threadgroup memory: %d KiB per threadgroup", d.MaxThreadgroupMemoryLength()>>10))
	out = append(out, "GPU family and recommended working-set size: not reported (aikit/gpu does not expose them)")
	ver, _ := syscall.Sysctl("kern.osproductversion")
	build, _ := syscall.Sysctl("kern.osversion")
	if ver != "" || build != "" {
		out = append(out, fmt.Sprintf("macOS %s (build %s)", orUnknown(ver), orUnknown(build)))
	}
	return out
}

func orUnknown(s string) string {
	if s == "" {
		return "unknown"
	}
	return s
}
