//go:build darwin || linux

package probe

import (
	"os"
	"syscall"
	"unsafe"
)

// residentSamples is how many evenly spaced pages ResidentFraction asks the kernel about.
const residentSamples = 512

// ResidentFraction samples how much of mapping is in the page cache: the share of residentSamples evenly spaced pages
// that mincore reports resident. It reports 1 (warm, so a cold-load plan does nothing) when the kernel cannot be asked.
func ResidentFraction(mapping []byte) float64 {
	page := os.Getpagesize()
	pages := len(mapping) / page
	if pages == 0 {
		return 1
	}
	n := min(residentSamples, pages)
	var vec [1]byte
	resident := 0
	for i := range n {
		off := (pages / n) * i * page
		if _, _, errno := syscall.Syscall(syscall.SYS_MINCORE, uintptr(unsafe.Pointer(&mapping[off])), uintptr(page), uintptr(unsafe.Pointer(&vec[0]))); errno != 0 {
			return 1
		}
		resident += int(vec[0] & 1)
	}
	return float64(resident) / float64(n)
}
