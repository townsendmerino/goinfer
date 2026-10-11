//go:build darwin

package decoder

import (
	"os"
	"syscall"
	"unsafe"
)

// coldPolicyHere is this system's rule.
var coldPolicyHere = coldPolicyDarwin

// coldResidentSamples is how many evenly spaced pages giwResidentFraction asks the kernel about.
const coldResidentSamples = 512

// giwResidentFraction samples how much of mapping is in memory: the share of coldResidentSamples evenly spaced pages
// that mincore reports resident. It reports 1 (warm, so the load does nothing) when the kernel cannot be asked.
func giwResidentFraction(mapping []byte) float64 {
	page := os.Getpagesize()
	pages := len(mapping) / page
	if pages == 0 {
		return 1
	}
	n := min(coldResidentSamples, pages)
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

// madvWillNeed is MADV_WILLNEED, which the syscall package does not export for darwin.
const madvWillNeed = 3

// giwReadThrough brings mapping into the page cache by asking the kernel to read all of it ahead. Touching it a page
// at a time, as Linux does, is bound here by the faults and not by the disk. Best effort: a refusal leaves the load
// as it was.
func giwReadThrough(mapping []byte) {
	if len(mapping) == 0 {
		return
	}
	_, _, _ = syscall.Syscall(syscall.SYS_MADVISE, uintptr(unsafe.Pointer(&mapping[0])), uintptr(len(mapping)), madvWillNeed)
}

// giwAdviseReadAround does nothing: the read-around on a page fault is small on this system (coldPolicyDarwin).
func giwAdviseReadAround(mapping []byte, on bool) {}
