//go:build linux

package decoder

import (
	"os"
	"syscall"
	"unsafe"
)

// coldLoadSupported: the plan is measured on Linux and runs there. A variable so a test can run the plan, with its
// probes and actions replaced, on any system.
var coldLoadSupported = true

// coldResidentSamples is how many evenly spaced pages giwResidentFraction asks the kernel about.
const coldResidentSamples = 512

// giwResidentFraction samples how much of mapping is in the page cache: the share of coldResidentSamples evenly
// spaced pages that mincore reports resident. It reports 1 (warm, so the load does nothing) when the kernel cannot
// be asked.
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

// giwReadThrough touches every page of mapping in order, so the kernel reads the file front to back.
func giwReadThrough(mapping []byte) { _ = sumOnePerPage(mapping, os.Getpagesize()) }

// sumOnePerPage adds up one byte of every page. It is not inlined, so the sum is a result a caller might use and the
// reads that make it cannot be removed.
//
//go:noinline
func sumOnePerPage(mapping []byte, page int) (sum byte) {
	for off := 0; off < len(mapping); off += page {
		sum += mapping[off]
	}
	return sum
}

// giwAdviseReadAround turns the kernel's read-around on page faults of mapping on (the default) or off. Best effort:
// a refusal leaves the default in force.
func giwAdviseReadAround(mapping []byte, on bool) {
	advice := syscall.MADV_RANDOM
	if on {
		advice = syscall.MADV_NORMAL
	}
	_ = syscall.Madvise(mapping, advice)
}
