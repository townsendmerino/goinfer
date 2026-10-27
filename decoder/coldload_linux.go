//go:build linux

package decoder

import (
	"os"
	"syscall"
)

// coldPolicyHere is this system's rule.
var coldPolicyHere = coldPolicyLinux

// giwReadThrough brings mapping into the page cache: it touches every page in order, so the kernel reads the file
// front to back. (MADV_WILLNEED over a whole mapping is honoured for one read-ahead window here.)
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
