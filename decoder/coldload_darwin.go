//go:build darwin

package decoder

import (
	"syscall"
	"unsafe"
)

// coldPolicyHere is this system's rule.
var coldPolicyHere = coldPolicyDarwin

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
