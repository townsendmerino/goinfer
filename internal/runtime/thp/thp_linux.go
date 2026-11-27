//go:build linux

// Package thp keeps chosen memory on ordinary pages where the kernel would back it with transparent huge pages.
package thp

import (
	"os"
	"strings"
	"sync"
	"syscall"
	"unsafe"
)

const (
	prSetTHPDisable = 41 // PR_SET_THP_DISABLE
	prGetTHPDisable = 42 // PR_GET_THP_DISABLE
	madvNoHugePage  = 15 // MADV_NOHUGEPAGE
	hugePage        = 2 << 20
)

var window sync.Mutex // PR_SET_THP_DISABLE is process-wide: one Small window at a time

// always reports whether the host backs anonymous memory with huge pages unasked (THP mode "always").
var always = sync.OnceValue(func() bool {
	b, err := os.ReadFile("/sys/kernel/mm/transparent_hugepage/enabled")
	return err == nil && strings.Contains(string(b), "[always]")
})

// Small runs alloc with transparent huge pages off for this process (PR_SET_THP_DISABLE, restored after), so the pages
// alloc touches, which include the memory Go zeroes as it hands a reused span to a large allocation, are ordinary ones and
// stay so; then it marks each returned buffer of at least a huge page MADV_NOHUGEPAGE. The mark holds only until the next
// window: re-enabling THP clears every MADV_NOHUGEPAGE in the process on current kernels, so Small acts only where the
// host's mode is "always" (elsewhere nothing would give the buffer huge pages anyway). It is for a large buffer taken beside
// resident memory that keeps its huge pages: on an "always" host, a burst of huge-page faults for it wakes the kernel's
// high-order reclaim, which swaps the process even with memory free. Advisory: a refusal leaves the kernel's default.
func Small(alloc func() [][]byte) {
	if !always() {
		alloc()
		return
	}
	window.Lock()
	prev, _, _ := syscall.RawSyscall(syscall.SYS_PRCTL, prGetTHPDisable, 0, 0)
	if prev == 0 {
		syscall.RawSyscall(syscall.SYS_PRCTL, prSetTHPDisable, 1, 0)
	}
	bufs := alloc()
	if prev == 0 {
		syscall.RawSyscall(syscall.SYS_PRCTL, prSetTHPDisable, 0, 0)
	}
	window.Unlock()
	ps := uintptr(syscall.Getpagesize())
	for _, b := range bufs {
		if cap(b) < hugePage {
			continue // too small to be a huge page
		}
		lo := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
		hi := (lo + uintptr(cap(b))) &^ (ps - 1)
		lo = (lo + ps - 1) &^ (ps - 1)
		syscall.Syscall(syscall.SYS_MADVISE, lo, hi-lo, madvNoHugePage)
	}
}
