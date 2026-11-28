//go:build cuda && unix

package towers

import (
	"fmt"
	"runtime"
	"syscall"
	"unsafe"
)

// makeWritable lets a test edit tower weights in place. An f32 tower weight aliases its checkpoint's read-only private
// mapping (aikit vision aliasQMat), so a write to it faults until its pages are made writable. The mapping is MAP_PRIVATE,
// so the edits land in copy-on-write pages, which the CPU tower reads too, and never reach the file. A heap slice is
// already writable; mprotecting its pages read-write again changes nothing.
//
// The page-rounded range is passed to mprotect as integers, never as a Go pointer. Where the OS page is larger than the
// runtime's 8 KiB page (as on some arm64 Linux kernels), a heap slice's rounded-down address can lie in another span, and
// the GC crashes on a pointer it finds there in any frame it scans.
func makeWritable(xs ...[]float32) {
	pg := uintptr(syscall.Getpagesize())
	for _, x := range xs {
		if len(x) == 0 {
			continue
		}
		p := uintptr(unsafe.Pointer(&x[0]))
		base := p &^ (pg - 1)
		n := (p - base + uintptr(len(x))*4 + pg - 1) &^ (pg - 1)
		_, _, errno := syscall.Syscall(syscall.SYS_MPROTECT, base, n, syscall.PROT_READ|syscall.PROT_WRITE)
		runtime.KeepAlive(x)
		if errno != 0 {
			panic(fmt.Sprintf("makeWritable: mprotect %d bytes: %v", n, errno))
		}
	}
}
