//go:build gpu && goinfer_testhooks && unix

package gpu

import (
	"fmt"
	"syscall"
	"unsafe"
)

// makeWritable lets a test edit tower weights in place. An f32 tower weight aliases its checkpoint's read-only private
// mapping (aikit vision aliasQMat), so a write to it faults until its pages are made writable. The mapping is MAP_PRIVATE,
// so the edits land in copy-on-write pages, which the CPU tower reads too, and never reach the file. Heap slices are
// already writable and are left as they are.
func makeWritable(xs ...[]float32) {
	pg := uintptr(syscall.Getpagesize())
	for _, x := range xs {
		if len(x) == 0 {
			continue
		}
		p := unsafe.Pointer(&x[0])
		off := uintptr(p) & (pg - 1)
		n := (off + uintptr(len(x))*4 + pg - 1) &^ (pg - 1)
		b := unsafe.Slice((*byte)(unsafe.Add(p, -int(off))), n)
		if err := syscall.Mprotect(b, syscall.PROT_READ|syscall.PROT_WRITE); err != nil {
			panic(fmt.Sprintf("makeWritable: mprotect %d bytes: %v", n, err))
		}
	}
}
