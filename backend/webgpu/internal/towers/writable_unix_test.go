//go:build gpu && goinfer_testhooks && unix

package towers

import (
	"fmt"
	"runtime"
	"sync/atomic"
	"syscall"
	"testing"
	"unsafe"
)

// makeWritable lets a test edit tower weights in place. An f32 tower weight aliases its checkpoint's read-only private
// mapping (aikit vision aliasQMat), so a write to it faults until its pages are made writable. The mapping is MAP_PRIVATE,
// so the edits land in copy-on-write pages, which the CPU tower reads too, and never reach the file. A heap slice is
// already writable; mprotecting its pages read-write again changes nothing.
//
// The page-rounded range is passed to mprotect as integers, never as a Go pointer. Where the OS page is larger than the
// runtime's 8 KiB page (16 KiB on darwin/arm64), a heap slice's rounded-down address can lie in another span, and the GC
// crashes on a pointer it finds there in any frame it scans.
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

// TestMakeWritable_heapSlicesUnderGC runs makeWritable on heap slices while the GC runs continuously, and passes by
// surviving. It can fail only where the OS page is larger than the runtime's, so a pass on a 4 KiB-page host says nothing.
func TestMakeWritable_heapSlicesUnderGC(t *testing.T) {
	var stop atomic.Bool
	done := make(chan struct{})
	go func() {
		defer close(done)
		for !stop.Load() {
			runtime.GC()
		}
	}()
	defer func() { stop.Store(true); <-done }()
	var keep [][]float32
	for i := range 20000 {
		// Small slices share a size-class span and large ones get spans of their own. Dropping every other one of each
		// leaves free slots and free spans below the survivors, where their page-rounded starts fall.
		n := 16 + (i%13)*150
		if i%4 >= 2 {
			n = 5000 + (i%7)*1024
		}
		x := make([]float32, n)
		if i%2 == 0 {
			keep = append(keep, x)
			if len(keep) > 64 {
				keep = keep[1:]
			}
		}
		makeWritable(x)
		x[len(x)-1] = 1
	}
}
