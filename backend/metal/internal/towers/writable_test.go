//go:build darwin

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
// so the edits land in copy-on-write pages, which the CPU tower reads too, and never reach the file. A heap slice (a
// widened bf16 weight) is already writable; mprotecting its pages read-write again changes nothing.
//
// The page-rounded range is passed to mprotect as integers, never as a Go pointer: for a heap slice the rounded-down
// address can fall in a neighbouring, free span, and the GC aborts on a pointer it finds there in any frame it scans.
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

// TestMakeWritable_heapSlicesUnderGC runs makeWritable on heap slices whose page-rounded start lies below their own span,
// often in a freed one, while the GC runs continuously. A Go pointer to that rounded address held in any frame makes the
// GC abort the process with "found bad pointer in Go heap"; the test passes by surviving.
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
		// Sizes past the small-object limit get spans of their own, 8 KiB-aligned against 16 KiB OS pages; dropping
		// every other one leaves free spans beneath the survivors.
		x := make([]float32, 5000+(i%7)*1024)
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
