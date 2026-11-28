//go:build gpu && goinfer_testhooks && unix

package towers

import (
	"fmt"
	"math/rand/v2"
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
// The page-rounded range is passed to mprotect as integers, never as a Go pointer. A heap slice's rounded-down address can
// be a free slot of its span, or lie in another span where the OS page is larger than the runtime's 8 KiB page, and the
// GC aborts the process on a pointer to either in any frame it scans.
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
// surviving. Fresh slices reach the other-span case, which needs an OS page over 8 KiB; long-lived small ones with half
// their neighbours freed reach the free-slot case at any page size.
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

	var keep [64][]float32
	for i := range 20000 {
		n := 16 + (i%13)*150
		if i%4 >= 2 {
			n = 5000 + (i%7)*1024
		}
		x := make([]float32, n)
		if i%2 == 0 {
			keep[i/2%len(keep)] = x
		}
		makeWritable(x)
		x[len(x)-1] = 1
	}

	// Sizes whose slots do not start on a 4 KiB boundary. Two collections free the dropped slots before the first call.
	rng := rand.New(rand.NewPCG(1, 2))
	var settled [][]float32
	for i := range 30000 {
		x := make([]float32, []int{24, 100, 1500}[i%3])
		if rng.IntN(2) == 0 {
			settled = append(settled, x)
		}
	}
	runtime.GC()
	runtime.GC()
	for range 10 {
		for _, x := range settled {
			makeWritable(x)
			x[len(x)-1] = 1
		}
	}
}
