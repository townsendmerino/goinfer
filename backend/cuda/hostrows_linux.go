//go:build cuda && linux

package cuda

import (
	"syscall"
	"unsafe"
)

// allocHostRows is words float32s of ordinary (pageable) host memory, zero-filled, and the function that returns it to the system. It is an anonymous
// mapping, not the Go heap: a snapshot's rows are released the moment it is, so the resident set holds exactly the snapshots held, where heap memory
// lingers until a collection and the server's peak grew by about a tenth. It falls back to the heap where the mapping fails.
func allocHostRows(words int) ([]float32, func()) {
	if words <= 0 {
		return nil, func() {}
	}
	b, err := syscall.Mmap(-1, 0, words*4, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		return make([]float32, words), func() {}
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), words), func() { _ = syscall.Munmap(b) }
}
