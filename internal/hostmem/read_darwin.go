//go:build darwin

package hostmem

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
	"golang.org/x/sys/unix"
)

// hostVMInfo64 is HOST_VM_INFO64 (mach/host_info.h).
const hostVMInfo64 = 4

// machHost is libSystem's three calls, bound once without cgo, and this task's host port (mach_host_self hands out a send right
// per call, so it is asked once).
type machHost struct {
	port       uint32
	statistics func(host uint32, flavor int32, info unsafe.Pointer, count *uint32) int32
	pageSize   func(host uint32, size *uint64) int32
}

var machOnce = sync.OnceValues(func() (h *machHost, err error) {
	defer func() {
		if v := recover(); v != nil { // RegisterLibFunc panics on a missing symbol
			h, err = nil, fmt.Errorf("hostmem: binding libSystem: %v", v)
		}
	}()
	lib, err := purego.Dlopen("/usr/lib/libSystem.B.dylib", purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return nil, err
	}
	var self func() uint32
	h = &machHost{}
	purego.RegisterLibFunc(&self, lib, "mach_host_self")
	purego.RegisterLibFunc(&h.statistics, lib, "host_statistics64")
	purego.RegisterLibFunc(&h.pageSize, lib, "host_page_size")
	h.port = self()
	return h, nil
})

func read() (Reading, bool) {
	h, err := machOnce()
	if err != nil {
		return Reading{}, false
	}
	var buf [vmStats64Size]byte
	n := uint32(vmStats64Size / 4)
	if h.statistics(h.port, hostVMInfo64, unsafe.Pointer(&buf[0]), &n) != 0 || n != vmStats64Size/4 {
		return Reading{}, false
	}
	var ps uint64
	if h.pageSize(h.port, &ps) != 0 || ps == 0 {
		return Reading{}, false
	}
	phys, err := unix.SysctlUint64("hw.memsize")
	if err != nil || phys == 0 {
		return Reading{}, false
	}
	s, ok := decodeVMStats64(buf[:])
	if !ok {
		return Reading{}, false
	}
	return fromVMStats64(s, int64(ps), int64(phys)), true
}
