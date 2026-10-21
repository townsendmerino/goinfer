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
	taskSelf   uint32
	nameForPid func(task uint32, pid int32, name *uint32) int32
	taskInfo   func(task uint32, flavor int32, info unsafe.Pointer, count *uint32) int32
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
	var taskSelf func() uint32
	purego.RegisterLibFunc(&taskSelf, lib, "mach_task_self")
	purego.RegisterLibFunc(&h.nameForPid, lib, "task_name_for_pid")
	purego.RegisterLibFunc(&h.taskInfo, lib, "task_info")
	h.port, h.taskSelf = self(), taskSelf()
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

// taskVMInfo is TASK_VM_INFO (mach/task_info.h); taskVMInfoBytes covers struct task_vm_info through phys_footprint.
const (
	taskVMInfo      = 22
	taskVMInfoBytes = 152
)

// readTask reads pid's task_vm_info through its task name port (task_name_for_pid needs no entitlement for the user's own
// processes, where task_for_pid does). The offsets are struct task_vm_info's: resident_size 16, internal 48, compressed 120,
// compressed_peak 128, phys_footprint 144.
func readTask(pid int) (TaskVM, bool) {
	h, err := machOnce()
	if err != nil {
		return TaskVM{}, false
	}
	var port uint32
	if h.nameForPid(h.taskSelf, int32(pid), &port) != 0 {
		return TaskVM{}, false
	}
	var buf [512]byte
	n := uint32(len(buf) / 4)
	if h.taskInfo(port, taskVMInfo, unsafe.Pointer(&buf[0]), &n) != 0 || int(n)*4 < taskVMInfoBytes {
		return TaskVM{}, false
	}
	return decodeTaskVM(buf[:taskVMInfoBytes]), true
}

// jetsamAvailable is vm.page_pageable_external_count + vm.page_free_count times the page size: what XNU counts as available pages
// under CONFIG_JETSAM (osfmk/vm/vm_page.h VM_CHECK_MEMORYSTATUS, purgeable left out where a compressor runs).
func jetsamAvailable(pageSize int64) (int64, bool) {
	ext, err1 := unix.SysctlUint32("vm.page_pageable_external_count")
	free, err2 := unix.SysctlUint32("vm.page_free_count")
	if err1 != nil || err2 != nil {
		return 0, false
	}
	return (int64(ext) + int64(free)) * pageSize, true
}

func readWithOracle() (OracleSample, bool) {
	h, err := machOnce()
	if err != nil {
		return OracleSample{}, false
	}
	var ps uint64
	if h.pageSize(h.port, &ps) != 0 || ps == 0 {
		return OracleSample{}, false
	}
	before, ok1 := jetsamAvailable(int64(ps))
	r, ok2 := read()
	after, ok3 := jetsamAvailable(int64(ps))
	if !ok1 || !ok2 || !ok3 {
		return OracleSample{}, false
	}
	return OracleSample{Ours: r, JetsamBefore: before, JetsamAfter: after}, true
}
