//go:build darwin

package hostmem

import (
	"sync"
	"unsafe"

	"golang.org/x/sys/unix"
)

// hostVMInfo64 is HOST_VM_INFO64 and hostCPULoadInfo HOST_CPU_LOAD_INFO (mach/host_info.h).
const (
	hostVMInfo64    = 4
	hostCPULoadInfo = 3
)

// machHost is the libSystem calls hostmem makes (libsystem_darwin.go) and this task's host and task ports, asked for once:
// mach_host_self hands out a send right per call.
type machHost struct {
	port       uint32
	statistics func(host uint32, flavor int32, info unsafe.Pointer, count *uint32) int32
	pageSize   func(host uint32, size *uint64) int32
	taskSelf   uint32
	nameForPid func(task uint32, pid int32, name *uint32) int32
	taskInfo   func(task uint32, flavor int32, info unsafe.Pointer, count *uint32) int32
}

var machOnce = sync.OnceValue(func() *machHost {
	return &machHost{port: machHostSelf(), statistics: hostStatistics64, pageSize: hostPageSize, taskSelf: machTaskSelf(),
		nameForPid: taskNameForPid, taskInfo: taskInfo}
})

func read() (Reading, bool) {
	h := machOnce()
	buf, ok := h.vmStats()
	if !ok {
		return Reading{}, false
	}
	return h.reading(buf)
}

// readCPU decodes host_cpu_load_info: user, system, idle and nice ticks, in that order.
func readCPU() (CPUTicks, bool) {
	h := machOnce()
	var t [4]uint32
	n := uint32(len(t))
	if h.statistics(h.port, hostCPULoadInfo, unsafe.Pointer(&t[0]), &n) != 0 || n != uint32(len(t)) {
		return CPUTicks{}, false
	}
	busy := uint64(t[0]) + uint64(t[1]) + uint64(t[3])
	return CPUTicks{Busy: busy, Total: busy + uint64(t[2])}, true
}

// vmStats is one host_statistics64(HOST_VM_INFO64) reply, undecoded.
func (h *machHost) vmStats() (buf [vmStats64Size]byte, ok bool) {
	n := uint32(vmStats64Size / 4)
	ok = h.statistics(h.port, hostVMInfo64, unsafe.Pointer(&buf[0]), &n) == 0 && n == vmStats64Size/4
	return buf, ok
}

func (h *machHost) reading(buf [vmStats64Size]byte) (Reading, bool) {
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
	level, err := unix.SysctlUint32("kern.memorystatus_level")
	if err != nil || level > 100 {
		return Reading{}, false
	}
	r := fromVMStats64(s, int64(ps), int64(phys))
	r.Level, r.Available = int(level), fromLevel(int(level), int64(phys))
	r.Source = "macOS kern.memorystatus_level × physical, less PressureBuffer"
	return r, true
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
	h := machOnce()
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
	h := machOnce()
	var ps uint64
	if h.pageSize(h.port, &ps) != 0 || ps == 0 {
		return OracleSample{}, false
	}
	page, err := unix.Mmap(-1, 0, int(ps), unix.PROT_READ|unix.PROT_WRITE, unix.MAP_ANON|unix.MAP_PRIVATE)
	if err != nil {
		return OracleSample{}, false
	}
	defer unix.Munmap(page)
	before, ok1 := jetsamAvailable(int64(ps))
	first, ok2 := h.vmStats()
	page[0] = 1 // a page fault, which a fresh second reply counts and the first did not
	second, ok3 := h.vmStats()
	after, ok4 := jetsamAvailable(int64(ps))
	if !ok1 || !ok2 || !ok3 || !ok4 {
		return OracleSample{}, false
	}
	r, ok := h.reading(second)
	if !ok {
		return OracleSample{}, false
	}
	// The rate limit's cache changes only when some caller takes a fresh reading, and the fault between the two calls changes a
	// fresh reply, so the second reply differs from the first exactly when it was computed after the first call began.
	return OracleSample{Ours: r, JetsamBefore: before, JetsamAfter: after, Fresh: first != second}, true
}
