//go:build darwin

package hostmem

import (
	"sync"
	"unsafe"

	"golang.org/x/sys/unix"
)

// hostVMInfo64 is HOST_VM_INFO64 (mach/host_info.h).
const hostVMInfo64 = 4

// machPorts is this task's host and task ports, asked for once: mach_host_self hands out a send right per call.
var machPorts = sync.OnceValues(func() (host, task uint32) { return machHostSelf(), machTaskSelf() })

func read() (Reading, bool) {
	host, _ := machPorts()
	var buf [vmStats64Size]byte
	n := uint32(vmStats64Size / 4)
	if hostStatistics64(host, hostVMInfo64, unsafe.Pointer(&buf[0]), &n) != 0 || n != vmStats64Size/4 {
		return Reading{}, false
	}
	var ps uint64
	if hostPageSize(host, &ps) != 0 || ps == 0 {
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
	_, self := machPorts()
	var port uint32
	if taskNameForPid(self, int32(pid), &port) != 0 {
		return TaskVM{}, false
	}
	var buf [512]byte
	n := uint32(len(buf) / 4)
	if taskInfo(port, taskVMInfo, unsafe.Pointer(&buf[0]), &n) != 0 || int(n)*4 < taskVMInfoBytes {
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
	host, _ := machPorts()
	var ps uint64
	if hostPageSize(host, &ps) != 0 || ps == 0 {
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
