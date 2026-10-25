//go:build darwin

package hostmem

import (
	"syscall"
	"unsafe"
)

// The libSystem calls hostmem makes, bound without cgo the way golang.org/x/sys/unix binds its own: a dynamic import, an assembly
// trampoline per function (libsystem_darwin.s), and the runtime's syscall.syscall6 to call it on the system stack. purego would do
// the same and put a module in the root's graph that the root's cleanliness guard (ci.yml) refuses.

//go:linkname syscall_syscall6 syscall.syscall6
func syscall_syscall6(fn, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err syscall.Errno)

var (
	libc_mach_host_self_trampoline_addr      uintptr
	libc_mach_task_self_trampoline_addr      uintptr
	libc_host_statistics64_trampoline_addr   uintptr
	libc_host_page_size_trampoline_addr      uintptr
	libc_task_name_for_pid_trampoline_addr   uintptr
	libc_task_info_trampoline_addr           uintptr
	libc_host_processor_info_trampoline_addr uintptr
	libc_vm_deallocate_trampoline_addr       uintptr
)

//go:cgo_import_dynamic libc_mach_host_self mach_host_self "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_mach_task_self mach_task_self "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_host_statistics64 host_statistics64 "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_host_page_size host_page_size "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_task_name_for_pid task_name_for_pid "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_task_info task_info "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_host_processor_info host_processor_info "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_vm_deallocate vm_deallocate "/usr/lib/libSystem.B.dylib"

// Each wrapper returns the C function's result: a mach port, or a kern_return_t that is 0 on success.

func machHostSelf() uint32 {
	r, _, _ := syscall_syscall6(libc_mach_host_self_trampoline_addr, 0, 0, 0, 0, 0, 0)
	return uint32(r)
}

func machTaskSelf() uint32 {
	r, _, _ := syscall_syscall6(libc_mach_task_self_trampoline_addr, 0, 0, 0, 0, 0, 0)
	return uint32(r)
}

func hostStatistics64(host uint32, flavor int32, info unsafe.Pointer, count *uint32) int32 {
	r, _, _ := syscall_syscall6(libc_host_statistics64_trampoline_addr, uintptr(host), uintptr(flavor), uintptr(info),
		uintptr(unsafe.Pointer(count)), 0, 0)
	return int32(r)
}

func hostPageSize(host uint32, size *uint64) int32 {
	r, _, _ := syscall_syscall6(libc_host_page_size_trampoline_addr, uintptr(host), uintptr(unsafe.Pointer(size)), 0, 0, 0, 0)
	return int32(r)
}

func taskNameForPid(task uint32, pid int32, name *uint32) int32 {
	r, _, _ := syscall_syscall6(libc_task_name_for_pid_trampoline_addr, uintptr(task), uintptr(pid), uintptr(unsafe.Pointer(name)), 0, 0, 0)
	return int32(r)
}

func taskInfo(task uint32, flavor int32, info unsafe.Pointer, count *uint32) int32 {
	r, _, _ := syscall_syscall6(libc_task_info_trampoline_addr, uintptr(task), uintptr(flavor), uintptr(info),
		uintptr(unsafe.Pointer(count)), 0, 0)
	return int32(r)
}

func hostProcessorInfo(host uint32, flavor int32, count *uint32, info **uint32, infoCnt *uint32) int32 {
	r, _, _ := syscall_syscall6(libc_host_processor_info_trampoline_addr, uintptr(host), uintptr(flavor), uintptr(unsafe.Pointer(count)),
		uintptr(unsafe.Pointer(info)), uintptr(unsafe.Pointer(infoCnt)), 0)
	return int32(r)
}

func vmDeallocate(task uint32, addr, size uintptr) int32 {
	r, _, _ := syscall_syscall6(libc_vm_deallocate_trampoline_addr, uintptr(task), addr, size, 0, 0, 0)
	return int32(r)
}
