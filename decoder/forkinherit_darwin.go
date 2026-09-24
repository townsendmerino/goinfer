//go:build darwin

package decoder

import (
	"syscall"
	"unsafe"
)

// vmInheritNone is mach/vm_inherit.h's VM_INHERIT_NONE: the region is absent from a fork()ed child.
const vmInheritNone = 2

// excludeFromFork marks b's pages VM_INHERIT_NONE, so a fork() of this process leaves them out of the
// child instead of copying them.
//
// Why a read-only weight mapping needs this (docs/measurements/m26-alias-fork-collapse-2026-09-24.md):
// a .giw is mapped PROT_READ|MAP_PRIVATE. Once anything wires a page of it through IOKit — Metal does,
// for every page a command buffer reads through a no-copy buffer — the kernel copy-on-write-copies the
// wired page (vm_object_iopl_request faults with write intent) and marks the mapping's object
// true_share. From then on vm_map_fork cannot share the entry lazily: vm_object_copy_delayed refuses
// wired pages and the kernel copies the WHOLE entry, page by page, on the forking thread. Every
// os/exec in the process is a fork on darwin; on a 15 GB mapping on a 16 GB Mac the first one after
// the first GPU request paged the whole server out. Measured with metal/alias_forkprobe_test.go: a
// fork+exec went 2.8 ms → 1,092 ms after one GPU touch of a 16 MiB window of a 2 GiB mapping, and
// stayed ~3.8 ms with the mapping VM_INHERIT_NONE. A child never needs the weights (the only children
// are exec'd helpers, which discard the address space immediately), so leaving them out is free.
//
// minherit(2) has no stdlib wrapper; SYS_MINHERIT is in the stdlib's own darwin syscall table.
func excludeFromFork(b []byte) error {
	if len(b) == 0 {
		return nil
	}
	if _, _, e := syscall.Syscall(syscall.SYS_MINHERIT, uintptr(unsafe.Pointer(&b[0])), uintptr(len(b)), vmInheritNone); e != 0 {
		return e
	}
	return nil
}
