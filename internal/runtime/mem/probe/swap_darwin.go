//go:build darwin

package probe

import (
	"encoding/binary"
	"os"
	"syscall"
)

// SwapUsed reads the vm.swapusage sysctl's used field as bytes, or (0, false) when it cannot be
// determined.
//
// A bare sysctl(2) through the stdlib, NOT `exec sysctl -n vm.swapusage`: the serving swap guard calls this every 2 s,
// and every exec is a fork(), which is expensive in a process whose .giw mapping a GPU backend has wired unless the
// mapping is VM_INHERIT_NONE (docs/measurements/m26-alias-fork-collapse-2026-09-24.md; Load marks it so, an
// independent fix). The value is `struct xsw_usage` (sys/sysctl.h): u64 xsu_total, u64 xsu_avail, u64 xsu_used, u32
// xsu_pagesize, boolean_t xsu_encrypted: 32 bytes, little-endian on every darwin target. syscall.Sysctl drops only a
// single trailing NUL (the high byte of xsu_encrypted, always 0), so the struct arrives intact through byte 24.
func SwapUsed() (usedBytes int64, ok bool) {
	v, err := syscall.Sysctl("vm.swapusage")
	if err != nil {
		return 0, false
	}
	return decodeXswUsed([]byte(v))
}

// decodeXswUsed extracts xsu_used from the raw vm.swapusage bytes.
func decodeXswUsed(b []byte) (usedBytes int64, ok bool) {
	if len(b) < 24 {
		return 0, false
	}
	used := binary.LittleEndian.Uint64(b[16:24])
	if used > 1<<62 {
		return 0, false
	}
	return int64(used), true
}

// SelfSwapName and SelfSwapWhat name SelfSwapReader's figure in the swap guard's messages.
const (
	SelfSwapName = "server compressed"
	SelfSwapWhat = "this process's compressed pages, task_vm_info: macOS has no per-process swap figure"
)

// SelfSwapReader returns the per-process probe the swap guard watches: this process's compressed pages (ReadTask's Compressed), the
// pages macOS has taken from the server itself, which it swaps from. Other processes' pages do not move it, as they move system
// swap. It is nil, with the reason, when this process's task_vm_info cannot be read. Unlike Linux's VmSwap it does not count
// descendants. B41's gate 2 read the server this way (docs/measurements/hostmem-available-2026-10/README.md, Amendment A1).
func SelfSwapReader() (r func() (usedBytes int64, ok bool), unavailable string) {
	pid := os.Getpid()
	if _, ok := readTask(pid); !ok {
		return nil, "this process's task_vm_info is not readable"
	}
	return func() (int64, bool) {
		t, ok := readTask(pid)
		return t.Compressed, ok
	}, ""
}
