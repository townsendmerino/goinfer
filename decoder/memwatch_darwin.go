//go:build darwin

package decoder

import (
	"encoding/binary"
	"syscall"
)

// SwapUsedBytes reads the vm.swapusage sysctl's used field as bytes, or (0, false) when it cannot be
// determined.
//
// A bare sysctl(2) through the stdlib, NOT `exec sysctl -n vm.swapusage`: the serving swap guard calls
// this every 2 s, and every exec is a fork() — which, while a GPU backend has wired a page of the .giw
// mapping, used to copy the whole mapping (docs/measurements/m26-alias-fork-collapse-2026-09-24.md;
// Load now also marks the mapping VM_INHERIT_NONE, so this is the second of two independent fixes).
// The value is `struct xsw_usage` (sys/sysctl.h): u64 xsu_total, u64 xsu_avail, u64 xsu_used,
// u32 xsu_pagesize, boolean_t xsu_encrypted — 32 bytes, little-endian on every darwin target.
// syscall.Sysctl drops only a single trailing NUL (the high byte of xsu_encrypted, always 0), so the
// struct arrives intact through byte 24. An earlier comment here claimed it "truncates at the first
// NUL" and that there was therefore no bare-syscall alternative; neither is true.
func SwapUsedBytes() (usedBytes int64, ok bool) {
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
