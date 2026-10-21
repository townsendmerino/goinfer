//go:build darwin

package decoder

import (
	"encoding/binary"
	"sync"
	"syscall"

	"github.com/townsendmerino/goinfer/internal/hostmem"
)

// HostRAMBytes is this machine's physical RAM, or 0 when it cannot be determined; 0 is a real answer that every caller
// must treat as "proceed", never as "no memory".
//
// It reads the hw.memsize sysctl with a bare sysctl(2) through the stdlib, once per process. The value is a
// little-endian u64 and syscall.Sysctl drops a single trailing NUL byte (16 GB = 0x4_0000_0000 loses its zero high
// byte), so it is zero-extended back to 8 bytes. It does not shell out to `sysctl`: every exec is a fork(), and a fork
// of a process whose .giw mapping a GPU backend has wired copies the whole mapping
// (docs/measurements/m26-alias-fork-collapse-2026-09-24.md).
func HostRAMBytes() int64 { return hostRAMOnce() }

var hostRAMOnce = sync.OnceValue(func() int64 {
	v, err := syscall.Sysctl("hw.memsize")
	if err != nil {
		return 0
	}
	return decodeSysctlU64([]byte(v))
})

// decodeSysctlU64 zero-extends a little-endian u64 sysctl value that syscall.Sysctl may have shortened
// by one trailing NUL. 0 for anything that is not 7 or 8 bytes.
func decodeSysctlU64(b []byte) int64 {
	if len(b) != 7 && len(b) != 8 {
		return 0
	}
	var p [8]byte
	copy(p[:], b)
	n := binary.LittleEndian.Uint64(p[:])
	if n == 0 || n > 1<<62 {
		return 0
	}
	return int64(n)
}

// HostRAMAvailableBytes is this machine's currently available memory, or 0 when it cannot be determined: internal/hostmem's
// reading, the free, file-backed and purgeable pages the kernel can hand back without compressing or swapping, so
// cached files count as available. Not cached: it changes as other processes run. It is a Mach call (host_statistics64), not an
// exec, so it costs no fork.
func HostRAMAvailableBytes() int64 { return hostmem.Available() }
