//go:build darwin

package probe

import (
	"encoding/binary"
	"syscall"

	"golang.org/x/sys/unix"
)

// Physical is compared against the Metal module's former read (unix.SysctlUint64), which it now replaces, as well as
// decoder's (syscall.Sysctl, decoded); Available was internal/hostmem's before and after the move, so it has no frozen copy.
func frozenReaders() frozen {
	return frozen{
		physical: func() int64 {
			ram, err := unix.SysctlUint64("hw.memsize")
			if err != nil {
				return 0
			}
			if d := frozenDecoderHostRAM(); d != int64(ram) {
				return -1 // the two former reads disagree: no single frozen figure
			}
			return int64(ram)
		},
		swapUsed: frozenSwapUsedBytes,
	}
}

// decoder/hostram_darwin.go

func frozenDecoderHostRAM() int64 {
	v, err := syscall.Sysctl("hw.memsize")
	if err != nil {
		return 0
	}
	b := []byte(v)
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

// decoder/memwatch_darwin.go

func frozenSwapUsedBytes() (usedBytes int64, ok bool) {
	v, err := syscall.Sysctl("vm.swapusage")
	if err != nil {
		return 0, false
	}
	b := []byte(v)
	if len(b) < 24 {
		return 0, false
	}
	used := binary.LittleEndian.Uint64(b[16:24])
	if used > 1<<62 {
		return 0, false
	}
	return int64(used), true
}
