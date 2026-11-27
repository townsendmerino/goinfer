//go:build windows

package serveapp

import (
	"os"
	"syscall"
)

// fsctlSetSparse is NTFS's FSCTL_SET_SPARSE control code so multi-GB File.Truncate test fixtures take zero disk blocks.
const fsctlSetSparse = 0x000900c4

func markSparseForTest(f *os.File) {
	var ret uint32
	_ = syscall.DeviceIoControl(syscall.Handle(f.Fd()), fsctlSetSparse, nil, 0, nil, 0, &ret, nil)
}
