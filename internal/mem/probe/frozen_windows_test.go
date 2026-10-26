package probe

import (
	"syscall"
	"unsafe"
)

func frozenReaders() frozen {
	return frozen{physical: frozenHostRAMBytes, available: frozenHostRAMAvailableBytes}
}

// decoder/hostram_windows.go

type frozenMemoryStatusEx struct {
	length               uint32
	memoryLoad           uint32
	totalPhys            uint64
	availPhys            uint64
	totalPageFile        uint64
	availPageFile        uint64
	totalVirtual         uint64
	availVirtual         uint64
	availExtendedVirtual uint64
}

var frozenProc = syscall.NewLazyDLL("kernel32.dll").NewProc("GlobalMemoryStatusEx")

func frozenGlobalMemoryStatus() (st frozenMemoryStatusEx, ok bool) {
	st.length = uint32(unsafe.Sizeof(st))
	if r, _, _ := frozenProc.Call(uintptr(unsafe.Pointer(&st))); r == 0 {
		return frozenMemoryStatusEx{}, false
	}
	return st, true
}

func frozenHostRAMBytes() int64 {
	st, ok := frozenGlobalMemoryStatus()
	if !ok || st.totalPhys == 0 || st.totalPhys > 1<<62 {
		return 0
	}
	return int64(st.totalPhys)
}

func frozenHostRAMAvailableBytes() int64 {
	st, ok := frozenGlobalMemoryStatus()
	if !ok || st.availPhys == 0 || st.availPhys > st.totalPhys {
		return 0
	}
	return int64(st.availPhys)
}
