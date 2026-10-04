package decoder

import (
	"sync"
	"syscall"
	"unsafe"
)

// memoryStatusEx is kernel32's MEMORYSTATUSEX: the fields GlobalMemoryStatusEx fills, in its order and widths.
type memoryStatusEx struct {
	length               uint32 // the struct's own size, which the caller must set
	memoryLoad           uint32
	totalPhys            uint64
	availPhys            uint64
	totalPageFile        uint64
	availPageFile        uint64
	totalVirtual         uint64
	availVirtual         uint64
	availExtendedVirtual uint64
}

var procGlobalMemoryStatusEx = syscall.NewLazyDLL("kernel32.dll").NewProc("GlobalMemoryStatusEx")

// globalMemoryStatus calls GlobalMemoryStatusEx through the stdlib's syscall package (no x/sys dependency, as the
// darwin and linux probes take none). ok is false when the call fails; every caller then answers 0, unknown.
func globalMemoryStatus() (st memoryStatusEx, ok bool) {
	st.length = uint32(unsafe.Sizeof(st))
	if r, _, _ := procGlobalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&st))); r == 0 {
		return memoryStatusEx{}, false
	}
	return st, true
}

// HostRAMBytes is this machine's physical RAM (MEMORYSTATUSEX.ullTotalPhys), or 0 when it cannot be determined, and 0
// is a real answer every caller treats as "proceed". Read once per process, like the darwin and linux probes.
func HostRAMBytes() int64 { return hostRAMOnce() }

var hostRAMOnce = sync.OnceValue(func() int64 {
	st, ok := globalMemoryStatus()
	if !ok || st.totalPhys == 0 || st.totalPhys > 1<<62 {
		return 0
	}
	return int64(st.totalPhys)
})

// HostRAMAvailableBytes is the physical memory available now (MEMORYSTATUSEX.ullAvailPhys: the standby, free and
// zero page lists, which Windows hands out without paging anything else out), or 0 when it cannot be determined. Not
// cached: it changes as other processes run, which is why it exists (see the darwin probe's comment).
func HostRAMAvailableBytes() int64 {
	st, ok := globalMemoryStatus()
	if !ok || st.availPhys == 0 || st.availPhys > st.totalPhys {
		return 0
	}
	return int64(st.availPhys)
}
