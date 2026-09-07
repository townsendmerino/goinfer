//go:build linux

package decoder

import (
	"os"
	"strconv"
	"strings"
	"sync"
)

// HostRAMBytes is this machine's physical RAM, or 0 when it cannot be determined — and 0 is a
// real answer that every caller must treat as "proceed", never as "no memory".
//
// Read from /proc/meminfo rather than through a syscall wrapper so the pure-Go root module stays
// free of golang.org/x/sys (audit M-19 keeps the root dependency graph minimal; the Metal module,
// which already depends on it, uses unix.SysctlUint64 for the same figure).
//
// KNOWN LIMIT: MemTotal is the HOST's memory. Inside a container with a cgroup memory limit this
// over-reports, so the guard can let a doomed load through there — the safe direction, and the
// direction the whole guard errs in. Reading memory.max is a follow-on, not this pass.
func HostRAMBytes() int64 { return hostRAMOnce() }

var hostRAMOnce = sync.OnceValue(func() int64 {
	return meminfoField(readMeminfo(), "MemTotal:")
})

// HostRAMAvailableBytes is this machine's CURRENTLY AVAILABLE memory — what the kernel estimates
// can be allocated by a new process without swapping (free pages plus reclaimable cache) — or 0
// when it cannot be determined. Unlike HostRAMBytes, this is NOT cached: it changes continuously
// as other processes run, which is the entire reason it exists (R13-follow-on,
// docs/measurements/... the Mac re-run that found this). HostRAMBytes reports total physical RAM,
// which never changes and is safe to read once; MemAvailable is the opposite by nature.
//
// KNOWN LIMIT: same container caveat as HostRAMBytes — /proc/meminfo reports the HOST's memory
// inside a container with a cgroup limit, over-reporting availability there. The safe direction.
func HostRAMAvailableBytes() int64 {
	return meminfoField(readMeminfo(), "MemAvailable:")
}

func readMeminfo() string {
	b, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return ""
	}
	return string(b)
}

// meminfoField extracts one "Key:   NNNNN kB" line's value in bytes, or 0 if the key is absent or
// the line is not shaped as expected (a missing or unexpected unit means "we do not know", never
// a guess off by a factor of 1024).
func meminfoField(meminfo, key string) int64 {
	for line := range strings.SplitSeq(meminfo, "\n") {
		rest, ok := strings.CutPrefix(line, key)
		if !ok {
			continue
		}
		f := strings.Fields(rest)
		if len(f) != 2 || f[1] != "kB" {
			return 0
		}
		kb, err := strconv.ParseInt(f[0], 10, 64)
		if err != nil || kb <= 0 {
			return 0
		}
		return kb * 1024
	}
	return 0
}
