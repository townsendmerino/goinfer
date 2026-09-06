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
	b, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	for line := range strings.SplitSeq(string(b), "\n") {
		rest, ok := strings.CutPrefix(line, "MemTotal:")
		if !ok {
			continue
		}
		f := strings.Fields(rest)
		// "MemTotal:  16311288 kB" — the unit is always kB in practice, but a missing or
		// unexpected unit means we do not know, rather than off by 1024.
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
})
