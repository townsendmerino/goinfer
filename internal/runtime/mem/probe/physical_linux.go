//go:build linux

package probe

import (
	"os"
	"strconv"
	"strings"
	"sync"
)

// Physical is this machine's physical RAM, /proc/meminfo's MemTotal, or 0 when it cannot be determined; 0 is a real
// answer that every caller must treat as "proceed", never as "no memory". Read once per process.
//
// KNOWN LIMIT: MemTotal is the HOST's memory. Inside a container with a cgroup memory limit this over-reports, so the
// guard can let a doomed load through there, which is the safe direction and the direction the whole guard errs in.
// Reading memory.max is not done. Available has the same limit.
func Physical() int64 { return physicalOnce() }

var physicalOnce = sync.OnceValue(func() int64 {
	return meminfoField(readMeminfo(), "MemTotal:")
})

// available is Read's: /proc/meminfo's MemAvailable, the kernel's estimate of what a new process can allocate without
// swapping (free pages plus reclaimable cache).
func available() int64 { return readAvailable() }

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
