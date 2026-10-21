//go:build linux

package hostmem

import (
	"os"
	"strconv"
	"strings"
)

func read() (Reading, bool) {
	b, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return Reading{}, false
	}
	total, avail := meminfoKB(string(b), "MemTotal:"), meminfoKB(string(b), "MemAvailable:")
	if avail <= 0 {
		return Reading{}, false
	}
	return Reading{Physical: total * 1024, Available: avail * 1024, Source: "Linux /proc/meminfo MemAvailable"}, true
}

// meminfoKB is a /proc/meminfo field's value in kB, or 0 when it is missing.
func meminfoKB(meminfo, key string) int64 {
	for _, ln := range strings.Split(meminfo, "\n") {
		if f := strings.Fields(ln); len(f) >= 2 && f[0] == key {
			v, _ := strconv.ParseInt(f[1], 10, 64)
			return v
		}
	}
	return 0
}

func readTask(int) (TaskVM, bool) { return TaskVM{}, false }
