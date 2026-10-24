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
	return Reading{Physical: total * 1024, Available: avail * 1024, Reclaimable: avail * 1024, Source: "Linux /proc/meminfo MemAvailable"}, true
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

// readCPU reads /proc/stat's aggregate cpu line: user, nice, system, idle, iowait, irq, softirq, steal (guest time is inside user).
func readCPU() (CPUTicks, bool) {
	b, err := os.ReadFile("/proc/stat")
	if err != nil {
		return CPUTicks{}, false
	}
	line, _, _ := strings.Cut(string(b), "\n")
	f := strings.Fields(line)
	if len(f) < 9 || f[0] != "cpu" {
		return CPUTicks{}, false
	}
	var v [8]uint64
	for i := range v {
		if v[i], err = strconv.ParseUint(f[i+1], 10, 64); err != nil {
			return CPUTicks{}, false
		}
	}
	busy := v[0] + v[1] + v[2] + v[5] + v[6] + v[7]
	return CPUTicks{Busy: busy, Total: busy + v[3] + v[4]}, true
}

func readTask(int) (TaskVM, bool) { return TaskVM{}, false }

func readWithOracle() (OracleSample, bool) { return OracleSample{}, false }
