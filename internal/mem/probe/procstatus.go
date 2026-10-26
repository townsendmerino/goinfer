package probe

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// ProcMemory is a process's memory as /proc/<pid>/status reports it, in kB. HWM is the kernel's high-water mark of the
// resident set, so one late sample reads the run's peak. A field the file does not carry reads 0.
type ProcMemory struct {
	HWMKB, RSSKB, SwapKB int64
}

// ReadProcMemory reads pid's ProcMemory. ok is false where the file cannot be read: a process that has exited, or a
// system without /proc.
func ReadProcMemory(pid int) (m ProcMemory, ok bool) {
	b, err := os.ReadFile(filepath.Join("/proc", strconv.Itoa(pid), "status"))
	if err != nil {
		return ProcMemory{}, false
	}
	return parseProcMemory(string(b)), true
}

// parseProcMemory reads the VmHWM, VmRSS and VmSwap kB fields of a /proc/<pid>/status text.
func parseProcMemory(s string) ProcMemory {
	kb := map[string]int64{}
	for ln := range strings.SplitSeq(s, "\n") {
		k, v, ok := strings.Cut(ln, ":")
		if !ok {
			continue
		}
		f := strings.Fields(v)
		if len(f) == 2 && f[1] == "kB" {
			if n, err := strconv.ParseInt(f[0], 10, 64); err == nil {
				kb[k] = n
			}
		}
	}
	return ProcMemory{HWMKB: kb["VmHWM"], RSSKB: kb["VmRSS"], SwapKB: kb["VmSwap"]}
}
