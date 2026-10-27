//go:build linux

package probe

import (
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
)

// SwapUsed reads /proc/meminfo's SwapTotal - SwapFree as bytes, or (0, false) when either
// field is missing or unparseable. Reuses readMeminfo rather than a second
// os.ReadFile of the same file. Unlike meminfoField (which folds "missing" and
// "genuinely zero" into the same 0 return — fine for a RAM figure that is never legitimately
// zero on a running machine, wrong here: zero swap-used is a real, common reading, and
// the swap watch's baseline capture must not mistake "field absent" for "zero swap"), this keeps
// found-ness explicit.
func SwapUsed() (usedBytes int64, ok bool) {
	return parseSwapUsedLinuxMeminfo(readMeminfo())
}

// parseSwapUsedLinuxMeminfo is separated from the file read so it is unit-testable with real,
// committed /proc/meminfo content — readMeminfo / meminfoField's own split, applied here.
func parseSwapUsedLinuxMeminfo(meminfo string) (usedBytes int64, ok bool) {
	if meminfo == "" {
		return 0, false
	}
	total, haveTotal := meminfoFieldOK(meminfo, "SwapTotal:")
	free, haveFree := meminfoFieldOK(meminfo, "SwapFree:")
	if !haveTotal || !haveFree || total < free {
		return 0, false
	}
	return total - free, true
}

// meminfoFieldOK is meminfoField's found-aware twin: same "Key:   NNNNN kB" shape, but reports
// whether the key was present and well-formed rather than collapsing that into 0.
func meminfoFieldOK(meminfo, key string) (int64, bool) {
	for line := range strings.SplitSeq(meminfo, "\n") {
		rest, ok := strings.CutPrefix(line, key)
		if !ok {
			continue
		}
		f := strings.Fields(rest)
		if len(f) != 2 || f[1] != "kB" {
			return 0, false
		}
		kb, err := strconv.ParseInt(f[0], 10, 64)
		if err != nil || kb < 0 {
			return 0, false
		}
		return kb * 1024, true
	}
	return 0, false
}

// SelfSwapReader returns the per-process swap probe: the swap the server and its descendants hold, from /proc/<pid>/status.
func SelfSwapReader() (r func() (usedBytes int64, ok bool), unavailable string) { return SelfSwap, "" }

// SelfSwap is the swap this process and every process descended from it hold, in bytes: the sum of VmSwap over the pid and its
// children, grandchildren and so on. It is (0, false) when this process's own VmSwap line is missing (a kernel without swap accounting)
// or its status cannot be read, so a caller falls back to the system-wide figure instead of watching a number that is not there.
// A descendant that exits mid-scan, or has no VmSwap line, is skipped.
func SelfSwap() (usedBytes int64, ok bool) { return selfSwapBytes("/proc", os.Getpid()) }

// selfSwapBytes is SelfSwap over a /proc tree rooted at root, so a test can build a machine in a temp dir.
func selfSwapBytes(root string, pid int) (usedBytes int64, ok bool) {
	self, haveSelf := procVmSwap(filepath.Join(root, strconv.Itoa(pid), "status"))
	if !haveSelf {
		return 0, false
	}
	ents, err := os.ReadDir(root)
	if err != nil {
		return 0, false
	}
	children := map[int][]int{}
	for _, e := range ents {
		p, err := strconv.Atoi(e.Name())
		if err != nil || p == pid {
			continue
		}
		if pp, ok := procParent(filepath.Join(root, e.Name(), "stat")); ok {
			children[pp] = append(children[pp], p)
		}
	}
	total := self
	for stack := slices.Clone(children[pid]); len(stack) > 0; {
		p := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if v, ok := procVmSwap(filepath.Join(root, strconv.Itoa(p), "status")); ok {
			total += v
		}
		stack = append(stack, children[p]...)
	}
	return total, true
}

// procVmSwap reads the VmSwap line of a /proc/<pid>/status file as bytes.
func procVmSwap(path string) (int64, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, false
	}
	return meminfoFieldOK(string(b), "VmSwap:")
}

// procParent reads the parent pid from a /proc/<pid>/stat line. The command name is parenthesised and may itself contain spaces and
// parentheses, so the fields are counted from the last ')'.
func procParent(path string) (int, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, false
	}
	s := string(b)
	i := strings.LastIndexByte(s, ')')
	if i < 0 {
		return 0, false
	}
	f := strings.Fields(s[i+1:])
	if len(f) < 2 {
		return 0, false
	}
	pp, err := strconv.Atoi(f[1])
	return pp, err == nil
}
