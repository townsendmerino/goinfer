//go:build linux

package probe

import (
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
)

func frozenReaders() frozen {
	return frozen{
		physical:  func() int64 { return frozenMeminfoField(frozenReadMeminfo(), "MemTotal:") },
		available: frozenHostmemAvailable,
		swapUsed:  func() (int64, bool) { return frozenParseSwapUsedLinuxMeminfo(frozenReadMeminfo()) },
		selfSwap:  func() (int64, bool) { return frozenSelfSwapBytes("/proc", os.Getpid()) },
	}
}

// decoder/hostram_linux.go

func frozenReadMeminfo() string {
	b, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return ""
	}
	return string(b)
}

func frozenMeminfoField(meminfo, key string) int64 {
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

// internal/hostmem: Available over read_linux.go's read

func frozenHostmemAvailable() int64 {
	b, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	avail := frozenMeminfoKB(string(b), "MemAvailable:")
	if avail <= 0 {
		return 0
	}
	return avail * 1024
}

func frozenMeminfoKB(meminfo, key string) int64 {
	for _, ln := range strings.Split(meminfo, "\n") {
		if f := strings.Fields(ln); len(f) >= 2 && f[0] == key {
			v, _ := strconv.ParseInt(f[1], 10, 64)
			return v
		}
	}
	return 0
}

// decoder/memwatch_linux.go

func frozenParseSwapUsedLinuxMeminfo(meminfo string) (usedBytes int64, ok bool) {
	if meminfo == "" {
		return 0, false
	}
	total, haveTotal := frozenMeminfoFieldOK(meminfo, "SwapTotal:")
	free, haveFree := frozenMeminfoFieldOK(meminfo, "SwapFree:")
	if !haveTotal || !haveFree || total < free {
		return 0, false
	}
	return total - free, true
}

func frozenMeminfoFieldOK(meminfo, key string) (int64, bool) {
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

func frozenSelfSwapBytes(root string, pid int) (usedBytes int64, ok bool) {
	self, haveSelf := frozenProcVmSwap(filepath.Join(root, strconv.Itoa(pid), "status"))
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
		if pp, ok := frozenProcParent(filepath.Join(root, e.Name(), "stat")); ok {
			children[pp] = append(children[pp], p)
		}
	}
	total := self
	for stack := slices.Clone(children[pid]); len(stack) > 0; {
		p := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if v, ok := frozenProcVmSwap(filepath.Join(root, strconv.Itoa(p), "status")); ok {
			total += v
		}
		stack = append(stack, children[p]...)
	}
	return total, true
}

func frozenProcVmSwap(path string) (int64, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, false
	}
	return frozenMeminfoFieldOK(string(b), "VmSwap:")
}

func frozenProcParent(path string) (int, bool) {
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
