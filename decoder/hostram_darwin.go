//go:build darwin

package decoder

import (
	"encoding/binary"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"syscall"
)

// HostRAMBytes is this machine's physical RAM, or 0 when it cannot be determined; 0 is a real answer that every caller
// must treat as "proceed", never as "no memory".
//
// It reads the hw.memsize sysctl with a bare sysctl(2) through the stdlib, once per process. The value is a
// little-endian u64 and syscall.Sysctl drops a single trailing NUL byte (16 GB = 0x4_0000_0000 loses its zero high
// byte), so it is zero-extended back to 8 bytes. It does not shell out to `sysctl`: every exec is a fork(), and a fork
// of a process whose .giw mapping a GPU backend has wired copies the whole mapping
// (docs/measurements/m26-alias-fork-collapse-2026-09-24.md).
func HostRAMBytes() int64 { return hostRAMOnce() }

var hostRAMOnce = sync.OnceValue(func() int64 {
	v, err := syscall.Sysctl("hw.memsize")
	if err != nil {
		return 0
	}
	return decodeSysctlU64([]byte(v))
})

// decodeSysctlU64 zero-extends a little-endian u64 sysctl value that syscall.Sysctl may have shortened
// by one trailing NUL. 0 for anything that is not 7 or 8 bytes.
func decodeSysctlU64(b []byte) int64 {
	if len(b) != 7 && len(b) != 8 {
		return 0
	}
	var p [8]byte
	copy(p[:], b)
	n := binary.LittleEndian.Uint64(p[:])
	if n == 0 || n > 1<<62 {
		return 0
	}
	return int64(n)
}

// HostRAMAvailableBytes is this machine's currently available memory, free plus reclaimable pages (an approximation of
// what `vm_stat` and Activity Monitor's "memory pressure" draw from), or 0 when it cannot be determined. Unlike
// HostRAMBytes it is not cached: it changes continuously as other processes run, which is why it exists. A budget
// computed as a fraction of total RAM assumes nothing else on the machine needs more than the remaining fraction,
// which a laptop with a browser and an IDE open routinely breaks
// (docs/measurements/cold-user-2026-09-07-macbook-arm64.md).
//
// Still an exec of vm_stat (the inactive-page count has no sysctl): it runs at load and from the web UI's status, not
// on a timer, and the .giw mapping is VM_INHERIT_NONE (forkinherit_darwin.go), so the fork it costs does not copy the
// weights.
//
// The approximation is free + inactive + speculative + purgeable pages: the pages vm_stat reports and the ones macOS
// reclaims before it would swap (inactive is the disk-cache equivalent; purgeable is discardable on demand). Wired and
// active pages are excluded as genuinely in use. Apple Silicon uses 16 KB pages, not 4 KB; the page size is read from
// vm_stat's own header, never assumed.
func HostRAMAvailableBytes() int64 {
	out, err := exec.Command("vm_stat").Output()
	if err != nil {
		return 0
	}
	return parseVMStatAvailable(string(out))
}

var vmStatPageSizeRe = regexp.MustCompile(`page size of (\d+) bytes`)

// parseVMStatAvailable is separated from the exec.Command call so it can be unit-tested with
// real, committed vm_stat output on any platform, not only when running on darwin itself.
func parseVMStatAvailable(out string) int64 {
	m := vmStatPageSizeRe.FindStringSubmatch(out)
	if m == nil {
		return 0
	}
	pageSize, err := strconv.ParseInt(m[1], 10, 64)
	if err != nil || pageSize <= 0 {
		return 0
	}
	var pages int64
	found := false
	for _, key := range []string{"Pages free:", "Pages inactive:", "Pages speculative:", "Pages purgeable:"} {
		v, ok := vmStatField(out, key)
		if !ok {
			return 0 // a field this parse depends on is missing or reshaped — unknown, not a guess
		}
		pages += v
		found = true
	}
	if !found {
		return 0
	}
	return pages * pageSize
}

// vmStatField reads one "Label:                 NNNNN." line's integer value.
func vmStatField(out, key string) (int64, bool) {
	for line := range strings.SplitSeq(out, "\n") {
		rest, ok := strings.CutPrefix(strings.TrimSpace(line), key)
		if !ok {
			continue
		}
		rest = strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(rest), "."))
		v, err := strconv.ParseInt(rest, 10, 64)
		if err != nil || v < 0 {
			return 0, false
		}
		return v, true
	}
	return 0, false
}
