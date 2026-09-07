//go:build darwin

package decoder

import (
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"sync"
)

// HostRAMBytes is this machine's physical RAM, or 0 when it cannot be determined — and 0 is a
// real answer that every caller must treat as "proceed", never as "no memory".
//
// `sysctl -n hw.memsize`, shelled out ONCE per process, because the pure-Go root module does not
// depend on golang.org/x/sys and the standard library exposes no 64-bit sysctl (syscall.Sysctl
// returns a string truncated at the first NUL, which a byte count of 16 GB contains). The Metal
// module reads the same value through unix.SysctlUint64; it already has that dependency and the
// root module deliberately does not (audit M-19). cmd/gate/gpu.go shells out for sysctl values
// the same way.
func HostRAMBytes() int64 { return hostRAMOnce() }

var hostRAMOnce = sync.OnceValue(func() int64 {
	out, err := exec.Command("sysctl", "-n", "hw.memsize").Output()
	if err != nil {
		return 0
	}
	n, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
	if err != nil || n <= 0 {
		return 0
	}
	return n
})

// HostRAMAvailableBytes is this machine's CURRENTLY AVAILABLE memory — free plus reclaimable
// pages, an approximation of what `vm_stat` and Activity Monitor's "memory pressure" both draw
// from — or 0 when it cannot be determined. Unlike HostRAMBytes, this is NOT cached: it changes
// continuously as other processes run, which is the entire reason it exists (R13-follow-on,
// docs/measurements/cold-user-2026-09-07-macbook-arm64.md's live re-run of the R13 fix): a
// budget computed as a fraction of TOTAL RAM assumes nothing else on the machine ever needs more
// than the remaining fraction, which a real laptop with a browser and an IDE open routinely
// breaks. `serve check`'s own requests pushed a load that the load-time guard had already
// correctly auto-pinned into 9.7 GB of swap, because "70% of 16 GB" was never actually free —
// this reads what IS actually free instead.
//
// APPROXIMATION, STATED RATHER THAN HIDDEN: free + inactive + speculative + purgeable pages,
// matching the pages vm_stat itself reports and the ones macOS reclaims before it would ever
// swap (inactive is the disk-cache equivalent; purgeable is explicitly discardable on demand).
// wired and active pages are excluded — those are genuinely in use. This is the same kind of
// measured-approximation-with-a-stated-residual as quantBytesPerElem's probe matrix (fitguard.go)
// — real numbers from the real tool, with the one simplification named.
//
// Apple Silicon uses 16 KB pages, not the traditional 4 KB (confirmed on the M1 Pro that found
// this bug) — the page size is read from vm_stat's own header, never assumed.
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
