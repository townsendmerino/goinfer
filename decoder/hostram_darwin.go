//go:build darwin

package decoder

import (
	"os/exec"
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
