//go:build darwin

package probe

import (
	"os/exec"
	"strconv"
	"strings"
	"testing"
)

func TestDecodeSysctlU64(t *testing.T) {
	sixteen := []byte{0, 0, 0, 0, 4, 0, 0, 0} // 16 GiB, little-endian
	for _, tc := range []struct {
		name string
		b    []byte
		want int64
	}{
		{"full 8 bytes", sixteen, 16 << 30},
		{"trailing NUL dropped (what syscall.Sysctl returns)", sixteen[:7], 16 << 30},
		{"96 GiB", []byte{0, 0, 0, 0, 24, 0, 0}, 96 << 30},
		{"wrong length is unknown", []byte{1, 2, 3}, 0},
		{"zero is unknown", make([]byte, 8), 0},
	} {
		if got := decodeSysctlU64(tc.b); got != tc.want {
			t.Errorf("%s: decodeSysctlU64 = %d, want %d", tc.name, got, tc.want)
		}
	}
}

// The bare-sysctl reader must match `sysctl -n hw.memsize` exactly (the exec reader it replaced).
func TestPhysical_matchesSysctlCommand(t *testing.T) {
	out, err := exec.Command("sysctl", "-n", "hw.memsize").Output()
	if err != nil {
		t.Skipf("sysctl command: %v", err)
	}
	want, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
	if err != nil {
		t.Fatalf("parse %q: %v", out, err)
	}
	if got := Physical(); got != want {
		t.Errorf("Physical = %d, `sysctl -n hw.memsize` = %d", got, want)
	}
}
