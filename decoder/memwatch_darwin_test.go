//go:build darwin

package decoder

import (
	"encoding/binary"
	"os/exec"
	"strconv"
	"strings"
	"testing"
)

// xswUsage builds a vm.swapusage value the way syscall.Sysctl returns it: the 32-byte struct with its
// final byte (the high byte of xsu_encrypted, always 0) dropped.
func xswUsage(total, avail, used uint64) []byte {
	b := make([]byte, 32)
	binary.LittleEndian.PutUint64(b[0:], total)
	binary.LittleEndian.PutUint64(b[8:], avail)
	binary.LittleEndian.PutUint64(b[16:], used)
	binary.LittleEndian.PutUint32(b[24:], 16384)
	binary.LittleEndian.PutUint32(b[28:], 1)
	return b[:31]
}

func TestDecodeXswUsed(t *testing.T) {
	for _, tc := range []struct {
		name   string
		b      []byte
		want   int64
		wantOK bool
	}{
		{"real shape, 2052.75 MiB used", xswUsage(3072<<20, 1019<<20, 2152464384), 2152464384, true},
		{"zero swap-used is a real reading", xswUsage(0, 0, 0), 0, true},
		{"short value is unknown", make([]byte, 20), 0, false},
		{"empty is unknown", nil, 0, false},
		{"implausible value is unknown", xswUsage(0, 0, 1<<63), 0, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := decodeXswUsed(tc.b)
			if ok != tc.wantOK || got != tc.want {
				t.Errorf("decodeXswUsed = %d,%v want %d,%v", got, ok, tc.want, tc.wantOK)
			}
		})
	}
}

// The bare-sysctl reader must agree with what `sysctl -n vm.swapusage` prints on this machine (the exec
// reader it replaced). The string only has two decimals of MiB, so agree within 0.01 MiB plus whatever
// swap moved between the two reads.
func TestSwapUsedBytes_matchesSysctlCommand(t *testing.T) {
	got, ok := SwapUsedBytes()
	if !ok {
		t.Fatal("SwapUsedBytes: unknown on darwin")
	}
	out, err := exec.Command("sysctl", "-n", "vm.swapusage").Output()
	if err != nil {
		t.Skipf("sysctl command: %v", err)
	}
	f := strings.Fields(string(out))
	var mb float64
	for i := range f {
		if f[i] == "used" && i+2 < len(f) {
			mb, err = strconv.ParseFloat(strings.TrimSuffix(f[i+2], "M"), 64)
			if err != nil {
				t.Fatalf("parse %q: %v", out, err)
			}
		}
	}
	want := int64(mb * (1 << 20))
	if d := got - want; d > 64<<20 || d < -(64<<20) {
		t.Errorf("SwapUsedBytes = %d, `sysctl -n` says %d (%q)", got, want, strings.TrimSpace(string(out)))
	}
}
