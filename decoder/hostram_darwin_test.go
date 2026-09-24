//go:build darwin

package decoder

import (
	"os/exec"
	"strconv"
	"strings"
	"testing"
)

// R13-follow-on (docs/measurements/cold-user-2026-09-07-macbook-arm64.md's live re-run): real
// vm_stat output shape, 16 KB pages (Apple Silicon — the M1 Pro that found this bug, not the
// traditional 4 KB), so a parser tested only against a 4 KB assumption would pass here and still
// be wrong on the machine that matters.
const sampleVMStat = `Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                5945.
Pages active:                             123456.
Pages inactive:                            98765.
Pages speculative:                          1234.
Pages throttled:                               0.
Pages wired down:                         271736.
Pages purgeable:                           61118.
"Translation faults":                  123456789.
Pages copy-on-write:                     1234567.
Pages zero filled:                      12345678.
Pages reactivated:                          1234.
Pages purged:                              12345.
File-backed pages:                        123456.
Anonymous pages:                          234567.
Pages stored in compressor:               345678.
Pages occupied by compressor:             123456.
Decompressions:                            12345.
Compressions:                              23456.
Pageins:                                  123456.
Pageouts:                                    1234.
Swapins:                                       0.
Swapouts:                                 621588.
`

func TestParseVMStatAvailable_realShape(t *testing.T) {
	got := parseVMStatAvailable(sampleVMStat)
	want := int64(5945+98765+1234+61118) * 16384
	if got != want {
		t.Errorf("parseVMStatAvailable = %d, want %d", got, want)
	}
	if got <= 0 {
		t.Errorf("available = %d, want a positive figure for this fixture", got)
	}
}

func TestParseVMStatAvailable_missingHeaderIsUnknown(t *testing.T) {
	if got := parseVMStatAvailable("garbage, no page size header\n"); got != 0 {
		t.Errorf("missing page-size header: got %d, want 0 (unknown)", got)
	}
}

func TestParseVMStatAvailable_missingFieldIsUnknown(t *testing.T) {
	// Mutation-relevant: if vm_stat's output ever drops one of the fields this depends on, the
	// parse must say "unknown", never silently sum the fields it did find — a partial sum would
	// UNDER-report usage in exactly the wrong direction (looks like more room than there is).
	const missingPurgeable = `Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                5945.
Pages inactive:                            98765.
Pages speculative:                          1234.
`
	if got := parseVMStatAvailable(missingPurgeable); got != 0 {
		t.Errorf("missing a required field: got %d, want 0 (unknown, not a partial sum)", got)
	}
}

func TestParseVMStatAvailable_zeroPageSizeIsUnknown(t *testing.T) {
	if got := parseVMStatAvailable("Mach Virtual Memory Statistics: (page size of 0 bytes)\nPages free: 100.\n"); got != 0 {
		t.Errorf("zero page size: got %d, want 0", got)
	}
}

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
func TestHostRAMBytes_matchesSysctlCommand(t *testing.T) {
	out, err := exec.Command("sysctl", "-n", "hw.memsize").Output()
	if err != nil {
		t.Skipf("sysctl command: %v", err)
	}
	want, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
	if err != nil {
		t.Fatalf("parse %q: %v", out, err)
	}
	if got := HostRAMBytes(); got != want {
		t.Errorf("HostRAMBytes = %d, `sysctl -n hw.memsize` = %d", got, want)
	}
}
