//go:build linux

package decoder

import "testing"

// Representative /proc/meminfo shape (documented kernel field order and units — SwapTotal/
// SwapFree are the ones this file's parser reads).
const sampleMeminfoSwap = `MemTotal:       16311288 kB
MemFree:         1234567 kB
MemAvailable:    5766224 kB
SwapTotal:       8388604 kB
SwapFree:        3145728 kB
`

func TestParseSwapUsedLinuxMeminfo_realShape(t *testing.T) {
	got, ok := parseSwapUsedLinuxMeminfo(sampleMeminfoSwap)
	if !ok {
		t.Fatal("expected ok=true on well-formed meminfo")
	}
	want := int64(8388604-3145728) * 1024
	if got != want {
		t.Errorf("parseSwapUsedLinuxMeminfo = %d, want %d", got, want)
	}
}

func TestParseSwapUsedLinuxMeminfo_zeroSwapUsed(t *testing.T) {
	got, ok := parseSwapUsedLinuxMeminfo("SwapTotal:       8388604 kB\nSwapFree:        8388604 kB\n")
	if !ok {
		t.Fatal("zero swap-used (total == free) is a real, valid reading — must not report unknown")
	}
	if got != 0 {
		t.Errorf("parseSwapUsedLinuxMeminfo = %d, want 0", got)
	}
}

func TestParseSwapUsedLinuxMeminfo_noSwapConfigured(t *testing.T) {
	// A machine with swap disabled reports SwapTotal: 0 kB, SwapFree: 0 kB — a real, valid
	// zero, not an absent field.
	got, ok := parseSwapUsedLinuxMeminfo("SwapTotal:             0 kB\nSwapFree:              0 kB\n")
	if !ok || got != 0 {
		t.Errorf("no-swap-configured machine: got (%d, %v), want (0, true)", got, ok)
	}
}

func TestParseSwapUsedLinuxMeminfo_missingFieldsAreUnknown(t *testing.T) {
	for _, in := range []string{
		"",
		"MemTotal:       16311288 kB\n", // no Swap* fields at all
		"SwapTotal:       8388604 kB\n", // SwapFree missing
		"SwapFree:        3145728 kB\n", // SwapTotal missing
	} {
		if _, ok := parseSwapUsedLinuxMeminfo(in); ok {
			t.Errorf("parseSwapUsedLinuxMeminfo(%q): got ok=true, want false", in)
		}
	}
}
