//go:build darwin

package decoder

import "testing"

// Real `sysctl -n vm.swapusage` output shape, captured 2026-09-22 on the M1 Pro.
const sampleSwapUsage = `total = 3072.00M  used = 2052.75M  free = 1019.25M  (encrypted)
`

func TestParseSwapUsageDarwin_realShape(t *testing.T) {
	got, ok := parseSwapUsageDarwin(sampleSwapUsage)
	if !ok {
		t.Fatal("expected ok=true on well-formed output")
	}
	want := int64(2052.75 * 1024 * 1024)
	if got != want {
		t.Errorf("parseSwapUsageDarwin = %d, want %d", got, want)
	}
}

func TestParseSwapUsageDarwin_zeroSwap(t *testing.T) {
	got, ok := parseSwapUsageDarwin("total = 0.00M  used = 0.00M  free = 0.00M\n")
	if !ok {
		t.Fatal("zero swap-used is a real, valid reading — must not report unknown")
	}
	if got != 0 {
		t.Errorf("parseSwapUsageDarwin = %d, want 0", got)
	}
}

func TestParseSwapUsageDarwin_malformedIsUnknown(t *testing.T) {
	for _, in := range []string{
		"",
		"garbage output\n",
		"total = 3072.00M  free = 1019.25M\n", // "used" field absent
	} {
		if _, ok := parseSwapUsageDarwin(in); ok {
			t.Errorf("parseSwapUsageDarwin(%q): got ok=true, want false (unparseable input)", in)
		}
	}
}
