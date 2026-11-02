package decoder

import "testing"

// Q4KTinyGGUFForTest is q4kTinyGGUF for the external test package, which reaches Load through internal/modelload.
var Q4KTinyGGUFForTest = q4kTinyGGUF

// HostRAMAvailableForTest makes the fit guard's memory probe report v until the test ends.
func HostRAMAvailableForTest(t *testing.T, v int64) {
	t.Helper()
	prev := hostRAMAvailable
	hostRAMAvailable = func() int64 { return v }
	resetAvailProbeCache()
	t.Cleanup(func() {
		hostRAMAvailable = prev
		resetAvailProbeCache()
	})
}
