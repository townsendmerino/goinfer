package probe

import (
	"testing"
	"unsafe"
)

// The Windows memory probe answers from the real OS: some RAM, an available amount no larger than it, and a
// MEMORYSTATUSEX laid out at the 64 bytes kernel32 expects (a wrong length makes the call fail, which would read as
// "unknown" everywhere and silently turn the fit guard off).
func TestPhysical_windowsProbeAnswers(t *testing.T) {
	if got := unsafe.Sizeof(memoryStatusEx{}); got != 64 {
		t.Fatalf("memoryStatusEx is %d bytes, MEMORYSTATUSEX is 64", got)
	}
	total, avail := Physical(), Available()
	if total < 1<<30 {
		t.Fatalf("Physical = %d: the probe did not answer (a CI runner has GBs)", total)
	}
	if avail <= 0 || avail > total {
		t.Fatalf("Available = %d against a total of %d", avail, total)
	}
	t.Logf("total %.1f GB, available %.1f GB", float64(total)/(1<<30), float64(avail)/(1<<30))
}
