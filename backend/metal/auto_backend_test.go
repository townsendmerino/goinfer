//go:build darwin

package metal

import (
	"runtime"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestAutoBackend_appleSilicon (R17): with this module linked, auto picks metal on Apple silicon, and on an Intel Mac
// (the darwin/amd64 release asset links this module too) falls to the CPU, since Metal has only been run on Apple
// silicon.
func TestAutoBackend_appleSilicon(t *testing.T) {
	c := decoder.AutoBackend()
	if runtime.GOARCH == "arm64" {
		if c.Backend != "metal" || c.Skipped {
			t.Errorf("Apple silicon, but AutoBackend() = %+v", c)
		}
		return
	}
	if c.Backend != "cpu" || !c.Skipped || !strings.Contains(c.Reason, "Apple silicon") {
		t.Errorf("GOARCH %s, but AutoBackend() = %+v", runtime.GOARCH, c)
	}
}
