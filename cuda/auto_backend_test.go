//go:build cuda

package cuda

import (
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestAutoBackend_followsTheDevice (R17): with this module linked, auto picks cuda exactly when a device opens, and
// otherwise falls to the CPU naming cuda as passed over. CI's GPU-less runner takes the second branch on every push,
// which is what a CPU-only box running the release's cuda binary meets at every start; nobara takes the first.
func TestAutoBackend_followsTheDevice(t *testing.T) {
	c := decoder.AutoBackend()
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		if c.Backend != "cpu" || !c.Skipped || !strings.Contains(c.Reason, "no CUDA device answered") {
			t.Errorf("no device opens (%v), but AutoBackend() = %+v", err, c)
		}
		return
	}
	dev.ReleaseObjects()
	if c.Backend != "cuda" || c.Skipped {
		t.Errorf("a device opens, but AutoBackend() = %+v", c)
	}
}
