//go:build darwin

package metal

import (
	"testing"

	"github.com/townsendmerino/goinfer/internal/towerid"
)

// TestTowerIdentity_metal is AX2a-T on Metal (docs/tasks/task-subsystems-2026-10.md §7.8): every tower Metal declares
// attaches through its production registry and gives the same bits twice.
func TestTowerIdentity_metal(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no Metal device: %v", err)
	}
	d.ReleaseObjects()
	towerid.Check(t, "..", "metal")
}
