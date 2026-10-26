//go:build gpu

package gpu

import (
	"testing"

	"github.com/townsendmerino/goinfer/internal/towerid"
)

// TestTowerIdentity_webgpu is AX2a-T on WebGPU (docs/tasks/task-subsystems-2026-10.md §7.8): every tower WebGPU declares
// attaches through its production registry and gives the same bits twice.
func TestTowerIdentity_webgpu(t *testing.T) {
	c, err := New()
	if err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	c.Close()
	towerid.Check(t, "..", "webgpu")
}
