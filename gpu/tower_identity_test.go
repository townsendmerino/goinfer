//go:build gpu

package gpu

import (
	"flag"
	"os"
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
	towerid.Check(t, "..", "webgpu", os.Getenv("GOINFER_TOWER_ID_OUT"))
}

var towerUpdate = flag.Bool("towerid.update", false, "rewrite this platform's WebGPU tower golden (review the diff)")

// TestTowerGolden_webgpu is G-SB9-1, G-SB9-2, G-SB9-3 and G-SB9M-2 on WebGPU: the record equals the committed golden recorded on this
// adapter (another adapter fails until it is re-anchored, and skips on CI, whose runner's adapter has no golden), each
// family's plant turns exactly its own cases red, each plant in SigLIP's block program turns exactly the families that
// walk it red, and each device case agrees with the CPU within the registered bound.
func TestTowerGolden_webgpu(t *testing.T) {
	c, err := New()
	if err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	anchor := "webgpu " + c.AdapterAnchor()
	c.Close()
	rec := towerid.Check(t, "..", "webgpu", "")
	golden := towerid.CheckGolden(t, "..", rec, anchor, towerid.FailOtherAnchorSkipOnCI, *towerUpdate)
	towerid.CheckPlants(t, "..", "webgpu", golden)
	towerid.CheckProgramPlants(t, "..", "webgpu", golden, "siglip", "qwen3", "glm-ocr")
	towerid.CompareBackends(t, "..", "webgpu")
}
