//go:build cuda

package cuda

import (
	"flag"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/towerid"
)

// TestTowerIdentity_cuda is AX2a-T on CUDA (docs/tasks/task-subsystems-2026-10.md §7.8): every tower CUDA declares, and
// the Whisper encoders, attach through their production registries and give the same bits twice.
func TestTowerIdentity_cuda(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	d.ReleaseObjects()
	towerid.Check(t, "..", "cuda", os.Getenv("GOINFER_TOWER_ID_OUT"))
}

var towerUpdate = flag.Bool("towerid.update", false, "rewrite this platform's CUDA tower golden (review the diff)")

// TestTowerGolden_cuda is G-SB9-1, G-SB9-2 and G-SB9-3 on CUDA: the record equals the committed golden recorded on this GPU
// and NVIDIA driver (another fails until it is re-anchored), each family's plant turns exactly its own cases red, and each
// device case agrees with the CPU within the registered bound.
func TestTowerGolden_cuda(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	d.ReleaseObjects()
	smi, err := exec.Command("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader").Output()
	if err != nil {
		t.Fatalf("the golden's anchor (GPU and driver) cannot be read: nvidia-smi: %v", err)
	}
	anchor := "cuda " + strings.TrimSpace(string(smi))
	rec := towerid.Check(t, "..", "cuda", "")
	golden := towerid.CheckGolden(t, "..", rec, anchor, towerid.FailOtherAnchor, *towerUpdate)
	towerid.CheckPlants(t, "..", "cuda", golden)
	towerid.CompareBackends(t, "..", "cuda")
}
