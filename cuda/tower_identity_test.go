//go:build cuda

package cuda

import (
	"flag"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/towerid"
	"github.com/townsendmerino/goinfer/internal/vision"
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

// TestTowerGolden_cuda is G-SB9-1, G-SB9-2, G-SB9-3 and G-SB9M-2 on CUDA: the record equals the committed golden recorded on this GPU
// and NVIDIA driver (another fails until it is re-anchored), each family's plant turns exactly its own cases red, each
// plant in SigLIP's block program turns exactly the families that walk it red, and each device case agrees with the CPU within the registered bound.
func TestTowerGolden_cuda(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	bus, err := d.PCIBusID()
	d.ReleaseObjects()
	if err != nil {
		t.Fatalf("the golden's anchor cannot name the device under test: %v", err)
	}
	// Scoped by PCI bus id, so on a host with several GPUs the anchor is the device under test's, whatever CUDA's ordinal order
	// and CUDA_VISIBLE_DEVICES make of it.
	smi, err := exec.Command("nvidia-smi", "-i", bus, "--query-gpu=name,driver_version", "--format=csv,noheader").Output()
	if err != nil {
		t.Fatalf("the golden's anchor (GPU and driver) cannot be read: nvidia-smi -i %s: %v", bus, err)
	}
	if n := strings.Count(strings.TrimSpace(string(smi)), "\n"); n != 0 {
		t.Fatalf("nvidia-smi -i %s named %d GPUs, want the one under test: %q", bus, n+1, smi)
	}
	anchor := "cuda " + strings.TrimSpace(string(smi))
	rec := towerid.Check(t, "..", "cuda", "")
	golden := towerid.CheckGolden(t, "..", rec, anchor, towerid.FailOtherAnchor, *towerUpdate)
	towerid.CheckPlants(t, "..", "cuda", golden)
	towerid.CheckProgramPlants(t, "..", "cuda", golden, "siglip", "qwen3", "glm-ocr")
	towerid.CompareBackends(t, "..", "cuda")
}

// TestGridTowerWalkBlock_refusesMisroutedOps: CUDA's tower walker refuses each misrouted op before it queues a kernel.
// The tower has no device, so an op that reached a dispatch would panic.
func TestGridTowerWalkBlock_refusesMisroutedOps(t *testing.T) {
	const hidden, inter = 8, 32
	a := &gridTower{hidden: hidden, inter: inter, heads: 2, hd: hidden / 2}
	hp := gridProj{out: hidden, in: hidden}
	b := &gridBlock{q: hp, k: hp, v: hp, proj: hp, fc1: gridProj{out: inter, in: hidden}, fc2: gridProj{out: hidden, in: inter},
		gate: gridProj{out: inter, in: hidden}, up: gridProj{out: inter, in: hidden}, down: gridProj{out: hidden, in: inter}}
	for _, c := range vision.MisroutedTowerOpsForTest() {
		err := func() (err error) {
			defer func() {
				if recover() != nil {
					t.Errorf("%s: queued instead of refused", c.Name)
				}
			}()
			return a.walkBlock(vision.TowerBlock{c.Op}, b, &towerRegs{}, 4, &towerWalkEnv{segs: []int{0, 4}})
		}()
		if err == nil {
			t.Errorf("%s: accepted", c.Name)
		}
	}
}
