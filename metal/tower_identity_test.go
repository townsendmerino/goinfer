//go:build darwin

package metal

import (
	"flag"
	"fmt"
	"os"
	"syscall"
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
	towerid.Check(t, "..", "metal", os.Getenv("GOINFER_TOWER_ID_OUT"))
}

var towerUpdate = flag.Bool("towerid.update", false, "rewrite this platform's Metal tower golden (review the diff)")

// metalAnchor names what a Metal record's bits depend on: the device, and the macOS version and build, since Metal's shader
// compiler and driver ship with the OS.
func metalAnchor(device string) (string, error) {
	ver, err := syscall.Sysctl("kern.osproductversion")
	if err != nil {
		return "", err
	}
	build, err := syscall.Sysctl("kern.osversion")
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("metal %s, macOS %s (%s)", device, ver, build), nil
}

// TestTowerGolden_metal is G-SB9-1M, G-SB9-2M and G-SB9-3M (docs/tasks/task-subsystems-2026-10.md, "SB9 step S on the
// Mac"): the record equals the committed golden recorded on this device and macOS build (another fails until it is
// re-anchored), each family's plant turns exactly its own cases red, and each device case agrees with the CPU within the
// registered bound.
func TestTowerGolden_metal(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no Metal device: %v", err)
	}
	name := d.Name()
	d.ReleaseObjects()
	anchor, err := metalAnchor(name)
	if err != nil {
		t.Fatalf("the golden's anchor (macOS version and build) cannot be read: %v", err)
	}
	rec := towerid.Check(t, "..", "metal", "")
	golden := towerid.CheckGolden(t, "..", rec, anchor, towerid.FailOtherAnchor, *towerUpdate)
	towerid.CheckPlants(t, "..", "metal", golden)
	towerid.CompareBackends(t, "..", "metal")
}
