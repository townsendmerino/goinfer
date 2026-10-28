package towerid

import (
	"flag"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/internal/vision"
)

// TestTowerIdentity_cpu is AX2a-T for the CPU: every tower family's record, twice, agreeing.
func TestTowerIdentity_cpu(t *testing.T) {
	r := Check(t, "../..", "cpu", os.Getenv("GOINFER_TOWER_ID_OUT"))
	families := map[string]bool{}
	for _, c := range r.Cases {
		families[c.Family] = true
	}
	if len(families) != 12 {
		t.Errorf("%d families ran on the CPU, want 12 (§7.8's list): %v", len(families), families)
	}
}

var towerUpdate = flag.Bool("towerid.update", false, "rewrite this platform's CPU tower golden (review the diff)")

// TestTowerGolden_cpu is G-SB9-1 and G-SB9-2 for the CPU: the record equals the committed golden under its kernel
// configuration, and each family's plant turns exactly its own cases red. A CPU configuration without a golden skips.
func TestTowerGolden_cpu(t *testing.T) {
	rec := Check(t, "../..", "cpu", "")
	golden := CheckGolden(t, "../..", rec, CPUAnchor(), SkipOtherAnchor, *towerUpdate)
	CheckPlants(t, "../..", "cpu", golden)
}

// TestRun_unregisteredDeviceIsAnError holds the rule a registry split depends on: a declared family whose device
// registration is missing (none is linked into this package's test binary) is an error, never a CPU result.
func TestRun_unregisteredDeviceIsAnError(t *testing.T) {
	for _, backend := range []string{"metal", "cuda", "webgpu"} {
		if _, err := Run("../..", backend); err == nil {
			t.Errorf("%s: Run succeeded with no %s registration in the binary", backend, backend)
		} else {
			t.Logf("%s: %v", backend, err)
		}
	}
}

// TestRun_whisperDeclaredButUnregisteredIsAnError: CUDA serves the Whisper encoders, and with no CUDA registration linked
// into this binary the Whisper and Voxtral families must error, not be skipped (a skip would let the CUDA record pass
// without them).
func TestRun_whisperDeclaredButUnregisteredIsAnError(t *testing.T) {
	for _, run := range []func(*runner) error{(*runner).whisper, (*runner).voxtral} {
		r := &runner{root: "../..", backend: "cuda"}
		if err := run(r); err == nil {
			t.Errorf("a Whisper-family tower ran (or was skipped) on cuda with no cuda Whisper registration: cases %v", r.cases)
		} else {
			t.Logf("%v", err)
		}
	}
}

// TestRun_siglipResidentFactoryReturningNothingIsAnError: a resident factory that returns no encoder and no error leaves
// the tower on the CPU; siglip must say so rather than record a device case.
func TestRun_siglipResidentFactoryReturningNothingIsAnError(t *testing.T) {
	vision.RegisterResident(func(*vision.Encoder) (vision.ResidentEncoder, error) { return nil, nil })
	defer vision.RegisterResident(nil)
	r := &runner{root: "../..", backend: "metal"}
	if err := r.siglip(); err == nil {
		t.Errorf("siglip recorded %v with a resident factory that attached nothing", r.cases)
	} else {
		t.Logf("%v", err)
	}
}
