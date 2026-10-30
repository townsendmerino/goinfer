package towerid

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
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

// TestTowerGolden_cpu is G-SB9-1, G-SB9-2 and G-SB9M-2 for the CPU: the record equals the committed golden under its kernel
// configuration, each family's plant turns exactly its own cases red, and each plant in SigLIP's block program turns
// exactly the families that walk it red. A CPU configuration without a golden skips.
func TestTowerGolden_cpu(t *testing.T) {
	rec := Check(t, "../..", "cpu", "")
	golden := CheckGolden(t, "../..", rec, CPUAnchor(), SkipOtherAnchor, *towerUpdate)
	CheckPlants(t, "../..", "cpu", golden)
	CheckProgramPlants(t, "../..", "cpu", golden, "siglip", "qwen3", "glm-ocr", "gemma4", "qwen2.5", "pixtral", "whisper", "gemma4-audio")
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

// outcomeTB records how CheckGolden ended: Skipf and Fatalf stop it, as testing's do, by a panic outcome recovers.
type outcomeTB struct {
	testing.TB
	how, msg string
}

type stopped struct{}

func (o *outcomeTB) Helper()                   {}
func (o *outcomeTB) Logf(string, ...any)       {}
func (o *outcomeTB) Errorf(f string, a ...any) { o.how, o.msg = "error", fmt.Sprintf(f, a...) }
func (o *outcomeTB) Fatal(a ...any)            { o.how, o.msg = "fail", fmt.Sprint(a...); panic(stopped{}) }
func (o *outcomeTB) Fatalf(f string, a ...any) {
	o.how, o.msg = "fail", fmt.Sprintf(f, a...)
	panic(stopped{})
}
func (o *outcomeTB) Skipf(f string, a ...any) {
	o.how, o.msg = "skip", fmt.Sprintf(f, a...)
	panic(stopped{})
}
func (o *outcomeTB) outcome(run func(*outcomeTB)) (res *outcomeTB) {
	res = o
	defer func() {
		if r := recover(); r != nil {
			if _, ok := r.(stopped); !ok {
				panic(r)
			}
		}
	}()
	run(o)
	return
}

// TestCheckGolden_otherAnchor is G-SB9-5 (docs/tasks/task-subsystems-2026-10.md, "SB9 step S on the Mac"): against a golden
// recorded on another adapter, FailOtherAnchorSkipOnCI skips on CI (GITHUB_ACTIONS=true) and fails anywhere else, naming both
// anchors either way; SkipOtherAnchor skips and FailOtherAnchor fails whatever the environment.
func TestCheckGolden_otherAnchor(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "internal", "towerid", "testdata"), 0o755); err != nil {
		t.Fatal(err)
	}
	rec := Record{Schema: 1, Backend: "webgpu", GOOS: runtime.GOOS, GOARCH: runtime.GOARCH,
		Cases: []Case{{Family: "siglip", Input: "normal-9", Device: true, Values: 4, SHA256: "00"}}}
	if o := new(outcomeTB).outcome(func(o *outcomeTB) { CheckGolden(o, root, rec, "webgpu adapter A", FailOtherAnchorSkipOnCI, true) }); o.how != "" {
		t.Fatalf("writing the golden: %s %s", o.how, o.msg)
	}
	for _, c := range []struct {
		ci   string
		mode OnOtherAnchor
		want string
	}{
		{"true", FailOtherAnchorSkipOnCI, "skip"},
		{"", FailOtherAnchorSkipOnCI, "fail"},
		{"false", FailOtherAnchorSkipOnCI, "fail"},
		{"true", FailOtherAnchor, "fail"},
		{"", FailOtherAnchor, "fail"},
		{"true", SkipOtherAnchor, "skip"},
		{"", SkipOtherAnchor, "skip"},
	} {
		t.Setenv("GITHUB_ACTIONS", c.ci)
		o := new(outcomeTB).outcome(func(o *outcomeTB) { CheckGolden(o, root, rec, "webgpu adapter B", c.mode, false) })
		if o.how != c.want || !strings.Contains(o.msg, "adapter A") || !strings.Contains(o.msg, "adapter B") {
			t.Errorf("GITHUB_ACTIONS=%q, mode %d: %s %q; want a %s naming both anchors", c.ci, c.mode, o.how, o.msg, c.want)
		}
	}
	t.Setenv("GITHUB_ACTIONS", "true")
	if o := new(outcomeTB).outcome(func(o *outcomeTB) { CheckGolden(o, root, rec, "webgpu adapter A", FailOtherAnchorSkipOnCI, false) }); o.how != "" {
		t.Errorf("the golden's own adapter on CI: %s %q; want it graded and equal", o.how, o.msg)
	}
}

// TestProgramPlantVerdict_vacuous holds G-SB9M-2's check to a golden it can fail: a record with no case of a family that
// walks SigLIP's program proves nothing about the program, so it is an error, not a pass.
func TestProgramPlantVerdict_vacuous(t *testing.T) {
	other := Case{Family: "qwen3", Input: "a", Values: 4, SHA256: "x"}
	golden := Record{Cases: []Case{other}}
	if errs, covered := programPlantVerdict("cpu", programFamilies["siglip"], golden, golden); len(errs) == 0 {
		t.Errorf("a golden with no SigLIP-program family passed (covered %v)", covered)
	}
	sig := Case{Family: "siglip", Input: "b", Device: true, Values: 4, SHA256: "y"}
	planted := sig
	planted.SHA256 = "z"
	errs, covered := programPlantVerdict("metal", programFamilies["siglip"], Record{Cases: []Case{other, sig}}, Record{Cases: []Case{other, planted}})
	if len(errs) != 0 || !slices.Equal(covered, []string{"siglip"}) {
		t.Errorf("siglip red on the device, qwen3 equal: errs %v, covered %v; want none and [siglip]", errs, covered)
	}
}

// TestProgramPlantVerdict_hostOnlyFamily: off the CPU, a program family whose cases all ran on the host proves nothing about
// the device walker, so each covered family needs a red case that ran on the device.
func TestProgramPlantVerdict_hostOnlyFamily(t *testing.T) {
	sig := Case{Family: "siglip", Input: "a", Device: true, Values: 4, SHA256: "x"}
	nax := Case{Family: "siglip2-naflex", Input: "b", Values: 4, SHA256: "y"}
	ps, pn := sig, nax
	ps.SHA256, pn.SHA256 = "x2", "y2"
	if errs, covered := programPlantVerdict("metal", programFamilies["siglip"], Record{Cases: []Case{sig, nax}}, Record{Cases: []Case{ps, pn}}); len(errs) == 0 {
		t.Errorf("siglip2-naflex, red only on the host, counted as covered on metal (%v)", covered)
	}
}
