package main

import (
	"bytes"
	"strings"
	"testing"
)

// TestGradePTX_plants is SB12-C4 for step 11 (docs/tasks/task-subsystems-2026-10.md, SB12 step 11 (CUDA)): from a green
// input the group passes; one file that differs fails naming it; none verified fails; some unverified add a counted SKIP
// naming them beside the verdict; a script that cannot run skips with its reason; every verdict that ran counts as ran.
func TestGradePTX_plants(t *testing.T) {
	green := ptxInput{Toolchains: []string{"12.6.85", "12.9.86"}, Files: []ptxFile{
		{Base: "a", Want: "12.6.85", Outcome: "ok"}, {Base: "b", Want: "12.9.86", Outcome: "ok"}}}
	v := gradePTX(green)
	if len(v) != 1 || v[0].State != groupPass || v[0].Line != "2/2 PTX regenerate byte-identically at their recorded NVRTC" ||
		v[0].Pre[0] != "  toolchains available: 12.6.85 12.9.86" {
		t.Fatalf("green: %+v", v)
	}

	differs := green
	differs.Files = []ptxFile{green.Files[0], {Base: "b", Want: "12.9.86", Outcome: "differs", KeepErr: "disk full"}}
	v = gradePTX(differs)
	pre := strings.Join(v[0].Pre, "\n")
	if v[0].State != groupFail || !v[0].Ran || !strings.Contains(pre, "DIFFERS: b.ptx (rebuilt at its recorded V12.9.86)") ||
		!strings.Contains(pre, "could not keep the rebuilt file: disk full") {
		t.Errorf("one file differs: state %d ran %v pre:\n%s", v[0].State, v[0].Ran, pre)
	}

	none := ptxInput{Files: []ptxFile{{Base: "a", Want: "12.6.85", Outcome: "unverified", Why: "needs V12.6.85"}}}
	v = gradePTX(none)
	if len(v) != 2 || v[0].State != groupFail || !v[0].Ran || !strings.Contains(v[0].Line, "zero of 1 artifacts verified") ||
		v[0].Pre[0] != "  toolchains available: none" {
		t.Errorf("none verified: %+v", v)
	}

	some := green
	some.Files = append(append([]ptxFile{}, green.Files...), ptxFile{Base: "c", Outcome: "unverified", Why: "no recorded version"},
		ptxFile{Base: "d", Want: "12.8.1", Outcome: "unverified", Why: "build failed at V12.8.1"})
	v = gradePTX(some)
	if len(v) != 2 || v[0].State != groupPass || v[0].Line != "2/4 PTX regenerate byte-identically at their recorded NVRTC" ||
		v[1].State != groupSkip || v[1].Line != "2/4 PTX NOT verified on this box (toolchain absent): c(no recorded version) d(build failed at V12.8.1)" {
		t.Errorf("some unverified: %+v", v)
	}

	v = gradePTX(ptxInput{Decline: "is missing"})
	if len(v) != 1 || v[0].State != groupSkip || v[0].Line != "PTX reproducibility (cuda/build_ptx.sh is missing)" || v[0].Ran {
		t.Errorf("a script that cannot run: %+v", v)
	}
}

// TestReplayGPU_ptxEqualsTheRun: a recorded ptx input replays to the section the run printed, through the same ptxGroup, and
// a record without one leaves ptx among the groups not regraded.
func TestReplayGPU_ptxEqualsTheRun(t *testing.T) {
	in := ptxInput{Toolchains: []string{"12.6.85"}, Files: []ptxFile{
		{Base: "a", Want: "12.6.85", Outcome: "differs"}, {Base: "b", Outcome: "unverified", Why: "no recorded version"}}}
	var live, again bytes.Buffer
	run := &gpuGate{w: &live, emitted: map[string]bool{}, runIn: gpuRunInputs{PTX: &in}}
	rep := &gpuGate{w: &again, emitted: map[string]bool{}, runIn: gpuRunInputs{PTX: &in}}
	if err := run.ptxGroup(fakeLiveInputs{g: run}); err != nil {
		t.Fatal(err)
	}
	if err := rep.ptxGroup(&recordedInputs{g: rep}); err != nil {
		t.Fatal(err)
	}
	if live.String() != again.String() || !strings.Contains(live.String(), "DIFFERS: a.ptx") {
		t.Errorf("the replayed ptx section differs from the run's:\n%s", lineDiff(live.String(), again.String()))
	}
	if run.fail != 1 || run.skipped != 1 || run.ran != 1 {
		t.Errorf("tally: fail %d skip %d ran %d, want 1 1 1", run.fail, run.skipped, run.ran)
	}
	none := &gpuGate{w: &bytes.Buffer{}, emitted: map[string]bool{}}
	if err := none.ptxGroup(&recordedInputs{g: none}); err != nil || none.emitted["ptx"] {
		t.Errorf("a record with no ptx input regraded ptx (err %v)", err)
	}
}
