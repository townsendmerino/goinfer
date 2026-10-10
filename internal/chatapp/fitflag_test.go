package chatapp

import (
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// The --fit value parser itself (on/off spellings, rejection, IsBoolFlag) is cliutil.OnOff,
// shared with the other binary and unit-tested in internal/cliutil. What stays here is the part
// that is per-binary: that THIS binary registers --fit with it.

// TestFitFlag_realBinaryAcceptsOff pins that --fit=off (the spelling docs/tasks/task-fit-to-hardware.md promises) parses
// through the FULL registered flag.CommandLine, not just cliutil.OnOff.Set in isolation: a plain flag.BoolVar would exit
// 2 with "invalid boolean value". --version exits before touching a model, so this is cheap.
func TestFitFlag_realBinaryAcceptsOff(t *testing.T) {
	if testing.Short() {
		t.Skip("builds a binary")
	}
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("no go toolchain")
	}
	bin := filepath.Join(t.TempDir(), "goinfer-chat")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	build := exec.Command("go", "build", "-o", bin, "github.com/townsendmerino/goinfer/demo/chat")
	build.Dir = ".."
	if out, err := build.CombinedOutput(); err != nil {
		t.Skipf("cannot build demo/chat here: %v\n%s", err, out)
	}
	for _, spelling := range []string{"off", "on"} {
		out, err := exec.Command(bin, "--fit="+spelling, "--version").CombinedOutput()
		if err != nil {
			t.Fatalf("--fit=%s --version: %v\n%s", spelling, err, out)
		}
		s := string(out)
		if strings.Contains(s, "invalid boolean value") || strings.Contains(s, "invalid value") {
			t.Errorf("--fit=%s was rejected by the flag parser:\n%s", spelling, s)
		}
		if !strings.Contains(s, "backends:") {
			t.Errorf("--fit=%s --version output missing a backends: line:\n%s", spelling, s)
		}
	}
}

// TestExactPrefillFlag_realBinaryParses pins that --exact-prefill (documented in docs/completed/task-prefill-gap.md as a
// flag of this REPL) is REGISTERED on the real flag.CommandLine, not just present in source: a typo'd flag.Bool name
// would exit 2 with "flag provided but not defined". Same discipline as TestFitFlag_realBinaryAcceptsOff; --version
// exits before touching a model. Origin (M-26):
// docs/code-notes/internal-chatapp.md#TestExactPrefillFlag_realBinaryParses.
func TestExactPrefillFlag_realBinaryParses(t *testing.T) {
	if testing.Short() {
		t.Skip("builds a binary")
	}
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("no go toolchain")
	}
	bin := filepath.Join(t.TempDir(), "goinfer-chat")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	build := exec.Command("go", "build", "-o", bin, "github.com/townsendmerino/goinfer/demo/chat")
	build.Dir = ".."
	if out, err := build.CombinedOutput(); err != nil {
		t.Skipf("cannot build demo/chat here: %v\n%s", err, out)
	}
	out, err := exec.Command(bin, "--exact-prefill", "--version").CombinedOutput()
	if err != nil {
		t.Fatalf("--exact-prefill --version: %v\n%s", err, out)
	}
	s := string(out)
	if strings.Contains(s, "flag provided but not defined") {
		t.Errorf("--exact-prefill is not a registered flag:\n%s", s)
	}
	if !strings.Contains(s, "backends:") {
		t.Errorf("--exact-prefill --version output missing a backends: line:\n%s", s)
	}
}
