package serveapp

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

// TestFitFlag_realBinaryAcceptsOff proves the fix through the FULL registered flag.CommandLine,
// not just cliutil.OnOff.Set in isolation — --version exits before touching a model, so this is a
// cheap way to prove "--fit=off" (the spelling tasks/task-fit-to-hardware.md and this flag's own help
// promise) parses cleanly end to end, where the original flag.BoolVar exited 2 with "invalid
// boolean value \"off\"".
func TestFitFlag_realBinaryAcceptsOff(t *testing.T) {
	if testing.Short() {
		t.Skip("builds a binary")
	}
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("no go toolchain")
	}
	bin := filepath.Join(t.TempDir(), "goinfer-serve")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	build := exec.Command("go", "build", "-o", bin, "github.com/townsendmerino/goinfer/cmd/serve")
	build.Dir = ".."
	if out, err := build.CombinedOutput(); err != nil {
		t.Skipf("cannot build cmd/serve here: %v\n%s", err, out)
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
