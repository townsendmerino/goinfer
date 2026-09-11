package chatapp

import (
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestFitFlag_acceptsOnOff is M-14 (audit-2026-09-10): chatapp had no --fit flag at all, and
// serve's own --fit rejected the "on"/"off" spelling its help and task-fit-to-hardware.md
// promise (a plain flag.Bool only understands strconv.ParseBool's spellings). This drives
// fitFlag.Set directly, the same way internal/serveapp's own test does.
func TestFitFlag_acceptsOnOff(t *testing.T) {
	cases := []struct {
		in   string
		want bool
	}{
		{"on", true}, {"off", false},
		{"true", true}, {"false", false},
		{"1", true}, {"0", false},
	}
	for _, c := range cases {
		var f fitFlag = true // start opposite of most expected results, so a no-op Set is caught
		if err := f.Set(c.in); err != nil {
			t.Errorf("Set(%q): unexpected error: %v", c.in, err)
			continue
		}
		if bool(f) != c.want {
			t.Errorf("Set(%q) = %v, want %v", c.in, bool(f), c.want)
		}
	}
}

func TestFitFlag_rejectsGarbage(t *testing.T) {
	var f fitFlag
	if err := f.Set("maybe"); err == nil {
		t.Error("Set(\"maybe\"): expected an error, got none")
	}
}

// TestFitFlag_isBoolFlag: a bare `--fit` (no `=value`) must still work.
func TestFitFlag_isBoolFlag(t *testing.T) {
	var f fitFlag
	if !f.IsBoolFlag() {
		t.Fatal("fitFlag.IsBoolFlag() = false, want true — a bare --fit would require a value")
	}
}

// TestFitFlag_realBinaryAcceptsOff proves the fix through the FULL registered flag.CommandLine,
// not just fitFlag.Set in isolation — --version exits before touching a model, so this is a
// cheap way to prove "--fit=off" (the spelling task-fit-to-hardware.md promises) parses cleanly
// end to end, where a plain flag.BoolVar would exit 2 with "invalid boolean value".
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
