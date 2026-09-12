package serveapp

import (
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestFitFlag_acceptsOnOff is M-14 (audit-2026-09-10): --fit=off is the spelling this flag's own
// help text and task-fit-to-hardware.md promise ("`--fit=off` restores today's behaviour"), but a
// plain flag.Bool only understands strconv.ParseBool's spellings and rejected it with exit 2. This
// drives fitFlag.Set directly rather than the real binary, since that is exactly where the parser
// lived.
func TestFitFlag_acceptsOnOff(t *testing.T) {
	cases := []struct {
		in   string
		want bool
	}{
		{"on", true}, {"On", true}, {"ON", true},
		{"off", false}, {"Off", false}, {"OFF", false},
		{"true", true}, {"false", false},
		{"1", true}, {"0", false},
		{"t", true}, {"f", false},
		{" off ", false}, // flag values can arrive with incidental whitespace from shell quoting
	}
	for _, c := range cases {
		var f fitFlag = true // start from the opposite of most expected results, so a no-op Set would be caught
		if err := f.Set(c.in); err != nil {
			t.Errorf("Set(%q): unexpected error: %v", c.in, err)
			continue
		}
		if bool(f) != c.want {
			t.Errorf("Set(%q) = %v, want %v", c.in, bool(f), c.want)
		}
	}
}

// TestFitFlag_rejectsGarbage: an unrecognised value must error, not silently pick a default —
// the same discipline strconv.ParseBool itself follows.
func TestFitFlag_rejectsGarbage(t *testing.T) {
	for _, bad := range []string{"yes", "no", "2", "maybe", ""} {
		var f fitFlag
		if err := f.Set(bad); err == nil {
			t.Errorf("Set(%q): expected an error, got none (f=%v)", bad, f)
		}
	}
}

// TestFitFlag_isBoolFlag: a bare `--fit` (no `=value`) must still work, matching flag.Bool's own
// UX — IsBoolFlag is what tells the stdlib flag package that's legal.
func TestFitFlag_isBoolFlag(t *testing.T) {
	var f fitFlag
	if !f.IsBoolFlag() {
		t.Fatal("fitFlag.IsBoolFlag() = false, want true — a bare --fit would require a value")
	}
}

func TestFitFlag_string(t *testing.T) {
	on, off := fitFlag(true), fitFlag(false)
	if on.String() != "true" || off.String() != "false" {
		t.Fatalf("String() = %q / %q, want \"true\" / \"false\"", on.String(), off.String())
	}
}

// TestFitFlag_realBinaryAcceptsOff proves the fix through the FULL registered flag.CommandLine,
// not just fitFlag.Set in isolation — --version exits before touching a model, so this is a
// cheap way to prove "--fit=off" (the spelling task-fit-to-hardware.md and this flag's own help
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
