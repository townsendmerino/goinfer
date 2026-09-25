package chatapp

import (
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestLoadFlags_realBinaryParses: chat had no --ctx, --stream-weights or --moe-cache-experts/
// --moe-cache-slots, and a cold-user run reached for --moe-cache-experts and got "flag provided but
// not defined". They are internal/loadflags' now, registered for both binaries; this proves the REAL
// flag.CommandLine carries them (--version exits before touching a model, the same discipline as
// TestExactPrefillFlag_realBinaryParses).
func TestLoadFlags_realBinaryParses(t *testing.T) {
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
	out, err := exec.Command(bin, "--ctx", "16384", "--stream-weights", "--weight-cache=4",
		"--moe-cache-experts", "--moe-cache-slots", "8", "--moe-pager=pool", "--accept-slow",
		"--cpu-exact-prefill", "--version").CombinedOutput()
	s := string(out)
	if err != nil {
		t.Fatalf("loading flags + --version: %v\n%s", err, s)
	}
	if strings.Contains(s, "flag provided but not defined") || strings.Contains(s, "belongs to goinfer-serve") {
		t.Errorf("a shared loading flag is missing from goinfer-chat:\n%s", s)
	}
	if !strings.Contains(s, "backends:") {
		t.Errorf("--version output missing a backends: line:\n%s", s)
	}
}
