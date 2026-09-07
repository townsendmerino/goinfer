package chatapp

import (
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// R6 (docs/measurements/cold-user-2026-09-06-nobara-pc.md): `goinfer-chat --version` was
// unanswerable — "flag provided but not defined: -version" — and a bare `version` positional
// (a plausible typo reaching for it) was silently swallowed and started an interactive chat
// session with the embedded model instead of erroring or printing anything.
func TestIsVersionArg_recognizesAllForms(t *testing.T) {
	for _, a := range []string{"--version", "-version", "version"} {
		if !isVersionArg(a) {
			t.Errorf("isVersionArg(%q) = false, want true", a)
		}
	}
	for _, a := range []string{"versions", "-v", "--Version", "models", ""} {
		if isVersionArg(a) {
			t.Errorf("isVersionArg(%q) = true, want false", a)
		}
	}
}

// The mutation this guards against: deleting the dispatch check makes this go red with exactly
// v0.17.0's own error text, "flag provided but not defined: -version".
func TestChatVersionFlag_answersWithoutAModel(t *testing.T) {
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

	// Both spellings, and both positions (bare first arg, and the registered flag after one
	// that would otherwise require a model) — R6 found the first broken and the second untested.
	for _, args := range [][]string{{"--version"}, {"version"}, {"-model", "/nonexistent.gguf", "--version"}} {
		out, err := exec.Command(bin, args...).CombinedOutput()
		if err != nil {
			t.Fatalf("%s %v: %v\n%s", bin, args, err, out)
		}
		s := string(out)
		if !strings.Contains(s, "backends:") {
			t.Errorf("%v output missing a backends: line:\n%s", args, s)
		}
		if !strings.Contains(s, "go: "+runtime.Version()) {
			t.Errorf("%v output missing the go toolchain line:\n%s", args, s)
		}
		if strings.Contains(s, "flag provided but not defined") {
			t.Errorf("%v regressed to v0.17.0's unanswerable-version error:\n%s", args, s)
		}
	}
}

// The other half of R6: a bare unrecognized positional must error and name the real
// subcommands, not fall through into loading a model. The mutation this guards against:
// removing the flag.Args() check reproduces v0.17.0's silent fallthrough exactly — no error,
// exit 0, and (on an embed build) an interactive session started on a typo.
func TestChatUnknownPositional_namesTheSubcommands(t *testing.T) {
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

	out, err := exec.Command(bin, "bogus-subcommand").CombinedOutput()
	if err == nil {
		t.Fatalf("bogus-subcommand exited 0, want a non-zero exit naming it as unrecognized:\n%s", out)
	}
	s := string(out)
	if !strings.Contains(s, `unrecognized argument "bogus-subcommand"`) {
		t.Errorf("stderr does not name the bad argument:\n%s", s)
	}
	for _, want := range []string{"pull", "models", "--version"} {
		if !strings.Contains(s, want) {
			t.Errorf("stderr does not name the known subcommand %q:\n%s", want, s)
		}
	}
}
