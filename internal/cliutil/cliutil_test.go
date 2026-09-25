package cliutil

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestOnOff_accepts is M-14 (audit-2026-09-10): --fit=off is the spelling --fit's own help text
// and tasks/task-fit-to-hardware.md promise, but a plain flag.Bool rejected it with exit 2.
func TestOnOff_accepts(t *testing.T) {
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
		f := OnOff(!c.want) // start from the opposite, so a no-op Set would be caught
		if err := f.Set(c.in); err != nil {
			t.Errorf("Set(%q): unexpected error: %v", c.in, err)
			continue
		}
		if bool(f) != c.want {
			t.Errorf("Set(%q) = %v, want %v", c.in, bool(f), c.want)
		}
	}
}

// TestOnOff_rejectsGarbage: an unrecognised value must error, not silently pick a default — the
// same discipline strconv.ParseBool itself follows.
func TestOnOff_rejectsGarbage(t *testing.T) {
	for _, bad := range []string{"yes", "no", "2", "maybe", ""} {
		var f OnOff
		if err := f.Set(bad); err == nil {
			t.Errorf("Set(%q): expected an error, got none (f=%v)", bad, f)
		}
	}
}

// TestOnOff_isBoolFlag: a bare `--fit` (no `=value`) must still work — IsBoolFlag is what tells the
// stdlib flag package that is legal.
func TestOnOff_isBoolFlag(t *testing.T) {
	var f OnOff
	if !f.IsBoolFlag() {
		t.Fatal("OnOff.IsBoolFlag() = false, want true — a bare --fit would require a value")
	}
}

func TestOnOff_string(t *testing.T) {
	on, off := OnOff(true), OnOff(false)
	if on.String() != "true" || off.String() != "false" {
		t.Fatalf("String() = %q / %q, want \"true\" / \"false\"", on.String(), off.String())
	}
}

func TestBuildIdent_injectedWins(t *testing.T) {
	if v, _ := BuildIdent("v9.9.9"); v != "v9.9.9" {
		t.Errorf("BuildIdent(\"v9.9.9\") version = %q, want the injected value", v)
	}
	if v, _ := BuildIdent(""); v == "" {
		t.Error("BuildIdent(\"\") returned an empty version; want the stamped one or \"(unknown)\"")
	}
}

// ldflagsX matches one `-X github.com/townsendmerino/goinfer/<pkg>.<var>=` in a build script.
var ldflagsX = regexp.MustCompile(`-X github\.com/townsendmerino/goinfer/([\w/]+)\.(\w+)=`)

// TestLdflagsTargetsExist: `go build -ldflags -X pkg.name=v` does NOTHING when pkg.name does not
// exist — no error, no warning, the binary builds and reports its stamped version instead. So a
// rename or move of injectedVersion / embeddedTier would ship a release reporting "+dirty" (R6)
// with every check green. This holds each -X target in the release scripts to a string var that
// is actually declared in that package's non-test source.
func TestLdflagsTargetsExist(t *testing.T) {
	root := filepath.Join("..", "..")
	scripts := []string{".github/workflows/release-assets.yml", "demo/chat/build-embed.sh"}
	n := 0
	for _, s := range scripts {
		src, err := os.ReadFile(filepath.Join(root, s))
		if err != nil {
			t.Fatalf("read %s: %v", s, err)
		}
		for _, m := range ldflagsX.FindAllStringSubmatch(string(src), -1) {
			n++
			pkg, name := m[1], m[2]
			decl := regexp.MustCompile(`(?m)^var\s+[\w\s,]*\b` + name + `\b[\w\s,]*\bstring\b`)
			files, _ := filepath.Glob(filepath.Join(root, pkg, "*.go"))
			found := false
			for _, f := range files {
				if strings.HasSuffix(f, "_test.go") {
					continue
				}
				b, err := os.ReadFile(f)
				if err != nil {
					t.Fatal(err)
				}
				if decl.Match(b) {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("%s sets -X %s.%s, but no non-test file in %s declares `var %s ... string` — the -X would be silently ignored", s, pkg, name, pkg, name)
			}
		}
	}
	// The release scripts set at least serve's and chat's injectedVersion and chat's embeddedTier;
	// finding fewer means the pattern stopped matching, and the check above covered nothing.
	if n < 4 {
		t.Fatalf("found only %d -X targets across %v — the pattern no longer matches the scripts", n, scripts)
	}
}
