package testcmdtags

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// swallowedPackage returns the package argument of a `go test` command line that a test-binary flag in front of it swallows. go test
// does not know a test binary's own flags (-census.update, -memgate.rows.update, any -name.with.a.dot), so it reads the next word as
// that flag's value: "go test -census.update ./decoder/" runs no package at all and reports "no Go files in ." where it was run. The
// package goes first, or after a flag go test knows ("-tags x", "-run y", "-timeout 30m").
func swallowedPackage(line string) (flag, pkg string, found bool) {
	i := strings.Index(line, "go test ")
	if i < 0 {
		return "", "", false
	}
	words := strings.Fields(line[i+len("go test "):])
	for k := 0; k+1 < len(words); k++ {
		if strings.HasPrefix(words[k], "-") && strings.Contains(words[k], ".") && !strings.Contains(words[k], "=") &&
			(strings.HasPrefix(words[k+1], "./") || strings.HasPrefix(words[k+1], "github.com/")) {
			return words[k], words[k+1], true
		}
	}
	return "", "", false
}

// TestSwallowedPackage plants the CI census step's mistake and its fix: the flag before the package swallows it, the package first does not.
func TestSwallowedPackage(t *testing.T) {
	for _, c := range []struct {
		line string
		want bool
	}{
		{"go test -timeout 30m -tags goinfer_testhooks -run '^TestDecisionCensus$' -census.update ./decoder/ || true", true},
		{"go test ./decoder/ -timeout 30m -tags goinfer_testhooks -run '^TestDecisionCensus$' -census.update || true", false},
		{"go test -tags goinfer_testhooks -run X -census.update=true ./decoder/", false},
		{"go test -timeout 25m -tags goinfer_testhooks ./...", false},
	} {
		if _, _, got := swallowedPackage(c.line); got != c.want {
			t.Errorf("%q: swallowed = %v, want %v", c.line, got, c.want)
		}
	}
}

// TestWorkflows_noTestBinaryFlagSwallowsAPackage reads every workflow's `go test` lines: none may put a test-binary flag directly before a
// package (the census artifact step did, so a red census job uploaded nothing).
func TestWorkflows_noTestBinaryFlagSwallowsAPackage(t *testing.T) {
	root := filepath.Join("..", "..", "..", ".github", "workflows")
	files, err := filepath.Glob(filepath.Join(root, "*.yml"))
	if err != nil || len(files) == 0 {
		t.Fatalf("VOID: no workflow files under %s (%v)", root, err)
	}
	var lines int
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		for n, l := range strings.Split(string(b), "\n") {
			if !strings.Contains(l, "go test ") {
				continue
			}
			lines++
			if fl, pkg, bad := swallowedPackage(l); bad {
				t.Errorf("%s:%d: %s is a test-binary flag go test does not know, so it takes %s as its value and no package runs: put the package first", filepath.Base(f), n+1, fl, pkg)
			}
		}
	}
	if lines == 0 {
		t.Fatal("VOID: no `go test` line was read")
	}
}
