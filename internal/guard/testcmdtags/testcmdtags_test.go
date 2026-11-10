package testcmdtags

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// plant writes a package under root/p with one test per build constraint.
func plant(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	dir := filepath.Join(root, "p")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	files := map[string]string{
		"plain_test.go":    "package p\n\nimport \"testing\"\n\nfunc TestPlain(t *testing.T) {}\n",
		"hooks_test.go":    "//go:build darwin && goinfer_testhooks\n\npackage p\n\nimport \"testing\"\n\nfunc TestHooked(t *testing.T) {}\n",
		"notwin_test.go":   "//go:build !windows\n\npackage p\n\nimport \"testing\"\n\nfunc TestNotWindows(t *testing.T) {}\n",
		"cudatags_test.go": "//go:build cuda && goinfer_testhooks\n\npackage p\n\nimport \"testing\"\n\nfunc TestCUDAHook(t *testing.T) {}\n",
	}
	for n, src := range files {
		if err := os.WriteFile(filepath.Join(dir, n), []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func check(t *testing.T, root, line string) []Finding {
	t.Helper()
	c, ok := Parse(line)
	if !ok {
		t.Fatalf("not a go test command: %q", line)
	}
	f, err := Check(root, c)
	if err != nil {
		t.Fatal(err)
	}
	return f
}

// TestCheck_namesATestItsTagsDoNotCompile is the shape that hid two of #827's four Metal tests: an untagged run of a
// darwin && goinfer_testhooks test. It also checks the cases that must not be flagged: a platform term alone, and the tag given
// in each spelling a command uses.
func TestCheck_namesATestItsTagsDoNotCompile(t *testing.T) {
	root := plant(t)
	f := check(t, root, "go test -count=1 ./p/ -run 'TestPlain|TestHooked|TestNotWindows' -v")
	if len(f) != 1 || f[0].Test != "TestHooked" || !strings.Contains(f[0].Why, "goinfer_testhooks") {
		t.Fatalf("findings %v, want exactly TestHooked naming goinfer_testhooks", f)
	}
	for _, line := range []string{
		"go test -tags goinfer_testhooks -count=1 ./p/ -run 'TestPlain|TestHooked|TestNotWindows' -v",
		"go test -tags=goinfer_testhooks ./p -run TestHooked",
		"GOFLAGS=-tags=goinfer_testhooks go test ./p -run '^TestHooked$'",
		"go test -tags 'cuda goinfer_testhooks' ./p/ -run '^(TestCUDAHook)$'",
		"cd p && go test -tags cuda,goinfer_testhooks . -run TestCUDAHook",
	} {
		if f := check(t, root, line); len(f) != 0 {
			t.Errorf("%q: findings %v, want none", line, f)
		}
	}
	if f := check(t, root, "go test -tags goinfer_testhooks ./p -run TestCUDAHook"); len(f) != 1 || !strings.Contains(f[0].Why, "cuda") {
		t.Errorf("a missing cuda tag: findings %v", f)
	}
}

// TestCheck_namesARunAlternativeThatMatchesNothing: an alternative that names no test runs nothing for that name.
func TestCheck_namesARunAlternativeThatMatchesNothing(t *testing.T) {
	root := plant(t)
	f := check(t, root, "go test ./p -run '^(TestPlain|TestGone)$'")
	if len(f) != 1 || !strings.Contains(f[0].Test, "TestGone") {
		t.Fatalf("findings %v, want TestGone alone", f)
	}
	if f := check(t, root, "go test ./p -run '^(TestPlain)$'"); len(f) != 0 {
		t.Errorf("an anchored alternative that matches: findings %v", f)
	}
}
