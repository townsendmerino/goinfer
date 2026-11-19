//go:build quality_scan

package qualityscan

import (
	"strings"
	"sync"
	"testing"
)

var (
	treeOnce sync.Once
	tree     *Tree
	treeErr  error
)

// loadTree parses the repository once per test binary.
func loadTree(t *testing.T) *Tree {
	t.Helper()
	treeOnce.Do(func() {
		root, err := FindRoot(".")
		if err != nil {
			treeErr = err
			return
		}
		tree, treeErr = Load(root)
	})
	if treeErr != nil {
		t.Fatal(treeErr)
	}
	return tree
}

// checkAllowlist fails the test on every site found that its allowlist does not allow, and logs the lines that can be
// tightened.
func checkAllowlist(t *testing.T, name string, found map[string]int) {
	t.Helper()
	allow, err := ReadAllowlist("testdata/" + name + ".allow")
	if err != nil {
		t.Fatal(err)
	}
	grown, shrunk := Compare(found, allow)
	for _, s := range shrunk {
		t.Logf("allowlist %s.allow can shrink: %s", name, s)
	}
	if len(grown) > 0 {
		t.Errorf("%d site(s) not in testdata/%s.allow or over their ceiling (fix the site, or allow it with its reason):\n\t%s",
			len(grown), name, strings.Join(grown, "\n\t"))
	}
}

// The scans read the whole tree: every module's sources, test and non-test. A scan that read a fraction of it would pass.
func TestQualityScan_readsEveryModule(t *testing.T) {
	tr := loadTree(t)
	seen := map[string]int{}
	tests := 0
	for _, f := range tr.Files {
		seg := strings.Split(f.Path, "/")
		top := seg[0]
		if top == "backend" && len(seg) > 2 {
			top += "/" + seg[1]
		}
		seen[top]++
		if f.Test {
			tests++
		}
	}
	for _, m := range []string{"decoder", "internal", "backend/webgpu", "backend/cuda", "backend/metal", "demo", "apps"} {
		if seen[m] == 0 {
			t.Errorf("read no Go file under %s/", m)
		}
	}
	if len(tr.Files) < 1000 || tests < 300 {
		t.Errorf("read %d files, %d of them tests: the tree walk is broken", len(tr.Files), tests)
	}
	t.Logf("read %d Go files (%d tests)", len(tr.Files), tests)
}
