package layout

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// forTestGlobalsRatchet is the committed count of distinct mutable package-level *ForTest var declarations per
// non-test package directory. New test seams belong on injected options, struct fields or constructor parameters,
// not package globals (CLAUDE.md, "Go style and structure"). A count above its figure fails as a regression; a
// count below it fails until lowered in the commit that removes the hook.
var forTestGlobalsRatchet = map[string]int{
	"backend/cuda":                    53,
	"backend/cuda/internal/gpudev":    2,
	"backend/metal":                   67,
	"backend/metal/internal/gpudev":   4,
	"backend/webgpu":                  9,
	"backend/webgpu/internal/kernels": 29,
	"decoder":                         22,
	"internal/diag/towerid":           1,
	"internal/vision":                 3,
}

// scanForTestGlobals walks root for non-test .go files (skipping testdata, hidden/underscore directories and
// //go:build ignore files) and returns the sorted distinct package-level var names containing "ForTest" per
// relative package directory. Complementary build-tag pairs in the same directory share one name and count once.
func scanForTestGlobals(t *testing.T, root string) map[string][]string {
	t.Helper()
	found := map[string]map[string]bool{}
	fset := token.NewFileSet()
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			n := d.Name()
			if path != root && (n == "testdata" || n == "node_modules" || n == "vendor" || strings.HasPrefix(n, ".") || strings.HasPrefix(n, "_")) {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		f, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
		if err != nil {
			return err
		}
		if ignored(f.Comments, f.Package) {
			return nil
		}
		rel, err := filepath.Rel(root, filepath.Dir(path))
		if err != nil {
			return err
		}
		dir := filepath.ToSlash(rel)
		for _, decl := range f.Decls {
			gd, ok := decl.(*ast.GenDecl)
			if !ok || gd.Tok != token.VAR {
				continue
			}
			for _, spec := range gd.Specs {
				vs, ok := spec.(*ast.ValueSpec)
				if !ok {
					continue
				}
				for _, name := range vs.Names {
					if name.Name != "_" && strings.Contains(name.Name, "ForTest") {
						if found[dir] == nil {
							found[dir] = map[string]bool{}
						}
						found[dir][name.Name] = true
					}
				}
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("scanning %s: %v", root, err)
	}
	out := make(map[string][]string, len(found))
	for dir, names := range found {
		list := make([]string, 0, len(names))
		for n := range names {
			list = append(list, n)
		}
		slices.Sort(list)
		out[dir] = list
	}
	return out
}

// TestForTestGlobals_onlyFall enforces forTestGlobalsRatchet across every non-test Go package in the workspace:
// adding a mutable package-global *ForTest variable fails, and removing one requires lowering the ratchet in the
// same commit.
func TestForTestGlobals_onlyFall(t *testing.T) {
	got := scanForTestGlobals(t, repoRoot)
	dirs := map[string]bool{}
	for d := range got {
		dirs[d] = true
	}
	for d := range forTestGlobalsRatchet {
		dirs[d] = true
	}
	sortedDirs := make([]string, 0, len(dirs))
	for d := range dirs {
		sortedDirs = append(sortedDirs, d)
	}
	slices.Sort(sortedDirs)
	for _, dir := range sortedDirs {
		names := got[dir]
		want := forTestGlobalsRatchet[dir]
		switch n := len(names); {
		case n > want:
			t.Errorf("%s has %d mutable package-global *ForTest vars, above ratchet %d: inject a test seam via options/fields instead. Vars: %s",
				dir, n, want, strings.Join(names, ", "))
		case n < want:
			t.Errorf("%s has %d mutable package-global *ForTest vars, below ratchet %d: lower forTestGlobalsRatchet[%q] to %d (or delete the entry if 0)",
				dir, n, want, dir, n)
		}
	}
}

// TestForTestGlobals_scannerAccuracy checks that scanForTestGlobals catches top-level *ForTest vars, deduplicates
// complementary build-tag declarations in one directory, and ignores consts, funcs, locals, test files and
// //go:build ignore files.
func TestForTestGlobals_scannerAccuracy(t *testing.T) {
	dir := t.TempDir()
	write := func(name, body string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("a.go", "package p\n\nconst okConstForTest = 1\nvar hookForTest bool\nvar (\n\tgroupedForTest func()\n\tplainGlobal int\n)\nfunc HelperForTest() {\n\tvar localForTest int\n\t_ = localForTest\n}\n")
	write("b_testhooks.go", "//go:build goinfer_testhooks\n\npackage p\n\nvar hookForTest bool\n")
	write("c_test.go", "package p\n\nvar testOnlyForTest bool\n")
	write("d_ignore.go", "//go:build ignore\n\npackage p\n\nvar ignoredForTest bool\n")

	got := scanForTestGlobals(t, dir)
	want := []string{"groupedForTest", "hookForTest"}
	if !slices.Equal(got["."], want) {
		t.Fatalf("got %v, want %v", got["."], want)
	}
}
