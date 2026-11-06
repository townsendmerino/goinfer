package layout

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
)

// The weekly fuzz job names its targets by hand, so a package move or a new target leaves it silently wrong: a row for a directory that is gone fails on
// a Monday, and a target nobody listed never runs for more than its seeds.
func TestFuzzWeekly_listsEveryTarget(t *testing.T) {
	root := filepath.Join("..", "..")
	raw, err := os.ReadFile(filepath.Join(root, ".github", "workflows", "fuzz-weekly.yml"))
	if err != nil {
		t.Fatal(err)
	}
	var listed []string
	for _, m := range regexp.MustCompile(`\{ pkg: (\S+), fn: (Fuzz\w+) \}`).FindAllStringSubmatch(string(raw), -1) {
		listed = append(listed, m[1]+" "+m[2])
	}
	if len(listed) < 10 {
		t.Fatalf("read %d rows from the matrix: the workflow's shape changed", len(listed))
	}
	var found []string
	fset := token.NewFileSet()
	err = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if n := d.Name(); n == ".git" || n == ".claude" || n == "testdata" || n == "node_modules" {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(p, "_test.go") {
			return nil
		}
		f, err := parser.ParseFile(fset, p, nil, parser.SkipObjectResolution)
		if err != nil {
			return err
		}
		for _, decl := range f.Decls {
			if name := funcName(decl); strings.HasPrefix(name, "Fuzz") {
				rel, err := filepath.Rel(root, filepath.Dir(p))
				if err != nil {
					return err
				}
				found = append(found, filepath.ToSlash(rel)+" "+name)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(found) < 10 {
		t.Fatalf("found %d fuzz targets: this walked the wrong tree", len(found))
	}
	for _, row := range listed {
		if !slices.Contains(found, row) {
			t.Errorf("fuzz-weekly.yml lists %q, and no such target is in that directory", row)
		}
	}
	for _, target := range found {
		if !slices.Contains(listed, target) {
			t.Errorf("%q is not in fuzz-weekly.yml, so it only ever runs its seeds", target)
		}
	}
}

func funcName(d ast.Decl) string {
	if fd, ok := d.(*ast.FuncDecl); ok && fd.Recv == nil {
		return fd.Name.Name
	}
	return ""
}
