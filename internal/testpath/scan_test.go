package testpath

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// relativeLiterals returns the relative paths a test file spells out: each string literal beginning with "../", and the
// leading run of literal arguments of a filepath.Join or path.Join call, joined, when it begins with "..".
func relativeLiterals(t *testing.T, path string) map[token.Pos]string {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, nil, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("%s: %v", path, err)
	}
	out := map[token.Pos]string{}
	inJoin := map[token.Pos]bool{}
	lit := func(e ast.Expr) (string, bool) {
		b, ok := e.(*ast.BasicLit)
		if !ok || b.Kind != token.STRING {
			return "", false
		}
		s, err := strconv.Unquote(b.Value)
		return s, err == nil
	}
	ast.Inspect(f, func(n ast.Node) bool {
		c, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		sel, ok := c.Fun.(*ast.SelectorExpr)
		if !ok || sel.Sel.Name != "Join" {
			return true
		}
		if x, ok := sel.X.(*ast.Ident); !ok || (x.Name != "filepath" && x.Name != "path") {
			return true
		}
		var parts []string
		for _, a := range c.Args {
			s, ok := lit(a)
			if !ok {
				break
			}
			parts = append(parts, s)
			inJoin[a.Pos()] = true
		}
		if len(parts) > 0 && (parts[0] == ".." || strings.HasPrefix(parts[0], "../")) {
			out[c.Pos()] = filepath.Join(parts...)
		}
		return true
	})
	ast.Inspect(f, func(n ast.Node) bool {
		if b, ok := n.(*ast.BasicLit); ok && !inJoin[b.Pos()] {
			if s, ok := lit(b); ok && strings.HasPrefix(s, "../") {
				out[b.Pos()] = s
			}
		}
		return true
	})
	return out
}

// strayRelative says what is wrong with rel, a relative path in a test of the package at pkgDir under root: it leaves
// root, or the first directory it names after climbing does not exist (a "../" count that went stale when the package
// moved). "" when neither. An absent asset under an existing directory is not stray: a test may skip on it.
func strayRelative(root, pkgDir, rel string) string {
	abs := filepath.Join(pkgDir, rel)
	if r, err := filepath.Rel(root, abs); err != nil || r == ".." || strings.HasPrefix(r, "../") {
		return "it leaves the module root"
	}
	climb := pkgDir
	for _, part := range strings.Split(filepath.ToSlash(filepath.Clean(rel)), "/") {
		if part == ".." {
			climb = filepath.Dir(climb)
			continue
		}
		first := filepath.Join(climb, part)
		if _, err := os.Stat(first); err != nil {
			r, _ := filepath.Rel(root, first)
			return "it names " + filepath.ToSlash(r) + ", which does not exist"
		}
		break
	}
	return ""
}

// TestNoStrayRelativePathInInternalTests: no test under internal/ spells a relative path that leaves the module root or
// climbs to a directory that does not exist. Such a path makes its test skip, not fail, after the package moves; Testdata
// finds the root instead. It first proves it can go red on the shape that went stale.
func TestNoStrayRelativePathInInternalTests(t *testing.T) {
	root := Root(t)
	pkg := filepath.Join(root, "internal", "app", "serveapp")
	if strayRelative(root, pkg, "../../testdata/x") == "" || strayRelative(root, pkg, "../../../../x") == "" {
		t.Fatal("VOID: the check does not flag a stale climb or one that leaves the root")
	}
	if why := strayRelative(root, pkg, "../../../testdata/an-absent-asset"); why != "" {
		t.Fatalf("VOID: an absent asset under testdata/ was flagged: %s", why)
	}
	var files, paths int
	err := filepath.WalkDir(filepath.Join(root, "internal"), func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() && d.Name() == "testdata" {
			return filepath.SkipDir
		}
		if d.IsDir() || !strings.HasSuffix(p, "_test.go") || p == filepath.Join(root, "internal", "testpath", "scan_test.go") {
			return nil // this file's own cases are relative to another package, on purpose
		}
		files++
		for _, rel := range relativeLiterals(t, p) {
			paths++
			if why := strayRelative(root, filepath.Dir(p), rel); why != "" {
				r, _ := filepath.Rel(root, p)
				t.Errorf("%s: %q: %s; use testpath.Testdata or testpath.Root", filepath.ToSlash(r), rel, why)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if files == 0 || paths == 0 {
		t.Fatalf("VOID: read %d test files and %d relative paths", files, paths)
	}
	t.Logf("%d test files, %d relative paths", files, paths)
}
