//go:build quality_scan

package qualityscan

import (
	"go/ast"
	"go/token"
	"os/exec"
	"path"
	"strconv"
	"strings"
	"testing"
)

// trackedTestdataDirs is every directory under a testdata/ directory that git tracks a file in, repository-relative.
func trackedTestdataDirs(t *testing.T, root string) map[string]bool {
	t.Helper()
	out, err := exec.Command("git", "-C", root, "ls-files", "-z", "--", "*testdata/*").Output()
	if err != nil {
		t.Fatal(err)
	}
	dirs := map[string]bool{}
	for p := range strings.SplitSeq(string(out), "\x00") {
		for d := path.Dir(p); d != "." && strings.Contains(d, "testdata"); d = path.Dir(d) {
			dirs[d] = true
		}
		if strings.Contains(p, "testdata/") {
			dirs[p] = true // a file fixture (a .gguf) is named directly
		}
	}
	return dirs
}

// namesCommittedFixture reports whether a string literal in fn names a git-tracked testdata path, resolved against the
// test file's directory.
func namesCommittedFixture(fn ast.Node, dir string, tracked map[string]bool) bool {
	hit := false
	ast.Inspect(fn, func(n ast.Node) bool {
		lit, ok := n.(*ast.BasicLit)
		if !ok || lit.Kind != token.STRING || hit {
			return !hit
		}
		s, err := strconv.Unquote(lit.Value)
		if err != nil || !strings.Contains(s, "testdata") {
			return true
		}
		if tracked[path.Clean(path.Join(dir, s))] {
			hit = true
		}
		return !hit
	})
	return hit
}

// isSkip reports whether stmt is a t.Skip, t.Skipf or t.SkipNow call.
func isSkip(stmt ast.Stmt) bool {
	es, ok := stmt.(*ast.ExprStmt)
	if !ok {
		return false
	}
	c, ok := es.X.(*ast.CallExpr)
	if !ok {
		return false
	}
	sel, ok := c.Fun.(*ast.SelectorExpr)
	return ok && strings.HasPrefix(sel.Sel.Name, "Skip")
}

// callsLoad reports whether stmt calls a function whose name starts with or contains "Load".
func callsLoad(stmt ast.Stmt) bool {
	found := false
	ast.Inspect(stmt, func(n ast.Node) bool {
		if c, ok := n.(*ast.CallExpr); ok {
			var name string
			switch f := c.Fun.(type) {
			case *ast.Ident:
				name = f.Name
			case *ast.SelectorExpr:
				name = f.Sel.Name
			}
			if strings.Contains(name, "Load") || strings.Contains(name, "load") {
				found = true
			}
		}
		return !found
	})
	return found
}

// A test whose fixture is committed under testdata/ cannot be missing it, so a load that fails is a regression: it
// fails the test. A Skip on that error turns a broken loader into `ok`. The scan flags `if err != nil { t.Skip… }`
// right after a load, in a test function that names a git-tracked testdata path. Skip stays right for
// an uncommitted checkpoint and for a device probe; the backend modules' loads probe a device, so they are not read.
// Each kept site is a line in testdata/fixtureskip.allow with why.
func TestQualityScan_committedFixtureLoadsFailNotSkip(t *testing.T) {
	tr := loadTree(t)
	tracked := trackedTestdataDirs(t, tr.Root)
	found := map[string]int{}
	for _, f := range tr.Files {
		if !f.Test || strings.HasPrefix(f.Path, "backend/") {
			continue
		}
		dir := path.Dir(f.Path)
		for _, d := range f.AST.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if !ok || fd.Body == nil || !namesCommittedFixture(fd, dir, tracked) {
				continue
			}
			ast.Inspect(fd.Body, func(n ast.Node) bool {
				var list []ast.Stmt
				switch b := n.(type) {
				case *ast.BlockStmt:
					list = b.List
				case *ast.CaseClause:
					list = b.Body
				default:
					return true
				}
				for i, s := range list {
					is, ok := s.(*ast.IfStmt)
					if !ok || len(is.Body.List) == 0 || !isSkip(is.Body.List[0]) {
						continue
					}
					if !strings.Contains(types(is.Cond), "err") {
						continue
					}
					if (is.Init != nil && callsLoad(is.Init)) || (is.Init == nil && i > 0 && callsLoad(list[i-1])) {
						found[FuncKey(f.Path, fd)]++
					}
				}
				return true
			})
		}
	}
	checkAllowlist(t, "fixtureskip", found)
}

// types renders a condition's identifiers, enough to see whether it tests an error.
func types(e ast.Expr) string {
	var b strings.Builder
	ast.Inspect(e, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok {
			b.WriteString(id.Name + " ")
		}
		return true
	})
	return b.String()
}
