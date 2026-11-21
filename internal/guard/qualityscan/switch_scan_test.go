//go:build quality_scan

package qualityscan

import (
	"go/ast"
	"go/token"
	"path"
	"strconv"
	"strings"
	"testing"
)

// switchScanDirs are the packages whose tests set package-level switches and seams; backend/metal keeps its own
// guard (TestSwitchGuard_coversEveryAssignedSwitch).
var switchScanDirs = []string{"decoder", "backend/cuda", "backend/webgpu"}

func inSwitchScan(p string) bool {
	for _, d := range switchScanDirs {
		if strings.HasPrefix(p, d+"/") {
			return true
		}
	}
	return false
}

// A test that assigns a package-level variable by hand has to restore it itself, and a forgotten restore leaks into every
// later test in the binary. New tests take internal/testgate.SetForTest, which restores it in t.Cleanup. The files that
// assign by hand are allowlisted at their count (testdata/switch.allow), so the raw assignments can only shrink.
func TestQualityScan_testSwitchAssignmentsRatchet(t *testing.T) {
	tr := loadTree(t)
	pkgVars := map[string]map[string]bool{} // dir to its package-level var names, from non-test files
	for _, f := range tr.Files {
		if f.Test || !inSwitchScan(f.Path) {
			continue
		}
		dir := path.Dir(f.Path)
		if pkgVars[dir] == nil {
			pkgVars[dir] = map[string]bool{}
		}
		for _, d := range f.AST.Decls {
			if g, ok := d.(*ast.GenDecl); ok && g.Tok == token.VAR {
				for _, sp := range g.Specs {
					for _, n := range sp.(*ast.ValueSpec).Names {
						if n.Name != "_" { // `var _ I = (*T)(nil)` declares nothing a test can set
							pkgVars[dir][n.Name] = true
						}
					}
				}
			}
		}
	}
	found := map[string]int{}
	for _, f := range tr.Files {
		if !f.Test || !inSwitchScan(f.Path) {
			continue
		}
		own := pkgVars[path.Dir(f.Path)]
		imported := map[string]bool{}
		for _, im := range f.AST.Imports {
			p, _ := strconv.Unquote(im.Path.Value)
			name := path.Base(p)
			if im.Name != nil {
				name = im.Name.Name
			}
			imported[name] = true
		}
		ast.Inspect(f.AST, func(n ast.Node) bool {
			a, ok := n.(*ast.AssignStmt)
			if !ok || a.Tok != token.ASSIGN {
				return true
			}
			for _, l := range a.Lhs {
				switch x := l.(type) {
				case *ast.Ident:
					if own[x.Name] && x.Obj == nil {
						found[f.Path]++
					}
				case *ast.SelectorExpr:
					if id, ok := x.X.(*ast.Ident); ok && imported[id.Name] && id.Obj == nil {
						found[f.Path]++
					}
				}
			}
			return true
		})
	}
	checkAllowlist(t, "switch", found)
}
