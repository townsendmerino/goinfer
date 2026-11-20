//go:build quality_scan

package qualityscan

import (
	"go/ast"
	"strings"
	"testing"
)

// forwardedTo is the pkg.Name a one-statement function body calls with the function's own parameters, in order, or "".
func forwardedTo(fd *ast.FuncDecl) string {
	if fd.Recv != nil || fd.Body == nil || len(fd.Body.List) != 1 {
		return ""
	}
	var call *ast.CallExpr
	switch s := fd.Body.List[0].(type) {
	case *ast.ReturnStmt:
		if len(s.Results) == 1 {
			call, _ = s.Results[0].(*ast.CallExpr)
		}
	case *ast.ExprStmt:
		call, _ = s.X.(*ast.CallExpr)
	}
	if call == nil {
		return ""
	}
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return ""
	}
	pkg, ok := sel.X.(*ast.Ident)
	if !ok {
		return ""
	}
	var params []string
	for _, f := range fd.Type.Params.List {
		for _, n := range f.Names {
			params = append(params, n.Name)
		}
	}
	if len(params) != len(call.Args) {
		return ""
	}
	for i, a := range call.Args {
		id, ok := a.(*ast.Ident)
		if !ok || id.Name != params[i] {
			return ""
		}
	}
	return pkg.Name + "." + sel.Sel.Name
}

// A function whose whole body forwards to a same-named function in another package is a shim a package move left: every
// search for it has two hits and a hop. Callers call the target. The deliberate ones (public API that re-exports an
// internal package, a command's main) are lines in testdata/shim.allow with why.
func TestQualityScan_noSameNameForwardingShims(t *testing.T) {
	tr := loadTree(t)
	found := map[string]int{}
	for _, f := range tr.Files {
		if f.Test {
			continue
		}
		for _, d := range f.AST.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if !ok {
				continue
			}
			to := forwardedTo(fd)
			if _, name, _ := strings.Cut(to, "."); to != "" && strings.EqualFold(name, fd.Name.Name) {
				found[FuncKey(f.Path, fd)]++
			}
		}
	}
	checkAllowlist(t, "shim", found)
}
