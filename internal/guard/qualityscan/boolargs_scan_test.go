//go:build quality_scan

package qualityscan

import (
	"go/ast"
	"testing"
)

// boolRun is the longest run of adjacent true/false literals in a call's arguments.
func boolRun(c *ast.CallExpr) int {
	best, run := 0, 0
	for _, a := range c.Args {
		if id, ok := a.(*ast.Ident); ok && (id.Name == "true" || id.Name == "false") {
			run++
			best = max(best, run)
			continue
		}
		run = 0
	}
	return best
}

// A call passing three or more bool literals side by side cannot be read without the signature open, and two swapped
// flags compile. Such a call takes a struct with named fields instead; testdata/boolargs.allow lists any kept, with why.
func TestQualityScan_noRunsOfBoolLiteralArguments(t *testing.T) {
	tr := loadTree(t)
	found := map[string]int{}
	for _, f := range tr.Files {
		if f.Test {
			continue
		}
		for _, d := range f.AST.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if !ok || fd.Body == nil {
				continue
			}
			ast.Inspect(fd.Body, func(n ast.Node) bool {
				if c, ok := n.(*ast.CallExpr); ok && boolRun(c) >= 3 {
					found[FuncKey(f.Path, fd)]++
				}
				return true
			})
		}
	}
	checkAllowlist(t, "boolargs", found)
}
