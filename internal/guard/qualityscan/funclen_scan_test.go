//go:build quality_scan

package qualityscan

import (
	"go/ast"
	"testing"
)

// funcLenCeiling is the longest a non-test function may be without an allowlist line.
const funcLenCeiling = 200

// A function longer than funcLenCeiling lines hides its structure in its closures, and every change to it diffs inside one
// body. The ones that exist are allowlisted at their current length (testdata/funclen.allow), so they can only shrink;
// a new one fails.
func TestQualityScan_functionLengthRatchet(t *testing.T) {
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
			if n := tr.Line(fd.End()) - tr.Line(fd.Pos()) + 1; n > funcLenCeiling {
				found[FuncKey(f.Path, fd)] = n
			}
		}
	}
	checkAllowlist(t, "funclen", found)
}
