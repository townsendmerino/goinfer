//go:build quality_scan

package qualityscan

import (
	"go/ast"
	"testing"
)

// preambleFields is how many receiver fields a method may copy into locals in plain multi-assignments before the scan
// counts it as an unpacking preamble.
const preambleFields = 8

// receiverFieldsUnpacked counts the receiver fields fd copies into locals in assignments whose every right-hand side is
// recv.field: `a, b, c := r.a, r.b, r.c`.
func receiverFieldsUnpacked(fd *ast.FuncDecl) int {
	if fd.Recv == nil || len(fd.Recv.List) == 0 || len(fd.Recv.List[0].Names) == 0 || fd.Body == nil {
		return 0
	}
	recv := fd.Recv.List[0].Names[0].Name
	n := 0
	ast.Inspect(fd.Body, func(x ast.Node) bool {
		if _, ok := x.(*ast.FuncLit); ok {
			return false
		}
		as, ok := x.(*ast.AssignStmt)
		if !ok || len(as.Rhs) < 2 || len(as.Lhs) != len(as.Rhs) {
			return true
		}
		for _, r := range as.Rhs {
			sel, ok := r.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			if id, ok := sel.X.(*ast.Ident); !ok || id.Name != recv {
				return true
			}
		}
		n += len(as.Rhs)
		return true
	})
	return n
}

// A method that opens by copying many receiver fields into locals makes the reader scan a long line to learn where a name
// comes from. The ones that exist are allowlisted at their field count (testdata/preamble.allow), so they can only shrink;
// a new one fails. Reading b.x directly, or grouping the fields into a sub-struct, is the fix.
func TestQualityScan_receiverUnpackingRatchet(t *testing.T) {
	tr := loadTree(t)
	found := map[string]int{}
	for _, f := range tr.Files {
		if f.Test {
			continue
		}
		for _, d := range f.AST.Decls {
			if fd, ok := d.(*ast.FuncDecl); ok {
				if n := receiverFieldsUnpacked(fd); n >= preambleFields {
					found[FuncKey(f.Path, fd)] = n
				}
			}
		}
	}
	checkAllowlist(t, "preamble", found)
}
