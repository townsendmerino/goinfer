//go:build quality_scan

package qualityscan

import (
	"go/ast"
	"go/token"
	"testing"
)

// A test that reads GOINFER_HEAVY_TESTS itself repeats a gate internal/testgate.RequireHeavy already states once. The
// files that do are allowlisted at their count (testdata/heavyenv.allow), so the raw reads can only shrink; a new one fails.
// Migrate a file's reads when it is next touched.
func TestQualityScan_heavyGateReadsRatchet(t *testing.T) {
	tr := loadTree(t)
	found := map[string]int{}
	for _, f := range tr.Files {
		if f.Path == "internal/testgate/testgate.go" {
			continue // the helper's own read
		}
		ast.Inspect(f.AST, func(n ast.Node) bool {
			c, ok := n.(*ast.CallExpr)
			if !ok || len(c.Args) == 0 {
				return true
			}
			sel, ok := c.Fun.(*ast.SelectorExpr)
			if !ok || (sel.Sel.Name != "Getenv" && sel.Sel.Name != "LookupEnv") {
				return true
			}
			if lit, ok := c.Args[0].(*ast.BasicLit); ok && lit.Kind == token.STRING && lit.Value == `"GOINFER_HEAVY_TESTS"` {
				found[f.Path]++
			}
			return true
		})
	}
	checkAllowlist(t, "heavyenv", found)
}
