//go:build quality_scan

package qualityscan

import (
	"go/ast"
	"go/token"
	"regexp"
	"testing"
)

// handGB is a format verb printing a float followed by "GB": a memory size formatted by hand.
var handGB = regexp.MustCompile(`%[-+ 0-9]*(\.\d+)?f ?GB\b`)

// Fit, decline and banner text prints memory through internal/runtime/mem.GiB, so one free-VRAM figure reads the same
// from every layer. A hand-formatted "%.Nf GB" is allowed only where testdata/gib.allow says why: a disk or download size,
// which stays SI.
func TestQualityScan_memorySizesUseTheGiBFormatter(t *testing.T) {
	tr := loadTree(t)
	found := map[string]int{}
	for _, f := range tr.Files {
		if f.Test {
			continue
		}
		ast.Inspect(f.AST, func(n ast.Node) bool {
			if lit, ok := n.(*ast.BasicLit); ok && lit.Kind == token.STRING && handGB.MatchString(lit.Value) {
				found[f.Path] += len(handGB.FindAllString(lit.Value, -1))
			}
			return true
		})
	}
	checkAllowlist(t, "gib", found)
}
