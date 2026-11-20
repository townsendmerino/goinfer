//go:build quality_scan

package qualityscan

import (
	"go/ast"
	"go/token"
	"regexp"
	"strings"
	"testing"
)

// trackerID is a task or audit id as the tree writes them: "M-09", "C-10", or the compact "K2", "KV8a", "MC3c", "S16".
var trackerID = regexp.MustCompile(`(?:^|[^\w-])([A-Z]{1,2}-\d{1,3}[a-z]?|[A-Z]{1,3}\d{1,3}[a-z]?)(?:$|[^\w-])`)

// notTrackers are tokens of the compact shape that name a thing, not a task.
var notTrackers = map[string]bool{
	"F16": true, "F32": true, "BF16": true, "FP8": true, "FP16": true, "U8": true, "U32": true, "I8": true, "Q8": true,
	"AVX2": true, "AVX512": true, "SSE4": true, "LFM2": true, "M1": true, "M2": true, "M3": true, "M4": true, "L2": true,
	"UTF8": true, "SHA256": true, "H100": true, "A100": true, "RTX2070": true, "MXFP4": true, "NVFP4": true, "Q4": true,
}

// userCall reports whether a call's string arguments reach a user: an error, a panic, a log line, or a flag's help.
func userCall(c *ast.CallExpr) bool {
	switch fn := c.Fun.(type) {
	case *ast.Ident:
		return fn.Name == "panic"
	case *ast.SelectorExpr:
		x, _ := fn.X.(*ast.Ident)
		switch {
		case x != nil && x.Name == "fmt" && fn.Sel.Name == "Errorf", x != nil && x.Name == "errors" && fn.Sel.Name == "New":
			return true
		case x != nil && x.Name == "logs":
			return true
		}
		switch fn.Sel.Name {
		case "String", "StringVar", "Int", "IntVar", "Int64", "Int64Var", "Bool", "BoolVar", "Float64", "Float64Var",
			"Duration", "DurationVar", "Uint", "UintVar", "Var", "Func", "BoolFunc", "TextVar":
			return x != nil && (x.Name == "flag" || x.Name == "fs" || strings.HasSuffix(x.Name, "Flags"))
		}
	}
	return false
}

// A message an operator reads says its reason in words: an id resolves only in a private task doc, and a reader of an
// error or a flag's help cannot follow it. Each id left in such a string is a line in testdata/tracker.allow with why.
func TestQualityScan_userStringsCarryNoTrackerIDs(t *testing.T) {
	tr := loadTree(t)
	found := map[string]int{}
	for _, f := range tr.Files {
		if f.Test {
			continue
		}
		ast.Inspect(f.AST, func(n ast.Node) bool {
			c, ok := n.(*ast.CallExpr)
			if !ok || !userCall(c) {
				return true
			}
			ast.Inspect(c, func(m ast.Node) bool {
				if lit, ok := m.(*ast.BasicLit); ok && lit.Kind == token.STRING {
					for _, sm := range trackerID.FindAllStringSubmatch(lit.Value, -1) {
						if !notTrackers[sm[1]] {
							found[f.Path+"#"+sm[1]]++
						}
					}
				}
				return true
			})
			return false
		})
	}
	checkAllowlist(t, "tracker", found)
}
