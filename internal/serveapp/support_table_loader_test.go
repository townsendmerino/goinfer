package serveapp

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// S8 (docs/tasks/task-multimodal-support-2026-10.md, "S8, plan and gates"): every family docs/multimodal.md's support
// table lists has a serve loader. loadVisionTower dispatches on the checkpoint's model_type (`mt == "..."` branches),
// and every other model_type falls through to Gemma 3's SigLIP tower; a listed model_type must be one of the branches or
// gemma3 itself.

// visionDispatchTypes returns the model_type literals loadVisionTower compares mt against.
func visionDispatchTypes(t *testing.T) map[string]bool {
	t.Helper()
	fset := token.NewFileSet()
	af, err := parser.ParseFile(fset, "main.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	types := map[string]bool{}
	ast.Inspect(af, func(n ast.Node) bool {
		fn, ok := n.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "loadVisionTower" {
			return true
		}
		ast.Inspect(fn, func(n ast.Node) bool {
			be, ok := n.(*ast.BinaryExpr)
			if !ok || be.Op != token.EQL {
				return true
			}
			if id, ok := be.X.(*ast.Ident); ok && id.Name == "mt" {
				if lit, ok := be.Y.(*ast.BasicLit); ok && lit.Kind == token.STRING {
					if s, err := strconv.Unquote(lit.Value); err == nil {
						types[s] = true
					}
				}
			}
			return true
		})
		return false
	})
	if len(types) == 0 {
		t.Fatal("loadVisionTower compares mt against no literal: this check is watching nothing")
	}
	return types
}

// tableModelTypes returns the model_types of the support table's rows in docs/multimodal.md.
func tableModelTypes(t *testing.T) []string {
	t.Helper()
	raw, err := os.ReadFile("../../docs/multimodal.md")
	if err != nil {
		t.Fatal(err)
	}
	doc := string(raw)
	i, j := strings.Index(doc, "<!-- BEGIN images-and-audio support table"), strings.Index(doc, "<!-- END images-and-audio support table -->")
	if i < 0 || j < i {
		t.Fatal("docs/multimodal.md has no support-table block")
	}
	var out []string
	re := regexp.MustCompile("^\\| [^|]*\\(`([a-z0-9_]+)`\\) \\|")
	for _, ln := range strings.Split(doc[i:j], "\n") {
		if m := re.FindStringSubmatch(ln); m != nil && m[1] != "model_type" { // not the header row
			out = append(out, m[1])
		}
	}
	if len(out) == 0 {
		t.Fatal("the support table has no rows")
	}
	return out
}

// unloaded returns the model_types with no loadVisionTower branch (gemma3 is the fallback's).
func unloaded(types []string, dispatch map[string]bool) []string {
	var miss []string
	for _, mt := range types {
		if mt != "gemma3" && !dispatch[mt] {
			miss = append(miss, mt)
		}
	}
	return miss
}

// TestSupportTable_everyFamilyHasALoader (G-S8a's serve half, and G-S8b's planted "a row with no loader").
func TestSupportTable_everyFamilyHasALoader(t *testing.T) {
	dispatch := visionDispatchTypes(t)
	types := tableModelTypes(t)
	if miss := unloaded(types, dispatch); len(miss) > 0 {
		t.Errorf("the support table lists model_types serve has no vision loader for: %v (loadVisionTower dispatches on %v, gemma3 by fallback)", miss, dispatch)
	}
	// Planted: a family the table might grow before serve does (Ministral 3's Pixtral tower, S10) must be reported.
	if miss := unloaded(append(types, "mistral3"), dispatch); len(miss) != 1 || miss[0] != "mistral3" {
		t.Errorf("a planted row with no loader was not reported: %v", miss)
	}
}
