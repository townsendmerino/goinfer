package wbuild

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// tensorReads are the calls that touch a tensor's mapped pages: aikit's GGUF readers and this package's helpers over them.
var tensorReads = map[string]bool{
	"Tensor": true, "RowDequantizer": true, "Q4KRaw": true, "Q5KRaw": true, "Q6KRaw": true,
	"GGUFScaledRowDequantizer": true, "q4kMat": true,
}

// releaseFree are the helpers that read a tensor for a caller that releases it, so the release is the caller's.
var releaseFree = map[string]string{
	"GGUFScaledRowDequantizer": "a reader over RowDequantizer and the .scale companion; its callers release both (releaseTensor)",
	"q4kMat":                   "the K-quant fast path of streamMat, streamMatBatched and fusedSplit, which release",
}

// ggufRead reports whether c reads a GGUF tensor: a helper above, or one of aikit's readers on this package's
// *embed.GGUFFile, which it names g (b.g on a ggufBuild). The safetensors readers (st) are not this guard's source.
func ggufRead(c *ast.CallExpr) bool {
	name := callName(c)
	if !tensorReads[name] {
		return false
	}
	sel, ok := c.Fun.(*ast.SelectorExpr)
	if !ok {
		return true
	}
	switch x := sel.X.(type) {
	case *ast.Ident:
		return x.Name == "g"
	case *ast.SelectorExpr:
		return x.Sel.Name == "g"
	}
	return false
}

func callName(c *ast.CallExpr) string {
	switch f := c.Fun.(type) {
	case *ast.Ident:
		return f.Name
	case *ast.SelectorExpr:
		return f.Sel.Name
	}
	return ""
}

// TestWbuild_everyTensorReadReleases holds the fit guard's source term (ggufInFlightSourceBytes prices one tensor per layer
// worker, not the file) to the code: in every non-test file of this package, a function that reads a GGUF tensor also
// releases one (release, b.release, releaseTensor or ReleaseTensors), so no builder holds its source resident to the end.
func TestWbuild_everyTensorReadReleases(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	reads := 0
	for _, path := range files {
		if strings.HasSuffix(path, "_test.go") {
			continue
		}
		src, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		f, err := parser.ParseFile(fset, path, src, 0)
		if err != nil {
			t.Fatal(err)
		}
		// The innermost function enclosing each read, with its name for a declaration.
		var stack []ast.Node
		ast.Inspect(f, func(n ast.Node) bool {
			if n == nil {
				stack = stack[:len(stack)-1]
				return true
			}
			stack = append(stack, n)
			c, ok := n.(*ast.CallExpr)
			if !ok || !ggufRead(c) {
				return true
			}
			var fn ast.Node
			name := ""
			for i := len(stack) - 1; i >= 0 && fn == nil; i-- {
				switch d := stack[i].(type) {
				case *ast.FuncLit:
					fn = d
				case *ast.FuncDecl:
					fn, name = d, d.Name.Name
				}
			}
			if _, ok := releaseFree[name]; ok && fn != nil {
				if _, isDecl := fn.(*ast.FuncDecl); isDecl {
					return true
				}
			}
			reads++
			released := false
			ast.Inspect(fn, func(m ast.Node) bool {
				if mc, ok := m.(*ast.CallExpr); ok {
					switch callName(mc) {
					case "release", "releaseTensor", "ReleaseTensors":
						released = true
					}
				}
				return !released
			})
			if !released {
				t.Errorf("%s: %s reads a tensor in a function that never releases it: defer b.release(name) beside the read (releaseTensor)", fset.Position(c.Pos()), callName(c))
			}
			return true
		})
	}
	if reads < 10 {
		t.Fatalf("found %d tensor reads: the scan is not seeing this package's builders", reads)
	}
}
