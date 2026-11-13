package kernels

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// tryResultCalls are the device calls in internal/kernels whose error must be read: each returns nil beside its error, and
// the next line uses the result, so a discarded error becomes a nil-pointer panic with the cause gone (an encoder, a command
// buffer, a bind group, a buffer). tryEffectCalls return only an error, which a discard loses outright (a pass that did not
// end, a copy that did not happen).
var (
	tryResultCalls = map[string]bool{"TryCreateCommandEncoder": true, "TryFinish": true, "TryCreateBindGroup": true,
		"tryCreateBuffer": true, "tryCreateBufferInit": true, "dims4": true}
	tryEffectCalls = map[string]bool{"TryEnd": true, "TryCopyBufferToBuffer": true}
)

// droppedTryErrors lists file:line of each call in dir's non-test Go files that discards the error of a tryResultCalls or
// tryEffectCalls method: assigned to _, or a bare statement.
func droppedTryErrors(t *testing.T, dir string) []string {
	t.Helper()
	files, err := filepath.Glob(filepath.Join(dir, "*.go"))
	if err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		head, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(strings.SplitN(string(head), "\n", 2)[0], "goinfer_testhooks") {
			continue // a test seam, built only into test binaries
		}
		fset := token.NewFileSet()
		file, err := parser.ParseFile(fset, f, nil, parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}
		method := func(c *ast.CallExpr) string {
			if sel, ok := c.Fun.(*ast.SelectorExpr); ok {
				return sel.Sel.Name
			}
			return ""
		}
		at := func(n ast.Node, name string) {
			out = append(out, filepath.Base(f)+":"+strconv.Itoa(fset.Position(n.Pos()).Line)+" "+name)
		}
		ast.Inspect(file, func(n ast.Node) bool {
			switch s := n.(type) {
			case *ast.ExprStmt:
				if c, ok := s.X.(*ast.CallExpr); ok {
					if m := method(c); tryEffectCalls[m] || tryResultCalls[m] {
						at(s, m)
					}
				}
			case *ast.AssignStmt:
				if len(s.Rhs) != 1 {
					return true
				}
				c, ok := s.Rhs[0].(*ast.CallExpr)
				if !ok {
					return true
				}
				m := method(c)
				if !tryEffectCalls[m] && !tryResultCalls[m] {
					return true
				}
				if id, ok := s.Lhs[len(s.Lhs)-1].(*ast.Ident); ok && id.Name == "_" {
					at(s, m)
				}
			}
			return true
		})
	}
	sort.Strings(out)
	return out
}

// TestKernels_noDroppedTryError: no call in internal/kernels discards the error of a device call whose result or effect the
// next step relies on. It first proves the scan finds each shape on a plant.
func TestKernels_noDroppedTryError(t *testing.T) {
	plant := t.TempDir()
	src := "package p\n\nfunc f(d D, e E) error {\n\tenc, _ := d.TryCreateCommandEncoder(nil)\n\te.TryCopyBufferToBuffer(nil, 0, nil, 0, 0)\n" +
		"\t_ = e.TryEnd()\n\tcmd, err := enc.TryFinish(nil)\n\t_ = cmd\n\treturn err\n}\n"
	if err := os.WriteFile(filepath.Join(plant, "p.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(droppedTryErrors(t, plant), ","); got != "p.go:4 TryCreateCommandEncoder,p.go:5 TryCopyBufferToBuffer,p.go:6 TryEnd" {
		t.Fatalf("VOID: the scan finds %q in the plant", got)
	}
	found := droppedTryErrors(t, ".")
	for _, s := range found {
		t.Errorf("internal/kernels/%s discards its error: return it, with the call named", s)
	}
}
