package layout

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"testing"
)

// TestCUDASoloShapes_takeTodaysCalls holds RF2's decode kill: Forward, ForwardArgmax, ForwardSample and ForwardN run today's launches,
// not a one-sequence ragged step, which G-RF2-S read slower than them; MC3's batched step keeps the ragged step. It fails on a
// call named StepRagged or soloRagged in those methods' bodies (a direct call only, by name).
// Docs: docs/tasks/task-forward-and-memory-2026-11.md, "RF2: registration" (the kill).
func TestCUDASoloShapes_takeTodaysCalls(t *testing.T) {
	solo := map[string]bool{"Forward": true, "ForwardArgmax": true, "ForwardSample": true, "ForwardN": true}
	banned := map[string]bool{"StepRagged": true, "soloRagged": true}
	found := map[string]bool{}
	fset := token.NewFileSet()
	for _, file := range []string{"resident.go", "gumbel.go"} {
		f, err := parser.ParseFile(fset, filepath.Join("..", "..", "backend", "cuda", file), nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range f.Decls {
			fn, ok := d.(*ast.FuncDecl)
			if !ok || fn.Recv == nil || !solo[fn.Name.Name] || fn.Body == nil {
				continue
			}
			if star, ok := fn.Recv.List[0].Type.(*ast.StarExpr); !ok || star.X.(*ast.Ident).Name != "cudaResident" {
				continue
			}
			found[fn.Name.Name] = true
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				if sel, ok := n.(*ast.SelectorExpr); ok && banned[sel.Sel.Name] {
					t.Errorf("%s: cudaResident.%s calls %s: a solo shape must take today's launches", fset.Position(sel.Pos()), fn.Name.Name, sel.Sel.Name)
				}
				return true
			})
		}
	}
	for name := range solo {
		if !found[name] {
			t.Errorf("cudaResident.%s not found in resident.go or gumbel.go: the scan would pass vacuously", name)
		}
	}
}
