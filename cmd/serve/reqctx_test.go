package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestDriveUsesRequestContext gates M6 (and the whole class it belongs to): every
// lm.drive / lm.driveVL call in a request path must pass a request-derived context,
// never context.Background(). A detached context means a disconnected client can't
// cancel the decode — the model runs to max_output_tokens holding lm.mu and a queue
// slot, and retries amplify into a DoS. responses.go's tool path was the one caller
// that got this wrong; this scans the whole package so a future one can't regress.
func TestDriveUsesRequestContext(t *testing.T) {
	fset := token.NewFileSet()
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("readdir: %v", err)
	}
	var offenders []string
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, filepath.Join(".", name), nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}
		ast.Inspect(f, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			sel, ok := call.Fun.(*ast.SelectorExpr)
			if !ok || (sel.Sel.Name != "drive" && sel.Sel.Name != "driveVL") {
				return true
			}
			if len(call.Args) == 0 {
				return true
			}
			// First arg is the context. Flag a literal context.Background() call.
			if bg, ok := call.Args[0].(*ast.CallExpr); ok {
				if s, ok := bg.Fun.(*ast.SelectorExpr); ok {
					if pkg, ok := s.X.(*ast.Ident); ok && pkg.Name == "context" && s.Sel.Name == "Background" {
						pos := fset.Position(call.Pos())
						offenders = append(offenders, pos.String()+" "+sel.Sel.Name)
					}
				}
			}
			return true
		})
	}
	if len(offenders) > 0 {
		t.Errorf("drive/driveVL called with context.Background() (client disconnect can't cancel — M6):\n\t%s",
			strings.Join(offenders, "\n\t"))
	}
}
