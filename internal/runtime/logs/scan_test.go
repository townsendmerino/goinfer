package logs

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// libraryRoots are the packages an embedding program links. Each may write only through this package. Left out on purpose: pull's download
// progress and the internal/app command-line front ends (and internal/guard/servecheck, internal/diag/hwreport: subcommands), whose output
// is their interface. cmd directories under a root are binaries, and are skipped.
var libraryRoots = []string{
	"backend/cuda", "backend/metal", "backend/webgpu", "chat", "constrain", "decoder", "engine", "multimodal", "tokenizer",
	"internal/audio", "internal/batchio", "internal/clef", "internal/codec", "internal/config", "internal/confidence", "internal/cost",
	"internal/decide", "internal/fidelity", "internal/guard/swapguard", "internal/runtime", "internal/vision",
}

var errName = regexp.MustCompile(`^(err|e)$|^[a-z]*[eE]rr[A-Za-z0-9]*$`)

// Library code reaches stderr, stdout and any logger through this package only, which is what makes decoder.SetLogger true. And no block
// both reports an error and returns it: the caller reports it again, and one failure reads as two.
func TestLibraryOutput_goesThroughLogs(t *testing.T) {
	root := filepath.Join("..", "..", "..")
	scanned := 0
	for _, dir := range libraryRoots {
		err := filepath.WalkDir(filepath.Join(root, dir), func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				if n := d.Name(); n == "cmd" || n == "testdata" {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") || strings.Contains(filepath.ToSlash(path), "internal/runtime/logs/") {
				return nil
			}
			scanned++
			scanOutput(t, path)
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if scanned < 500 {
		t.Fatalf("scanned %d files: this walked the wrong tree", scanned)
	}
}

func scanOutput(t *testing.T, path string) {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, nil, parser.SkipObjectResolution)
	if err != nil {
		t.Fatal(err)
	}
	ast.Inspect(f, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.CallExpr:
			if what := directOutput(x); what != "" {
				t.Errorf("%s: %s: library code writes through internal/runtime/logs", fset.Position(x.Pos()), what)
			}
		case *ast.BlockStmt:
			checkReportAndReturn(t, fset, x.List)
		case *ast.CaseClause:
			checkReportAndReturn(t, fset, x.Body)
		}
		return true
	})
}

func directOutput(c *ast.CallExpr) string {
	switch f := c.Fun.(type) {
	case *ast.Ident:
		if f.Name == "print" || f.Name == "println" {
			return f.Name
		}
	case *ast.SelectorExpr:
		x, ok := f.X.(*ast.Ident)
		if ok {
			switch x.Name {
			case "log", "slog":
				return x.Name + "." + f.Sel.Name
			case "fmt":
				switch f.Sel.Name {
				case "Print", "Printf", "Println":
					return "fmt." + f.Sel.Name
				case "Fprint", "Fprintf", "Fprintln":
					if len(c.Args) > 0 && isStdStream(c.Args[0]) {
						return "fmt." + f.Sel.Name + " to a standard stream"
					}
				}
			}
		}
		if isStdStream(f.X) && strings.HasPrefix(f.Sel.Name, "Write") {
			return "a write to a standard stream"
		}
	}
	return ""
}

func isStdStream(e ast.Expr) bool {
	s, ok := e.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	x, ok := s.X.(*ast.Ident)
	return ok && x.Name == "os" && (s.Sel.Name == "Stderr" || s.Sel.Name == "Stdout")
}

// checkReportAndReturn: a logs call naming an error, then a return of that error, in one statement list.
func checkReportAndReturn(t *testing.T, fset *token.FileSet, list []ast.Stmt) {
	t.Helper()
	reported := map[string]token.Pos{}
	for _, st := range list {
		switch s := st.(type) {
		case *ast.ExprStmt:
			c, ok := s.X.(*ast.CallExpr)
			if !ok {
				continue
			}
			sel, ok := c.Fun.(*ast.SelectorExpr)
			if !ok {
				continue
			}
			if x, ok := sel.X.(*ast.Ident); !ok || x.Name != "logs" {
				continue
			}
			for _, a := range c.Args {
				ast.Inspect(a, func(n ast.Node) bool {
					if id, ok := n.(*ast.Ident); ok && errName.MatchString(id.Name) {
						reported[id.Name] = c.Pos()
					}
					return true
				})
			}
		case *ast.ReturnStmt:
			for _, res := range s.Results {
				ast.Inspect(res, func(n ast.Node) bool {
					if id, ok := n.(*ast.Ident); ok {
						if pos, hit := reported[id.Name]; hit {
							t.Errorf("%s: reports %s and returns it at line %d: the caller reports it again", fset.Position(pos), id.Name, fset.Position(s.Pos()).Line)
						}
					}
					return true
				})
			}
		}
	}
}
