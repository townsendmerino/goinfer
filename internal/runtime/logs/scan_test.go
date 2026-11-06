package logs

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
)

// libraryRoots are the packages an embedding program links. Each may write only through this package. Left out on purpose: pull's download
// progress and the internal/app command-line front ends (and internal/guard/servecheck, internal/diag/hwreport: subcommands), whose output
// is their interface. cmd directories under a root are binaries, and are skipped.
var libraryRoots = []string{
	"backend/cuda", "backend/metal", "backend/webgpu", "chat", "constrain", "decoder", "embeddinggemma2", "engine", "multimodal", "server",
	"tokenizer", "internal/audio", "internal/batchio", "internal/clef", "internal/codec", "internal/config", "internal/confidence",
	"internal/cost", "internal/decide", "internal/fidelity", "internal/guard/memgate", "internal/guard/paritykey", "internal/guard/swapguard",
	"internal/runtime", "internal/vision", "internal/wrappedupload",
}

// errName is what the scan takes for an error variable, by name (it parses and does not type-check): err and e, a one-letter prefix
// and e (le, ge), and any name containing err or Err.
var errName = regexp.MustCompile(`^[a-z]?e$|^[a-z]*[eE]rr[A-Za-z0-9]*$`)

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

func scanOutput(t *testing.T, file string) {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, file, nil, parser.SkipObjectResolution)
	if err != nil {
		t.Fatal(err)
	}
	logPkgs := map[string]bool{} // the names this file imports log and log/slog under
	for _, im := range f.Imports {
		if p := strings.Trim(im.Path.Value, `"`); p == "log" || p == "log/slog" {
			name := path.Base(p)
			if im.Name != nil {
				name = im.Name.Name
			}
			logPkgs[name] = true
		}
	}
	ast.Inspect(f, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.CallExpr:
			if what := directOutput(x, logPkgs); what != "" {
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

// directOutput names the output a call writes outside this package: print and println, any call into log or log/slog (under
// whatever name the file imports them; a *slog.Logger a library holds comes from one, or from its caller), fmt's printing to
// stdout or a standard stream, and a write to one.
func directOutput(c *ast.CallExpr, logPkgs map[string]bool) string {
	switch f := c.Fun.(type) {
	case *ast.Ident:
		if f.Name == "print" || f.Name == "println" {
			return f.Name
		}
	case *ast.SelectorExpr:
		x, ok := f.X.(*ast.Ident)
		if ok && logPkgs[x.Name] {
			return x.Name + "." + f.Sel.Name
		}
		if ok && x.Name == "fmt" {
			switch f.Sel.Name {
			case "Print", "Printf", "Println":
				return "fmt." + f.Sel.Name
			case "Fprint", "Fprintf", "Fprintln":
				if len(c.Args) > 0 && isStdStream(c.Args[0]) {
					return "fmt." + f.Sel.Name + " to a standard stream"
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

// checkReportAndReturn: a logs call naming an error, then a return of that error, in one statement list. A report inside an if
// statement, in any branch of its else-if chain and at any depth of nested ifs, counts for the statements after it when its own
// branch falls through, unless that if declared the name; assigning the name again clears it.
func checkReportAndReturn(t *testing.T, fset *token.FileSet, list []ast.Stmt) {
	t.Helper()
	reported := map[string]token.Pos{}
	for _, st := range list {
		if s, ok := st.(*ast.ReturnStmt); ok {
			for _, res := range s.Results {
				for _, id := range returnedErrors(res) {
					if pos, hit := reported[id.Name]; hit {
						t.Errorf("%s: reports %s and returns it at line %d: the caller reports it again", fset.Position(pos), id.Name, fset.Position(s.Pos()).Line)
					}
				}
			}
			continue
		}
		stmtReports(st, reported)
	}
}

// stmtReports updates reported for one statement that does not return: a logs call adds its error names, an if adds what its
// falling-through branches report, a block adds what it reports when it falls through, and an assignment clears what it assigns.
func stmtReports(st ast.Stmt, reported map[string]token.Pos) {
	switch s := st.(type) {
	case *ast.ExprStmt:
		for name, pos := range logsReports(s) {
			reported[name] = pos
		}
	case *ast.IfStmt:
		for name, pos := range ifReports(s) {
			reported[name] = pos
		}
	case *ast.BlockStmt:
		for name, pos := range listReports(s.List) {
			reported[name] = pos
		}
	case *ast.AssignStmt:
		for name := range assigned(s, token.ASSIGN, token.DEFINE) {
			delete(reported, name)
		}
	}
}

// listReports is what a statement list has reported, and not assigned again, by its end: nothing when it ends by leaving.
func listReports(list []ast.Stmt) map[string]token.Pos {
	out := map[string]token.Pos{}
	if terminates(list) {
		return out
	}
	for _, st := range list {
		stmtReports(st, out)
	}
	return out
}

// ifReports is what an if statement's branches report when they fall through, through its else-if chain, without the names its
// own init declared.
func ifReports(s *ast.IfStmt) map[string]token.Pos {
	out := listReports(s.Body.List)
	var rest map[string]token.Pos
	switch e := s.Else.(type) {
	case *ast.BlockStmt:
		rest = listReports(e.List)
	case *ast.IfStmt:
		rest = ifReports(e)
	}
	for name, pos := range rest {
		out[name] = pos
	}
	for name := range assigned(s.Init, token.DEFINE) {
		delete(out, name)
	}
	return out
}

// returnedErrors is the names a return result hands back as an error, at any depth of wrapping: the result itself, a composite
// literal's fields (an error type wrapping it), or the arguments of a fmt.Errorf or errors.Join. A name passed to any other call is
// an argument, not the error returned.
func returnedErrors(res ast.Expr) []*ast.Ident {
	if u, ok := res.(*ast.UnaryExpr); ok && u.Op == token.AND {
		res = u.X
	}
	var inner []ast.Expr
	switch x := res.(type) {
	case *ast.Ident:
		return []*ast.Ident{x}
	case *ast.CompositeLit: // an error type wrapping it
		for _, e := range x.Elts {
			if kv, ok := e.(*ast.KeyValueExpr); ok {
				e = kv.Value
			}
			inner = append(inner, e)
		}
	case *ast.CallExpr:
		sel, ok := x.Fun.(*ast.SelectorExpr)
		if !ok {
			return nil
		}
		if pkg, ok := sel.X.(*ast.Ident); !ok || !(pkg.Name == "fmt" && sel.Sel.Name == "Errorf" || pkg.Name == "errors" && sel.Sel.Name == "Join") {
			return nil
		}
		inner = x.Args
	}
	var out []*ast.Ident
	for _, e := range inner {
		out = append(out, returnedErrors(e)...)
	}
	return out
}

// logsReports is the error names a logs call within n passes, where each was reported; function literals are not looked in.
func logsReports(n ast.Node) map[string]token.Pos {
	out := map[string]token.Pos{}
	ast.Inspect(n, func(n ast.Node) bool {
		if _, ok := n.(*ast.FuncLit); ok {
			return false
		}
		c, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		if sel, ok := c.Fun.(*ast.SelectorExpr); ok {
			if x, ok := sel.X.(*ast.Ident); ok && x.Name == "logs" {
				for _, a := range c.Args {
					ast.Inspect(a, func(n ast.Node) bool {
						if id, ok := n.(*ast.Ident); ok && errName.MatchString(id.Name) {
							out[id.Name] = c.Pos()
						}
						return true
					})
				}
			}
		}
		return true
	})
	return out
}

// assigned is the names st, an assignment with one of toks, assigns.
func assigned(st ast.Stmt, toks ...token.Token) map[string]bool {
	out := map[string]bool{}
	if a, ok := st.(*ast.AssignStmt); ok && slices.Contains(toks, a.Tok) {
		for _, l := range a.Lhs {
			if id, ok := l.(*ast.Ident); ok {
				out[id.Name] = true
			}
		}
	}
	return out
}

// terminates reports whether a statement list ends in a return, a panic, or a branch out of it.
func terminates(list []ast.Stmt) bool {
	if len(list) == 0 {
		return false
	}
	switch s := list[len(list)-1].(type) {
	case *ast.ReturnStmt, *ast.BranchStmt:
		return true
	case *ast.ExprStmt:
		if c, ok := s.X.(*ast.CallExpr); ok {
			id, ok := c.Fun.(*ast.Ident)
			return ok && id.Name == "panic"
		}
	}
	return false
}
