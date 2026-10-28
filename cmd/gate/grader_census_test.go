package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// sb12CensusPath is SB12's census record; the table in its section 1 names one subcommand per row, in backticks, in the
// first column.
const sb12CensusPath = "../../docs/measurements/sb12-2026-10.md"

// TestGraderCensus_everySubcommand is SB12-C1: the census table names exactly the subcommands run dispatches on, once
// each, so a new gate cannot appear without a row saying where it decides and a removed one cannot leave a stale row;
// and every function a row names as where its gate decides is declared in cmd/gate.
func TestGraderCensus_everySubcommand(t *testing.T) {
	disp := dispatchedSubcommands(t)
	b, err := os.ReadFile(sb12CensusPath)
	if err != nil {
		t.Fatal(err)
	}
	sec, ok := strings.CutPrefix(string(b)[strings.Index(string(b), "\n## 1. ")+1:], "## 1. ")
	if !ok {
		t.Fatalf("%s has no section 1", sb12CensusPath)
	}
	if i := strings.Index(sec, "\n## "); i >= 0 {
		sec = sec[:i]
	}
	row := regexp.MustCompile("(?m)^\\| `([a-z-]+)` \\|[^|]*\\|([^|]*)\\|")
	inTable := map[string]bool{}
	decls := packageDecls(t)
	for _, m := range row.FindAllStringSubmatch(sec, -1) {
		if inTable[m[1]] {
			t.Errorf("%s has two rows for %s", sb12CensusPath, m[1])
		}
		inTable[m[1]] = true
		// The "decides in" column names functions in backticks; each must be declared in this package, or be a
		// qualified name (pkg.Func) of another package.
		for _, fm := range regexp.MustCompile("`([A-Za-z_][A-Za-z0-9_.]*)`").FindAllStringSubmatch(m[2], -1) {
			name := fm[1]
			if strings.Contains(name, ".") {
				continue
			}
			if !decls[name] {
				t.Errorf("the census row for %s names %s as where it decides, and cmd/gate declares no %s", m[1], name, name)
			}
		}
	}
	for _, n := range disp {
		if !inTable[n] {
			t.Errorf("gate %s is dispatched by run but has no row in %s", n, sb12CensusPath)
		}
		delete(inTable, n)
	}
	for n := range inTable {
		t.Errorf("%s has a row for %s, which run does not dispatch", sb12CensusPath, n)
	}
	t.Logf("%d subcommands: %s", len(disp), strings.Join(disp, " "))
}

// dispatchedSubcommands reads run in main.go: every string compared with name (name == "x") or a case label of a switch
// on name.
func dispatchedSubcommands(t *testing.T) []string {
	t.Helper()
	f, err := parser.ParseFile(token.NewFileSet(), "main.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	lit := func(e ast.Expr) {
		if bl, ok := e.(*ast.BasicLit); ok && bl.Kind == token.STRING {
			if s, err := strconv.Unquote(bl.Value); err == nil {
				seen[s] = true
			}
		}
	}
	isName := func(e ast.Expr) bool { id, ok := e.(*ast.Ident); return ok && id.Name == "name" }
	var mapDispatch bool
	for _, d := range f.Decls {
		fn, ok := d.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "run" {
			continue
		}
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			switch x := n.(type) {
			case *ast.BinaryExpr:
				if x.Op == token.EQL && isName(x.X) {
					lit(x.Y)
				}
				if x.Op == token.EQL && isName(x.Y) {
					lit(x.X)
				}
			case *ast.IndexExpr:
				if isName(x.Index) {
					mapDispatch = true // a table keyed by name: the scan cannot list it
				}
			case *ast.SwitchStmt:
				if isName(x.Tag) {
					for _, s := range x.Body.List {
						for _, e := range s.(*ast.CaseClause).List {
							lit(e)
						}
					}
				}
			}
			return true
		})
	}
	if mapDispatch {
		t.Fatal("run indexes a table by name: this scan reads only comparisons and switches on name, so teach it the table")
	}
	if len(seen) == 0 {
		t.Fatal("found no subcommand in main.go's run: the scan covered nothing")
	}
	out := make([]string, 0, len(seen))
	for s := range seen {
		out = append(out, s)
	}
	sort.Strings(out)
	return out
}

// packageDecls is every function and method name declared in this package's non-test files.
func packageDecls(t *testing.T) map[string]bool {
	t.Helper()
	names, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	out := map[string]bool{}
	for _, n := range names {
		if strings.HasSuffix(n, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(token.NewFileSet(), n, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range f.Decls {
			if fn, ok := d.(*ast.FuncDecl); ok {
				out[fn.Name.Name] = true
			}
		}
	}
	return out
}
