package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// sb12CensusPath is SB12's census record; the table in its section 1 names one subcommand per row, in backticks, in the
// first column.
const sb12CensusPath = "../../docs/measurements/sb12-2026-10.md"

// TestGraderCensus_everySubcommand is SB12-C1: the census table names exactly the subcommands run dispatches on, so a new
// gate cannot appear without a row saying where it decides, and a removed one cannot leave a stale row.
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
	row := regexp.MustCompile("(?m)^\\| `([a-z-]+)` \\|")
	inTable := map[string]bool{}
	for _, m := range row.FindAllStringSubmatch(sec, -1) {
		inTable[m[1]] = true
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
