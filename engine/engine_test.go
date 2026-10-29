package engine

import (
	"context"
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// exported is the package's whole exported surface, top-level names only. A new name is a public API change, made
// here on purpose.
var exported = []string{"Decoder", "Engine", "Report", "ReportOf", "Reporter", "Request", "Scripted", "SessionPool"}

func TestSurface_exportsOnlyItsContract(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	var got []string
	for _, p := range files {
		if strings.HasSuffix(p, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, p, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range f.Decls {
			switch d := d.(type) {
			case *ast.FuncDecl:
				if d.Recv == nil && d.Name.IsExported() {
					got = append(got, d.Name.Name)
				}
			case *ast.GenDecl:
				for _, sp := range d.Specs {
					switch sp := sp.(type) {
					case *ast.TypeSpec:
						if sp.Name.IsExported() {
							got = append(got, sp.Name.Name)
						}
					case *ast.ValueSpec:
						for _, n := range sp.Names {
							if n.IsExported() {
								got = append(got, n.Name)
							}
						}
					}
				}
			}
		}
	}
	slices.Sort(got)
	if !slices.Equal(got, exported) {
		t.Errorf("engine exports %v, want %v", got, exported)
	}
}

func TestReport_BudgetFor(t *testing.T) {
	for _, c := range []struct {
		name      string
		r         *Report
		requested int
		want      int
	}{
		{"clamped below request", &Report{Budget: 96, BudgetClamped: true}, 512, 96},
		{"clamped to zero", &Report{Budget: 0, BudgetClamped: true}, 512, 0},
		{"unclamped zero", &Report{Budget: 0}, 512, 512},
		{"nil report", nil, 512, 512},
	} {
		if got := c.r.BudgetFor(c.requested); got != c.want {
			t.Errorf("%s: BudgetFor = %d, want %d", c.name, got, c.want)
		}
	}
	if ReportOf(nil) != nil || Reporter(nil)() != nil {
		t.Error("no generation gives a non-nil report")
	}
}

func TestScripted_streamsItsIDsWithinTheBudget(t *testing.T) {
	var seen []Request
	s := Scripted{IDs: []int{5, 6, 7}, OnStart: func(r Request) { seen = append(seen, r) }}
	for _, c := range []struct{ max, want int }{{2, 2}, {3, 3}, {9, 3}, {0, 0}, {-1, 0}} {
		req := Request{MaxTokens: c.max}
		req.Scope.Keep = 4
		tokens, report, done := s.Start(context.Background(), req)
		n := 0
		for range tokens {
			n++
		}
		done()
		if n != c.want || report() != nil {
			t.Errorf("max %d: %d tokens (want %d), report %v (want nil)", c.max, n, c.want, report())
		}
	}
	if len(seen) != 5 || seen[0].Scope.Keep != 4 {
		t.Errorf("OnStart saw %+v, want five requests carrying their scope", seen)
	}
}
