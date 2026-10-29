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
var exported = []string{"Decoder", "Engine", "GenerationStream", "NewStream", "Report", "Request", "Scripted", "SessionPool", "Stream"}

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
}

func TestStream_reportIsNilSafeAndCloseRunsOnce(t *testing.T) {
	var none *Stream
	none.Close()
	if none.Report() != nil || none.Tokens() != nil {
		t.Error("a nil Stream has a report or tokens")
	}
	for name, st := range map[string]*Stream{
		"no report function": NewStream(nil, nil, nil),
		"a nil report":       NewStream(nil, func() *Report { return nil }, nil),
		"no generation":      GenerationStream(nil, nil, nil),
	} {
		if st.Report() != nil {
			t.Errorf("%s: Report = %+v, want nil", name, st.Report())
		}
		st.Close()
	}
	want := &Report{Budget: 7}
	closes := 0
	st := NewStream(nil, func() *Report { return want }, func() { closes++ })
	if st.Report() != want {
		t.Error("Report is not the report function's account")
	}
	st.Close()
	st.Close()
	if closes != 1 {
		t.Errorf("done ran %d times over two Closes, want 1", closes)
	}
}

func TestScripted_streamsItsIDsWithinTheBudget(t *testing.T) {
	var seen []Request
	s := Scripted{IDs: []int{5, 6, 7}, OnStart: func(r Request) { seen = append(seen, r) }}
	for _, c := range []struct{ max, want int }{{2, 2}, {3, 3}, {9, 3}, {0, 0}, {-1, 0}} {
		req := Request{MaxTokens: c.max}
		req.Scope.Keep = 4
		st := s.Start(context.Background(), req)
		n := 0
		for range st.Tokens() {
			n++
		}
		st.Close()
		if n != c.want || st.Report() != nil {
			t.Errorf("max %d: %d tokens (want %d), report %v (want nil)", c.max, n, c.want, st.Report())
		}
	}
	if len(seen) != 5 || seen[0].Scope.Keep != 4 {
		t.Errorf("OnStart saw %+v, want five requests carrying their scope", seen)
	}
}
