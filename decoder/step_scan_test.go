package decoder

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"
)

// stepShapes are the forward-shape interfaces step resolves: asserting one of them on a resident is stepDo's job alone.
var stepShapes = map[string]bool{
	"Prefiller": true, "ResidentGreedy": true, "ResidentSample": true, "ResidentTopK": true, "ResidentPrefillKV": true,
	"ResidentHiddenLast": true, "ResidentResidualAll": true, "ResidentMRoPE": true, "ResidentMRoPEPrefill": true,
	"ResidentMRoPEDeepstackPrefill": true, "ResidentHybridMRoPEPrefill": true, "ResidentImagePrefill": true,
	"ResidentImageBlocksPrefill": true, "ResidentSpanPrefill": true, "ResidentPrefillRows": true, "ResidentPrefillMarked": true,
	"ResidentSeedArgmax": true, "ResidentBatchStepper": true,
}

// TestStepShapes_assertedOnlyInStep refuses a type assertion or type-switch case naming a forward-shape interface in any
// non-test decoder file but step.go, whatever its build tags: every caller asks canStep and calls step, so the call shape has one
// place to change.
func TestStepShapes_assertedOnlyInStep(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	read := 0
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") || f == "step.go" {
			continue
		}
		af, err := parser.ParseFile(fset, f, nil, parser.SkipObjectResolution)
		if err != nil {
			t.Fatalf("%s: %v", f, err)
		}
		read++
		name := func(e ast.Expr) string {
			if id, ok := e.(*ast.Ident); ok {
				return id.Name
			}
			return ""
		}
		ast.Inspect(af, func(n ast.Node) bool {
			switch x := n.(type) {
			case *ast.TypeAssertExpr:
				if x.Type != nil && stepShapes[name(x.Type)] {
					t.Errorf("%s: asserts %s; ask canStep and call step (decoder/step.go)", fset.Position(x.Pos()), name(x.Type))
				}
			case *ast.TypeSwitchStmt:
				for _, c := range x.Body.List {
					for _, e := range c.(*ast.CaseClause).List {
						if stepShapes[name(e)] {
							t.Errorf("%s: switches on %s; ask canStep and call step (decoder/step.go)", fset.Position(e.Pos()), name(e))
						}
					}
				}
			}
			return true
		})
	}
	if read == 0 {
		t.Fatal("no decoder sources read")
	}
}
