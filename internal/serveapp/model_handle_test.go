package serveapp

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"
)

// TestModelHandle_assignedThroughHandleOf holds modelHandle's rule: a loadedModel's model is set only to handleOf(...),
// another handle (x.model) or nil, so a nil *decoder.Model never becomes a non-nil handle; its blockSpec likewise
// only to drafterOf(...) or nil. It reads this package's
// non-test files: every loadedModel literal, and every .model assignment on lm or base, the names this package gives a
// *loadedModel (other types here have a string field called model, which a scan without types cannot tell apart).
func TestModelHandle_assignedThroughHandleOf(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	wrapper := map[string]string{"model": "handleOf", "blockSpec": "drafterOf"}
	okFor := func(field string, e ast.Expr) bool {
		switch v := e.(type) {
		case *ast.CallExpr:
			id, isID := v.Fun.(*ast.Ident)
			return isID && id.Name == wrapper[field]
		case *ast.SelectorExpr:
			return v.Sel.Name == field
		case *ast.Ident:
			return v.Name == "nil"
		}
		return false
	}
	fset := token.NewFileSet()
	checked := 0
	for _, p := range files {
		if strings.HasSuffix(p, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, p, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		ast.Inspect(f, func(n ast.Node) bool {
			switch v := n.(type) {
			case *ast.CompositeLit:
				if id, isID := v.Type.(*ast.Ident); !isID || id.Name != "loadedModel" {
					return true
				}
				for _, el := range v.Elts {
					if kv, isKV := el.(*ast.KeyValueExpr); isKV {
						if k, isID := kv.Key.(*ast.Ident); isID && wrapper[k.Name] != "" {
							checked++
							if !okFor(k.Name, kv.Value) {
								t.Errorf("%s: a loadedModel's %s is set to %s, not %s(...)", fset.Position(kv.Pos()), k.Name, exprString(kv.Value), wrapper[k.Name])
							}
						}
					}
				}
			case *ast.AssignStmt:
				for i, l := range v.Lhs {
					sel, isSel := l.(*ast.SelectorExpr)
					if !isSel || wrapper[sel.Sel.Name] == "" || i >= len(v.Rhs) {
						continue
					}
					if x, isID := sel.X.(*ast.Ident); isID && (x.Name == "lm" || x.Name == "base") {
						checked++
						if !okFor(sel.Sel.Name, v.Rhs[i]) {
							t.Errorf("%s: .%s is assigned %s, not %s(...)", fset.Position(v.Pos()), sel.Sel.Name, exprString(v.Rhs[i]), wrapper[sel.Sel.Name])
						}
					}
				}
			}
			return true
		})
	}
	if checked == 0 {
		t.Fatal("found no loadedModel model assignment: the scan checks nothing")
	}
}

func exprString(e ast.Expr) string {
	switch v := e.(type) {
	case *ast.Ident:
		return v.Name
	case *ast.SelectorExpr:
		return exprString(v.X) + "." + v.Sel.Name
	case *ast.CallExpr:
		return exprString(v.Fun) + "(...)"
	}
	return "an expression"
}

// TestModelHandle_nilModelIsNoHandle: a nil model, as a nil *decoder.Model through handleOf or as a typed nil of any
// implementation, is never retained or released: liveness gains no entry, and a release of it reports nothing to close.
func TestModelHandle_nilModelIsNoHandle(t *testing.T) {
	if h := handleOf(nil); h != nil {
		t.Fatalf("handleOf(nil) = %v, want a nil handle", h)
	}
	var typedNil *fakeModel
	for _, h := range []modelHandle{nil, handleOf(nil), typedNil} {
		if !nilHandle(h) {
			t.Fatalf("nilHandle(%#v) = false", h)
		}
		s := &server{}
		s.retainLocked(h)
		if len(s.liveness) != 0 {
			t.Fatalf("retainLocked(%#v) recorded liveness %v, want none", h, s.liveness)
		}
		if ml, last := s.releaseLocked(h); ml != nil || last {
			t.Fatalf("releaseLocked(%#v) = (%v, %v), want (nil, false)", h, ml, last)
		}
	}
	if nilHandle(&fakeModel{}) {
		t.Fatal("nilHandle of a real (non-nil) handle is true")
	}
}
