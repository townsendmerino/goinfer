package serveapp

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
)

// TestAttachBlockDrafter_declinesWithoutResident gates the startup behaviour an operator
// actually depends on: --drafter on a backend that cannot host one FAILS, with a reason.
//
// The alternative — attaching silently and serving at 1x — is the worse outcome by far: block
// drafting is invisible when it does not engage, because output is identical either way
// (it is lossless by construction). An operator would see correct responses at plain speed and
// have nothing to look at. So the decline is the feature.
func TestAttachBlockDrafter_declinesWithoutResident(t *testing.T) {
	lm := &loadedModel{modelIdentity: modelIdentity{name: "test"}} // no model ⇒ not block-spec capable
	err := attachBlockDrafter(lm, nil, 0)                          // never dereferenced: the resident check declines first
	if err == nil {
		t.Fatal("attachBlockDrafter succeeded with no resident model — it must decline")
	}
	if !strings.Contains(err.Error(), "resident") {
		t.Errorf("decline should name the resident requirement, got: %v", err)
	}
	if lm.blockSpec != nil {
		t.Error("blockSpec set despite the decline")
	}
}

// TestWarnThinkingTemplate: the warning follows the served template's thinking, not the vocabulary: a drafter beside a template
// whose replies can think warns; -thinking off, a checkpoint whose own default is off, no template, and no drafter do not.
func TestWarnThinkingTemplate(t *testing.T) {
	nine := templateFromGolden(t, "qwen3.5-9b") // its template opens a think block by default
	if !nine.WithThinking(chat.ThinkTemplate).ThinkingPossible() {
		t.Fatal("qwen3.5-9b's golden template no longer thinks by default; pick a fixture that does")
	}
	bs := &fakeDrafter{}
	for _, c := range []struct {
		name string
		lm   *loadedModel
		want bool
	}{
		{"drafter, template default on", &loadedModel{modelIdentity: modelIdentity{name: "x", tmpl: nine.WithThinking(chat.ThinkTemplate)}, modelSpecConfig: modelSpecConfig{blockSpec: bs}}, true},
		{"drafter, -thinking on", &loadedModel{modelIdentity: modelIdentity{name: "x", tmpl: nine.WithThinking(chat.ThinkOn)}, modelSpecConfig: modelSpecConfig{blockSpec: bs}}, true},
		{"drafter, -thinking off", &loadedModel{modelIdentity: modelIdentity{name: "x", tmpl: nine.WithThinking(chat.ThinkOff)}, modelSpecConfig: modelSpecConfig{blockSpec: bs}}, false},
		{"drafter, no template", &loadedModel{modelIdentity: modelIdentity{name: "x"}, modelSpecConfig: modelSpecConfig{blockSpec: bs}}, false},
		{"no drafter", &loadedModel{modelIdentity: modelIdentity{name: "x", tmpl: nine.WithThinking(chat.ThinkTemplate)}}, false},
	} {
		if got := warnThinkingTemplate(c.lm); got != c.want {
			t.Errorf("%s: warned %v, want %v", c.name, got, c.want)
		}
	}
}

// TestDrafterWidth_validate: -drafter-width without -drafter is a startup error, not silently ignored, and so are 1 and negative
// widths; 0 and widths of 2 or more pass validation (the attach checks a width against the drafter and the target).
func TestDrafterWidth_validate(t *testing.T) {
	for _, c := range []struct {
		drafter string
		width   int
		ok      bool
	}{
		{"", 0, true}, {"", 4, false}, {"", -1, false},
		{"d", 0, true}, {"d", 4, true}, {"d", 16, true}, {"d", 1, false}, {"d", 2, false}, {"d", 3, false}, {"d", -1, false},
	} {
		// A bare config fails other required flags; only whether a -drafter-width refusal appears is this test's.
		err := (&config{drafter: c.drafter, drafterWidth: c.width}).validate()
		if refused := err != nil && strings.Contains(err.Error(), "-drafter-width"); refused == c.ok {
			t.Errorf("-drafter %q -drafter-width %d: err %v, want a -drafter-width refusal %v", c.drafter, c.width, err, !c.ok)
		}
	}
}

// TestBlockDrafter_assignedThroughDrafterOf holds blockDrafter's rule: a loadedModel's blockSpec is set only to
// drafterOf(...), another blockSpec (x.blockSpec) or nil, so a nil *decoder.BlockSpec never becomes a non-nil
// interface.
func TestBlockDrafter_assignedThroughDrafterOf(t *testing.T) {
	if drafterOf(nil) != nil {
		t.Fatal("drafterOf(nil) is not nil")
	}
	if asBlockSpec(&fakeDrafter{}) != nil {
		t.Fatal("asBlockSpec of a fake is not nil")
	}
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	okFor := func(e ast.Expr) bool {
		switch v := e.(type) {
		case *ast.CallExpr:
			id, isID := v.Fun.(*ast.Ident)
			return isID && id.Name == "drafterOf"
		case *ast.SelectorExpr:
			return v.Sel.Name == "blockSpec"
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
						if k, isID := kv.Key.(*ast.Ident); isID && k.Name == "blockSpec" {
							checked++
							if !okFor(kv.Value) {
								t.Errorf("%s: a loadedModel's blockSpec is set to %s, not drafterOf(...)", fset.Position(kv.Pos()), exprString(kv.Value))
							}
						}
					}
				}
			case *ast.AssignStmt:
				for i, l := range v.Lhs {
					sel, isSel := l.(*ast.SelectorExpr)
					if !isSel || sel.Sel.Name != "blockSpec" || i >= len(v.Rhs) {
						continue
					}
					if x, isID := sel.X.(*ast.Ident); isID && (x.Name == "lm" || x.Name == "base") {
						checked++
						if !okFor(v.Rhs[i]) {
							t.Errorf("%s: .blockSpec is assigned %s, not drafterOf(...)", fset.Position(v.Pos()), exprString(v.Rhs[i]))
						}
					}
				}
			}
			return true
		})
	}
	if checked == 0 {
		t.Fatal("found no loadedModel blockSpec assignment: the scan checks nothing")
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
