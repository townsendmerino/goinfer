package mem

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// guardThresholds are the names a guard threshold goes by, each mapped to the threshold it is: mem's constants and Policy
// fields, and every alias of them in the tree.
var guardThresholds = map[string]string{
	"WeightsFraction": "WeightsFraction", "WarnRatio": "WarnRatio", "CtxFloor": "CtxFloor", "GIWMargin": "GIWMargin",
	"PlanCtxFloor": "PlanCtxFloor", "MetalFitMargin": "MetalFitMargin", "RequestFraction": "RequestFraction",
	"WeightsMemFraction": "WeightsFraction", "fitMemFraction": "WeightsFraction", "prefillAvailFraction": "WeightsFraction",
	"residentMemFraction": "WeightsFraction", "giwMemMargin": "GIWMargin", "MetalFitMarginBytes": "MetalFitMargin",
	"ctxPlanFloor": "PlanCtxFloor", "fitWarnRatio": "WarnRatio", "ctxFloor": "CtxFloor",
}

// thresholdRatchet is each file outside mem that still applies a guard threshold itself, and how many references it makes.
// It is empty: every guard's arithmetic is in mem. A count may only fall, and a file leaves the list when it reaches 0.
var thresholdRatchet = map[string]int{}

// thresholdRules are mem's functions that apply a guard threshold to a figure their caller supplies (RAMShare takes
// WeightsFraction of any byte count), so calling one is applying the threshold. ruleCallers is every file outside mem
// allowed to, with its exact number of calls: a file not listed, or a count above or below its entry, is red.
var thresholdRules = map[string]bool{"RAMShare": true, "WeightsPercent": true}

var ruleCallers = map[string]int{
	"backend/metal/backend.go":   3, // metalStaticCeiling and the two refusal messages
	"backend/webgpu/kv_slots.go": 3, // darwinKVSlots, darwinSlotsFit and buildKVSlots' clamp log line
}

// ruleUses counts f's references to a threshold rule through a package selector (mem.RAMShare), calls and function
// values alike.
func ruleUses(f *ast.File) int {
	n := 0
	ast.Inspect(f, func(x ast.Node) bool {
		if s, ok := x.(*ast.SelectorExpr); ok && thresholdRules[s.Sel.Name] {
			if _, pkg := s.X.(*ast.Ident); pkg {
				n++
			}
		}
		return true
	})
	return n
}

// countsBreak lists how got (references per file) breaks want (the counts each file is allowed): a file over its count or
// not listed, and a file under its count. what names the references, fix the remedy for one under its entry.
func countsBreak(got, want map[string]int, what, fix string) []string {
	var files []string
	for k := range got {
		files = append(files, k)
	}
	for k := range want {
		if _, ok := got[k]; !ok {
			files = append(files, k)
		}
	}
	sort.Strings(files)
	var out []string
	for _, k := range files {
		switch w := want[k]; {
		case got[k] > w:
			out = append(out, fmt.Sprintf("%s %s %d time(s) (allowed %d): the decision belongs in internal/runtime/mem", k, what, got[k], w))
		case got[k] < w:
			out = append(out, fmt.Sprintf("%s now %s %d time(s), fewer than its entry's %d: %s", k, what, got[k], w, fix))
		}
	}
	return out
}

// thresholdUses counts the references to a guard threshold in f, less the ones in a constant declaration that only aliases
// one under a name for the same threshold (const fitMemFraction = mem.WeightsFraction). An alias of one threshold under
// another's name (const ctxPlanFloor = ctxFloor) counts.
func thresholdUses(f *ast.File) int {
	alias := map[*ast.Ident]bool{}
	for _, d := range f.Decls {
		g, ok := d.(*ast.GenDecl)
		if !ok || g.Tok != token.CONST {
			continue
		}
		for _, sp := range g.Specs {
			vs := sp.(*ast.ValueSpec)
			if len(vs.Names) != len(vs.Values) {
				continue
			}
			pure := true
			var ids []*ast.Ident
			for i, n := range vs.Names {
				var v *ast.Ident
				switch x := vs.Values[i].(type) {
				case *ast.Ident:
					v = x
				case *ast.SelectorExpr:
					v = x.Sel
				}
				if v == nil || guardThresholds[n.Name] == "" || guardThresholds[n.Name] != guardThresholds[v.Name] {
					pure = false
					break
				}
				ids = append(ids, n, v)
			}
			if pure {
				for _, id := range ids {
					alias[id] = true
				}
			}
		}
	}
	n := 0
	ast.Inspect(f, func(x ast.Node) bool {
		if id, ok := x.(*ast.Ident); ok && guardThresholds[id.Name] != "" && !alias[id] {
			n++
		}
		return true
	})
	return n
}

// TestNoGuardThresholdOutsideMem (MM1 gate G-MM1c-3, I6 as far as MM1 reaches): no file outside this package applies a guard
// threshold. It goes red on a planted reference first.
func TestNoGuardThresholdOutsideMem(t *testing.T) {
	plant := "package x\nimport \"github.com/townsendmerino/goinfer/decoder\"\nconst fitMemFraction = decoder.WeightsMemFraction\nvar budget = int64(decoder.WeightsMemFraction * 8)\n"
	f, err := parser.ParseFile(token.NewFileSet(), "plant.go", plant, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got := thresholdUses(f); got != 1 {
		t.Fatalf("the planted file counts %d references, want 1 (the alias exempt, the arithmetic counted)", got)
	}
	cross := "package x\nconst ctxPlanFloor = ctxFloor\nvar p = mem.DefaultPolicy().RequestFraction\n"
	if f, err = parser.ParseFile(token.NewFileSet(), "cross.go", cross, 0); err != nil {
		t.Fatal(err)
	}
	if got := thresholdUses(f); got != 3 {
		t.Fatalf("the planted cross-alias and Policy field read count %d references, want 3", got)
	}
	rule := "package x\nimport \"github.com/townsendmerino/goinfer/internal/runtime/mem\"\nvar b = mem.RAMShare(8)\nvar pct = mem.WeightsPercent\n"
	if f, err = parser.ParseFile(token.NewFileSet(), "rule.go", rule, 0); err != nil {
		t.Fatal(err)
	}
	if got := ruleUses(f); got != 2 {
		t.Fatalf("the planted RAMShare call and WeightsPercent value count %d rule references, want 2", got)
	}
	planted := map[string]int{"x/rule.go": 2}
	for k, v := range ruleCallers {
		planted[k] = v
	}
	if bad := countsBreak(planted, ruleCallers, "applies a threshold rule", "lower ruleCallers"); len(bad) != 1 || !strings.Contains(bad[0], "x/rule.go") {
		t.Fatalf("a caller outside ruleCallers was not refused: %v", bad)
	}

	root, err := filepath.Abs("../../..")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(root, "go.mod")); err != nil {
		t.Fatalf("no repository root at %s", root)
	}
	self, _ := filepath.Abs(".")
	got, rules := map[string]int{}, map[string]int{}
	read := 0
	err = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			n := d.Name()
			if path != root && (n == "testdata" || n == "node_modules" || strings.HasPrefix(n, ".")) {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") || filepath.Dir(path) == self {
			return nil
		}
		f, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
		if err != nil {
			return err
		}
		read++
		rel, _ := filepath.Rel(root, path)
		if n := thresholdUses(f); n > 0 {
			got[filepath.ToSlash(rel)] = n
		}
		if n := ruleUses(f); n > 0 {
			rules[filepath.ToSlash(rel)] = n
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if read < 500 {
		t.Fatalf("read %d non-test files: a scan that reads little proves nothing", read)
	}
	for _, b := range countsBreak(got, thresholdRatchet, "applies a guard threshold", "lower thresholdRatchet") {
		t.Error(b)
	}
	for _, b := range countsBreak(rules, ruleCallers, "applies a threshold rule", "lower ruleCallers") {
		t.Error(b)
	}
	t.Logf("scanned %d non-test files; ratcheted: %v; rule callers: %v", read, got, rules)
}
