package mem

import (
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

// thresholdRatchet is each file outside mem that still applies a guard threshold itself, and how many references it makes:
// the backends' own guards, which MM2 moves into mem. A count may only fall, and a file leaves the list when it reaches 0.
var thresholdRatchet = map[string]int{
	"gpu/kv_slots.go":  2,
	"metal/backend.go": 3,
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
// threshold, except the ratcheted backend guards. It goes red on a planted reference first.
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

	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(root, "go.mod")); err != nil {
		t.Fatalf("no repository root at %s", root)
	}
	self, _ := filepath.Abs(".")
	got := map[string]int{}
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
		if n := thresholdUses(f); n > 0 {
			rel, _ := filepath.Rel(root, path)
			got[filepath.ToSlash(rel)] = n
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if read < 500 {
		t.Fatalf("read %d non-test files: a scan that reads little proves nothing", read)
	}
	var files []string
	for k := range got {
		files = append(files, k)
	}
	for k := range thresholdRatchet {
		if _, ok := got[k]; !ok {
			files = append(files, k)
		}
	}
	sort.Strings(files)
	for _, k := range files {
		switch want := thresholdRatchet[k]; {
		case got[k] > want:
			t.Errorf("%s applies a guard threshold %d time(s) (allowed %d): the decision belongs in internal/mem", k, got[k], want)
		case got[k] < want:
			t.Errorf("%s now applies a guard threshold %d time(s), fewer than its ratchet's %d: lower thresholdRatchet", k, got[k], want)
		}
	}
	t.Logf("scanned %d non-test files; ratcheted: %v", read, got)
}
