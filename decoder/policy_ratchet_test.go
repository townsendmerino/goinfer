package decoder

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

const loadedPolicyTestsPath = "../testdata/sb5/loaded_policy_tests.txt"

// policyModelLoaders are the calls by which a decoder test loads a model.
var policyModelLoaders = map[string]bool{
	"Load": true, "LoadWeights": true, "loadWeights": true, "loadBenchModel": true, "loadWithFakeResident": true,
	"loadTestModel": true, "loadFixtureModel": true, "loadWithMC3Fake": true, "loadWithDivergingResident": true,
	"assetPath": true,
}

// TestPolicyTests_loadedRatchet is SB5-P5: the policy tests (spec_*_test.go, speculative*_test.go and
// session*_test.go) that load a model
// are exactly the ones listed in loadedPolicyTestsPath. A new one fails until it is listed, so a policy check that
// could run on the scripted fake does not silently need a checkpoint; a listed one that no longer loads fails until its
// line is removed, so the list only shrinks as tests move to the fake. -update rewrites the list.
func TestPolicyTests_loadedRatchet(t *testing.T) {
	var files []string
	for _, pat := range []string{"spec_*_test.go", "speculative*_test.go", "session*_test.go"} {
		m, err := filepath.Glob(pat)
		if err != nil {
			t.Fatal(err)
		}
		files = append(files, m...)
	}
	if len(files) == 0 {
		t.Fatal("no policy test file found: the scan covered nothing")
	}
	var loaded []string
	for _, f := range files {
		af, err := parser.ParseFile(token.NewFileSet(), f, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range af.Decls {
			fn, ok := d.(*ast.FuncDecl)
			if !ok || fn.Recv != nil || !strings.HasPrefix(fn.Name.Name, "Test") {
				continue
			}
			loads := false
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				if c, ok := n.(*ast.CallExpr); ok {
					if id, ok := c.Fun.(*ast.Ident); ok && policyModelLoaders[id.Name] {
						loads = true
					}
				}
				return !loads
			})
			if loads {
				loaded = append(loaded, f+" "+fn.Name.Name)
			}
		}
	}
	sort.Strings(loaded)
	got := strings.Join(loaded, "\n") + "\n"
	if *updateMatrix {
		if err := os.WriteFile(loadedPolicyTestsPath, []byte(got), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %d loaded policy tests", len(loaded))
		return
	}
	b, err := os.ReadFile(loadedPolicyTestsPath)
	if err != nil {
		t.Fatalf("read %s: %v (run with -update)", loadedPolicyTestsPath, err)
	}
	want := map[string]bool{}
	for _, ln := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		want[ln] = true
	}
	for _, l := range loaded {
		if !want[l] {
			t.Errorf("%s loads a model to check a policy and is not in %s: check the policy on the scripted fake, or list it", l, loadedPolicyTestsPath)
		}
		delete(want, l)
	}
	for l := range want {
		t.Errorf("%s lists %s, which no longer loads a model: remove the line (the list only shrinks)", loadedPolicyTestsPath, l)
	}
	t.Logf("%d policy tests load a model", len(loaded))
}
