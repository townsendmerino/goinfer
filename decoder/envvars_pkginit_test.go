package decoder

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// pkgInitEnvReadsKnown is every production package-level variable initialised from the environment. Such a read happens
// before any test runs, so t.Setenv cannot reach it: a test that sets the variable silently gets the default. Each is
// listed in testdata/env_reads.txt. The list only shrinks: a new setting is a decoder.Options
// field or a load-time knob (docs/env-vars.md, "Policy — where a new setting goes"), and a new diagnostic a test hook.
var pkgInitEnvReadsKnown = map[string]bool{
	"backend/webgpu/internal/kernels/attention.go:AttnKeysDisabled": true,
	"decoder/cpu_gateup_fused.go:cpuFusedGateUp":                    true,
	"decoder/deltanet.go:deltaNetTiming":                            true,
	"decoder/fakequant.go:fakeQuantExpertsOnly":                     true,
	"decoder/fakequant.go:fakeQuantF32Act":                          true,
	"decoder/fakequant.go:fakeQuantScheme":                          true,
	"decoder/forward_granite.go:ssmNoMul":                           true,
	"decoder/forward_granite.go:ssmSkipFFN":                         true,
	"decoder/mamba2.go:ssmQ8CPU":                                    true,
	"decoder/model.go:decodeTiming":                                 true,
	"decoder/weightmat.go:w4a8BatchEnabled":                         true,
	"decoder/weightmat.go:w4a8SplitHalfRepackEnabled":               true,
	"internal/app/serveapp/tools.go:toolUnionEnabled":               true,
}

// readsEnvAtInit reports whether e, a package-level initialiser, reads the environment when the package initialises: a
// call to os.Getenv or os.LookupEnv, or a GOINFER_ literal passed to any call (a helper that reads it). A function
// literal's body runs later, so it is not looked into.
func readsEnvAtInit(e ast.Expr) bool {
	found := false
	ast.Inspect(e, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.FuncLit:
			return false
		case *ast.CallExpr:
			if sel, ok := n.Fun.(*ast.SelectorExpr); ok {
				if x, ok := sel.X.(*ast.Ident); ok && x.Name == "os" && (sel.Sel.Name == "Getenv" || sel.Sel.Name == "LookupEnv") {
					found = true
				}
			}
			for _, a := range n.Args {
				if b, ok := a.(*ast.BasicLit); ok && b.Kind == token.STRING {
					if s, err := strconv.Unquote(b.Value); err == nil && strings.HasPrefix(s, "GOINFER_") {
						found = true
					}
				}
			}
		}
		return !found
	})
	return found
}

// pkgInitEnvReads lists file:var for each package-level var in the non-test Go files under root whose initialiser reads the
// environment.
func pkgInitEnvReads(t *testing.T, root string) []string {
	t.Helper()
	var out []string
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			switch d.Name() {
			case "testdata", "node_modules", ".git", ".claude":
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
			return nil
		}
		f, err := parser.ParseFile(token.NewFileSet(), p, nil, parser.SkipObjectResolution)
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(root, p)
		for _, decl := range f.Decls {
			g, ok := decl.(*ast.GenDecl)
			if !ok || g.Tok != token.VAR {
				continue
			}
			for _, spec := range g.Specs {
				vs := spec.(*ast.ValueSpec)
				for i, v := range vs.Values {
					if readsEnvAtInit(v) {
						name := vs.Names[min(i, len(vs.Names)-1)].Name
						out = append(out, filepath.ToSlash(rel)+":"+name)
					}
				}
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	sort.Strings(out)
	return out
}

// TestEnvVars_noNewPackageInitRead: no production package-level variable reads the environment at initialisation beyond
// pkgInitEnvReadsKnown, and every entry there still does (the list only shrinks). It first proves the check can go red.
func TestEnvVars_noNewPackageInitRead(t *testing.T) {
	plant := filepath.Join(t.TempDir(), "plant")
	if err := os.MkdirAll(plant, 0o755); err != nil {
		t.Fatal(err)
	}
	src := "package p\n\nimport \"os\"\n\nvar viaOS = os.Getenv(\"X\")\nvar viaHelper = flag(\"GOINFER_Y\")\nvar later = func() string { return os.Getenv(\"Z\") }\n\nfunc flag(string) bool { return false }\n"
	if err := os.WriteFile(filepath.Join(plant, "p.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(pkgInitEnvReads(t, plant), ","); got != "p.go:viaHelper,p.go:viaOS" {
		t.Fatalf("VOID: the scan finds %q in the plant, want p.go:viaHelper,p.go:viaOS", got)
	}

	found := pkgInitEnvReads(t, "..")
	seen := map[string]bool{}
	for _, r := range found {
		seen[r] = true
		if !pkgInitEnvReadsKnown[r] {
			t.Errorf("%s reads the environment at package initialisation, where t.Setenv cannot reach it: make it a "+
				"decoder.Options field or a load-time knob, or a test hook (docs/env-vars.md, \"Policy — where a new setting goes\")", r)
		}
	}
	for k := range pkgInitEnvReadsKnown {
		if !seen[k] {
			t.Errorf("%s no longer reads the environment at initialisation: delete it from pkgInitEnvReadsKnown", k)
		}
	}
	if len(found) == 0 {
		t.Fatal("VOID: the scan found no package-level environment read at all")
	}
}
