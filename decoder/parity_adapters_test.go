package decoder

import (
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"sort"
	"strings"
	"testing"
)

// TestParityManifest_adapterFilesOwned is VK2's census (docs/tasks/task-subsystems-2026-10.md §5.5, A4): every parity
// family's key covers the file holding its registry adapter, and the arch_*.go file of every adapter or helper that
// adapter reaches, so an edit to a family's adapter re-stales that family and an edit to another's does not reach it.
// The adapter is found through the live registry map, the calls by parsing the adapters' files.
func TestParityManifest_adapterFilesOwned(t *testing.T) {
	raw, err := os.ReadFile(parityManifestPath)
	if err != nil {
		t.Fatal(err)
	}
	var m parityManifest
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}

	// Every top-level function in decoder's non-test files, by name, with its file and declaration.
	fset := token.NewFileSet()
	paths, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	declFile := map[string]string{}
	decls := map[string]*ast.FuncDecl{}
	for _, p := range paths {
		if strings.HasSuffix(p, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, p, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range f.Decls {
			if fd, ok := d.(*ast.FuncDecl); ok && fd.Recv == nil && fd.Body != nil {
				declFile[fd.Name.Name] = "decoder/" + p
				decls[fd.Name.Name] = fd
			}
		}
	}
	// reach is the arch_*.go files an adapter's body reaches through calls, transitively through those files' functions.
	reach := func(name string) []string {
		seen, files := map[string]bool{}, map[string]bool{}
		var walk func(string)
		walk = func(n string) {
			fd := decls[n]
			if seen[n] || fd == nil {
				return
			}
			seen[n] = true
			ast.Inspect(fd.Body, func(x ast.Node) bool {
				if id, ok := x.(*ast.Ident); ok && decls[id.Name] != nil && strings.HasPrefix(filepath.Base(declFile[id.Name]), "arch_") {
					files[declFile[id.Name]] = true
					walk(id.Name)
				}
				return true
			})
		}
		walk(name)
		out := make([]string, 0, len(files))
		for f := range files {
			out = append(out, f)
		}
		sort.Strings(out)
		return out
	}

	for fam, fp := range m.Families {
		adapter, ok := registry[strings.ReplaceAll(fam, "-", "_")]
		if !ok {
			t.Errorf("%s: no registry entry for the family (model_type %q)", fam, strings.ReplaceAll(fam, "-", "_"))
			continue
		}
		fn := runtime.FuncForPC(reflect.ValueOf(adapter).Pointer())
		file, _ := fn.FileLine(fn.Entry())
		name := fn.Name()[strings.LastIndex(fn.Name(), ".")+1:]
		covered := familyDepFiles(&m, fp)
		need := append([]string{"decoder/" + filepath.Base(file)}, reach(name)...)
		for _, f := range need {
			if !slices.Contains(covered, f) {
				t.Errorf("%s: %s reaches %s, which the family's key does not cover (add it to its own list)", fam, name, f)
			}
		}
	}
}
