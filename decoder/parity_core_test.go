package decoder

import (
	"encoding/json"
	"go/ast"
	"go/build"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// forwardEntryFiles hold the forward's entry points: the per-token forward (model_forward.go) and the batched one
// (forwardn.go). Both are in core.
var forwardEntryFiles = map[string]bool{"decoder/model_forward.go": true, "decoder/forwardn.go": true}

// TestParityManifest_coreClosed is VK2's forward census (docs/tasks/task-subsystems-2026-10.md §5.5, B2): every decoder
// function the forward reaches, by a go/types call graph from the functions in forwardEntryFiles over the files this
// GOOS and GOARCH build, lives in a file some shared set or some family's own list covers, so an edit to code the forward
// runs re-stales the families that run it. A reached file's other functions are not roots: a file that mixes forward and
// plumbing code is covered for its forward code only. Calls through interfaces and func values are not followed.
func TestParityManifest_coreClosed(t *testing.T) {
	raw, err := os.ReadFile(parityManifestPath)
	if err != nil {
		t.Fatal(err)
	}
	var m parityManifest
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	covered := map[string]bool{}
	for _, files := range m.SharedSets {
		for _, f := range files {
			covered[f] = true
		}
	}
	for _, fam := range m.Families {
		for _, f := range fam.Own {
			covered[f] = true
		}
	}
	core := map[string]bool{}
	for _, f := range m.SharedSets["core"] {
		core[f] = true
	}

	fset := token.NewFileSet()
	ents, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	var files []*ast.File
	for _, e := range ents {
		n := e.Name()
		if !strings.HasSuffix(n, ".go") || strings.HasSuffix(n, "_test.go") {
			continue
		}
		if ok, err := build.Default.MatchFile(".", n); err != nil || !ok {
			continue
		}
		f, err := parser.ParseFile(fset, n, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		files = append(files, f)
	}
	info := &types.Info{Uses: map[*ast.Ident]types.Object{}, Defs: map[*ast.Ident]types.Object{}}
	conf := types.Config{Importer: importer.ForCompiler(fset, "source", nil)}
	if _, err := conf.Check("github.com/townsendmerino/goinfer/decoder", fset, files, info); err != nil {
		t.Fatalf("type-checking decoder: %v", err)
	}
	decl := map[types.Object]*ast.FuncDecl{}
	for _, f := range files {
		for _, d := range f.Decls {
			if fd, ok := d.(*ast.FuncDecl); ok && fd.Body != nil {
				decl[info.Defs[fd.Name]] = fd
			}
		}
	}
	file := func(fd *ast.FuncDecl) string { return "decoder/" + filepath.Base(fset.Position(fd.Pos()).Filename) }

	var queue []*ast.FuncDecl
	for _, fd := range decl {
		if forwardEntryFiles[file(fd)] {
			queue = append(queue, fd)
		}
	}
	if len(queue) == 0 {
		t.Fatal("no function in a forward entry file: the scan covered nothing")
	}
	for f := range forwardEntryFiles {
		if !core[f] {
			t.Errorf("forward entry file %s is not in core", f)
		}
	}
	seen := map[*ast.FuncDecl]bool{}
	via := map[string]string{}
	for len(queue) > 0 {
		fd := queue[0]
		queue = queue[1:]
		if seen[fd] {
			continue
		}
		seen[fd] = true
		ast.Inspect(fd.Body, func(n ast.Node) bool {
			id, ok := n.(*ast.Ident)
			if !ok {
				return true
			}
			fn, ok := info.Uses[id].(*types.Func)
			if !ok {
				return true
			}
			if to := decl[fn]; to != nil && !seen[to] {
				if _, ok := via[file(to)]; !ok {
					via[file(to)] = file(fd) + ":" + fd.Name.Name + " calls " + to.Name.Name
				}
				queue = append(queue, to)
			}
			return true
		})
	}
	var missing []string
	for f, why := range via {
		if !covered[f] {
			missing = append(missing, f+" (first reached: "+why+")")
		}
	}
	sort.Strings(missing)
	for _, f := range missing {
		t.Errorf("core reaches %s, which no shared set or own list covers", f)
	}
	t.Logf("core reaches %d functions in %d files", len(seen), len(via))
}
