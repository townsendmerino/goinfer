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
	"slices"
	"strings"
	"testing"
)

// loaderRootFuncs are the two load entry points: safetensors and GGUF, before either dispatches to a family.
var loaderRootFuncs = map[string]bool{"loadWeights": true, "loadGGUFWeights": true}

// loadersExempt lists files the load path reaches that the parity key deliberately omits, each with its reason. A file
// that changes what a loader reads, decodes or lays out belongs in a shared set or a family's own list instead.
var loadersExempt = map[string]string{
	"decoder/config/sampling.go":              "reads recommended sampling defaults from generation_config.json; no tensor or forward numerics",
	"internal/runtime/ggufconfig/sampling.go": "GGUF sampling-default metadata; no tensor or forward numerics",
	"decoder/load_errors.go":                  "classifies a load failure for the caller; changes no loaded value",
	"decoder/loadprofile.go":                  "load-time byte counters for diagnostics; changes no loaded value",
}

// TestParityManifest_loadersClosed is the loader twin of TestParityManifest_coreClosed: every function reachable by a
// go/types call graph from loadWeights and loadGGUFWeights (this GOOS and GOARCH, calls through interfaces and func values
// not followed) lives in a file some shared set or family's own list covers, or in loadersExempt. The walk follows calls
// into the module's other packages (decoder/quant, say), so a loader body moved out of decoder is still seen.
func TestParityManifest_loadersClosed(t *testing.T) {
	raw, err := parityManifestJSON()
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
	root, err := filepath.Abs("..")
	if err != nil {
		t.Fatal(err)
	}
	const modPath = "github.com/townsendmerino/goinfer"

	fset := token.NewFileSet()
	imp := importer.ForCompiler(fset, "source", nil)
	type pkg struct {
		info *types.Info
		decl map[string]*ast.FuncDecl // by the position of the declared name
	}
	pkgs := map[string]*pkg{}
	posKey := func(p token.Pos) string { return fset.Position(p).String() }
	rel := func(p token.Pos) string {
		name := fset.Position(p).Filename
		if !filepath.IsAbs(name) {
			name, _ = filepath.Abs(name)
		}
		r, err := filepath.Rel(root, name)
		if err != nil || strings.HasPrefix(r, "..") {
			return ""
		}
		return filepath.ToSlash(r)
	}
	// load type-checks the module package at import path ip (non-test files only) and indexes its function bodies.
	load := func(ip string) *pkg {
		if p, ok := pkgs[ip]; ok {
			return p
		}
		dir := filepath.Join(root, strings.TrimPrefix(strings.TrimPrefix(ip, modPath), "/"))
		ents, err := os.ReadDir(dir)
		if err != nil {
			t.Fatal(err)
		}
		var files []*ast.File
		for _, e := range ents {
			n := e.Name()
			if !strings.HasSuffix(n, ".go") || strings.HasSuffix(n, "_test.go") {
				continue
			}
			if ok, err := build.Default.MatchFile(dir, n); err != nil || !ok {
				continue
			}
			f, err := parser.ParseFile(fset, filepath.Join(dir, n), nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			files = append(files, f)
		}
		info := &types.Info{Uses: map[*ast.Ident]types.Object{}, Defs: map[*ast.Ident]types.Object{}}
		if _, err := (&types.Config{Importer: imp}).Check(ip, fset, files, info); err != nil {
			t.Fatalf("type-checking %s: %v", ip, err)
		}
		p := &pkg{info: info, decl: map[string]*ast.FuncDecl{}}
		for _, f := range files {
			for _, d := range f.Decls {
				if fd, ok := d.(*ast.FuncDecl); ok && fd.Body != nil {
					p.decl[posKey(fd.Name.Pos())] = fd
				}
			}
		}
		pkgs[ip] = p
		return p
	}

	type item struct {
		fd *ast.FuncDecl
		p  *pkg
	}
	via := map[string]string{}
	dec := load(modPath + "/decoder")
	var queue []item
	for _, fd := range dec.decl {
		if fd.Recv == nil && loaderRootFuncs[fd.Name.Name] {
			queue = append(queue, item{fd, dec})
			via[rel(fd.Pos())] = "root " + fd.Name.Name
		}
	}
	if len(queue) != len(loaderRootFuncs) {
		t.Fatalf("found %d of the %d loader roots: the scan would cover nothing", len(queue), len(loaderRootFuncs))
	}
	seen := map[string]bool{}
	for len(queue) > 0 {
		it := queue[0]
		queue = queue[1:]
		k := posKey(it.fd.Name.Pos())
		if seen[k] {
			continue
		}
		seen[k] = true
		ast.Inspect(it.fd.Body, func(n ast.Node) bool {
			id, ok := n.(*ast.Ident)
			if !ok {
				return true
			}
			fn, ok := it.p.info.Uses[id].(*types.Func)
			if !ok || fn.Pkg() == nil || !strings.HasPrefix(fn.Pkg().Path(), modPath) {
				return true
			}
			cp := load(fn.Pkg().Path())
			to := cp.decl[posKey(fn.Pos())]
			if to == nil || seen[posKey(to.Name.Pos())] {
				return true
			}
			if f := rel(to.Pos()); f != "" {
				if _, ok := via[f]; !ok {
					via[f] = rel(it.fd.Pos()) + ":" + it.fd.Name.Name + " calls " + to.Name.Name
				}
			}
			queue = append(queue, item{to, cp})
			return true
		})
	}
	var missing []string
	for f, why := range via {
		if !covered[f] && loadersExempt[f] == "" {
			missing = append(missing, f+" (first reached: "+why+")")
		}
	}
	slices.Sort(missing)
	for _, f := range missing {
		t.Errorf("the load path reaches %s, which no shared set or own list covers and loadersExempt does not list", f)
	}
	for f, why := range loadersExempt {
		if why == "" || covered[f] {
			t.Errorf("loadersExempt[%q]: needs a reason, and must not be a covered file", f)
		}
	}
	t.Logf("the load path reaches %d functions in %d files", len(seen), len(via))
}
