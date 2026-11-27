package main

import (
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
)

const gatePkgPrefix = "github.com/townsendmerino/goinfer/internal/guard/gate"

var graderAllowedImports = map[string]bool{
	"cmp": true, "fmt": true, "io": true, "maps": true, "slices": true, "sort": true, "strings": true, "regexp": true,
	"time": true, "math": true, "bytes": true, "strconv": true, "encoding/binary": true, "encoding/json": true,
}

var graderClockCalls = map[string]bool{
	"Now": true, "Since": true, "Until": true, "Sleep": true, "After": true,
	"AfterFunc": true, "Tick": true, "NewTimer": true, "NewTicker": true,
}

type multiGateImporter struct {
	base types.Importer
	pkgs map[string]*types.Package
}

func (m *multiGateImporter) Import(path string) (*types.Package, error) {
	if p, ok := m.pkgs[path]; ok {
		return p, nil
	}
	return m.base.Import(path)
}

// TestGraders_arePure is SB12-C3 across cmd/gate and internal/guard/gate*: every grade_*.go file imports only
// graderAllowedImports (or another internal/guard/gate* package), calls nothing in time that reads the clock or waits,
// reads no package variable declared outside a grader file, and calls no gate function or method declared outside a grader file.
func TestGraders_arePure(t *testing.T) {
	fset := token.NewFileSet()
	info := &types.Info{Uses: map[*ast.Ident]types.Object{}}
	imp := &multiGateImporter{
		base: importer.ForCompiler(fset, "source", nil),
		pkgs: map[string]*types.Package{},
	}

	type pkgSpec struct {
		importPath string
		dir        string
	}
	var specs []pkgSpec
	subDirs, err := filepath.Glob("../../internal/guard/gate*")
	if err != nil {
		t.Fatal(err)
	}
	slices.Sort(subDirs)
	// Order gatecore first so sibling gate* packages can import it cleanly.
	slices.SortStableFunc(subDirs, func(a, b string) int {
		if filepath.Base(a) == "gatecore" {
			return -1
		}
		if filepath.Base(b) == "gatecore" {
			return 1
		}
		return strings.Compare(a, b)
	})
	for _, d := range subDirs {
		st, err := os.Stat(d)
		if err != nil || !st.IsDir() {
			continue
		}
		specs = append(specs, pkgSpec{
			importPath: "github.com/townsendmerino/goinfer/internal/guard/" + filepath.Base(d),
			dir:        d,
		})
	}
	specs = append(specs, pkgSpec{importPath: "main", dir: "."})

	isGatePkg := func(p *types.Package) bool {
		if p == nil {
			return false
		}
		return p.Path() == "main" || strings.HasPrefix(p.Path(), gatePkgPrefix)
	}
	isGrader := func(file string) bool {
		return strings.HasPrefix(file, "grade_") && !strings.HasSuffix(file, "_test.go")
	}

	var allFiles []*ast.File
	for _, sp := range specs {
		names, err := filepath.Glob(filepath.Join(sp.dir, "*.go"))
		if err != nil {
			t.Fatal(err)
		}
		var files []*ast.File
		for _, name := range names {
			if strings.HasSuffix(name, "_test.go") {
				continue
			}
			f, err := parser.ParseFile(fset, name, nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			files = append(files, f)
		}
		conf := types.Config{Importer: imp}
		pkg, err := conf.Check(sp.importPath, fset, files, info)
		if err != nil {
			t.Fatalf("type-checking %s: %v", sp.importPath, err)
		}
		imp.pkgs[sp.importPath] = pkg
		allFiles = append(allFiles, files...)
	}

	var graders []string
	for _, f := range allFiles {
		fullPath := fset.Position(f.Pos()).Filename
		base := filepath.Base(fullPath)
		if !isGrader(base) {
			continue
		}
		graders = append(graders, filepath.ToSlash(fullPath))
		for _, im := range f.Imports {
			p, _ := strconv.Unquote(im.Path.Value)
			if !graderAllowedImports[p] && !strings.HasPrefix(p, gatePkgPrefix) {
				t.Errorf("%s imports %s, which is not on the grader allowlist", fullPath, p)
			}
		}
		ast.Inspect(f, func(n ast.Node) bool {
			if id, ok := n.(*ast.Ident); ok {
				if v, ok := info.Uses[id].(*types.Var); ok && isGatePkg(v.Pkg()) && v.Parent() == v.Pkg().Scope() {
					if decl := filepath.Base(fset.Position(v.Pos()).Filename); !isGrader(decl) {
						t.Errorf("%s: reads %s, a package variable declared in %s, which is not a grader file", fset.Position(id.Pos()), v.Name(), decl)
					}
				}
			}
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			var id *ast.Ident
			switch x := call.Fun.(type) {
			case *ast.Ident:
				id = x
			case *ast.SelectorExpr:
				id = x.Sel
			}
			if id == nil {
				return true
			}
			fn, ok := info.Uses[id].(*types.Func)
			if ok && fn.Pkg() != nil && fn.Pkg().Path() == "time" && graderClockCalls[fn.Name()] {
				t.Errorf("%s: calls time.%s", fset.Position(call.Pos()), fn.Name())
			}
			if !ok || !isGatePkg(fn.Pkg()) {
				return true
			}
			if decl := filepath.Base(fset.Position(fn.Pos()).Filename); !isGrader(decl) {
				t.Errorf("%s: calls %s, declared in %s, which is not a grader file", fset.Position(call.Pos()), fn.FullName(), decl)
			}
			return true
		})
	}
	if len(graders) == 0 {
		t.Fatal("no grade_*.go file: the scan covered nothing")
	}
	slices.Sort(graders)
	t.Logf("grader files (%d): %s", len(graders), strings.Join(graders, " "))
}
