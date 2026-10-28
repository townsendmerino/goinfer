package main

import (
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// graderAllowedImports is everything a grader file may import: formatting, text, sorting, numbers, decoding and time's
// types.
// Anything else (os, exec, net, filepath, runtime) is the runner's.
var graderAllowedImports = map[string]bool{"fmt": true, "io": true, "sort": true, "strings": true, "regexp": true,
	"time": true, "math": true, "bytes": true, "strconv": true, "encoding/binary": true, "encoding/json": true}

// graderClockCalls is the time package's functions that read the clock or wait; a grader may use time's types only.
var graderClockCalls = map[string]bool{"Now": true, "Since": true, "Until": true, "Sleep": true, "After": true,
	"AfterFunc": true, "Tick": true, "NewTimer": true, "NewTicker": true}

// TestGraders_arePure is SB12-C3: every grade_*.go file in this package imports only graderAllowedImports, calls
// nothing in time that reads the clock or waits, reads no package variable declared outside a grader file, and every package function or method a grader calls is itself declared in a grader
// file, so a grader cannot reach I/O through a helper. Calls are resolved by type-checking the package, so a method is
// told from a function of the same name.
func TestGraders_arePure(t *testing.T) {
	fset := token.NewFileSet()
	names, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	var files []*ast.File // the package's non-test files
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
	info := &types.Info{Uses: map[*ast.Ident]types.Object{}}
	conf := types.Config{Importer: importer.ForCompiler(fset, "source", nil)}
	if _, err := conf.Check("main", fset, files, info); err != nil {
		t.Fatalf("type-checking cmd/gate: %v", err)
	}
	isGrader := func(file string) bool {
		return strings.HasPrefix(file, "grade_") && !strings.HasSuffix(file, "_test.go")
	}
	var graders []string
	for _, f := range files {
		base := filepath.Base(fset.Position(f.Pos()).Filename)
		if !isGrader(base) {
			continue
		}
		graders = append(graders, base)
		for _, im := range f.Imports {
			p, _ := strconv.Unquote(im.Path.Value)
			if !graderAllowedImports[p] {
				t.Errorf("%s imports %s, which is not on the grader allowlist", base, p)
			}
		}
		ast.Inspect(f, func(n ast.Node) bool {
			if id, ok := n.(*ast.Ident); ok {
				// A package-level variable declared outside a grader file is state the record does not carry.
				if v, ok := info.Uses[id].(*types.Var); ok && v.Pkg() != nil && v.Pkg().Path() == "main" && v.Parent() == v.Pkg().Scope() {
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
			if !ok || fn.Pkg() == nil || fn.Pkg().Path() != "main" {
				return true // a builtin, a conversion, or a function of an imported package, which the import list governs
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
	sort.Strings(graders)
	t.Logf("grader files: %s", strings.Join(graders, " "))
}
