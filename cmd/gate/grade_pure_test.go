package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// graderForbiddenImports is what a grader file may not import: running, the environment, the clock, the network and
// files are the runner's.
var graderForbiddenImports = map[string]bool{"os": true, "os/exec": true, "time": true, "net": true, "net/http": true,
	"io/ioutil": true, "path/filepath": true, "syscall": true, "runtime": true}

// TestGraders_arePure is SB12-C3: every grade_*.go file in this package imports nothing that runs, reads the
// environment, the clock or files, and every package function or method a grader calls is itself declared in a grader
// file, so a grader cannot reach I/O through a helper. A method or function with the same name as one declared in a
// grader file counts as declared there.
func TestGraders_arePure(t *testing.T) {
	fset := token.NewFileSet()
	names, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	files := map[string]*ast.File{} // the package's non-test files
	for _, name := range names {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, name, nil, parser.ParseComments)
		if err != nil {
			t.Fatal(err)
		}
		files[name] = f
	}
	declaredIn := map[string]string{} // function or method name -> file
	for name, f := range files {
		for _, d := range f.Decls {
			if fn, ok := d.(*ast.FuncDecl); ok {
				declaredIn[fn.Name.Name] = filepath.Base(name)
			}
		}
	}
	isGrader := func(file string) bool {
		return strings.HasPrefix(file, "grade_") && !strings.HasSuffix(file, "_test.go")
	}
	graders := 0
	for name, f := range files {
		base := filepath.Base(name)
		if !isGrader(base) {
			continue
		}
		graders++
		for _, im := range f.Imports {
			p, _ := strconv.Unquote(im.Path.Value)
			if graderForbiddenImports[p] {
				t.Errorf("%s imports %s", base, p)
			}
		}
		for _, d := range f.Decls {
			fn, ok := d.(*ast.FuncDecl)
			if !ok {
				continue
			}
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				var callee string
				switch x := call.Fun.(type) {
				case *ast.Ident:
					callee = x.Name
				case *ast.SelectorExpr:
					if id, ok := x.X.(*ast.Ident); ok && isImportName(f, id.Name) {
						return true // a call into an imported package, which the import list already allows
					}
					callee = x.Sel.Name
				}
				if file, ok := declaredIn[callee]; ok && !isGrader(file) {
					t.Errorf("%s: %s calls %s, declared in %s, which is not a grader file", base, fn.Name.Name, callee, file)
				}
				return true
			})
		}
	}
	if graders == 0 {
		t.Fatal("no grade_*.go file: the scan covered nothing")
	}
	var gs []string
	for name := range files {
		if isGrader(filepath.Base(name)) {
			gs = append(gs, filepath.Base(name))
		}
	}
	sort.Strings(gs)
	t.Logf("grader files: %s", strings.Join(gs, " "))
}

func isImportName(f *ast.File, name string) bool {
	for _, im := range f.Imports {
		p, _ := strconv.Unquote(im.Path.Value)
		local := p[strings.LastIndex(p, "/")+1:]
		if im.Name != nil {
			local = im.Name.Name
		}
		if local == name {
			return true
		}
	}
	return false
}
