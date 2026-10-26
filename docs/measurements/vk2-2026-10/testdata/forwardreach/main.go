// forwardreach: the decoder files holding code the forward path reaches (a go/types call graph from model.go's forward,
// forwardFromEmbed, forwardCapture, runLayers and logitsFromHidden; calls through interfaces and func values are not
// followed), each with the parity hash set that covers it, or NONE.
//
//	go run ./docs/measurements/vk2-2026-10/testdata/forwardreach

package main

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

func main() {
	fset := token.NewFileSet()
	paths, _ := filepath.Glob("decoder/*.go")
	var files []*ast.File
	for _, p := range paths {
		if strings.HasSuffix(p, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, p, nil, 0)
		if err != nil {
			panic(err)
		}
		// honor build tags roughly: skip files whose build constraint excludes darwin/arm64 default build
		files = append(files, f)
	}
	conf := types.Config{Importer: importer.ForCompiler(fset, "source", nil), Error: func(error) {}}
	info := &types.Info{Uses: map[*ast.Ident]types.Object{}, Defs: map[*ast.Ident]types.Object{}}
	conf.Check("github.com/townsendmerino/goinfer/decoder", fset, files, info)
	decl := map[types.Object]*ast.FuncDecl{}
	for _, f := range files {
		for _, d := range f.Decls {
			if fd, ok := d.(*ast.FuncDecl); ok && fd.Body != nil {
				if o := info.Defs[fd.Name]; o != nil {
					decl[o] = fd
				}
			}
		}
	}
	file := func(fd *ast.FuncDecl) string { return filepath.Base(fset.Position(fd.Pos()).Filename) }
	var roots []*ast.FuncDecl
	for o, fd := range decl {
		n := o.Name()
		if file(fd) == "model.go" && (n == "forward" || n == "forwardFromEmbed" || n == "forwardCapture" || n == "runLayers" || n == "logitsFromHidden") {
			roots = append(roots, fd)
		}
	}
	seen := map[*ast.FuncDecl]bool{}
	q := roots
	for len(q) > 0 {
		fd := q[0]
		q = q[1:]
		if seen[fd] {
			continue
		}
		seen[fd] = true
		ast.Inspect(fd.Body, func(n ast.Node) bool {
			if id, ok := n.(*ast.Ident); ok {
				if o, ok := info.Uses[id].(*types.Func); ok {
					if t, ok := decl[o]; ok && !seen[t] {
						q = append(q, t)
					}
				}
			}
			return true
		})
	}
	var m struct {
		SharedSets map[string][]string                     `json:"shared_sets"`
		Families   map[string]struct{ Uses, Own []string } `json:"families"`
	}
	b, _ := os.ReadFile("testdata/parity_manifest.json")
	json.Unmarshal(b, &m)
	inSet := map[string]string{}
	for s, fs := range m.SharedSets {
		for _, f := range fs {
			inSet[filepath.Base(f)] += s + " "
		}
	}
	for fam, f := range m.Families {
		for _, o := range f.Own {
			if !strings.Contains(inSet[filepath.Base(o)], "own") {
				inSet[filepath.Base(o)] += "own(" + fam + ") "
			}
		}
	}
	byFile := map[string][]string{}
	for fd := range seen {
		byFile[file(fd)] = append(byFile[file(fd)], fd.Name.Name)
	}
	var fs []string
	for f := range byFile {
		fs = append(fs, f)
	}
	sort.Strings(fs)
	for _, f := range fs {
		sort.Strings(byFile[f])
		set := inSet[f]
		if set == "" {
			set = "NONE"
		}
		fmt.Printf("%-28s %-30s %d funcs\n", f, set, len(byFile[f]))
	}
}
