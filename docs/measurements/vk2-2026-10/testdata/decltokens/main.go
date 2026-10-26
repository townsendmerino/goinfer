// decltokens: VK2's A1. For every top-level declaration in a package directory's non-test files (a func or method, or
// each name a var, const or type declares), the SHA-256 of its Go token stream with comments dropped, and the file
// holding it. Two runs are compared by declaration: equal digests mean the code is unchanged wherever it now lives.
//
//	go run ./docs/measurements/vk2-2026-10/testdata/decltokens <dir> > out.json
package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/scanner"
	"go/token"
	"os"
	"path/filepath"
	"strings"
)

type entry struct {
	File   string `json:"file"`
	Digest string `json:"digest"`
}

func main() {
	dir := os.Args[1]
	out := map[string]entry{}
	paths, _ := filepath.Glob(filepath.Join(dir, "*.go"))
	for _, p := range paths {
		if strings.HasSuffix(p, "_test.go") {
			continue
		}
		src, err := os.ReadFile(p)
		if err != nil {
			panic(err)
		}
		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, p, src, 0)
		if err != nil {
			panic(err)
		}
		add := func(key string, n ast.Node) {
			lo, hi := fset.Position(n.Pos()).Offset, fset.Position(n.End()).Offset
			key = f.Name.Name + "." + key
			if _, dup := out[key]; dup {
				key += "@" + filepath.Base(p) // init funcs and blank names
			}
			out[key] = entry{filepath.Base(p), tokens(src[lo:hi])}
		}
		for _, d := range f.Decls {
			switch x := d.(type) {
			case *ast.FuncDecl:
				name := x.Name.Name
				if x.Recv != nil && len(x.Recv.List) > 0 {
					var b bytes.Buffer
					printer.Fprint(&b, fset, x.Recv.List[0].Type)
					name = "(" + b.String() + ")." + name
				}
				add(name, x)
			case *ast.GenDecl:
				if x.Tok == token.IMPORT {
					continue
				}
				for _, s := range x.Specs {
					switch sp := s.(type) {
					case *ast.TypeSpec:
						add("type "+sp.Name.Name, sp)
					case *ast.ValueSpec:
						var names []string
						for _, n := range sp.Names {
							names = append(names, n.Name)
						}
						add(x.Tok.String()+" "+strings.Join(names, ","), sp)
					}
				}
			}
		}
	}
	b, _ := json.MarshalIndent(out, "", " ")
	fmt.Println(string(b))
}

func tokens(src []byte) string {
	fset := token.NewFileSet()
	file := fset.AddFile("", -1, len(src))
	var s scanner.Scanner
	s.Init(file, src, nil, 0) // comments dropped
	h := sha256.New()
	for {
		_, tok, lit := s.Scan()
		if tok == token.EOF {
			break
		}
		fmt.Fprintf(h, "%s %s\n", tok, lit)
	}
	return hex.EncodeToString(h.Sum(nil))[:16]
}
