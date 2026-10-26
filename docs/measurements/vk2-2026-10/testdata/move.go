// move.go: VK2's mechanical move. Each named top-level declaration (a func, or a method as "(*T).Name"), with its doc
// comment, is cut from its file and appended to a target file in the same package; goimports then fixes both sides'
// imports. Nothing is rewritten: the moved text is byte-for-byte the original.
//
//	go run docs/measurements/vk2-2026-10/testdata/move.go -plan docs/measurements/vk2-2026-10/move-plan.json
//
// plan.json: [{"decl": "llamaArchitecture", "from": "decoder/registry.go", "to": "decoder/arch_llama.go"}, ...]
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strings"
)

type move struct{ Decl, From, To string }

func declName(fset *token.FileSet, fd *ast.FuncDecl) string {
	if fd.Recv == nil || len(fd.Recv.List) == 0 {
		return fd.Name.Name
	}
	var b bytes.Buffer
	printer.Fprint(&b, fset, fd.Recv.List[0].Type)
	return "(" + b.String() + ")." + fd.Name.Name
}

func main() {
	planPath := flag.String("plan", "", "the move plan")
	flag.Parse()
	raw, err := os.ReadFile(*planPath)
	if err != nil {
		fail(err)
	}
	var plan []move
	if err := json.Unmarshal(raw, &plan); err != nil {
		fail(err)
	}
	byFrom := map[string][]move{}
	for _, m := range plan {
		byFrom[m.From] = append(byFrom[m.From], m)
	}
	appended := map[string][]string{}
	touched := map[string]bool{}
	for from, moves := range byFrom {
		src, err := os.ReadFile(from)
		if err != nil {
			fail(err)
		}
		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, from, src, parser.ParseComments)
		if err != nil {
			fail(err)
		}
		type cut struct {
			lo, hi int
			to     string
		}
		var cuts []cut
		for _, m := range moves {
			found := false
			for _, d := range f.Decls {
				fd, ok := d.(*ast.FuncDecl)
				if !ok || declName(fset, fd) != m.Decl {
					continue
				}
				start := fd.Pos()
				if fd.Doc != nil {
					start = fd.Doc.Pos()
				}
				lo, hi := fset.Position(start).Offset, fset.Position(fd.End()).Offset
				for hi < len(src) && src[hi] != '\n' {
					hi++ // a trailing line comment stays with its func
				}
				cuts = append(cuts, cut{lo, hi, m.To})
				found = true
			}
			if !found {
				fail(fmt.Errorf("%s: no declaration %s", from, m.Decl))
			}
		}
		sort.Slice(cuts, func(i, j int) bool { return cuts[i].lo < cuts[j].lo })
		for _, c := range cuts {
			appended[c.to] = append(appended[c.to], string(src[c.lo:c.hi]))
		}
		var out []byte
		prev := 0
		for _, c := range cuts {
			out = append(out, src[prev:c.lo]...)
			prev = c.hi
		}
		out = append(out, src[prev:]...)
		out = regexp.MustCompile(`\n{3,}`).ReplaceAll(out, []byte("\n\n"))
		if err := os.WriteFile(from, out, 0o644); err != nil {
			fail(err)
		}
		touched[from] = true
	}
	for to, decls := range appended {
		var b strings.Builder
		if old, err := os.ReadFile(to); err == nil {
			b.Write(bytes.TrimRight(old, "\n"))
			b.WriteString("\n\n")
		} else {
			b.WriteString("package decoder\n\n")
		}
		b.WriteString(strings.Join(decls, "\n\n"))
		b.WriteString("\n")
		if err := os.WriteFile(to, []byte(b.String()), 0o644); err != nil {
			fail(err)
		}
		touched[to] = true
	}
	var files []string
	for f := range touched {
		files = append(files, f)
	}
	sort.Strings(files)
	cmd := exec.Command("goimports", append([]string{"-w"}, files...)...)
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fail(fmt.Errorf("goimports: %w", err))
	}
	fmt.Printf("moved %d declarations; wrote %s\n", len(plan), strings.Join(files, " "))
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "move:", err)
	os.Exit(1)
}
