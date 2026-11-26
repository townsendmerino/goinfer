//go:build quality_scan

package qualityscan

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"path"
	"strings"
	"testing"
)

// decoderCensus is the size of one package directory, read from its files by the parser alone, every build tag at once.
type decoderCensus struct {
	Lines        int // non-test file lines
	ModelMethods int // methods with a body on Model, in non-test files
	KnotLines    int // declaration lines of the non-test declarations that reach Model and that Model reaches
}

// The decoder ceilings: the package's non-test lines, Model's methods and the Model knot, each held at its exact count in
// testdata/decoder.ceilings. A rise fails until its line is raised with a reason; a fall fails until its line is lowered,
// so each fall is recorded in the change that made it.
func TestQualityScan_decoderCeilings(t *testing.T) {
	tr := loadTree(t)
	c := censusOf(tr.Fset, tr.Files, "decoder")
	allow, err := ReadAllowlist("testdata/decoder.ceilings")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("decoder: %d non-test lines, %d Model methods, %d Model knot lines", c.Lines, c.ModelMethods, c.KnotLines)
	for _, e := range ceilingFailures(c, allow) {
		t.Error(e)
	}
}

// ceilingFailures lists every census number that differs from its line in allow, or has none.
func ceilingFailures(c decoderCensus, allow map[string]Entry) []string {
	var out []string
	found := map[string]int{"lines": c.Lines, "model_methods": c.ModelMethods, "model_knot_lines": c.KnotLines}
	for k := range found {
		if _, ok := allow[k]; !ok {
			out = append(out, "testdata/decoder.ceilings has no line for "+k)
		}
	}
	grown, stale := CompareExact(found, allow)
	if len(grown) > 0 {
		out = append(out, "decoder grew past a ceiling (shrink it back, or raise the line with a one-line reason):\n\t"+strings.Join(grown, "\n\t"))
	}
	if len(stale) > 0 {
		out = append(out, "decoder fell under a ceiling (lower the line in testdata/decoder.ceilings to the new count):\n\t"+strings.Join(stale, "\n\t"))
	}
	return out
}

// censusNode is one package-level declaration: a function, a method (named Recv.Name), a type, a var or a const.
type censusNode struct {
	kind  string
	lines int
	refs  map[string]bool
}

// censusOf reads the non-test files directly in dir (not its subdirectories) that belong to the package named after it. An
// edge is "this declaration's text names that package-level name", resolved by the parser's scopes; a type and its methods
// are tied both ways. The knot is computed with each method folded into its type.
func censusOf(fset *token.FileSet, files []*File, dir string) decoderCensus {
	var c decoderCensus
	nodes := map[string]*censusNode{}
	methodsOf := map[string][]string{}
	span := func(from, to token.Pos) int { return fset.Position(to).Line - fset.Position(from).Line + 1 }
	get := func(name, kind string) *censusNode {
		n := nodes[name]
		if n == nil {
			n = &censusNode{kind: kind, refs: map[string]bool{}}
			nodes[name] = n
		}
		return n
	}
	for _, f := range files {
		if f.Test || path.Dir(f.Path) != dir || f.AST.Name.Name != path.Base(dir) {
			continue
		}
		base := path.Base(f.Path)
		c.Lines += FileLines(f.Src)
		collect := censusCollector(f.AST)
		for _, d := range f.AST.Decls {
			switch d := d.(type) {
			case *ast.FuncDecl:
				from := d.Pos()
				if d.Doc != nil {
					from = d.Doc.Pos()
				}
				var n *censusNode
				if d.Recv != nil && len(d.Recv.List) > 0 {
					rb := recvBase(d.Recv.List[0].Type)
					n = get(rb+"."+d.Name.Name, "method")
					n.refs[rb] = true
					methodsOf[rb] = append(methodsOf[rb], rb+"."+d.Name.Name)
					if rb == "Model" && d.Body != nil {
						c.ModelMethods++
					}
				} else {
					name := d.Name.Name
					if name == "init" {
						name = "init@" + base
					}
					n = get(name, "func")
				}
				n.lines += span(from, d.End())
				collect(d, n.refs)
			case *ast.GenDecl:
				var prevRefs map[string]bool
				for i, s := range d.Specs {
					from, to := s.Pos(), s.End()
					if !d.Lparen.IsValid() {
						from, to = d.Pos(), d.End()
						if d.Doc != nil {
							from = d.Doc.Pos()
						}
					}
					switch s := s.(type) {
					case *ast.TypeSpec:
						if s.Doc != nil && d.Lparen.IsValid() {
							from = s.Doc.Pos()
						}
						n := get(s.Name.Name, "type")
						n.lines += span(from, to)
						collect(s, n.refs)
					case *ast.ValueSpec:
						if s.Doc != nil && d.Lparen.IsValid() {
							from = s.Doc.Pos()
						}
						refs := map[string]bool{}
						if s.Type != nil {
							collect(s.Type, refs)
						}
						for _, v := range s.Values {
							collect(v, refs)
						}
						if d.Tok == token.CONST && s.Type == nil && len(s.Values) == 0 && prevRefs != nil {
							for k := range prevRefs {
								refs[k] = true // an implicit repetition of the const spec above names what it names
							}
						}
						prevRefs = refs
						for j, nm := range s.Names {
							name := nm.Name
							if name == "_" {
								name = fmt.Sprintf("_@%s#%d.%d.%d", base, fset.Position(d.Pos()).Line, i, j)
							}
							n := get(name, d.Tok.String())
							if j == 0 {
								n.lines += span(from, to)
							}
							for k := range refs {
								n.refs[k] = true
							}
						}
					}
				}
			}
		}
	}
	for name, n := range nodes {
		for r := range n.refs {
			if nodes[r] == nil || r == name {
				delete(n.refs, r)
			}
		}
	}
	for t, ms := range methodsOf {
		if tn := nodes[t]; tn != nil {
			for _, m := range ms {
				tn.refs[m] = true
			}
		}
	}

	group := func(name string) string {
		if n := nodes[name]; n != nil && n.kind == "method" {
			if t, _, _ := strings.Cut(name, "."); nodes[t] != nil {
				return t
			}
		}
		return name
	}
	fwd, rev := map[string]map[string]bool{}, map[string]map[string]bool{}
	glines := map[string]int{}
	for name, n := range nodes {
		g := group(name)
		glines[g] += n.lines
		for r := range n.refs {
			if nodes[r] != nil {
				if rg := group(r); rg != g {
					if fwd[g] == nil {
						fwd[g] = map[string]bool{}
					}
					if rev[rg] == nil {
						rev[rg] = map[string]bool{}
					}
					fwd[g][rg], rev[rg][g] = true, true
				}
			}
		}
	}
	if _, ok := glines["Model"]; !ok {
		return c
	}
	down, up := closureOf("Model", fwd), closureOf("Model", rev)
	for g := range down {
		if up[g] {
			c.KnotLines += glines[g]
		}
	}
	return c
}

// censusCollector returns a function that adds to into every package-level name a node's text names: an identifier the
// parser resolved to the file's scope, or one it left unresolved that is not an import (a name declared in another file).
// A bare key in a composite literal is never one: the parser resolves it only when the name is declared in the same file,
// so counting it would let a move between files change the census.
func censusCollector(f *ast.File) func(ast.Node, map[string]bool) {
	imports := map[string]bool{}
	for _, is := range f.Imports {
		ip := strings.Trim(is.Path.Value, `"`)
		nm := ip[strings.LastIndex(ip, "/")+1:]
		if is.Name != nil {
			nm = is.Name.Name
		}
		imports[nm] = true
	}
	unres := map[*ast.Ident]bool{}
	for _, id := range f.Unresolved {
		unres[id] = true
	}
	return func(n ast.Node, into map[string]bool) {
		keys := map[*ast.Ident]bool{}
		ast.Inspect(n, func(x ast.Node) bool {
			if cl, ok := x.(*ast.CompositeLit); ok {
				for _, e := range cl.Elts {
					if kv, ok := e.(*ast.KeyValueExpr); ok {
						if id, ok := kv.Key.(*ast.Ident); ok {
							keys[id] = true
						}
					}
				}
			}
			return true
		})
		ast.Inspect(n, func(x ast.Node) bool {
			id, ok := x.(*ast.Ident)
			if !ok || id.Name == "_" || keys[id] {
				return true
			}
			if id.Obj != nil {
				if f.Scope.Objects[id.Name] == id.Obj {
					into[id.Name] = true
				}
			} else if unres[id] && !imports[id.Name] {
				into[id.Name] = true
			}
			return true
		})
	}
}

// recvBase is the name of a receiver's type, through pointers, parentheses and type parameters.
func recvBase(e ast.Expr) string {
	for {
		switch t := e.(type) {
		case *ast.StarExpr:
			e = t.X
		case *ast.ParenExpr:
			e = t.X
		case *ast.IndexExpr:
			e = t.X
		case *ast.IndexListExpr:
			e = t.X
		case *ast.Ident:
			return t.Name
		default:
			return ""
		}
	}
}

// closureOf is every node reachable from start along edges, start included.
func closureOf(start string, edges map[string]map[string]bool) map[string]bool {
	seen := map[string]bool{start: true}
	q := []string{start}
	for len(q) > 0 {
		x := q[0]
		q = q[1:]
		for y := range edges[x] {
			if !seen[y] {
				seen[y] = true
				q = append(q, y)
			}
		}
	}
	return seen
}

// The decoder ceilings' plants: on a small package, a new method on Model, a new declaration in Model's knot and a longer
// file each raise their number and fail the check against the base counts; a declaration below the knot raises only the
// line count; a declaration that comes to name Model on the line it had raises only the knot; and a lowered count fails
// until its line is lowered. A test file's method on Model counts nowhere.
func TestQualityScan_decoderCeilingsPlantsGoRed(t *testing.T) {
	const base = `package pkg

type Model struct{ w *Weights }

func (m *Model) Forward() { helper(m) }

func helper(m *Model) { _ = m.w }

type Weights struct{ n int }
`
	census := func(t *testing.T, src string) decoderCensus {
		t.Helper()
		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, "pkg/a.go", src, parser.ParseComments)
		if err != nil {
			t.Fatal(err)
		}
		testFile, err := parser.ParseFile(fset, "pkg/a_test.go", "package pkg\n\nfunc (m *Model) testOnly() { helper(m) }\n", 0)
		if err != nil {
			t.Fatal(err)
		}
		return censusOf(fset, []*File{{Path: "pkg/a.go", Src: []byte(src), AST: f}, {Path: "pkg/a_test.go", Test: true, AST: testFile}}, "pkg")
	}
	got := census(t, base)
	if want := (decoderCensus{Lines: 9, ModelMethods: 1, KnotLines: 3}); got != want {
		t.Fatalf("base census %+v, want %+v (the knot is Model and its method, with helper; Weights is below it)", got, want)
	}
	allow, err := ParseAllowlist([]byte(fmt.Sprintf("lines %d -- r\nmodel_methods %d -- r\nmodel_knot_lines %d -- r\n", got.Lines, got.ModelMethods, got.KnotLines)))
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range []struct {
		name, add string
		want      decoderCensus
	}{
		{"method", "\nfunc (m *Model) Extra() {}\n", decoderCensus{Lines: 11, ModelMethods: 2, KnotLines: 4}},
		{"knot", "\nfunc tied(m *Model) {}\n\nfunc (m *Model) Tie() { tied(m) }\n", decoderCensus{Lines: 13, ModelMethods: 2, KnotLines: 5}},
		{"below", "\nfunc below(w *Weights) {}\n", decoderCensus{Lines: 11, ModelMethods: 1, KnotLines: 3}},
	} {
		c := census(t, base+p.add)
		if c != p.want {
			t.Errorf("plant %s: census %+v, want %+v", p.name, c, p.want)
		}
		if len(ceilingFailures(c, allow)) == 0 {
			t.Errorf("plant %s: the check passed against the base ceilings", p.name)
		}
	}
	// Only the knot grows: Weights names Model on the line it had, so it joins the knot and nothing else changes.
	tied := census(t, strings.Replace(base, "type Weights struct{ n int }", "type Weights struct{ m *Model }", 1))
	if want := (decoderCensus{Lines: 9, ModelMethods: 1, KnotLines: 4}); tied != want {
		t.Errorf("plant knot-only: census %+v, want %+v", tied, want)
	}
	if f := ceilingFailures(tied, allow); len(f) != 1 || !strings.Contains(f[0], "model_knot_lines (4, ceiling 3)") {
		t.Errorf("plant knot-only: failures %q, want one naming model_knot_lines alone", f)
	}
	lowered := got
	lowered.KnotLines--
	if len(ceilingFailures(lowered, allow)) == 0 {
		t.Error("a count under its ceiling passed the check")
	}
}
