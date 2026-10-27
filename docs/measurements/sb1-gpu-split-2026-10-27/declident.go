//go:build ignore

// declident is G1 of the gpu split (docs/tasks/task-subsystems-2026-10.md, "SB1 — gpu's kernel layer"): every
// declaration of package gpu at the base appears at the head, in gpu or gpu/internal/kernels, with identical code once
// the split's mechanical edits are mapped back. Standard library only.
//
//	go run declident.go <base gpu dir> <head gpu dir> <renames.tsv> [hand_edits.tsv]
//
// The head is type-checked (go/types over the compiler's export data from go list -export, tags gpu goinfer_testhooks,
// linux/amd64) and normalised to the base's spelling: each object the split renamed, found by identity and matched to
// renames.tsv by kind, owner and name, gets its old name back; kernels. qualifiers are dropped; accessor calls become
// selectors, method calls and method values again; <T>Lit constructors become composite literals; functions that were
// methods (the pre-step's uploadProj and the copied Context helpers) become methods; the two adapter types are decoder's
// again. Both sides are then printed without comments or positions. A base declaration must appear exactly once at the
// head with the same text (a copied test helper at least once, every copy the same); the head's other declarations must
// be the generated aliases, wrappers and accessors, each in its generated shape. hand_edits.tsv lists the edits made by
// hand (key, head text, base text): each is applied to that declaration's head text only, and must occur there. Exit 0
// when all hold.
package main

import (
	"bufio"
	"bytes"
	"fmt"
	"go/ast"
	"go/build"
	"go/importer"
	"go/parser"
	"go/printer"
	"go/token"
	"go/types"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
)

const kernelsPath = "github.com/townsendmerino/goinfer/gpu/internal/kernels"

var (
	fset      = token.NewFileSet()
	revObj    = map[string]string{} // new package-level name -> old
	revMember = map[string]string{} // "<owner's old name>.<new name>" -> old, for fields and methods
	accField  = map[string]string{} // accessor -> field
	accMeth   = map[string]string{} // accessor (and accessor+"Func") -> method
	copied    = map[string]bool{}   // copied test decl names
	toMethod  = map[string]bool{}   // functions at the head that are methods at the base
	litKeys   = map[string][]string{}
	fails     []string

	info      = &types.Info{Defs: map[*ast.Ident]types.Object{}, Uses: map[*ast.Ident]types.Object{}, Selections: map[*ast.SelectorExpr]*types.Selection{}, Types: map[ast.Expr]types.TypeAndValue{}}
	kernelPkg *types.Package
	gpuPkg    *types.Package
	owners    = map[*types.Var]*types.Named{}
)

func fail(f string, a ...any) { fails = append(fails, fmt.Sprintf(f, a...)) }

type pfile struct {
	f    *ast.File
	path string
}

func main() {
	baseDir, headDir, table := os.Args[1], os.Args[2], os.Args[3]
	readTable(table)
	hand := map[string][][2]string{}
	if len(os.Args) > 4 {
		b, err := os.ReadFile(os.Args[4])
		if err != nil {
			panic(err)
		}
		for _, l := range strings.Split(strings.TrimSpace(string(b)), "\n") {
			c := strings.Split(l, "\t")
			hand[c[0]] = append(hand[c[0]], [2]string{c[1], c[2]})
		}
	}
	toMethod["uploadProj"] = true // pre-step: Context.uploadProj became uploadProj(c, w)

	base := declsOf(parseDir(baseDir), false)
	kf, gf := typecheckHead(headDir)
	readLits(kf)
	head := declsOf(append(append(append([]pfile{}, kf...), gf...), parseExt(headDir)...), true)

	var keys []string
	for k := range base {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	seenAll := map[string]bool{}
	for _, k := range keys {
		b, h := base[k], head[k]
		for _, e := range hand["*"] { // an edit made in every moved file: applied wherever it occurs, and it must occur somewhere
			for i := range h {
				if strings.Contains(h[i], e[0]) {
					seenAll[e[0]] = true
					h[i] = strings.ReplaceAll(h[i], e[0], e[1])
				}
			}
		}
		for _, e := range hand[k] {
			for i := range h {
				if !strings.Contains(h[i], e[0]) {
					fail("%s: hand edit %q does not occur", k, e[0])
				}
				h[i] = strings.ReplaceAll(h[i], e[0], e[1])
			}
		}
		name := k[strings.LastIndex(k, " ")+1:]
		name = name[strings.LastIndex(name, ".")+1:]
		switch {
		case len(h) == 0:
			fail("%s: missing at the head", k)
		case multiKey(k):
			if strings.Join(sorted(b), "\x00") != strings.Join(sorted(h), "\x00") {
				fail("%s: the set differs (base %d, head %d)", k, len(b), len(h))
			}
		case len(b) != 1:
			fail("%s: %d declarations at the base", k, len(b))
		case copied[name]:
			for _, t := range h {
				if t != b[0] {
					fail("%s: a copy differs:\n--- base\n%s\n--- head\n%s", k, b[0], t)
				}
			}
		default:
			if len(h) != 1 {
				fail("%s: %d declarations at the head", k, len(h))
			} else if h[0] != b[0] {
				fail("%s: differs:\n--- base\n%s\n--- head\n%s", k, b[0], h[0])
			}
		}
		delete(head, k)
	}
	for k := range head {
		fail("%s: at the head only", k)
	}
	for _, e := range hand["*"] {
		if !seenAll[e[0]] {
			fail("hand edit %q (every file) does not occur", e[0])
		}
	}
	checkGenerated(kf, gf)
	if len(fails) > 0 {
		for _, f := range fails {
			fmt.Println("FAIL", f)
		}
		fmt.Printf("declident: %d failure(s)\n", len(fails))
		os.Exit(1)
	}
	fmt.Printf("declident: PASS: %d base declaration keys, each at the head with identical code\n", len(keys))
}

func multiKey(k string) bool { return strings.HasSuffix(k, " init") || strings.HasSuffix(k, " _") }

func sorted(s []string) []string {
	c := append([]string(nil), s...)
	sort.Strings(c)
	return c
}

func readTable(p string) {
	f, err := os.Open(p)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		c := strings.Split(sc.Text(), "\t")
		if len(c) < 5 {
			continue
		}
		switch c[0] {
		case "rename":
			if c[1] == "obj" {
				revObj[c[4]] = c[3]
			} else {
				revMember[c[2]+"."+c[4]] = c[3]
			}
		case "accessor":
			if c[1] == "field" {
				accField[c[4]] = c[3]
			} else {
				accMeth[c[4]] = c[3]
				accMeth[c[4]+"Func"] = c[3]
			}
		case "copy":
			copied[c[3]] = true
			if c[1] == "to-gpu-method" {
				toMethod[c[3]] = true
			}
		}
	}
}

var bctx = func() build.Context {
	c := build.Default
	c.GOOS, c.GOARCH, c.CgoEnabled = "linux", "amd64", true
	c.BuildTags = []string{"gpu", "goinfer_testhooks"}
	return c
}()

// goFiles parses the files of dir that build under the tags, keeping those whose package clause is pkg (or any, "").
func goFiles(dir, pkg string) []pfile {
	ms, _ := filepath.Glob(filepath.Join(dir, "*.go"))
	var out []pfile
	for _, m := range ms {
		if ok, err := bctx.MatchFile(dir, filepath.Base(m)); err != nil || !ok {
			continue
		}
		f, err := parser.ParseFile(fset, m, nil, parser.SkipObjectResolution)
		if err != nil {
			panic(err)
		}
		if pkg == "" || f.Name.Name == pkg {
			out = append(out, pfile{f, m})
		}
	}
	return out
}

func parseDir(dir string) []pfile { return goFiles(dir, "") }
func parseExt(dir string) []pfile { return goFiles(dir, "gpu_test") }

// typecheckHead type-checks gpu/internal/kernels (with its in-package tests) and gpu (with its), filling info.
func typecheckHead(headDir string) (kf, gf []pfile) {
	cmd := exec.Command("go", "list", "-export", "-deps", "-test", "-tags", "gpu goinfer_testhooks", "-f", "{{.ImportPath}}={{.Export}}", "./", "./internal/kernels/")
	cmd.Dir = headDir
	out, err := cmd.Output()
	if err != nil {
		panic(fmt.Sprintf("go list -export: %v", err))
	}
	exports := map[string]string{}
	for _, l := range strings.Split(string(out), "\n") {
		if p, e, ok := strings.Cut(l, "="); ok && e != "" && !strings.Contains(p, " ") {
			exports[p] = e
		}
	}
	gc := importer.ForCompiler(fset, "gc", func(path string) (io.ReadCloser, error) {
		if e, ok := exports[path]; ok {
			return os.Open(e)
		}
		return nil, fmt.Errorf("no export data for %s", path)
	})
	imp := importerFunc(func(path string) (*types.Package, error) {
		if path == kernelsPath && kernelPkg != nil {
			return kernelPkg, nil
		}
		return gc.Import(path)
	})
	check := func(path string, files []pfile) *types.Package {
		var asts []*ast.File
		for _, f := range files {
			asts = append(asts, f.f)
		}
		conf := types.Config{Importer: imp, Error: func(err error) { fail("typecheck %s: %v", path, err) }}
		p, _ := conf.Check(path, fset, asts, info)
		return p
	}
	kf = goFiles(filepath.Join(headDir, "internal", "kernels"), "kernels")
	kernelPkg = check(kernelsPath, kf)
	gf = goFiles(headDir, "gpu")
	gpuPkg = check("github.com/townsendmerino/goinfer/gpu", gf)
	for _, p := range []*types.Package{kernelPkg, gpuPkg} {
		for _, n := range p.Scope().Names() {
			if nt, ok := p.Scope().Lookup(n).Type().(*types.Named); ok {
				if st, ok := nt.Underlying().(*types.Struct); ok {
					for i := range st.NumFields() {
						owners[st.Field(i)] = nt
					}
				}
			}
		}
	}
	return kf, gf
}

type importerFunc func(string) (*types.Package, error)

func (f importerFunc) Import(p string) (*types.Package, error) { return f(p) }

// readLits learns the key order of each <T>Lit constructor from accessors.go.
func readLits(files []pfile) {
	for _, pf := range files {
		if filepath.Base(pf.path) != "accessors.go" {
			continue
		}
		for _, d := range pf.f.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if !ok || !strings.HasSuffix(fd.Name.Name, "Lit") {
				continue
			}
			var ks []string
			for _, p := range fd.Type.Params.List {
				for _, n := range p.Names {
					ks = append(ks, n.Name)
				}
			}
			litKeys[strings.TrimSuffix(fd.Name.Name, "Lit")] = ks
		}
	}
}

// declsOf keys every declaration ("<pkg|ext> <name>", methods as Type.Method) to its printed code.
func declsOf(files []pfile, head bool) map[string][]string {
	out := map[string][]string{}
	for _, pf := range files {
		b := filepath.Base(pf.path)
		if head && (b == "api.go" || b == "accessors.go") {
			continue // checked by checkGenerated
		}
		class := "pkg"
		if strings.HasSuffix(pf.f.Name.Name, "_test") {
			class = "ext"
		}
		for _, d := range pf.f.Decls {
			if head && class == "pkg" {
				normalize(d)
			}
			canonLits(d)
			switch d := d.(type) {
			case *ast.FuncDecl:
				key := d.Name.Name
				if d.Recv != nil {
					key = recvName(d.Recv.List[0].Type) + "." + key
				}
				d.Doc = nil
				out[class+" "+key] = append(out[class+" "+key], show(d))
			case *ast.GenDecl:
				if d.Tok == token.IMPORT {
					continue
				}
				for _, s := range d.Specs {
					var key string
					switch s := s.(type) {
					case *ast.TypeSpec:
						key, s.Doc, s.Comment = s.Name.Name, nil, nil
					case *ast.ValueSpec:
						key, s.Doc, s.Comment = s.Names[0].Name, nil, nil
					}
					out[class+" "+key] = append(out[class+" "+key], d.Tok.String()+" "+show(s))
				}
			}
		}
	}
	return out
}

func recvName(e ast.Expr) string {
	switch t := e.(type) {
	case *ast.StarExpr:
		return recvName(t.X)
	case *ast.Ident:
		return t.Name
	case *ast.IndexExpr:
		return recvName(t.X)
	}
	return "?"
}

// show prints a node without positions or comments.
func show(n ast.Node) string {
	zero(reflect.ValueOf(n))
	var b bytes.Buffer
	if err := printer.Fprint(&b, token.NewFileSet(), n); err != nil {
		panic(err)
	}
	return b.String()
}

var (
	posType = reflect.TypeOf(token.Pos(0))
	cgType  = reflect.TypeOf((*ast.CommentGroup)(nil))
)

func zero(v reflect.Value) {
	switch v.Kind() {
	case reflect.Pointer, reflect.Interface:
		if !v.IsNil() {
			zero(v.Elem())
		}
	case reflect.Struct:
		for i := range v.NumField() {
			f := v.Field(i)
			switch {
			case f.Type() == posType && f.CanSet():
				f.SetInt(0)
			case f.Type() == cgType && f.CanSet():
				f.Set(reflect.Zero(f.Type()))
			default:
				zero(f)
			}
		}
	case reflect.Slice:
		for i := range v.Len() {
			zero(v.Index(i))
		}
	}
}

// canonLits writes &T{...} as an element of a composite literal in its elided form {...}, on both sides.
func canonLits(n ast.Node) {
	ast.Inspect(n, func(x ast.Node) bool {
		cl, ok := x.(*ast.CompositeLit)
		if !ok {
			return true
		}
		for i, e := range cl.Elts {
			if u, ok := e.(*ast.UnaryExpr); ok && u.Op == token.AND {
				if inner, ok := u.X.(*ast.CompositeLit); ok {
					inner.Type = nil
					cl.Elts[i] = inner
				}
			}
		}
		return true
	})
}

// oldName is the base's name for ident id at the head, from its object: renamed objects, fields and methods only.
func oldName(id *ast.Ident) (string, bool) {
	o := info.Uses[id]
	if o == nil {
		o = info.Defs[id]
	}
	if o == nil || (o.Pkg() != kernelPkg && o.Pkg() != gpuPkg) {
		return "", false
	}
	switch o := o.(type) {
	case *types.Var:
		if o.IsField() {
			if nt := owners[o]; nt != nil {
				return lookupMember(nt.Obj().Name(), o.Name())
			}
			return "", false
		}
	case *types.Func:
		if r := o.Type().(*types.Signature).Recv(); r != nil {
			t := r.Type()
			if p, ok := t.(*types.Pointer); ok {
				t = p.Elem()
			}
			if nt, ok := t.(*types.Named); ok {
				return lookupMember(nt.Obj().Name(), o.Name())
			}
			return "", false
		}
	}
	if o.Pkg() == kernelPkg && o.Parent() == kernelPkg.Scope() {
		old, ok := revObj[o.Name()]
		return old, ok
	}
	return "", false
}

func lookupMember(owner, name string) (string, bool) {
	if old, ok := revObj[owner]; ok {
		owner = old
	}
	old, ok := revMember[owner+"."+name]
	return old, ok
}

// normalize rewrites a head declaration into the base's spelling, in place.
func normalize(d ast.Decl) {
	// names are decided first, from the original nodes' objects
	ren := map[*ast.Ident]string{}
	ast.Inspect(d, func(x ast.Node) bool {
		if id, ok := x.(*ast.Ident); ok {
			if old, ok := oldName(id); ok {
				ren[id] = old
			}
		}
		return true
	})
	if fd, ok := d.(*ast.FuncDecl); ok && fd.Recv == nil && toMethod[fd.Name.Name] && isGpuFunc(fd.Name) {
		p := fd.Type.Params.List[0]
		recv := &ast.Field{Names: p.Names[:1], Type: p.Type}
		rest := fd.Type.Params.List[1:]
		if len(p.Names) > 1 {
			rest = append([]*ast.Field{{Names: p.Names[1:], Type: p.Type}}, rest...)
		}
		fd.Recv = &ast.FieldList{List: []*ast.Field{recv}}
		fd.Type.Params = &ast.FieldList{List: rest}
	}
	rewriteExprs(d)
	ast.Inspect(d, func(x ast.Node) bool {
		if id, ok := x.(*ast.Ident); ok {
			if old, ok := ren[id]; ok {
				id.Name = old
			}
		}
		return true
	})
}

func isGpuFunc(id *ast.Ident) bool {
	o := info.Defs[id]
	if o == nil {
		o = info.Uses[id]
	}
	f, ok := o.(*types.Func)
	return ok && f.Pkg() == gpuPkg && f.Type().(*types.Signature).Recv() == nil
}

// funcOf is the function a call's Fun names, through a kernels. qualifier or not.
func funcOf(e ast.Expr) (*types.Func, *ast.Ident) {
	var id *ast.Ident
	switch x := e.(type) {
	case *ast.Ident:
		id = x
	case *ast.SelectorExpr:
		id = x.Sel
	default:
		return nil, nil
	}
	f, _ := info.Uses[id].(*types.Func)
	return f, id
}

// made holds the selectors fix builds from accessor calls: a parenthesis around one was the accessor form's, not the base's.
var made = map[ast.Expr]bool{}

func deref(e ast.Expr) ast.Expr {
	if p, ok := e.(*ast.ParenExpr); ok {
		e = p.X
	}
	if u, ok := e.(*ast.UnaryExpr); ok && u.Op == token.AND {
		return u.X
	}
	if s, ok := e.(*ast.StarExpr); ok {
		return s.X
	}
	return e
}

// fix maps one expression from the split's form to the base's, or returns it unchanged.
func fix(e ast.Expr) ast.Expr {
	switch x := e.(type) {
	case *ast.ParenExpr: // (*kernels.TF(x)) -> x.f
		if made[x.X] {
			return x.X
		}
		if st, ok := x.X.(*ast.StarExpr); ok {
			if c, ok := st.X.(*ast.CallExpr); ok {
				if f, _ := funcOf(c.Fun); f != nil && f.Pkg() == kernelPkg && accField[f.Name()] != "" && len(c.Args) == 1 {
					return &ast.SelectorExpr{X: deref(c.Args[0]), Sel: ast.NewIdent(accField[f.Name()])}
				}
			}
		}
	case *ast.StarExpr: // *kernels.TF(x) -> x.f (gofmt drops the parentheses where they are redundant); *kernels.TLit(...) -> T{...}
		if c, ok := x.X.(*ast.CallExpr); ok {
			if f, _ := funcOf(c.Fun); f != nil && f.Pkg() == kernelPkg && accField[f.Name()] != "" && len(c.Args) == 1 {
				sel := &ast.SelectorExpr{X: deref(c.Args[0]), Sel: ast.NewIdent(accField[f.Name()])}
				made[sel] = true
				return sel
			}
			if lit := litOf(c); lit != nil {
				return lit
			}
		}
	case *ast.CallExpr:
		if lit := litOf(x); lit != nil { // kernels.TLit(...) -> &T{...}
			return &ast.UnaryExpr{Op: token.AND, X: lit}
		}
		f, id := funcOf(x.Fun)
		if f == nil {
			return e
		}
		if m, ok := accMeth[f.Name()]; ok && f.Pkg() == kernelPkg && len(x.Args) >= 1 {
			sel := &ast.SelectorExpr{X: deref(x.Args[0]), Sel: ast.NewIdent(m)}
			if strings.HasSuffix(f.Name(), "Func") && len(x.Args) == 1 {
				return sel
			}
			return &ast.CallExpr{Fun: sel, Args: x.Args[1:], Ellipsis: x.Ellipsis}
		}
		if toMethod[f.Name()] && f.Pkg() == gpuPkg && f.Type().(*types.Signature).Recv() == nil && len(x.Args) >= 1 {
			return &ast.CallExpr{Fun: &ast.SelectorExpr{X: deref(x.Args[0]), Sel: ast.NewIdent(id.Name)}, Args: x.Args[1:], Ellipsis: x.Ellipsis}
		}
	case *ast.SelectorExpr:
		if id, ok := x.X.(*ast.Ident); ok {
			if pn, ok := info.Uses[id].(*types.PkgName); ok {
				switch pn.Imported().Path() {
				case kernelsPath:
					return x.Sel
				case "github.com/townsendmerino/goinfer/internal/adapter":
					return &ast.SelectorExpr{X: ast.NewIdent("decoder"), Sel: x.Sel}
				}
			}
		}
	}
	return e
}

func litOf(c *ast.CallExpr) *ast.CompositeLit {
	f, _ := funcOf(c.Fun)
	if f == nil || f.Pkg() != kernelPkg || !strings.HasSuffix(f.Name(), "Lit") {
		return nil
	}
	t := strings.TrimSuffix(f.Name(), "Lit")
	ks, ok := litKeys[t]
	if !ok || len(ks) != len(c.Args) {
		return nil
	}
	lit := &ast.CompositeLit{Type: ast.NewIdent(t)}
	for i, k := range ks {
		lit.Elts = append(lit.Elts, &ast.KeyValueExpr{Key: ast.NewIdent(k), Value: c.Args[i]})
	}
	return lit
}

// rewriteExprs replaces, bottom-up, every expression the split rewrote.
func rewriteExprs(n ast.Node) {
	exprType := reflect.TypeOf((*ast.Expr)(nil)).Elem()
	var walk func(v reflect.Value)
	walk = func(v reflect.Value) {
		switch v.Kind() {
		case reflect.Pointer, reflect.Interface:
			if !v.IsNil() {
				walk(v.Elem())
			}
		case reflect.Struct:
			for i := range v.NumField() {
				f := v.Field(i)
				walk(f)
				if f.CanSet() && f.Kind() == reflect.Interface && f.Type() == exprType && !f.IsNil() {
					if r := fix(f.Interface().(ast.Expr)); r != f.Interface() {
						f.Set(reflect.ValueOf(r))
					}
				}
			}
		case reflect.Slice:
			for i := range v.Len() {
				el := v.Index(i)
				walk(el)
				if el.Kind() == reflect.Interface && el.Type() == exprType && !el.IsNil() {
					if r := fix(el.Interface().(ast.Expr)); r != el.Interface() {
						el.Set(reflect.ValueOf(r))
					}
				}
			}
		}
	}
	walk(reflect.ValueOf(n))
}

// checkGenerated checks api.go and accessors.go hold only the generated shapes.
func checkGenerated(kf, gf []pfile) {
	for _, pf := range append(append([]pfile{}, kf...), gf...) {
		switch filepath.Base(pf.path) {
		case "api.go":
			for _, d := range pf.f.Decls {
				switch d := d.(type) {
				case *ast.GenDecl:
					if d.Tok == token.IMPORT {
						continue
					}
					for _, s := range d.Specs {
						ts, ok := s.(*ast.TypeSpec)
						if !ok || !ts.Assign.IsValid() {
							fail("api.go: a declaration that is not a type alias")
							continue
						}
						sel, ok := ts.Type.(*ast.SelectorExpr)
						if !ok || sel.Sel.Name != ts.Name.Name || sel.X.(*ast.Ident).Name != "kernels" {
							fail("api.go: %s is not = kernels.%s", ts.Name.Name, ts.Name.Name)
						}
					}
				case *ast.FuncDecl:
					r, ok := d.Body.List[0].(*ast.ReturnStmt)
					if len(d.Body.List) != 1 || !ok || len(r.Results) != 1 {
						fail("api.go: %s is not a one-statement wrapper", d.Name.Name)
						continue
					}
					c, ok := r.Results[0].(*ast.CallExpr)
					if f, _ := funcOf(c.Fun); !ok || f == nil || f.Pkg() != kernelPkg || f.Name() != d.Name.Name {
						fail("api.go: %s forwards to something else", d.Name.Name)
					}
				}
			}
		case "accessors.go":
			for _, d := range pf.f.Decls {
				fd, ok := d.(*ast.FuncDecl)
				if !ok {
					continue
				}
				nm := fd.Name.Name
				body := show(fd.Body)
				switch {
				case accField[nm] != "":
					if !strings.Contains(body, "return &x."+accField[nm]+"\n") {
						fail("accessors.go: %s does not return &x.%s: %s", nm, accField[nm], body)
					}
				case accMeth[nm] != "":
					m := accMeth[nm]
					if !strings.Contains(body, "x."+m+"(") && !strings.Contains(body, "return x."+m+"\n") {
						fail("accessors.go: %s does not call x.%s: %s", nm, m, body)
					}
				case strings.HasSuffix(nm, "Lit"):
					if !strings.Contains(body, "return &"+strings.TrimSuffix(nm, "Lit")+"{") {
						fail("accessors.go: %s is not a literal: %s", nm, body)
					}
				default:
					fail("accessors.go: %s is not in renames.tsv", nm)
				}
			}
		}
	}
}
