package main

import (
	"bytes"
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
)

// ---- the graph ----

type goListPkg struct {
	ImportPath string
	Dir        string
	ForTest    string
	Module     *struct{ Path, Dir string }
	Deps       []string

	GoFiles, CgoFiles, IgnoredGoFiles, TestGoFiles, XTestGoFiles []string
	SFiles, CFiles, HFiles, CXXFiles, MFiles, SysoFiles          []string
	EmbedFiles, TestEmbedFiles, XTestEmbedFiles                  []string

	Error *struct{ Err string }
}

const goListFields = "ImportPath,Dir,ForTest,Module,Deps,GoFiles,CgoFiles,IgnoredGoFiles,TestGoFiles,XTestGoFiles," +
	"SFiles,CFiles,HFiles,CXXFiles,MFiles,SysoFiles,EmbedFiles,TestEmbedFiles,XTestEmbedFiles,Error"

// qPkg is one package of the five modules.
type qPkg struct {
	Path string
	Dir  string // absolute
	Mod  *quickModule
	Deps map[string]bool // non-test transitive deps, from the owning module's listing
	// Files are this package's own Go files (non-test and test), absolute.
	Files     []string
	TestFiles []string
}

// testRoot is one test binary: `go test <Pkg>` in module Mod.
type testRoot struct {
	Mod  *quickModule
	Pkg  string
	Dir  string
	Deps map[string]bool // every package the test binary links, the package itself included
	p    *qPkg

	scanned  bool
	src      string   // the package's Go sources, concatenated, for the "names it" scan
	srcs     []string // the same, per file (parallel to p.Files)
	tests    []string // top-level Test functions
	walkers  []string // the ones that walk the parent tree (split into their own cell)
	walkSite string   // where the package enumerates a directory ("" = nowhere)
	// loaders are the tests that name a real checkpoint present on THIS host (see checkpointLit),
	// directly or through a package-local helper or table; loadWhy[test] is the path each named.
	loaders []string
	loadWhy map[string]string
	repo    string   // the repository root, which checkpoint literals resolve against
	lits    []string // every string literal in the package's sources, unquoted
	litSet  map[string]bool
}

type fileOwner struct {
	Pkg  string
	Test bool // compiled only into the package's own test binary
}

type quickGraph struct {
	pkgs     map[string]*qPkg
	roots    []*testRoot
	owners   map[string][]fileOwner // absolute file -> packages compiling it
	dirPkgs  map[string][]string    // absolute dir -> packages in it
	warnings []string
}

func stripVariant(ip string) string {
	if before, _, ok := strings.Cut(ip, " ["); ok {
		return before
	}
	return ip
}

// loadGraph runs `go list -e -deps -test -json` once per module, with that module's tags,
// workspace and target, and merges the five listings into one graph.
func loadGraph(qc *quickConfig, overlay string) (*quickGraph, error) {
	g := &quickGraph{pkgs: map[string]*qPkg{}, owners: map[string][]fileOwner{}, dirPkgs: map[string][]string{}}
	local := map[string]*quickModule{}
	for _, m := range qc.Modules {
		local[m.Path] = m
	}
	type rootEntry struct {
		m    *quickModule
		base string
		deps []string
	}
	var rootEntries []rootEntry
	for _, m := range qc.Modules {
		env, err := qc.cmdEnv(m, m.listTarget(), overlay, false)
		if err != nil {
			return nil, err
		}
		args := []string{"list", "-e", "-deps", "-test", "-json=" + goListFields}
		if len(m.ListTags) > 0 {
			args = append(args, "-tags", strings.Join(m.ListTags, " "))
		}
		args = append(args, "./...")
		cmd := exec.Command("go", args...)
		cmd.Dir, cmd.Env = qc.modDir(m), env
		var stderr bytes.Buffer
		cmd.Stderr = &stderr
		out, err := cmd.Output()
		if err != nil {
			return nil, fmt.Errorf("go list in %s (%s): %w\n%s", m.Name, m.Dir, err, strings.TrimSpace(stderr.String()))
		}
		dec := json.NewDecoder(bytes.NewReader(out))
		for {
			var p goListPkg
			if err := dec.Decode(&p); errors.Is(err, io.EOF) {
				break
			} else if err != nil {
				return nil, fmt.Errorf("go list in %s: decoding: %w", m.Name, err)
			}
			base := stripVariant(p.ImportPath)
			if p.Module == nil || local[p.Module.Path] == nil {
				continue
			}
			if strings.HasSuffix(base, ".test") && p.ForTest == "" && !strings.Contains(p.ImportPath, " [") {
				pkg := strings.TrimSuffix(base, ".test")
				if local[p.Module.Path] == m {
					rootEntries = append(rootEntries, rootEntry{m, pkg, p.Deps})
				}
				continue
			}
			// The plain variant, from the listing of the module that OWNS it: its file lists and
			// non-test deps are then the ones its own tags produce.
			if p.ImportPath != base || p.ForTest != "" || local[p.Module.Path] != m {
				continue
			}
			if p.Error != nil {
				g.warnings = append(g.warnings, fmt.Sprintf("%s: %s", base, strings.TrimSpace(p.Error.Err)))
			}
			q := &qPkg{Path: base, Dir: p.Dir, Mod: m, Deps: map[string]bool{}}
			for _, d := range p.Deps {
				q.Deps[stripVariant(d)] = true
			}
			add := func(names []string, test bool) {
				for _, n := range names {
					abs := filepath.Join(p.Dir, n)
					t := test || strings.HasSuffix(n, "_test.go")
					g.owners[abs] = append(g.owners[abs], fileOwner{base, t})
					if strings.HasSuffix(n, ".go") {
						if t {
							q.TestFiles = append(q.TestFiles, abs)
						}
						q.Files = append(q.Files, abs)
					}
				}
			}
			add(p.GoFiles, false)
			add(p.CgoFiles, false)
			add(p.IgnoredGoFiles, false) // compiled under some other tag or GOOS: conservative
			for _, fl := range [][]string{p.SFiles, p.CFiles, p.HFiles, p.CXXFiles, p.MFiles, p.SysoFiles, p.EmbedFiles} {
				add(fl, false)
			}
			add(p.TestGoFiles, true)
			add(p.XTestGoFiles, true)
			add(p.TestEmbedFiles, true)
			add(p.XTestEmbedFiles, true)
			g.pkgs[base] = q
			g.dirPkgs[p.Dir] = append(g.dirPkgs[p.Dir], base)
		}
	}
	for _, re := range rootEntries {
		q := g.pkgs[re.base]
		if q == nil {
			continue
		}
		r := &testRoot{Mod: re.m, Pkg: re.base, Dir: q.Dir, Deps: map[string]bool{re.base: true}, p: q, repo: qc.Root}
		for _, d := range re.deps {
			r.Deps[stripVariant(d)] = true
		}
		g.roots = append(g.roots, r)
	}
	slices.SortFunc(g.roots, func(a, b *testRoot) int { return cmp.Compare(a.Pkg, b.Pkg) })
	return g, nil
}

// ---- static scan of a package's sources ----

// enumerators are the calls through which a test lists a directory rather than naming a file.
// Package-qualified, so a method that happens to be called ReadDir on some unrelated type does not
// count; Readdir/Readdirnames exist only on *os.File, so those match on any receiver.
var enumerators = map[string]map[string]bool{
	"filepath": {"WalkDir": true, "Walk": true, "Glob": true},
	"fs":       {"WalkDir": true, "Glob": true, "ReadDir": true},
	"os":       {"ReadDir": true, "DirFS": true},
	"ioutil":   {"ReadDir": true},
}

// scan parses the package once: its source text (for "names it"), its string literals (for
// crossModuleReader), its top-level tests, and which of them enumerate directories. The split that
// uses the last is a CACHE optimisation and never a correctness input — both halves of a split
// package run, and each half's cache is keyed on what that half actually read — so a heuristic
// miss costs time, not coverage.
func (r *testRoot) scan() {
	if r.scanned {
		return
	}
	r.scanned = true
	var src strings.Builder
	fset := token.NewFileSet()
	type fn struct {
		direct bool   // enumerates any directory
		site   string //
		tree   bool   // enumerates the PARENT tree: the repo, not its own package or testdata
		calls  []string
		ckpt   string // a real checkpoint the function names ("" = none)
	}
	funcs := map[string]*fn{}
	var tests []string
	r.srcs = make([]string, len(r.p.Files))
	type parsed struct {
		file   *ast.File
		isTest bool
	}
	var files []parsed
	isParent := func(lit *ast.BasicLit) bool { return lit.Value == `".."` || lit.Value == `"../.."` }
	// Package-level names bound to the parent: decoder/assets.go's `const repoRoot = ".."`, which
	// every tree walk in decoder passes by name.
	parentNames := map[string]bool{}
	// Package-level names whose value names a real checkpoint: `const ggufPath = "../testdata/…"`,
	// or a table of models that a test ranges over.
	ckptNames := map[string]string{}
	for i, f := range r.p.Files {
		b, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		r.srcs[i] = string(b)
		src.Write(b)
		src.WriteByte('\n')
		file, err := parser.ParseFile(fset, f, b, parser.SkipObjectResolution)
		if err != nil {
			continue
		}
		files = append(files, parsed{file, strings.HasSuffix(f, "_test.go")})
		for _, d := range file.Decls {
			gd, ok := d.(*ast.GenDecl)
			if !ok {
				continue
			}
			for _, sp := range gd.Specs {
				vs, ok := sp.(*ast.ValueSpec)
				if !ok {
					continue
				}
				for k, v := range vs.Values {
					if lit, ok := v.(*ast.BasicLit); ok && isParent(lit) && k < len(vs.Names) {
						parentNames[vs.Names[k].Name] = true
					}
					if ck := r.namesCheckpoint(v, strings.HasSuffix(f, "_test.go"), nil); ck != "" && k < len(vs.Names) {
						ckptNames[vs.Names[k].Name] = ck
					}
				}
			}
		}
	}
	// reachesParent: does this expression mention the parent — a ".." literal, or a name bound to one?
	reachesParent := func(n ast.Node) bool {
		hit := false
		ast.Inspect(n, func(n ast.Node) bool {
			switch x := n.(type) {
			case *ast.BasicLit:
				hit = hit || isParent(x)
			case *ast.Ident:
				hit = hit || parentNames[x.Name]
			}
			return !hit
		})
		return hit
	}
	for _, pf := range files {
		file, isTest := pf.file, pf.isTest
		ast.Inspect(file, func(n ast.Node) bool {
			if lit, ok := n.(*ast.BasicLit); ok && lit.Kind == token.STRING {
				if v, err := strconv.Unquote(lit.Value); err == nil && len(v) < 512 {
					r.lits = append(r.lits, v)
				}
			}
			return true
		})
		for _, d := range file.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if !ok || fd.Body == nil {
				continue
			}
			name := fd.Name.Name
			if fd.Recv != nil {
				name = "(method)" + name // methods take no part in the call graph below
			}
			e := &fn{ckpt: r.namesCheckpoint(fd.Body, isTest, ckptNames)}
			// The function reaches the parent itself: root := ".." / filepath.Abs("..") / repoRoot.
			parentLit := reachesParent(fd.Body)
			ast.Inspect(fd.Body, func(n ast.Node) bool {
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				switch fun := call.Fun.(type) {
				case *ast.Ident:
					e.calls = append(e.calls, fun.Name)
				case *ast.SelectorExpr:
					x, _ := fun.X.(*ast.Ident)
					hit := fun.Sel.Name == "Readdir" || fun.Sel.Name == "Readdirnames" ||
						(x != nil && enumerators[x.Name][fun.Sel.Name])
					// `git ls-files` / `git grep` read the tree as surely as a WalkDir.
					if !hit && x != nil && x.Name == "exec" && strings.HasPrefix(fun.Sel.Name, "Command") {
						for _, a := range call.Args {
							if lit, ok := a.(*ast.BasicLit); ok && lit.Kind == token.STRING && lit.Value == `"git"` {
								hit = true
							}
						}
					}
					if hit && !e.direct {
						e.direct = true
						pos := fset.Position(call.Pos())
						e.site = fmt.Sprintf("%s:%d", filepath.Base(pos.Filename), pos.Line)
					}
					if hit && len(call.Args) > 0 {
						// The walked root reaches the parent directly (WalkDir(repoRoot, …),
						// Glob(filepath.Join("..", "*.go"))), or is a local the function bound to it.
						_, local := call.Args[0].(*ast.Ident)
						e.tree = e.tree || reachesParent(call.Args[0]) || (local && parentLit)
					}
				}
				return true
			})
			if prev, dup := funcs[name]; dup { // same name under two build tags: union them
				prev.direct = prev.direct || e.direct
				prev.tree = prev.tree || e.tree
				if prev.ckpt == "" {
					prev.ckpt = e.ckpt
				}
				if prev.site == "" {
					prev.site = e.site
				}
				prev.calls = append(prev.calls, e.calls...)
			} else {
				funcs[name] = e
			}
			if isTest && fd.Recv == nil && isTestFunc(fd.Name.Name) {
				tests = append(tests, fd.Name.Name)
			}
		}
	}
	// Fixed point: a function walks if it enumerates directly or calls a package-local one that does.
	for changed := true; changed; {
		changed = false
		for _, e := range funcs {
			for _, c := range e.calls {
				callee := funcs[c]
				if callee == nil {
					continue
				}
				if callee.direct && !e.direct {
					e.direct, e.site, changed = true, callee.site, true
				}
				if callee.tree && !e.tree {
					e.tree, changed = true, true
				}
				if callee.ckpt != "" && e.ckpt == "" {
					e.ckpt, changed = callee.ckpt, true
				}
			}
		}
	}
	r.src = src.String()
	slices.Sort(tests)
	r.tests = dedupe(tests)
	for _, t := range r.tests {
		// Only the tree walkers split off. A fixture helper that globs ../testdata is called by hundreds of
		// decoder tests, and those re-run only when a fixture moves, which the rest cell's cache already handles.
		if e := funcs[t]; e != nil && e.tree {
			r.walkers = append(r.walkers, t)
		}
		if e := funcs[t]; e != nil && e.ckpt != "" {
			if r.loadWhy == nil {
				r.loadWhy = map[string]string{}
			}
			r.loaders = append(r.loaders, t)
			r.loadWhy[t] = e.ckpt
		}
	}
	for _, e := range funcs {
		if e.direct && (r.walkSite == "" || e.site < r.walkSite) {
			r.walkSite = e.site
		}
	}
}

// ---- tests that load a real checkpoint ----

// A real checkpoint is a model file of at least minCheckpointBytes. The tiny fixtures (the largest is a few
// MB) load beside anything; a real one is what the fit guard refuses when other test binaries hold memory,
// so a cell that loads one runs with nothing beside it (quickCell.Mem).
const minCheckpointBytes = 64 << 20

var checkpointExts = []string{".gguf", ".safetensors", ".giw"}

var (
	ckptMu     sync.Mutex
	ckptCache  = map[string]string{}              // repo + "\x00" + dir + "\x00" + literal -> resolved path or ""
	registries = map[string]map[string][]string{} // repo -> asset-registry key -> candidates
)

// namesCheckpoint returns the first real checkpoint that a string literal in n names, or one that a
// package-level name in names is bound to, or "".
func (r *testRoot) namesCheckpoint(n ast.Node, inTest bool, names map[string]string) string {
	hit := ""
	ast.Inspect(n, func(n ast.Node) bool {
		if hit != "" {
			return false
		}
		switch x := n.(type) {
		case *ast.BasicLit:
			if x.Kind == token.STRING {
				if v, err := strconv.Unquote(x.Value); err == nil {
					hit = r.checkpointLit(v, inTest)
				}
			}
		case *ast.Ident:
			hit = names[x.Name]
		}
		return hit == ""
	})
	return hit
}

// checkpointLit resolves one string literal to a real checkpoint present on this host, or "".
//
//   - A literal with a checkpoint extension is tried as written (after $VAR and ~ expansion), then
//     against the package directory and the repo root, and a bare file name also under the repo's
//     testdata/ and ~/models/ — the places this tree's tests look ("$HOME/models/qwen…gguf",
//     "../testdata/qwen2.5-coder…gguf", Join(home, "models", "qwen…gguf")).
//   - A bare name without one ("gpt2", from Join("..", "testdata", "gpt2")) counts only in a test
//     file, and only as a directory under testdata/ or ~/models/ holding a real checkpoint: in
//     production code "gpt2" is an architecture name that every test reaches.
//
// A miss costs parallelism, never coverage: the test still runs, only beside other loaders.
func (r *testRoot) checkpointLit(lit string, inTest bool) string {
	if lit == "" || len(lit) > 400 || strings.ContainsAny(lit, " \t\n%*?") {
		return ""
	}
	if strings.HasPrefix(lit, "GOINFER_") {
		return r.registryCheckpoint(lit)
	}
	ext := false
	for _, e := range checkpointExts {
		ext = ext || strings.HasSuffix(lit, e)
	}
	bare := !strings.ContainsAny(lit, "/\\$~")
	if !ext && !(bare && inTest && len(lit) >= 3) {
		return ""
	}
	key := r.repo + "\x00" + r.Dir + "\x00" + lit
	ckptMu.Lock()
	v, ok := ckptCache[key]
	ckptMu.Unlock()
	if ok {
		return v
	}
	home, _ := os.UserHomeDir()
	var got string
	if ext {
		s := os.ExpandEnv(lit)
		if strings.HasPrefix(s, "~/") && home != "" {
			s = filepath.Join(home, s[2:])
		}
		cands := []string{s}
		if !filepath.IsAbs(s) {
			cands = []string{filepath.Join(r.Dir, s), filepath.Join(r.repo, s)}
			if bare {
				cands = append(cands, filepath.Join(r.repo, "testdata", s))
				if home != "" {
					cands = append(cands, filepath.Join(home, "models", s))
				}
			}
		}
		for _, c := range cands {
			if st, err := os.Stat(c); err == nil && st.Mode().IsRegular() && st.Size() >= minCheckpointBytes {
				got = c
				break
			}
		}
	} else {
		dirs := []string{filepath.Join(r.repo, "testdata", lit)}
		if home != "" {
			dirs = append(dirs, filepath.Join(home, "models", lit))
		}
		for _, d := range dirs {
			if p := checkpointIn(d); p != "" {
				got = p
				break
			}
		}
	}
	ckptMu.Lock()
	ckptCache[key] = got
	ckptMu.Unlock()
	return got
}

// registryCheckpoint resolves an asset-registry key (testdata/assets.json) the way decoder's
// lookupAsset does — an explicit env value wins, else the first usable candidate, with $REPO and
// $MODELS expanded — and returns it when it is a real checkpoint, or a directory holding one. Tests
// reach most real models this way: assetPath(tb, "GOINFER_PREQUANT_GGUF") names no path at all.
func (r *testRoot) registryCheckpoint(key string) string {
	ckptMu.Lock()
	reg, ok := registries[r.repo]
	ckptMu.Unlock()
	if !ok {
		reg = map[string][]string{}
		var doc struct {
			Assets []struct {
				Env        string   `json:"env"`
				Candidates []string `json:"candidates"`
			} `json:"assets"`
		}
		if b, err := os.ReadFile(filepath.Join(r.repo, "testdata", "assets.json")); err == nil && json.Unmarshal(b, &doc) == nil {
			for _, a := range doc.Assets {
				reg[a.Env] = a.Candidates
			}
		}
		ckptMu.Lock()
		registries[r.repo] = reg
		ckptMu.Unlock()
	}
	cands, ok := reg[key]
	if !ok {
		return ""
	}
	if v := os.Getenv(key); v != "" {
		cands = []string{v}
	}
	models := os.Getenv("GOINFER_MODELS")
	if models == "" {
		if home, err := os.UserHomeDir(); err == nil {
			models = filepath.Join(home, "models")
		}
	}
	for _, c := range cands {
		switch {
		case strings.HasPrefix(c, "$REPO/"):
			c = filepath.Join(r.repo, strings.TrimPrefix(c, "$REPO/"))
		case strings.HasPrefix(c, "$MODELS/"):
			c = filepath.Join(models, strings.TrimPrefix(c, "$MODELS/"))
		}
		st, err := os.Stat(c)
		if err != nil {
			continue
		}
		if st.Mode().IsRegular() && st.Size() >= minCheckpointBytes {
			return c
		}
		if st.IsDir() {
			if p := checkpointIn(c); p != "" {
				return p
			}
		}
	}
	return ""
}

// checkpointIn returns a real checkpoint directly inside dir, or "".
func checkpointIn(dir string) string {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	for _, e := range ents {
		for _, x := range checkpointExts {
			if !strings.HasSuffix(e.Name(), x) {
				continue
			}
			p := filepath.Join(dir, e.Name())
			if st, err := os.Stat(p); err == nil && st.Mode().IsRegular() && st.Size() >= minCheckpointBytes {
				return p
			}
		}
	}
	return ""
}

func isTestFunc(name string) bool {
	if !strings.HasPrefix(name, "Test") || name == "TestMain" {
		return false
	}
	rest := name[len("Test"):]
	return rest == "" || !(rest[0] >= 'a' && rest[0] <= 'z')
}

func dedupe(xs []string) []string {
	var out []string
	for i, x := range xs {
		if i == 0 || x != xs[i-1] {
			out = append(out, x)
		}
	}
	return out
}

// names reports which of the package's source files contains name (a file's base name), or "".
func (r *testRoot) names(name string) string {
	r.scan()
	if len(name) < 4 || !strings.Contains(r.src, name) {
		return ""
	}
	for i, b := range r.srcs {
		if strings.Contains(b, name) {
			return filepath.Base(r.p.Files[i])
		}
	}
	return "?"
}

// needle is what a source must contain to "name" a changed file: its base name for data (fixtures
// are referenced by name — "config.json", "shared_sets.json"), and parent/base for a Go file,
// whose base name alone ("main.go", "model.go") is mentioned in half the tree's comments.
func needle(rel string) string {
	if strings.HasSuffix(rel, ".go") {
		if d := filepath.Base(filepath.Dir(rel)); d != "." {
			return d + "/" + filepath.Base(rel)
		}
	}
	return filepath.Base(rel)
}

// crossModuleReader reports why the test cache cannot be trusted to see a change to rel on this root's
// behalf, or "" when it can. cmd/go re-checks a test's recorded inputs only INSIDE the module root ("Do
// not recheck files outside the module, GOPATH, or GOROOT root", Go's computeTestInputsID), so a metal test
// that read ../docs/x.md replays a stale result after x.md changes. Where the read is determinable, the
// root is forced to re-run:
//
//   - its source names the file (see needle);
//   - a "../…" string literal, resolved against the package directory, IS the file or one of its
//     directories (so "../docs/audit-metal-2026-09-12.md" forces on that file only, not on every docs/
//     edit);
//   - it has a bare ".." literal and every component of the file's path as a literal
//     (filepath.Join("..", "testdata", "llama-tiny") plus "config.json"). Requiring every component, not
//     just the top directory, is what stops a Join("..", "docs", "measurements", …) from forcing on every
//     docs/ edit.
//
// A read by a path assembled at run time outside the module root, including a walk of the whole parent from
// a computed root, stays the cache's blind spot, exactly as it is for a hand-typed `go test`; that is the
// risk accepted for day loops.
func (r *testRoot) crossModuleReader(root, rel string) string {
	modRel := r.Mod.Dir
	if modRel == "." || rel == modRel || strings.HasPrefix(rel, modRel+"/") {
		return ""
	}
	r.scan()
	if n := needle(rel); r.names(n) != "" {
		return "names " + n + " outside its module root"
	}
	abs := filepath.Join(root, filepath.FromSlash(rel))
	for _, lit := range r.lits {
		if !strings.HasPrefix(lit, "../") {
			continue
		}
		p := lit
		if i := strings.IndexAny(p, "*?["); i >= 0 { // a glob reads the directory it starts in
			p = p[:i]
		}
		res := filepath.Clean(filepath.Join(r.Dir, filepath.FromSlash(p)))
		if res == abs || strings.HasPrefix(abs, res+string(filepath.Separator)) {
			return "reads " + strconv.Quote(lit) + " outside its module root"
		}
	}
	if r.hasLit("..") || r.hasLit("../..") {
		all := true
		for c := range strings.SplitSeq(rel, "/") {
			all = all && r.hasLit(c)
		}
		if all {
			return `joins ".." with every component of ` + rel
		}
	}
	return ""
}

func (r *testRoot) hasLit(s string) bool {
	if r.litSet == nil {
		r.litSet = map[string]bool{}
		for _, l := range r.lits {
			r.litSet[l] = true
		}
	}
	return r.litSet[s]
}
