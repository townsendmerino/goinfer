package layout

import (
	"bytes"
	"encoding/json"
	"go/parser"
	"go/scanner"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// testCachePath is PW5's data (docs/tasks/task-subsystems-2026-10.md §12.3): every package, by its directory relative to
// the repository root, as cacheable by Go's test cache or not, with the hidden inputs that make it not. scripts/touched.py
// reads it and drops -count=1 only for a cacheable package.
const testCachePath = "test_cache.json"

// testCacheData is test_cache.json. Packages is generated (-update); Declared is hand-edited and names reasons a scan
// cannot see, each making its package non-cacheable.
type testCacheData struct {
	About    string                    `json:"_about"`
	Declared map[string][]string       `json:"declared"`
	Packages map[string]testCacheEntry `json:"packages"`
}

type testCacheEntry struct {
	Cacheable bool     `json:"cacheable"`
	Reasons   []string `json:"reasons,omitempty"`
}

// The hidden inputs: what a test can depend on that Go's test cache does not key on. A test's own os.Getenv reads and its
// opens of files inside its module are keyed, so they are not here.
const (
	hiddenMemory  = "OS memory"
	hiddenDevice  = "a device"
	hiddenExec    = "a subprocess"
	hiddenEnviron = "the whole environment"
	hiddenFiles   = "files outside the module"
	hiddenNet     = "the network"
	hiddenTiming  = "timing"
)

// hiddenCalls finds hidden reads in a file's code, written as compactCode renders it (comments dropped, no spaces around
// "." or before "("). The patterns are broad on purpose: a false match costs a package its cache, a miss replays a stale
// pass. They count in every file, test or not: a production timeout or guard can decide what a test asserts.
var hiddenCalls = []struct {
	kind string
	re   *regexp.Regexp
}{
	{hiddenExec, regexp.MustCompile(`\bexec\.Command(Context)?\(|\bos\.StartProcess\(|\bsyscall\.(Exec|ForkExec)\(`)},
	{hiddenEnviron, regexp.MustCompile(`\bos\.Environ\(|\bsyscall\.(Getenv|Environ)\(`)},
	{hiddenFiles, regexp.MustCompile(`\bos\.(UserHomeDir|UserCacheDir|UserConfigDir|TempDir|Hostname|Executable)\(|\bAssetPathForTest\(`)},
	{hiddenNet, regexp.MustCompile(`\bnet\.(Dial|Listen)|\bhttp\.(Get|Post|Head|PostForm)\(|\bhttp\.(Client|DefaultClient)\b`)},
	{hiddenTiming, regexp.MustCompile(`\btime\.(Since|Until|Now|After|AfterFunc|NewTimer|NewTicker|Tick|Sleep)\b|\bcontext\.With(Timeout|Deadline)\b|\btesting\.Benchmark\b`)},
}

// systemDirs names the absolute roots, below "/", that a string literal may name and that lie outside any module. They are
// written without the slash so that this list does not read as such a path itself.
var systemDirs = []string{"Users", "home", "proc", "sys", "tmp", "dev", "var", "private", "opt", "etc", "Library", "Volumes", "srv", "mnt"}

// fileHidden returns a file's hidden reads; how many directories its relative paths can climb (the leading "../"
// segments of a string literal, summed over literals passed together as Join's arguments are); and whether it locates
// files by runtime.Caller, which yields paths in its own module.
func fileHidden(name string, src []byte) (kinds map[string]bool, climb int, caller bool) {
	kinds = map[string]bool{}
	code := compactCode(name, src)
	for _, h := range hiddenCalls {
		if h.re.MatchString(code) {
			kinds[h.kind] = true
		}
	}
	caller = strings.Contains(code, "runtime.Caller(")
	fset := token.NewFileSet()
	var sc scanner.Scanner
	sc.Init(fset.AddFile(name, -1, len(src)), src, nil, 0)
	run := 0 // the climb of the literals passed together so far, as Join("..", "..", "x") takes them
	for {
		_, tok, lit := sc.Scan()
		if tok == token.EOF {
			break
		}
		if tok == token.COMMA {
			continue
		}
		if tok != token.STRING {
			run = 0
			continue
		}
		v, err := strconv.Unquote(lit)
		if err != nil {
			run = 0
			continue
		}
		for _, d := range systemDirs {
			if v == "/"+d || strings.HasPrefix(v, "/"+d+"/") {
				kinds[hiddenFiles] = true
			}
		}
		n, rest := 0, strings.TrimPrefix(v, "./")
		for ; strings.HasPrefix(rest, "../"); rest = rest[3:] {
			n++
		}
		if rest == ".." {
			n++
		}
		if n == 0 {
			run = 0
			continue
		}
		run += n
		climb = max(climb, run)
	}
	return kinds, climb, caller
}

// hiddenImports maps an import to the hidden input it brings.
var hiddenImports = map[string]string{
	"github.com/townsendmerino/aikit/gpu":   hiddenDevice,
	"github.com/townsendmerino/aikit/bench": hiddenTiming,
}

// deviceModules are the modules whose every package drives a device.
var deviceModules = map[string]bool{"metal": true, "cuda": true, "gpu": true}

// compactCode renders a Go file's tokens with comments dropped and no space around "." or before "(", so a call reads
// as "exec.Command(" however it was laid out.
func compactCode(name string, src []byte) string {
	fset := token.NewFileSet()
	f := fset.AddFile(name, -1, len(src))
	var s scanner.Scanner
	s.Init(f, src, nil, 0)
	var b strings.Builder
	prev := token.ILLEGAL
	for {
		_, tok, lit := s.Scan()
		if tok == token.EOF {
			break
		}
		if b.Len() > 0 && tok != token.PERIOD && tok != token.LPAREN && prev != token.PERIOD {
			b.WriteByte(' ')
		}
		switch {
		case tok == token.STRING:
			b.WriteString(`""`) // a literal's text is not code; fileHidden reads literals on their own
		case lit != "" && tok != token.SEMICOLON:
			b.WriteString(lit)
		default:
			b.WriteString(tok.String())
		}
		prev = tok
	}
	return b.String()
}

// pkgScan is one package's own hidden reads: in its non-test files (which its importers' tests can run) and in all of
// its files; its module; and the goinfer packages its non-test and test files import.
type pkgScan struct {
	module           string
	prod, all        map[string]bool
	imports          map[string]bool
	testImports      map[string]bool
	depth            int  // directories from its module's root to it
	climb, climbProd int  // the most its relative paths can climb (fileHidden), in all its files / its non-test files
	callerProd       bool // a non-test file locates files by runtime.Caller
}

// scanTestCache scans the tree the way scanImports does and returns every package's own hidden reads.
func scanTestCache(t *testing.T) map[string]*pkgScan {
	t.Helper()
	return scanTestCacheAt(t, repoRoot)
}

// scanTestCacheAt is scanTestCache over the tree at root.
func scanTestCacheAt(t *testing.T, repoRoot string) map[string]*pkgScan {
	t.Helper()
	modules := map[string]bool{".": true}
	pkgs := map[string]*pkgScan{}
	fset := token.NewFileSet()
	err := filepath.WalkDir(repoRoot, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(repoRoot, path)
		rel = filepath.ToSlash(rel)
		if d.IsDir() {
			n := d.Name()
			if path != repoRoot && (n == "testdata" || n == "node_modules" || n == "vendor" || strings.HasPrefix(n, ".") || strings.HasPrefix(n, "_")) {
				return filepath.SkipDir
			}
			if _, err := os.Stat(filepath.Join(path, "go.mod")); err == nil {
				modules[rel] = true
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		src, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		f, err := parser.ParseFile(fset, path, src, parser.ImportsOnly|parser.ParseComments)
		if err != nil {
			return err
		}
		if ignored(f.Comments, f.Package) {
			return nil
		}
		dir := filepath.ToSlash(filepath.Dir(rel))
		p := pkgs[dir]
		if p == nil {
			p = &pkgScan{prod: map[string]bool{}, all: map[string]bool{}, imports: map[string]bool{}, testImports: map[string]bool{}}
			pkgs[dir] = p
		}
		test := strings.HasSuffix(path, "_test.go")
		mark := func(kind string) {
			p.all[kind] = true
			if !test {
				p.prod[kind] = true
			}
		}
		kinds, climb, caller := fileHidden(path, src)
		for kind := range kinds {
			mark(kind)
		}
		p.climb = max(p.climb, climb)
		if !test {
			p.climbProd = max(p.climbProd, climb)
			p.callerProd = p.callerProd || caller
		}
		for _, im := range f.Imports {
			ip, err := strconv.Unquote(im.Path.Value)
			if err != nil {
				return err
			}
			if kind, ok := hiddenImports[ip]; ok {
				mark(kind)
			}
			if to := goinferRel(ip); to != "" && to != dir {
				if test {
					p.testImports[to] = true
				} else {
					p.imports[to] = true
				}
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("scanning the tree: %v", err)
	}
	for dir, p := range pkgs {
		m, d := ".", dir
		for d != "." && d != "" {
			if modules[d] {
				m = d
				break
			}
			d = filepath.ToSlash(filepath.Dir(d))
		}
		p.module = m
		if dir != "." {
			p.depth = strings.Count(dir, "/") + 1
		}
		if m != "." {
			p.depth -= strings.Count(m, "/") + 1
		}
		// A test runs in its package's directory, so a relative path that climbs past the module's root leaves it.
		if p.climb > p.depth {
			p.all[hiddenFiles] = true
		}
		if p.climbProd > p.depth {
			p.prod[hiddenFiles] = true
		}
		if deviceModules[m] {
			p.prod[hiddenDevice], p.all[hiddenDevice] = true, true
		}
	}
	// The probe is the one OS memory reader (MM0).
	if p := pkgs["internal/mem/probe"]; p != nil {
		p.prod[hiddenMemory], p.all[hiddenMemory] = true, true
	}
	return pkgs
}

// classifyTestCache gives every package its reasons: its own hidden reads (tests included) and the reasons declared for
// it by hand, then those of the non-test files and the declared reasons of every goinfer package it reaches, each named
// through the nearest such package (breadth first, ties by name). A relative path in a reached package resolves from the
// test's directory, so one that climbs past this package's module root is a file outside the module; so are the paths
// runtime.Caller gives in a reached package of another module.
func classifyTestCache(pkgs map[string]*pkgScan, declared map[string][]string) map[string]testCacheEntry {
	for dir, rs := range declared {
		if p := pkgs[dir]; p != nil {
			for _, r := range rs {
				p.prod[r], p.all[r] = true, true
			}
		}
	}
	out := map[string]testCacheEntry{}
	for dir, p := range pkgs {
		reasons := map[string]string{}
		for kind := range p.all {
			reasons[kind] = kind
		}
		seen := map[string]bool{dir: true}
		var frontier []string
		for to := range p.imports {
			frontier = append(frontier, to)
		}
		for to := range p.testImports {
			frontier = append(frontier, to)
		}
		for len(frontier) > 0 {
			sort.Strings(frontier)
			var next []string
			for _, d := range frontier {
				if seen[d] {
					continue
				}
				seen[d] = true
				q := pkgs[d]
				if q == nil {
					continue
				}
				for kind := range q.prod {
					if _, ok := reasons[kind]; !ok {
						reasons[kind] = kind + " (through " + d + ")"
					}
				}
				if _, ok := reasons[hiddenFiles]; !ok {
					switch {
					case q.climbProd > p.depth:
						reasons[hiddenFiles] = hiddenFiles + " (a relative read in " + d + ")"
					case q.callerProd && q.module != p.module:
						reasons[hiddenFiles] = hiddenFiles + " (runtime.Caller paths in " + d + ")"
					}
				}
				for to := range q.imports {
					next = append(next, to)
				}
			}
			frontier = next
		}
		e := testCacheEntry{Cacheable: len(reasons) == 0}
		for _, r := range reasons {
			e.Reasons = append(e.Reasons, r)
		}
		sort.Strings(e.Reasons)
		out[dir] = e
	}
	return out
}

// TestTestCache_classified is PW5 (docs/tasks/task-subsystems-2026-10.md §12.3): test_cache.json classifies every
// package as the scan does, so a cacheable package that gains a hidden input goes red in the change that adds it.
//
//	go test -run TestTestCache_classified ./internal/layout/            # the check
//	go test -run TestTestCache_classified ./internal/layout/ -update    # rewrite the packages (never the declared reasons)
func TestTestCache_classified(t *testing.T) {
	var data testCacheData
	raw, err := os.ReadFile(testCachePath)
	switch {
	case err == nil:
		if err := json.Unmarshal(raw, &data); err != nil {
			t.Fatalf("%s: %v", testCachePath, err)
		}
	case !*updateArrows || !os.IsNotExist(err):
		t.Fatal(err)
	}
	got := classifyTestCache(scanTestCache(t), data.Declared)
	if *updateArrows {
		data.Packages = got
		if data.Declared == nil {
			data.Declared = map[string][]string{}
		}
		if data.About == "" {
			data.About = "PW5 (docs/tasks/task-subsystems-2026-10.md §12.3): which packages Go's test cache may replay. packages is generated by TestTestCache_classified -update; declared is hand-edited, for hidden inputs a scan cannot see."
		}
		var buf bytes.Buffer
		enc := json.NewEncoder(&buf)
		enc.SetEscapeHTML(false)
		enc.SetIndent("", "  ")
		if err := enc.Encode(data); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(testCachePath, buf.Bytes(), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %s", testCachePath)
		return
	}
	dirs := make([]string, 0, len(got))
	for d := range got {
		dirs = append(dirs, d)
	}
	sort.Strings(dirs)
	cacheable := 0
	for _, d := range dirs {
		g, w := got[d], data.Packages[d]
		if g.Cacheable {
			cacheable++
		}
		switch {
		case !g.Cacheable && w.Cacheable:
			t.Errorf("%s is marked cacheable, but the scan finds %s: Go's test cache would replay a result that depends on it. Remove the read, or record it (-update) so touched.py keeps -count=1", d, strings.Join(g.Reasons, "; "))
		case strings.Join(g.Reasons, "\x00") != strings.Join(w.Reasons, "\x00") || g.Cacheable != w.Cacheable:
			t.Errorf("%s: %s records %v (cacheable %v), the scan finds %v (cacheable %v): run with -update", d, testCachePath, w.Reasons, w.Cacheable, g.Reasons, g.Cacheable)
		}
	}
	for d := range data.Packages {
		if _, ok := got[d]; !ok {
			t.Errorf("%s records %s, which is no longer a package: run with -update", testCachePath, d)
		}
	}
	for d, rs := range data.Declared {
		if _, ok := got[d]; !ok || len(rs) == 0 {
			t.Errorf("%s declares %q for %s, which is not a package or names no reason", testCachePath, rs, d)
		}
	}
	t.Logf("%d packages, %d cacheable, %d declared by hand", len(got), cacheable, len(data.Declared))
}

// TestClassifyTestCache_propagates holds classifyTestCache's two rules a scan of today's tree cannot reach: a reason
// declared by hand makes its package and every importer non-cacheable, and a nested-module package takes a climbing
// relative read from the non-test files of a root-module package it reaches.
func TestClassifyTestCache_propagates(t *testing.T) {
	pkg := func(module string, imports ...string) *pkgScan {
		p := &pkgScan{module: module, prod: map[string]bool{}, all: map[string]bool{}, imports: map[string]bool{}, testImports: map[string]bool{}}
		for _, i := range imports {
			p.imports[i] = true
		}
		return p
	}
	pkgs := map[string]*pkgScan{
		"a":          pkg("."),
		"b":          pkg(".", "a"),
		"c":          pkg("."),
		"nested/x":   pkg("nested", "r"),
		"r":          pkg("."),
		"plain":      pkg("."),
		"nested/own": pkg("nested", "plain"),
	}
	pkgs["nested/x"].depth, pkgs["nested/own"].depth = 1, 1
	pkgs["r"].climb, pkgs["r"].climbProd = 2, 2
	got := classifyTestCache(pkgs, map[string][]string{"a": {"timing"}})
	want := map[string][]string{
		"a":        {"timing"},
		"b":        {"timing (through a)"},
		"nested/x": {"files outside the module (a relative read in r)"},
	}
	for dir, e := range got {
		w := want[dir]
		if strings.Join(e.Reasons, "|") != strings.Join(w, "|") || e.Cacheable != (len(w) == 0) {
			t.Errorf("%s: got %+v, want reasons %v", dir, e, w)
		}
	}
}

// TestFileHidden_markers runs source through fileHidden, as the scan does: each call shape it must catch, and reads it
// must leave alone.
func TestFileHidden_markers(t *testing.T) {
	cases := []struct {
		code   string
		kind   string // "" for none
		climb  int
		caller bool
	}{
		{`_ = filepath.Join("..", "x")`, "", 1, false},
		{`_ = filepath.Join("..", "..", "x")`, "", 2, false},
		{`_ = filepath.Join("..", "../..", "x")`, "", 3, false},
		{`if v == ".." || w == ".." { }`, "", 1, false},
		{`_, _ = os.Open("..")`, "", 1, false},
		{"_ = `../../a`", "", 2, false},
		{`_ = "./../b"`, "", 1, false},
		{`const repoRoot = "../../.."`, "", 3, false},
		{"_ = \"time.Now() exec.Command(\\\"x\\\")\"", "", 0, false},
		{`_ = filepath.Join("/proc", "meminfo")`, hiddenFiles, 0, false},
		{`_, _ = os.ReadDir("/tmp")`, hiddenFiles, 0, false},
		{"_ = `/var/log/x`", hiddenFiles, 0, false},
		{`_ = os.TempDir()`, hiddenFiles, 0, false},
		{`<-time.After(time.Second)`, hiddenTiming, 0, false},
		{`_, _ = context.WithTimeout(context.Background(), 1)`, hiddenTiming, 0, false},
		{`_ = time.Now()`, hiddenTiming, 0, false},
		{`f := time.After; _ = f`, hiddenTiming, 0, false},
		{`_, _ = http.Post("u", "t", nil)`, hiddenNet, 0, false},
		{`_, _ = net.Listen("tcp", ":0")`, hiddenNet, 0, false},
		{`_, _ = exec.Command("git").Output()`, hiddenExec, 0, false},
		{`_, _, _, _ = runtime.Caller(0)`, "", 0, true},
		{`_, _ = os.ReadFile("testdata/x.json")`, "", 0, false},
		{`_ = "a/../b"`, "", 0, false},
	}
	for _, c := range cases {
		src := []byte("package x\n\nfunc f() {\n\t" + c.code + "\n}\n")
		kinds, climb, caller := fileHidden("x.go", src)
		if c.kind != "" && !kinds[c.kind] {
			t.Errorf("%s: kinds %v, want %q", c.code, kinds, c.kind)
		}
		if c.kind == "" && len(kinds) > 0 {
			t.Errorf("%s: kinds %v, want none", c.code, kinds)
		}
		if climb != c.climb || caller != c.caller {
			t.Errorf("%s: climb %d caller %v, want %d %v", c.code, climb, caller, c.climb, c.caller)
		}
	}
}

// TestScanTestCache_tree scans a small tree with a nested module through scanTestCacheAt and classifyTestCache: a
// relative read that climbs past its own module root, one that climbs past an importer's in another module, and
// runtime.Caller paths reached from another module are files outside the module; the same reads inside a module are not,
// and a climb in a test file stays with its own package.
func TestScanTestCache_tree(t *testing.T) {
	root := t.TempDir()
	write := func(path, body string) {
		t.Helper()
		p := filepath.Join(root, path)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	pkg := func(name, imp, body string) string {
		s := "package " + name + "\n\n"
		if imp != "" {
			s += "import _ \"" + modulePath + "/" + imp + "\"\n\n"
		}
		return s + "func f() {\n\t" + body + "\n}\n"
	}
	write("go.mod", "module "+modulePath+"\n")
	write("a/b/b.go", pkg("b", "", `_ = filepath.Join("..", "..", "..", "x")`))
	write("a/c/c.go", pkg("c", "", `_ = filepath.Join("..", "..", "x")`))
	write("a/d/d.go", pkg("d", "", `_, _, _, _ = runtime.Caller(0)`))
	write("a/e/e.go", pkg("e", "a/d", ``))
	write("a/f/f.go", pkg("f", "", ``))
	write("a/f/f_test.go", pkg("f", "", `_ = filepath.Join("..", "..", "..", "x")`))
	write("a/g/g.go", pkg("g", "a/f", ``))
	write("n/go.mod", "module "+modulePath+"/n\n")
	write("n/p/p.go", pkg("p", "a/c", ``))
	write("n/q/q.go", pkg("q", "a/d", ``))
	got := classifyTestCache(scanTestCacheAt(t, root), nil)
	want := map[string]string{
		"a/b": hiddenFiles,
		"a/c": "",
		"a/d": "",
		"a/e": "",
		"a/f": hiddenFiles,
		"a/g": "",
		"n/p": hiddenFiles + " (a relative read in a/c)",
		"n/q": hiddenFiles + " (runtime.Caller paths in a/d)",
	}
	for dir, w := range want {
		e, ok := got[dir]
		if !ok {
			t.Errorf("%s: not scanned", dir)
			continue
		}
		if strings.Join(e.Reasons, "|") != w || e.Cacheable != (w == "") {
			t.Errorf("%s: got %+v, want %q", dir, e, w)
		}
	}
}
