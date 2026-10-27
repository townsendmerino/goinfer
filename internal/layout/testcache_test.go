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

// testCachePath is PW5's data (docs/tasks/task-subsystems-2026-10.md §12.2): every package, by its directory relative to
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
// "." or before "(").
var hiddenCalls = []struct {
	kind string
	re   *regexp.Regexp
}{
	{hiddenExec, regexp.MustCompile(`\bexec\.Command(Context)?\(`)},
	{hiddenEnviron, regexp.MustCompile(`\bos\.Environ\(`)},
	{hiddenFiles, regexp.MustCompile(`\bos\.UserHomeDir\(|\bAssetPathForTest\(|\bos\.Hostname\(|"/(Users|home|proc|sys|tmp|dev)/`)},
	{hiddenNet, regexp.MustCompile(`\bnet\.Dial|\bhttp\.Get\(`)},
	{hiddenTiming, regexp.MustCompile(`\btime\.Since\(|\btesting\.Benchmark\(`)},
}

// nestedEscape finds a relative path that climbs, in a package of a nested module: its module root is its own directory,
// so a file above it is outside the module.
var nestedEscape = regexp.MustCompile(`"\.\./`)

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
		if lit != "" && tok != token.SEMICOLON {
			b.WriteString(lit)
		} else {
			b.WriteString(tok.String())
		}
		prev = tok
	}
	return b.String()
}

// pkgScan is one package's own hidden reads: in its non-test files (which its importers' tests can run) and in all of
// its files; its module; and the goinfer packages its non-test and test files import.
type pkgScan struct {
	module       string
	prod, all    map[string]bool
	imports      map[string]bool
	testImports  map[string]bool
	prodExternal map[string]bool
}

// scanTestCache scans the tree the way scanImports does and returns every package's own hidden reads.
func scanTestCache(t *testing.T) map[string]*pkgScan {
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
			p = &pkgScan{prod: map[string]bool{}, all: map[string]bool{}, imports: map[string]bool{}, testImports: map[string]bool{}, prodExternal: map[string]bool{}}
			pkgs[dir] = p
		}
		test := strings.HasSuffix(path, "_test.go")
		mark := func(kind string) {
			if kind == hiddenTiming && !test {
				return // timing is a test that reads the clock; production code that times itself decides nothing a test checks
			}
			p.all[kind] = true
			if !test {
				p.prod[kind] = true
			}
		}
		code := compactCode(path, src)
		for _, h := range hiddenCalls {
			if h.re.MatchString(code) {
				mark(h.kind)
			}
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
		// A literal "../" in a nested module may climb out of it; the root module's root is the repository's.
		if nestedEscape.MatchString(code) {
			p.prodExternal[path] = true
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
		if m != "." && len(p.prodExternal) > 0 {
			for path := range p.prodExternal {
				if strings.HasSuffix(path, "_test.go") {
					p.all[hiddenFiles] = true
				} else {
					p.prod[hiddenFiles], p.all[hiddenFiles] = true, true
				}
			}
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

// classifyTestCache gives every package its reasons: its own hidden reads (tests included), then those of the non-test
// files of every goinfer package it reaches, each named through the nearest such package (breadth first, ties by name).
func classifyTestCache(pkgs map[string]*pkgScan) map[string]testCacheEntry {
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

// TestTestCache_classified is PW5 (docs/tasks/task-subsystems-2026-10.md §12.2): test_cache.json classifies every
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
	got := classifyTestCache(scanTestCache(t))
	if *updateArrows {
		data.Packages = got
		if data.Declared == nil {
			data.Declared = map[string][]string{}
		}
		if data.About == "" {
			data.About = "PW5 (docs/tasks/task-subsystems-2026-10.md §12.2): which packages Go's test cache may replay. packages is generated by TestTestCache_classified -update; declared is hand-edited, for hidden inputs a scan cannot see."
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
