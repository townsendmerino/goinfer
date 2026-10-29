package layout

import (
	"go/ast"
	"go/build/constraint"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
)

const (
	repoRoot    = "../.."
	modulePath  = "github.com/townsendmerino/goinfer"
	decoderPath = modulePath + "/decoder"
)

// decoderRatchet is the committed count of each backend module's test files that depend on decoder. A count above
// its figure fails as a regression; a count below it fails until the figure is lowered to match, so each fall is
// recorded in the commit that made it.
var decoderRatchet = map[string]struct{ Direct, Reach int }{
	"cuda":  {Direct: 244, Reach: 309},
	"metal": {Direct: 243, Reach: 328},
	"gpu":   {Direct: 86, Reach: 96},
}

// decoderRatchetExempt names the test files decoderRatchet does not count: the backend layers of SB2's admission census
// (docs/tasks/task-subsystems-2026-10.md, "SB2 step 2"). A census of the decoder's admission has to reach decoder, since it
// loads models through it. The exemption is by exact path, so no other test can shelter under it.
var decoderRatchetExempt = map[string]bool{
	"cuda/sb2_census_test.go":  true,
	"gpu/sb2_census_test.go":   true,
	"metal/sb2_census_test.go": true,
}

// goFile is one parsed Go file: its directory relative to the repository root, whether it is a test file, and the
// goinfer import paths it names.
type goFile struct {
	dir, name string
	test      bool
	imports   []string
}

// scanTree parses the imports of every Go file under the repository root, skipping testdata, hidden and underscore
// directories and files constrained by `//go:build ignore`. Build tags are otherwise ignored, so a directory's
// imports are the union over every tag combination.
func scanTree(t *testing.T) []goFile {
	t.Helper()
	var files []goFile
	fset := token.NewFileSet()
	err := filepath.WalkDir(repoRoot, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			n := d.Name()
			if path != repoRoot && (n == "testdata" || n == "node_modules" || n == "vendor" || strings.HasPrefix(n, ".") || strings.HasPrefix(n, "_")) {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		f, err := parser.ParseFile(fset, path, nil, parser.ImportsOnly|parser.ParseComments)
		if err != nil {
			return err
		}
		if ignored(f.Comments, f.Package) {
			return nil
		}
		rel, err := filepath.Rel(repoRoot, filepath.Dir(path))
		if err != nil {
			return err
		}
		g := goFile{dir: filepath.ToSlash(rel), name: filepath.Base(path), test: strings.HasSuffix(path, "_test.go")}
		for _, im := range f.Imports {
			p, err := strconv.Unquote(im.Path.Value)
			if err != nil {
				return err
			}
			if p == modulePath || strings.HasPrefix(p, modulePath+"/") {
				g.imports = append(g.imports, p)
			}
		}
		files = append(files, g)
		return nil
	})
	if err != nil {
		t.Fatalf("scanning the tree: %v", err)
	}
	return files
}

// ignored reports whether a file's `//go:build` line excludes it under every tag set that does not name "ignore".
func ignored(comments []*ast.CommentGroup, pkg token.Pos) bool {
	for _, cg := range comments {
		if cg.Pos() >= pkg {
			break
		}
		for _, c := range cg.List {
			if constraint.IsGoBuild(c.Text) {
				if x, err := constraint.Parse(c.Text); err == nil && !x.Eval(func(tag string) bool { return tag != "ignore" }) {
					return true
				}
			}
		}
	}
	return false
}

// importPath maps a directory relative to the repository root to its package path. Every goinfer module's path is the
// root module's path plus its directory, so one rule covers them all.
func importPath(dir string) string {
	if dir == "." {
		return modulePath
	}
	return modulePath + "/" + dir
}

type backendCount struct{ tests, direct, reach int }

// census counts, per backend module, its test files, those importing decoder directly, and those in a directory whose
// test build reaches decoder. byDir lists each reaching directory's test-file count for the -v output.
func census(files []goFile) (map[string]backendCount, map[string]int) {
	libImports := map[string][]string{}  // package path -> imports of its non-test files
	testImports := map[string][]string{} // directory -> imports of every file there, test files included
	for _, f := range files {
		p := importPath(f.dir)
		if !f.test {
			libImports[p] = append(libImports[p], f.imports...)
		}
		testImports[f.dir] = append(testImports[f.dir], f.imports...)
	}
	memo := map[string]bool{}
	var reaches func(p string, seen map[string]bool) bool
	reaches = func(p string, seen map[string]bool) bool {
		if p == decoderPath {
			return true
		}
		if v, ok := memo[p]; ok {
			return v
		}
		if seen[p] {
			return false
		}
		seen[p] = true
		r := false
		for _, q := range libImports[p] {
			if reaches(q, seen) {
				r = true
				break
			}
		}
		memo[p] = r
		return r
	}
	dirReaches := map[string]bool{}
	for dir, ims := range testImports {
		for _, q := range ims {
			if reaches(q, map[string]bool{}) {
				dirReaches[dir] = true
				break
			}
		}
	}
	counts := map[string]backendCount{}
	byDir := map[string]int{}
	for _, f := range files {
		if !f.test || decoderRatchetExempt[f.dir+"/"+f.name] {
			continue
		}
		mod := strings.SplitN(f.dir, "/", 2)[0]
		if _, ok := decoderRatchet[mod]; !ok {
			continue
		}
		c := counts[mod]
		c.tests++
		for _, q := range f.imports {
			if q == decoderPath {
				c.direct++
				break
			}
		}
		if dirReaches[f.dir] {
			c.reach++
			byDir[f.dir]++
		}
		counts[mod] = c
	}
	return counts, byDir
}

// TestBackendTests_decoderImportsOnlyFall holds decoderRatchet. A backend kernel test that needs nothing from decoder
// belongs in a package whose test build does not reach it (metal/kernels, cuda/kernels), so a decoder edit leaves its
// test cache alone.
func TestBackendTests_decoderImportsOnlyFall(t *testing.T) {
	files := scanTree(t)
	counts, byDir := census(files)
	if len(counts) != len(decoderRatchet) {
		t.Fatalf("found test files in %d of the %d backend modules; the scan is broken", len(counts), len(decoderRatchet))
	}
	mods := make([]string, 0, len(decoderRatchet))
	for m := range decoderRatchet {
		mods = append(mods, m)
	}
	sort.Strings(mods)
	for _, m := range mods {
		c, want := counts[m], decoderRatchet[m]
		t.Logf("%-6s test files %4d, importing decoder %4d, reaching decoder %4d", m, c.tests, c.direct, c.reach)
		check := func(what string, got, figure int) {
			switch {
			case got > figure:
				t.Errorf("%s: %d test files %s decoder, above the ratchet's %d. Move the kernel-level test into a package "+
					"that does not import decoder (cuda/kernels, metal/kernels) instead of raising the figure.", m, got, what, figure)
			case got < figure:
				t.Errorf("%s: %d test files %s decoder, below the ratchet's %d. Lower decoderRatchet[%q] to %d in this commit.",
					m, got, what, figure, m, got)
			}
		}
		check("importing", c.direct, want.Direct)
		check("reaching", c.reach, want.Reach)
	}
	dirs := make([]string, 0, len(byDir))
	for d := range byDir {
		dirs = append(dirs, d)
	}
	sort.Strings(dirs)
	for _, d := range dirs {
		t.Logf("  reaching: %-40s %4d", d, byDir[d])
	}
}

// TestCensus_countsAPlantedChain checks the reach rule on a synthetic tree: a test file reaches decoder through a
// package that imports it, a test file beside a non-test file that imports it, and not through a test file of another
// directory.
func TestCensus_countsAPlantedChain(t *testing.T) {
	mid := modulePath + "/internal/mid"
	files := []goFile{
		{dir: "decoder", name: "d.go"},
		{dir: "internal/mid", name: "m.go", imports: []string{decoderPath}},
		{dir: "internal/clean", name: "c_test.go", test: true, imports: []string{decoderPath}},
		{dir: "internal/clean", name: "c.go"},
		{dir: "cuda/kernels", name: "k_test.go", test: true, imports: []string{modulePath + "/internal/clean"}},
		{dir: "cuda", name: "a.go", imports: []string{mid}},
		{dir: "cuda", name: "a_test.go", test: true},
		{dir: "cuda", name: "b_test.go", test: true, imports: []string{decoderPath}},
		{dir: "gpu", name: "g_test.go", test: true, imports: []string{mid}},
		{dir: "metal", name: "x_test.go", test: true},
	}
	counts, _ := census(files)
	want := map[string]backendCount{
		"cuda":  {tests: 3, direct: 1, reach: 2},
		"gpu":   {tests: 1, direct: 0, reach: 1},
		"metal": {tests: 1, direct: 0, reach: 0},
	}
	for m, w := range want {
		if counts[m] != w {
			t.Errorf("%s: got %+v, want %+v", m, counts[m], w)
		}
	}
}

// TestScanTree_skipsIgnoredFiles checks that a `//go:build ignore` file is not counted and a tagged one is.
func TestScanTree_skipsIgnoredFiles(t *testing.T) {
	dir := t.TempDir()
	write := func(name, body string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("ignored.go", "//go:build ignore\n\npackage x\n\nimport _ \""+decoderPath+"\"\n")
	write("tagged_test.go", "//go:build cuda && goinfer_testhooks\n\npackage x\n\nimport _ \""+decoderPath+"\"\n")
	fset := token.NewFileSet()
	var kept []string
	for _, n := range []string{"ignored.go", "tagged_test.go"} {
		f, err := parser.ParseFile(fset, filepath.Join(dir, n), nil, parser.ImportsOnly|parser.ParseComments)
		if err != nil {
			t.Fatal(err)
		}
		if !ignored(f.Comments, f.Package) {
			kept = append(kept, n)
		}
	}
	if len(kept) != 1 || kept[0] != "tagged_test.go" {
		t.Errorf("kept %v, want [tagged_test.go]", kept)
	}
}

// serveappDecoderTests is the committed count of internal/serveapp's test files that import decoder (SB7). The serving
// shell's tests move onto a fake engine in batches; each batch lowers this figure in the commit that moves them, and a
// count above it fails as a regression.
const serveappDecoderTests = 31

// TestServeappTests_decoderImportsOnlyFall holds serveappDecoderTests: a count above it fails, and so does one below it
// until the figure is lowered to match.
func TestServeappTests_decoderImportsOnlyFall(t *testing.T) {
	var got []string
	for _, f := range scanTree(t) {
		if f.dir != "internal/serveapp" || !f.test {
			continue
		}
		for _, q := range f.imports {
			if q == decoderPath {
				got = append(got, f.name)
				break
			}
		}
	}
	sort.Strings(got)
	switch n := len(got); {
	case n > serveappDecoderTests:
		t.Errorf("%d internal/serveapp test files import decoder, above the ratchet's %d: test the serving shell on the fake engine instead (docs/tasks/task-subsystems-2026-10.md, SB7). Files: %s",
			n, serveappDecoderTests, strings.Join(got, ", "))
	case n < serveappDecoderTests:
		t.Errorf("%d internal/serveapp test files import decoder, below the ratchet's %d: lower serveappDecoderTests to %d in this change", n, serveappDecoderTests, n)
	}
}
