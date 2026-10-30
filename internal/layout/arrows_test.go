package layout

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"
	"testing"

	"go/parser"
	"go/token"
)

// importsPath is SB13's data: every import from one goinfer package to another, and the rules no import may break.
// SB14 and docs/ARCHITECTURE.md read it.
const importsPath = "imports.json"

var updateArrows = flag.Bool("update", false, "rewrite imports.json's arrows from the tree and docs/ARCHITECTURE.md's import appendix from the data (the rules are hand-edited only)")

// layoutData is imports.json. Arrows and TestArrows map a package, by its directory relative to the repository root, to
// the goinfer packages its non-test and test files import. Rules are the protected separations.
type layoutData struct {
	About      string              `json:"_about"`
	Rules      []layoutRule        `json:"rules"`
	Arrows     map[string][]string `json:"arrows"`
	TestArrows map[string][]string `json:"test_arrows"`
}

// layoutRule constrains the imports of From (package directories, or "module:root" for every root-module package).
// ForbidGoinfer lists goinfer package prefixes it may not import ("*" is all of them); AllowGoinfer maps one From
// package to goinfer directories it may import despite ForbidGoinfer, each matched exactly (a subdirectory is not
// allowed); AllowExternal, when set, is the only non-standard, non-goinfer import prefixes it may use; StdOnly allows
// none; ForbidExternal lists ones it may not. Tests says whether its test files are held too. GoModForbid lists module
// paths the root go.mod may not require (module:root only).
type layoutRule struct {
	Name           string              `json:"name"`
	From           []string            `json:"from"`
	Tests          bool                `json:"tests"`
	ForbidGoinfer  []string            `json:"forbid_goinfer,omitempty"`
	AllowGoinfer   map[string][]string `json:"allow_goinfer,omitempty"`
	AllowExternal  []string            `json:"allow_external,omitempty"`
	StdOnly        bool                `json:"std_only,omitempty"`
	ForbidExternal []string            `json:"forbid_external,omitempty"`
	GoModForbid    []string            `json:"gomod_forbid,omitempty"`
	Why            string              `json:"why"`
}

// importUse is one import as the tree makes it: the importing package, whether from a test file, the file, and the path.
type importUse struct {
	from, file, path string
	test             bool
}

// scanImports parses every Go file the way scanTree does and returns its non-standard imports, with the module each
// package belongs to ("." for the root module).
func scanImports(t *testing.T) ([]importUse, map[string]string) {
	t.Helper()
	modules := map[string]bool{".": true}
	var uses []importUse
	dirs := map[string]bool{} // every package directory, standard-library-only ones too, so a rule can name one
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
		f, err := parser.ParseFile(fset, path, nil, parser.ImportsOnly|parser.ParseComments)
		if err != nil {
			return err
		}
		if ignored(f.Comments, f.Package) {
			return nil
		}
		dir := filepath.ToSlash(filepath.Dir(rel))
		dirs[dir] = true
		for _, im := range f.Imports {
			p, err := strconv.Unquote(im.Path.Value)
			if err != nil {
				return err
			}
			if !strings.Contains(strings.SplitN(p, "/", 2)[0], ".") {
				continue // the standard library
			}
			uses = append(uses, importUse{from: dir, file: rel, path: p, test: strings.HasSuffix(path, "_test.go")})
		}
		return nil
	})
	if err != nil {
		t.Fatalf("scanning the tree: %v", err)
	}
	moduleOf := map[string]string{}
	for from := range dirs {
		m, d := ".", from
		for d != "." && d != "" {
			if modules[d] {
				m = d
				break
			}
			d = filepath.ToSlash(filepath.Dir(d))
		}
		moduleOf[from] = m
	}
	return uses, moduleOf
}

// goinferRel is a goinfer import path as a repository-relative directory, or "" for any other path.
func goinferRel(p string) string {
	switch {
	case p == modulePath:
		return "."
	case strings.HasPrefix(p, modulePath+"/"):
		return strings.TrimPrefix(p, modulePath+"/")
	}
	return ""
}

func hasPrefixPath(p, prefix string) bool {
	return prefix == "*" || p == prefix || strings.HasPrefix(p, prefix+"/")
}

// breaks reports whether an import of path from the package directory from (from a test file when test) breaks r.
// moduleOf maps each package directory to its module ("." for the root), for a "module:root" rule.
func (r layoutRule) breaks(from, path string, test bool, moduleOf map[string]string) bool {
	if test && !r.Tests {
		return false
	}
	held := false
	for _, f := range r.From {
		if f == from || (f == "module:root" && moduleOf[from] == ".") {
			held = true
		}
	}
	if !held {
		return false
	}
	if rel := goinferRel(path); rel != "" {
		if rel == from || slices.Contains(r.AllowGoinfer[from], rel) {
			return false
		}
		for _, f := range r.ForbidGoinfer {
			if hasPrefixPath(rel, f) {
				return true
			}
		}
		return false
	}
	for _, f := range r.ForbidExternal {
		if hasPrefixPath(path, f) {
			return true
		}
	}
	if r.StdOnly {
		return true
	}
	if len(r.AllowExternal) > 0 {
		for _, a := range r.AllowExternal {
			if hasPrefixPath(path, a) {
				return false
			}
		}
		return true
	}
	return false
}

// TestLayout_arrows is SB13 (docs/tasks/task-subsystems-2026-10.md §8, "SB13's gates"): the goinfer→goinfer imports the
// tree makes equal the arrows imports.json records, and no import, made or recorded, breaks one of its rules. A new
// arrow is added to the data in the change that makes it; an arrow the tree stops making is removed from it.
//
//	go test -run TestLayout_arrows ./internal/layout/            # the check
//	go test -run TestLayout_arrows ./internal/layout/ -update    # rewrite the arrows (never the rules) from the tree
func TestLayout_arrows(t *testing.T) {
	raw, err := os.ReadFile(importsPath)
	if err != nil {
		t.Fatal(err)
	}
	var data layoutData
	if err := json.Unmarshal(raw, &data); err != nil {
		t.Fatalf("%s: %v", importsPath, err)
	}
	if len(data.Rules) == 0 {
		t.Fatalf("%s has no rules", importsPath)
	}
	uses, moduleOf := scanImports(t)

	made := map[bool]map[string]map[string]string{false: {}, true: {}} // test? -> from -> to -> a file that makes it
	for _, u := range uses {
		to := goinferRel(u.path)
		if to == "" || to == u.from {
			continue
		}
		m := made[u.test]
		if m[u.from] == nil {
			m[u.from] = map[string]string{}
		}
		if _, ok := m[u.from][to]; !ok {
			m[u.from][to] = u.file
		}
	}

	// The rules, against every import the tree makes and every arrow the data records.
	breaks := func(r layoutRule, from, path string, test bool) bool { return r.breaks(from, path, test, moduleOf) }
	for _, r := range data.Rules {
		for _, f := range r.From {
			if _, ok := moduleOf[f]; !ok && f != "module:root" {
				t.Errorf("rule %q names %s, which is not a package in the tree: a rule that matches nothing checks nothing", r.Name, f)
			}
		}
		for _, u := range uses {
			if breaks(r, u.from, u.path, u.test) {
				t.Errorf("rule %q: %s → %s (%s). %s", r.Name, u.from, u.path, u.file, r.Why)
			}
		}
		for test, arrows := range map[bool]map[string][]string{false: data.Arrows, true: data.TestArrows} {
			for from, tos := range arrows {
				for _, to := range tos {
					if breaks(r, from, modulePath+"/"+to, test) {
						t.Errorf("rule %q: %s records the arrow %s → %s, which the rule forbids; remove it from the data", r.Name, importsPath, from, to)
					}
				}
			}
		}
		if len(r.GoModForbid) > 0 {
			gomod, err := os.ReadFile(filepath.Join(repoRoot, "go.mod"))
			if err != nil {
				t.Fatal(err)
			}
			for _, line := range strings.Split(string(gomod), "\n") {
				f := strings.Fields(strings.TrimPrefix(strings.TrimSpace(line), "require "))
				for _, m := range r.GoModForbid {
					if len(f) > 0 && hasPrefixPath(f[0], m) {
						t.Errorf("rule %q: the root go.mod requires %s. %s", r.Name, f[0], r.Why)
					}
				}
			}
		}
	}

	if *updateArrows {
		data.Arrows, data.TestArrows = flatten(made[false]), flatten(made[true])
		var buf bytes.Buffer
		enc := json.NewEncoder(&buf)
		enc.SetEscapeHTML(false)
		enc.SetIndent("", "  ")
		if err := enc.Encode(data); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(importsPath, buf.Bytes(), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %s", importsPath)
		return
	}

	// The arrows, both ways.
	for test, recorded := range map[bool]map[string][]string{false: data.Arrows, true: data.TestArrows} {
		kind := "import"
		if test {
			kind = "test import"
		}
		rec := map[string]bool{}
		for from, tos := range recorded {
			for _, to := range tos {
				rec[from+"\x00"+to] = true
				if _, ok := made[test][from][to]; !ok {
					t.Errorf("%s records the %s arrow %s → %s, which the tree no longer makes: remove it from the data (an arrow only comes back by editing the data)", importsPath, kind, from, to)
				}
			}
		}
		var fresh []string
		for from, tos := range made[test] {
			for to, file := range tos {
				if !rec[from+"\x00"+to] {
					fresh = append(fresh, fmt.Sprintf("%s → %s (%s)", from, to, file))
				}
			}
		}
		sort.Strings(fresh)
		for _, a := range fresh {
			t.Errorf("a new %s arrow %s is not in %s: add it to the data in the change that makes it (-update), so the review sees it", kind, a, importsPath)
		}
	}
	n, nt := 0, 0
	for _, tos := range data.Arrows {
		n += len(tos)
	}
	for _, tos := range data.TestArrows {
		nt += len(tos)
	}
	t.Logf("%d arrows and %d test arrows between goinfer packages; %d rules", n, nt, len(data.Rules))
}

func flatten(m map[string]map[string]string) map[string][]string {
	out := map[string][]string{}
	for from, tos := range m {
		for to := range tos {
			out[from] = append(out[from], to)
		}
		sort.Strings(out[from])
	}
	return out
}

// TestLayout_towersStandalone holds towers-standalone to its one exception: internal/audio may import internal/vision
// (the tower program type; Francis, 2026-10-29), and every other goinfer import from either tower still breaks it.
func TestLayout_towersStandalone(t *testing.T) {
	raw, err := os.ReadFile(importsPath)
	if err != nil {
		t.Fatal(err)
	}
	var data layoutData
	if err := json.Unmarshal(raw, &data); err != nil {
		t.Fatal(err)
	}
	i := slices.IndexFunc(data.Rules, func(r layoutRule) bool { return r.Name == "towers-standalone" })
	if i < 0 {
		t.Fatalf("%s has no towers-standalone rule", importsPath)
	}
	r := data.Rules[i]
	moduleOf := map[string]string{"internal/audio": ".", "internal/vision": "."}
	g := func(rel string) string { return modulePath + "/" + rel }
	for _, test := range []bool{false, true} {
		for _, c := range []struct {
			from, path string
			breaks     bool
		}{
			{"internal/audio", g("internal/vision"), false},
			{"internal/audio", g("internal/vision/x"), true},
			{"internal/audio", g("decoder"), true},
			{"internal/audio", g("internal/mem"), true},
			{"internal/audio", modulePath, true},
			{"internal/audio", g("multimodal"), true},
			{"internal/vision", g("internal/audio"), true},
			{"internal/vision", g("decoder"), true},
			{"internal/audio", "github.com/townsendmerino/aikit/linalg", false},
			{"internal/audio", "github.com/townsendmerino/aikit/gpu", true},
			{"internal/vision", "golang.org/x/sys/unix", true},
		} {
			if got := r.breaks(c.from, c.path, test, moduleOf); got != c.breaks {
				t.Errorf("%s → %s (test file %v): breaks = %v, want %v", c.from, c.path, test, got, c.breaks)
			}
		}
	}
}
