package layout

import (
	"encoding/json"
	"fmt"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// testFuncRE finds the tests a _test.go file declares.
var testFuncRE = regexp.MustCompile(`(?m)^func (Test[A-Za-z0-9_]*)\(`)

// testMentionRE finds a test named in prose, with the package before it when the prose gives one ("decoder's TestX",
// "internal/layout's TestY").
var testMentionRE = regexp.MustCompile(`(?:([A-Za-z0-9_./-]+)'s\s+)?\b(Test[A-Z][A-Za-z0-9_]*)\b`)

// goTestPkgRE finds the package of a `go test ./path` command in prose.
var goTestPkgRE = regexp.MustCompile(`go test\b[^\n]*?\s\./([A-Za-z0-9_./-]+)`)

// packageDoc is one package doc comment: the file it sits in, its package directory, and its text.
type packageDoc struct{ file, dir, text string }

// scanDocs returns every package doc comment in the tree's non-test files and the tests each directory declares, with
// scanImports' skips (testdata, hidden and underscore directories).
func scanDocs(t *testing.T) ([]packageDoc, map[string]map[string]bool) {
	t.Helper()
	var docs []packageDoc
	tests := map[string]map[string]bool{}
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
		rel, _ := filepath.Rel(repoRoot, path)
		rel = filepath.ToSlash(rel)
		dir := filepath.ToSlash(filepath.Dir(rel))
		if strings.HasSuffix(path, "_test.go") {
			b, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			for _, m := range testFuncRE.FindAllSubmatch(b, -1) {
				if tests[dir] == nil {
					tests[dir] = map[string]bool{}
				}
				tests[dir][string(m[1])] = true
			}
			return nil
		}
		f, err := parser.ParseFile(fset, path, nil, parser.PackageClauseOnly|parser.ParseComments)
		if err != nil {
			return err
		}
		if f.Doc != nil {
			docs = append(docs, packageDoc{file: rel, dir: dir, text: f.Doc.Text()})
		}
		return nil
	})
	if err != nil {
		t.Fatalf("scanning the tree: %v", err)
	}
	return docs, tests
}

// TestPackageDocs_namedTestsExist is SB14's first half (docs/tasks/task-subsystems-2026-10.md §8, "SB14's gates"): every
// test a package doc comment names exists, in the package the prose puts it in (its own when it names none), and every
// line of an "Invariants" block names one. A doc that promises coverage by a test that is gone, or by none, fails here.
func TestPackageDocs_namedTestsExist(t *testing.T) {
	docs, tests := scanDocs(t)
	named := 0
	for _, d := range docs {
		// A possessive may sit on the line before the name, so mentions are found in the whole text; in a
		// `go test ./pkg … -run TestX` command, the package is the nearest command's on the mention's line.
		var mentions [][]string
		cmds := goTestPkgRE.FindAllStringSubmatchIndex(d.text, -1)
		for _, ix := range testMentionRE.FindAllStringSubmatchIndex(d.text, -1) {
			m := []string{d.text[ix[0]:ix[1]], "", d.text[ix[4]:ix[5]]}
			if ix[2] >= 0 {
				m[1] = d.text[ix[2]:ix[3]]
			} else {
				lineStart := strings.LastIndex(d.text[:ix[0]], "\n") + 1
				for _, c := range cmds {
					if c[0] >= lineStart && c[1] <= ix[0] {
						m[1] = strings.TrimSuffix(strings.TrimSuffix(d.text[c[2]:c[3]], "/..."), "/")
					}
				}
			}
			mentions = append(mentions, m)
		}
		for _, m := range mentions {
			pkg, name := m[1], m[2]
			named++
			dirs := []string{d.dir}
			if pkg != "" {
				dirs = nil
				for dir := range tests {
					if dir == pkg || strings.HasSuffix(dir, "/"+pkg) {
						dirs = append(dirs, dir)
					}
				}
			}
			found := false
			for _, dir := range dirs {
				if tests[dir][name] {
					found = true
				}
			}
			if !found {
				where := "its own package (" + d.dir + ")"
				if pkg != "" {
					where = pkg
				}
				t.Errorf("%s names %s in %s, which declares no such test", d.file, name, where)
			}
		}
		lines := strings.Split(d.text, "\n")
		for i, l := range lines {
			if !strings.Contains(l, "Invariants") {
				continue
			}
			for _, inv := range lines[i+1:] {
				if strings.TrimSpace(inv) == "" {
					continue
				}
				if !strings.HasPrefix(inv, "\t") {
					break
				}
				if !testMentionRE.MatchString(inv) {
					t.Errorf("%s: the invariant %q names no test", d.file, strings.TrimSpace(inv))
				}
			}
		}
	}
	if named == 0 {
		t.Fatal("no package doc names a test: the scan covered nothing")
	}
	t.Logf("%d package docs, %d test names checked", len(docs), named)
}

const (
	architecturePath = repoRoot + "/docs/ARCHITECTURE.md"
	appendixBegin    = "<!-- imports:begin (generated from internal/layout/imports.json; go test -run TestArchitecture_importAppendix ./internal/layout/ -update) -->"
	appendixEnd      = "<!-- imports:end -->"
)

// importAppendix renders imports.json as docs/ARCHITECTURE.md's appendix: a row per importing package, then the rules.
func importAppendix(data layoutData) string {
	var b strings.Builder
	b.WriteString(appendixBegin + "\n\n## Appendix: which package imports which\n\n")
	b.WriteString("Every import from one goinfer package to another, by repository-relative directory, from `internal/layout/imports.json`. ")
	b.WriteString("`TestLayout_arrows` holds the data against the tree; this table is generated from it, and `TestArchitecture_importAppendix` fails when they differ.\n\n")
	b.WriteString("| package | imports | its tests also import |\n|---|---|---|\n")
	froms := map[string]bool{}
	for f := range data.Arrows {
		froms[f] = true
	}
	for f := range data.TestArrows {
		froms[f] = true
	}
	keys := make([]string, 0, len(froms))
	for f := range froms {
		keys = append(keys, f)
	}
	sort.Strings(keys)
	code := func(xs []string) string {
		out := make([]string, len(xs))
		for i, x := range xs {
			out[i] = "`" + x + "`"
		}
		return strings.Join(out, ", ")
	}
	for _, f := range keys {
		prod := map[string]bool{}
		for _, x := range data.Arrows[f] {
			prod[x] = true
		}
		var extra []string
		for _, x := range data.TestArrows[f] {
			if !prod[x] {
				extra = append(extra, x)
			}
		}
		fmt.Fprintf(&b, "| `%s` | %s | %s |\n", f, code(data.Arrows[f]), code(extra))
	}
	b.WriteString("\n### The protected separations\n\nNo import, made or recorded, may break one; they are edited by hand in `imports.json`.\n\n")
	b.WriteString("| rule | packages | may not import | why |\n|---|---|---|---|\n")
	for _, r := range data.Rules {
		var not []string
		for _, g := range r.ForbidGoinfer {
			if g == "*" {
				not = append(not, "any goinfer package")
			} else {
				not = append(not, "`"+g+"`")
			}
		}
		allowFrom := make([]string, 0, len(r.AllowGoinfer))
		for f := range r.AllowGoinfer {
			allowFrom = append(allowFrom, f)
		}
		sort.Strings(allowFrom)
		for _, f := range allowFrom {
			not = append(not, fmt.Sprintf("(`%s` may import %s)", f, strings.Join(codeEach(r.AllowGoinfer[f]), ", ")))
		}
		not = append(not, codeEach(r.ForbidExternal)...)
		if len(r.AllowExternal) > 0 {
			not = append(not, "anything outside the standard library but "+strings.Join(codeEach(r.AllowExternal), ", "))
		}
		if len(r.GoModForbid) > 0 {
			not = append(not, "(the root go.mod requires none of them)")
		}
		scope := ""
		if r.Tests {
			scope = ", tests included"
		}
		fmt.Fprintf(&b, "| %s | %s%s | %s | %s |\n", r.Name, strings.Join(codeEach(r.From), ", "), scope, strings.Join(not, "; "), r.Why)
	}
	b.WriteString("\n" + appendixEnd + "\n")
	return b.String()
}

func codeEach(xs []string) []string {
	out := make([]string, len(xs))
	for i, x := range xs {
		out[i] = "`" + x + "`"
	}
	return out
}

// TestArchitecture_importAppendix is SB14's second half: docs/ARCHITECTURE.md's import appendix is imports.json,
// rendered. -update regenerates it (appending it the first time).
func TestArchitecture_importAppendix(t *testing.T) {
	raw, err := os.ReadFile(importsPath)
	if err != nil {
		t.Fatal(err)
	}
	var data layoutData
	if err := json.Unmarshal(raw, &data); err != nil {
		t.Fatal(err)
	}
	want := importAppendix(data)
	doc, err := os.ReadFile(architecturePath)
	if err != nil {
		t.Fatal(err)
	}
	s := string(doc)
	i, j := strings.Index(s, appendixBegin), strings.Index(s, appendixEnd)
	if *updateArrows {
		if i < 0 || j < 0 {
			s = strings.TrimRight(s, "\n") + "\n\n" + want
		} else {
			s = s[:i] + want + strings.TrimPrefix(s[j+len(appendixEnd):], "\n")
		}
		if err := os.WriteFile(architecturePath, []byte(s), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote the import appendix of %s", architecturePath)
		return
	}
	if i < 0 || j < 0 || j < i {
		t.Fatalf("%s has no import appendix between its markers: run with -update", architecturePath)
	}
	if got := s[i : j+len(appendixEnd)+1]; got != want {
		t.Errorf("%s's import appendix differs from %s: run with -update (it is generated; do not edit it by hand)", architecturePath, importsPath)
	}
}
