package site

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const goodWriteup = `---
title: "A title"
area: "Area"
order: 7
summary: "One sentence."
stand: "A standfirst."
measured: 2026-09-01
reviewed: ""
doesnt:
  - {title: "It isn't this.", text: "Because."}
figures:
  - {text: "1.83", source: docs/rec.md}
sources: [docs/rec.md]
---
## The problem

It went 1.83 times faster.
`

func writeupRoot(t *testing.T, files map[string]string) string {
	t.Helper()
	root := t.TempDir()
	dir := filepath.Join(root, "site", "different")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for n, c := range files {
		p := filepath.Join(root, n)
		if !strings.Contains(n, "/") {
			p = filepath.Join(dir, n)
		}
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(c), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func TestLoadWriteups_readsAGoodOne(t *testing.T) {
	root := writeupRoot(t, map[string]string{"07-a-title.md": goodWriteup})
	ws, err := LoadWriteups(root)
	if err != nil || len(ws) != 1 {
		t.Fatalf("got %v, %v", ws, err)
	}
	if !ws[0].Draft() {
		t.Error("an empty reviewed date is a draft")
	}
	if !strings.Contains(string(ws[0].HTML), "<h2") {
		t.Errorf("the body was not rendered as Markdown: %q", ws[0].HTML)
	}
}

func TestLoadWriteups_refusesBadFiles(t *testing.T) {
	cases := map[string]struct{ file, content, want string }{
		"no front matter":     {"07-a-title.md", "just words\n", "no front matter"},
		"unknown key":         {"07-a-title.md", strings.Replace(goodWriteup, "area:", "colour: red\narea:", 1), "colour"},
		"order vs file name":  {"08-a-title.md", goodWriteup, "does not match the file name"},
		"bad slug":            {"a-title.md", goodWriteup, "13-some-words.md"},
		"no limits":           {"07-a-title.md", strings.Replace(goodWriteup, "doesnt:\n  - {title: \"It isn't this.\", text: \"Because.\"}\n", "", 1), "at least one thing it doesn't do"},
		"its own limits head": {"07-a-title.md", goodWriteup + "\n## What it doesn't do\n", "renders the front matter"},
		"bad date":            {"07-a-title.md", strings.Replace(goodWriteup, "2026-09-01", "September", 1), "measured"},
		"long summary":        {"07-a-title.md", strings.Replace(goodWriteup, "One sentence.", strings.Repeat("x", 161), 1), "a card holds 160"},
	}
	for name, c := range cases {
		root := writeupRoot(t, map[string]string{c.file: c.content})
		if _, err := LoadWriteups(root); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: want an error containing %q, got %v", name, c.want, err)
		}
	}
	root := writeupRoot(t, map[string]string{"07-a-title.md": goodWriteup, "07-other-one.md": goodWriteup})
	if _, err := LoadWriteups(root); err == nil || !strings.Contains(err.Error(), "order 7 is also") {
		t.Errorf("two writeups with one order must be refused, got %v", err)
	}
}

// The figure check is the point of listing figures: a number must be in the page AND in the record that prints it.
func TestCheckWriteups_figuresMustBeInBothPlaces(t *testing.T) {
	root := writeupRoot(t, map[string]string{"07-a-title.md": goodWriteup, "docs/rec.md": "the speed-up was 1.83x on the 7B\n"})
	ws, err := LoadWriteups(root)
	if err != nil {
		t.Fatal(err)
	}
	if err := CheckWriteups(root, ws); err != nil {
		t.Fatalf("a supported figure must pass: %v", err)
	}
	// the record says 1.8, not 1.83: the writeup would be claiming more digits than it has
	if err := os.WriteFile(filepath.Join(root, "docs", "rec.md"), []byte("the speed-up was 1.8x\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := CheckWriteups(root, ws); err == nil || !strings.Contains(err.Error(), `"1.83" does not appear in docs/rec.md`) {
		t.Errorf("a figure the record does not print must fail, got %v", err)
	}
	// listed but never in the page
	ws[0].Body = "nothing quantitative here"
	if err := os.WriteFile(filepath.Join(root, "docs", "rec.md"), []byte("1.83\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := CheckWriteups(root, ws); err == nil || !strings.Contains(err.Error(), "never appears in the page") {
		t.Errorf("a listed figure absent from the page must fail, got %v", err)
	}
	// a missing source
	ws[0].Body = goodWriteup
	ws[0].Sources = []string{"nope.md"}
	if err := CheckWriteups(root, ws); err == nil || !strings.Contains(err.Error(), "nope.md does not exist") {
		t.Errorf("a source that does not exist must fail, got %v", err)
	}
}

func TestVisible_hidesDraftsUnlessPreviewing(t *testing.T) {
	a := &Writeup{Slug: "01-a", Order: 1, Reviewed: "2026-09-29"}
	b := &Writeup{Slug: "02-b", Order: 2}
	c := &Writeup{Slug: "03-c", Order: 3, Reviewed: "2026-09-29"}
	deploy := Visible([]*Writeup{a, b, c}, false)
	if len(deploy) != 2 || deploy[0] != a || deploy[1] != c {
		t.Fatalf("a deploy shows reviewed writeups only, got %v", deploy)
	}
	if a.Next != c || c.Prev != a || a.Pos != 1 || c.Pos != 2 || c.Of != 2 {
		t.Errorf("neighbours and numbering must skip the draft: %+v %+v", a, c)
	}
	if got := Visible([]*Writeup{a, b, c}, true); len(got) != 3 || b.Pos != 2 {
		t.Errorf("a preview shows drafts too, got %d", len(got))
	}
}

// A draft in a deploy build is a mistake the output check must catch, even though Visible normally prevents it.
func TestVerify_refusesADraftInADeploy(t *testing.T) {
	needRepo(t)
	out := t.TempDir()
	m, err := Derive(realInputs(t))
	if err != nil {
		t.Fatal(err)
	}
	m.Book = []BookChapter{{Num: 1, Title: "T", File: "01.html"}}
	docs, err := LoadDocs(repoRoot, DefaultConfig().GitHubURL)
	if err != nil {
		t.Fatal(err)
	}
	m.Docs = docs
	draft := &Writeup{Slug: "09-draft", Order: 9, Title: "A draft", Area: "A", Summary: "s", Stand: "s", Measured: "2026-09-01", Doesnt: []WLimit{{"t", "x"}}, HTML: "<p>x</p>"}
	ws := Visible([]*Writeup{draft}, true)
	if _, err := m.Render(out, DefaultConfig(), ws); err != nil {
		t.Fatal(err)
	}
	if err := Verify(out, m, ws, true); err != nil {
		t.Fatalf("a preview may carry a draft: %v", err)
	}
	if err := Verify(out, m, ws, false); err == nil || !strings.Contains(err.Error(), "unreviewed draft") {
		t.Errorf("a deploy build with a draft must fail, got %v", err)
	}
	b, _ := os.ReadFile(filepath.Join(out, "different", "09-draft", "index.html"))
	if !strings.Contains(string(b), `name="robots" content="noindex"`) {
		t.Error("a draft page must ask not to be indexed")
	}
}

// The templates escape an apostrophe as &#39;, so the output check must compare escaped titles: a title like "Batching
// that doesn't change the answer" must not fail the build.
func TestVerify_acceptsATitleWithAnApostrophe(t *testing.T) {
	needRepo(t)
	out := t.TempDir()
	m, err := Derive(realInputs(t))
	if err != nil {
		t.Fatal(err)
	}
	m.Book = []BookChapter{{Num: 1, Title: "T", File: "01.html"}}
	if m.Docs, err = LoadDocs(repoRoot, DefaultConfig().GitHubURL); err != nil {
		t.Fatal(err)
	}
	w := &Writeup{Slug: "05-apostrophe", Order: 5, Title: "It doesn't break & it's <fine>", Area: "A", Summary: "s", Stand: "s", Measured: "2026-09-01", Reviewed: "2026-09-29", Doesnt: []WLimit{{"t", "x"}}, HTML: "<p>x</p>"}
	ws := Visible([]*Writeup{w}, false)
	if _, err := m.Render(out, DefaultConfig(), ws); err != nil {
		t.Fatal(err)
	}
	if err := Verify(out, m, ws, false); err != nil {
		t.Errorf("a title with an apostrophe, an ampersand and angle brackets must verify: %v", err)
	}
}

// The link check is what keeps cross-links between writeups honest: a link to a page the build did not write (a typo, a draft
// left out of a deploy, a renamed slug) fails the build.
func TestCheckLinks_findsABrokenInternalLink(t *testing.T) {
	out := t.TempDir()
	write := func(rel, body string) {
		p := filepath.Join(out, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("index.html", `<a href="/different/01-a/">a</a> <a href="/book/x.html">book</a> <img src="/assets/x.css"> <a href="/different/01-a/#h">h</a>`)
	write("different/01-a/index.html", `<a href="/">home</a>`)
	write("assets/x.css", "")
	if err := CheckLinks(out); err != nil {
		t.Fatalf("every link resolves (and /book/ is not this build's): %v", err)
	}
	write("different/01-a/index.html", `<a href="/different/02-draft/">next</a>`)
	if err := CheckLinks(out); err == nil || !strings.Contains(err.Error(), "different/01-a/index.html -> /different/02-draft/") {
		t.Errorf("a link to a page that was not written must fail, naming both ends: %v", err)
	}
}

func TestCheckWriteups_repoLinksMustResolve(t *testing.T) {
	body := goodWriteup + "\nSee [the record](https://github.com/townsendmerino/goinfer/blob/main/docs/rec.md#part) and [gone](https://github.com/townsendmerino/goinfer/blob/main/docs/moved.md).\n"
	root := writeupRoot(t, map[string]string{"07-a-title.md": body, "docs/rec.md": "1.83\n"})
	ws, err := LoadWriteups(root)
	if err != nil {
		t.Fatal(err)
	}
	err = CheckWriteups(root, ws)
	if err == nil || !strings.Contains(err.Error(), "links to docs/moved.md") || strings.Contains(err.Error(), "docs/rec.md, which") {
		t.Errorf("a GitHub link to a missing path must fail, and only that one: %v", err)
	}
}
