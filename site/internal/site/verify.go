package site

import (
	"fmt"
	"html"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// Verify is the gate on what a build wrote (docs/tasks/task-site-2026-09.md, S7): every family in the capability matrix
// has a page that names it, the Models page links every one, and every measured figure that a claim carries is on its
// checkpoint's page with its date. It reads the files back, so a template or a loop that quietly skips a family fails
// here and not in production.
func Verify(out string, m *Model, ws []*Writeup, drafts bool) error {
	read := func(rel string) (string, error) {
		b, err := os.ReadFile(filepath.Join(out, filepath.FromSlash(rel)))
		return string(b), err
	}
	idx, err := read("models/index.html")
	if err != nil {
		return fmt.Errorf("verify: the Models page is missing: %w", err)
	}
	var bad []string
	// Embeddings are not a family, so nothing above generates them; a hand-written section covers them.
	if !strings.Contains(idx, `id="embeddings"`) || !strings.Contains(idx, "/v1/embeddings") {
		bad = append(bad, "the Models page has no Embeddings section")
	}
	// The Ollama section: its heading, its headline sentence, and every tag in the data, in the built page.
	if m.Ollama != nil {
		if !strings.Contains(idx, `id="ollama"`) || !strings.Contains(idx, html.EscapeString(m.Ollama.Headline)) {
			bad = append(bad, "the Models page has no Coming from Ollama? section, or it lacks the headline")
		}
		for _, e := range m.Ollama.Entries {
			if !strings.Contains(idx, "<code>"+html.EscapeString(e.Tag)+"</code>") {
				bad = append(bad, fmt.Sprintf("the Ollama section does not list %s", e.Tag))
			}
		}
	}
	// Every visible writeup has a page that names it and lists what it doesn't do, and the index links it. A draft must
	// never be linked from a deploy: the index of a non-preview build has no draft in it.
	di, err := read("different/index.html")
	if err != nil {
		return fmt.Errorf("verify: the What's different page is missing: %w", err)
	}
	// Every doc has a page, and none keeps a relative link: each was rewritten to a docs page or to GitHub.
	if _, err := read("docs/index.html"); err != nil {
		return fmt.Errorf("verify: the Docs page is missing: %w", err)
	}
	relLink := regexp.MustCompile(`href="(?:\./|\.\./|[a-zA-Z0-9_-]+\.md)`)
	for _, d := range m.Docs {
		page, err := read("docs/" + d.Slug + "/index.html")
		switch {
		case err != nil:
			bad = append(bad, fmt.Sprintf("doc %q has no page", d.Slug))
		case !strings.Contains(page, html.EscapeString(d.Title)):
			bad = append(bad, fmt.Sprintf("doc %q: its page never carries its title", d.Slug))
		case relLink.MatchString(page):
			bad = append(bad, fmt.Sprintf("doc %q still has a relative link", d.Slug))
		}
	}
	// The Download page names every binary and carries its checksum, or the build had no release data and says so.
	dl, err := read("download/index.html")
	if err != nil {
		return fmt.Errorf("verify: the Download page is missing: %w", err)
	}
	if m.Download != nil {
		for _, k := range m.Download.Kinds {
			for key, f := range k.Files {
				if !strings.Contains(dl, f.Name) || !strings.Contains(dl, f.SHA) || !strings.Contains(dl, f.URL) {
					bad = append(bad, fmt.Sprintf("download: %s (%s) is not fully on the page", f.Name, key))
				}
			}
		}
	} else if !strings.Contains(dl, "built without release data") {
		bad = append(bad, "download: a build without release data must say so on the page")
	}
	for _, w := range ws {
		page, err := read("different/" + w.Slug + "/index.html")
		switch {
		case err != nil:
			bad = append(bad, fmt.Sprintf("writeup %q has no page", w.Slug))
			continue
		case !strings.Contains(page, html.EscapeString(w.Title)):
			bad = append(bad, fmt.Sprintf("writeup %q: its page never names it", w.Slug))
		case !strings.Contains(page, "What it doesn&#39;t do") && !strings.Contains(page, "What it doesn't do"):
			bad = append(bad, fmt.Sprintf("writeup %q: its page has no \"What it doesn't do\" section", w.Slug))
		}
		if !strings.Contains(di, `href="/different/`+w.Slug+`/"`) {
			bad = append(bad, fmt.Sprintf("writeup %q is not linked from the What's different page", w.Slug))
		}
		if w.Draft() && !drafts {
			bad = append(bad, fmt.Sprintf("writeup %q is an unreviewed draft in a deploy build", w.Slug))
		}
	}
	for _, f := range m.Families {
		page, err := read("models/" + f.Name + "/index.html")
		switch {
		case err != nil:
			bad = append(bad, fmt.Sprintf("family %q has no page", f.Name))
			continue
		case !strings.Contains(page, html.EscapeString(f.DisplayName)):
			bad = append(bad, fmt.Sprintf("family %q: its page never names it (%q)", f.Name, f.DisplayName))
		case !strings.Contains(page, "What hasn't been shown"):
			bad = append(bad, fmt.Sprintf("family %q: its page has no \"What hasn't been shown\" list", f.Name))
		}
		if !strings.Contains(idx, `href="/models/`+f.Name+`/"`) {
			bad = append(bad, fmt.Sprintf("family %q is not linked from the Models page", f.Name))
		}
		for _, c := range f.Checkpoints {
			// The decisions row: a measured checkpoint's page carries both figures, an unmeasured one says so.
			if d := c.Decision; d != nil {
				if !strings.Contains(page, d.Top1) || !strings.Contains(page, d.ECE) {
					bad = append(bad, fmt.Sprintf("checkpoint %q: decision figures %s / %s are not on its page", c.ID, d.Top1, d.ECE))
				}
			} else if !strings.Contains(page, "Label scoring, unmeasured.") {
				bad = append(bad, fmt.Sprintf("checkpoint %q: no decisions row", c.ID))
			}
			for key, s := range c.Speed {
				if s.Tok != nil && (!strings.Contains(page, num(s.Tok)) || !strings.Contains(page, s.Date)) {
					bad = append(bad, fmt.Sprintf("checkpoint %q: its %s speed %s (%s) is not on its page", c.ID, key, num(s.Tok), s.Date))
				}
			}
		}
	}
	if len(bad) > 0 {
		return fmt.Errorf("the site failed its gates (%d):\n  %s", len(bad), strings.Join(bad, "\n  "))
	}
	return nil
}
