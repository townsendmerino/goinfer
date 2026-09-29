package site

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Verify is the gate on what a build wrote (docs/tasks/task-site-2026-09.md, S7): every family in the capability matrix
// has a page that names it, the Models page links every one, and every measured figure that a claim carries is on its
// checkpoint's page with its date. It reads the files back, so a template or a loop that quietly skips a family fails
// here and not in production.
func Verify(out string, m *Model) error {
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
	for _, f := range m.Families {
		page, err := read("models/" + f.Name + "/index.html")
		switch {
		case err != nil:
			bad = append(bad, fmt.Sprintf("family %q has no page", f.Name))
			continue
		case !strings.Contains(page, f.DisplayName):
			bad = append(bad, fmt.Sprintf("family %q: its page never names it (%q)", f.Name, f.DisplayName))
		case !strings.Contains(page, "What hasn't been shown"):
			bad = append(bad, fmt.Sprintf("family %q: its page has no \"What hasn't been shown\" list", f.Name))
		}
		if !strings.Contains(idx, `href="/models/`+f.Name+`/"`) {
			bad = append(bad, fmt.Sprintf("family %q is not linked from the Models page", f.Name))
		}
		for _, c := range f.Checkpoints {
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
