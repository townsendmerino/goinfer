package site

import (
	"bytes"
	"fmt"
	"html/template"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/yuin/goldmark"
	"github.com/yuin/goldmark/extension"
	"github.com/yuin/goldmark/parser"
	"github.com/yuin/goldmark/renderer/html"
	"gopkg.in/yaml.v3"
)

// The "What's different" writeups (docs/tasks/task-site-2026-09.md S8c): one Markdown file each in site/different/, with
// YAML front matter. A writeup is published only when its front matter carries a `reviewed` date, which is the
// owner's; an unreviewed one is a draft, built only for a preview (Config.Drafts) and never linked from the site.

// WFact is one item of the strip under a writeup's title.
type WFact struct{ Label, Value string }

// WLimit is one entry of the closing "What it doesn't do" section.
type WLimit struct{ Title, Text string }

// WFigure ties a number in a writeup to the record that prints it.
type WFigure struct{ Text, Source string }

// Writeup is one parsed file.
type Writeup struct {
	Slug     string `yaml:"-"`
	Title    string
	Area     string
	Order    int
	Summary  string
	Stand    string
	Measured string
	Reviewed string
	Raw      bool // the body is HTML, not Markdown, and is used as written
	Facts    []WFact
	Doesnt   []WLimit
	Figures  []WFigure
	Sources  []string
	Body     string            `yaml:"-"`
	HTML     template.HTML     `yaml:"-"`
	CSS, JS  []string          `yaml:"-"` // files beside the .md, named <slug>.css / <slug>.js
	Files    map[string][]byte `yaml:"-"` // their bytes, by name
	Pos, Of  int               `yaml:"-"` // place among the published (or previewed) writeups
	Prev     *Writeup          `yaml:"-"`
	Next     *Writeup          `yaml:"-"`
}

// Draft reports whether the owner has not reviewed it.
func (w *Writeup) Draft() bool { return w.Reviewed == "" }

// Card is the limit a home-page or index card quotes.
func (w *Writeup) Card() string {
	if len(w.Doesnt) == 0 {
		return ""
	}
	return w.Doesnt[0].Title
}

var (
	slugRE     = regexp.MustCompile(`^(\d{2})-[a-z0-9]+(?:-[a-z0-9]+)*$`)
	dateRE     = regexp.MustCompile(`^\d{4}-\d{2}-\d{2}$`)
	repoLinkRE = regexp.MustCompile(`https://github\.com/townsendmerino/goinfer/(?:blob|tree)/main/([^)"#\s?]+)`)
	doesntRE   = regexp.MustCompile(`(?im)(^#{1,6}\s*|<h[1-6][^>]*>\s*)what it doesn.?t do`)
)

func splitFrontMatter(b []byte) (fm, body []byte, err error) {
	s := string(b)
	if !strings.HasPrefix(s, "---\n") {
		return nil, nil, fmt.Errorf("no front matter (the file must start with ---)")
	}
	rest := s[4:]
	i := strings.Index(rest, "\n---\n")
	if i < 0 {
		return nil, nil, fmt.Errorf("front matter is never closed (a line of ---)")
	}
	return []byte(rest[:i]), []byte(rest[i+5:]), nil
}

var md = goldmark.New(
	goldmark.WithExtensions(extension.GFM),
	goldmark.WithParserOptions(parser.WithAutoHeadingID()),
	// The writeups are the repo's own trusted files, and some carry a figure or a table as raw HTML.
	goldmark.WithRendererOptions(html.WithUnsafe()),
)

// LoadWriteups reads site/different/*.md under root. REVIEW-*.md files are the reviewer's notes, not writeups.
func LoadWriteups(root string) ([]*Writeup, error) {
	dir := filepath.Join(root, "site", "different")
	ents, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var out []*Writeup
	seen := map[int]string{}
	for _, e := range ents {
		n := e.Name()
		if e.IsDir() || !strings.HasSuffix(n, ".md") || strings.HasPrefix(n, "REVIEW") || strings.HasPrefix(n, "README") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, n))
		if err != nil {
			return nil, err
		}
		fm, body, err := splitFrontMatter(b)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", n, err)
		}
		w := &Writeup{Slug: strings.TrimSuffix(n, ".md")}
		dec := yaml.NewDecoder(bytes.NewReader(fm))
		dec.KnownFields(true)
		if err := dec.Decode(w); err != nil {
			return nil, fmt.Errorf("%s: front matter: %w", n, err)
		}
		w.Body = string(body)
		if err := w.validate(); err != nil {
			return nil, fmt.Errorf("%s: %w", n, err)
		}
		if prev, dup := seen[w.Order]; dup {
			return nil, fmt.Errorf("%s: order %d is also %s", n, w.Order, prev)
		}
		seen[w.Order] = n
		if w.Raw {
			w.HTML = template.HTML(body)
		} else {
			var buf bytes.Buffer
			if err := md.Convert(body, &buf); err != nil {
				return nil, fmt.Errorf("%s: markdown: %w", n, err)
			}
			w.HTML = template.HTML(buf.String())
		}
		for _, ext := range []string{".css", ".js"} {
			if fb, err := os.ReadFile(filepath.Join(dir, w.Slug+ext)); err == nil {
				if w.Files == nil {
					w.Files = map[string][]byte{}
				}
				w.Files[w.Slug+ext] = fb
				if ext == ".css" {
					w.CSS = append(w.CSS, w.Slug+ext)
				} else {
					w.JS = append(w.JS, w.Slug+ext)
				}
			}
		}
		out = append(out, w)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Order < out[j].Order })
	return out, nil
}

func (w *Writeup) validate() error {
	m := slugRE.FindStringSubmatch(w.Slug)
	if m == nil {
		return fmt.Errorf("the file name must look like 13-some-words.md")
	}
	if fmt.Sprintf("%02d", w.Order) != m[1] {
		return fmt.Errorf("order %d does not match the file name's %s", w.Order, m[1])
	}
	for name, v := range map[string]string{"title": w.Title, "area": w.Area, "summary": w.Summary, "stand": w.Stand} {
		if strings.TrimSpace(v) == "" {
			return fmt.Errorf("front matter %q is empty", name)
		}
	}
	if n := len([]rune(w.Summary)); n > 160 {
		return fmt.Errorf("summary is %d characters; a card holds 160", n)
	}
	if !dateRE.MatchString(w.Measured) {
		return fmt.Errorf("measured %q is not YYYY-MM-DD", w.Measured)
	}
	if _, err := time.Parse("2006-01-02", w.Measured); err != nil {
		return fmt.Errorf("measured: %w", err)
	}
	if w.Reviewed != "" {
		if _, err := time.Parse("2006-01-02", w.Reviewed); err != nil {
			return fmt.Errorf("reviewed %q is not YYYY-MM-DD", w.Reviewed)
		}
	}
	if len(w.Doesnt) == 0 {
		return fmt.Errorf("a writeup needs at least one thing it doesn't do")
	}
	for _, d := range w.Doesnt {
		if strings.TrimSpace(d.Title) == "" || strings.TrimSpace(d.Text) == "" {
			return fmt.Errorf("an entry of doesnt has no title or no text")
		}
	}
	if doesntRE.MatchString(w.Body) {
		return fmt.Errorf(`the body has its own "What it doesn't do" heading; the template renders the front matter's doesnt list`)
	}
	return nil
}

// CheckWriteups is S7's "no claim not in the repo" for the writeups: every figure a writeup lists must appear in its
// body and in the record it names, and every source it cites must exist. Drafts are checked too, so a draft that could
// not be reviewed is caught before the owner spends time on it.
func CheckWriteups(root string, ws []*Writeup) error {
	var bad []string
	for _, w := range ws {
		text := w.Body
		for _, f := range w.Doesnt {
			text += "\n" + f.Text + "\n" + f.Title
		}
		for _, f := range w.Facts {
			text += "\n" + f.Value
		}
		for _, f := range w.Figures {
			src, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(f.Source)))
			switch {
			case strings.TrimSpace(f.Text) == "":
				bad = append(bad, fmt.Sprintf("%s: a figure has no text", w.Slug))
			case err != nil:
				bad = append(bad, fmt.Sprintf("%s: figure %q names a record that is not there: %s", w.Slug, f.Text, f.Source))
			case !strings.Contains(text, f.Text):
				bad = append(bad, fmt.Sprintf("%s: figure %q is listed but never appears in the page", w.Slug, f.Text))
			case !hasToken(string(src), f.Text):
				bad = append(bad, fmt.Sprintf("%s: figure %q does not appear in %s", w.Slug, f.Text, f.Source))
			}
		}
		// A link into the repo on GitHub must name a path that exists, or it is a dead link on a public page.
		for _, m := range repoLinkRE.FindAllStringSubmatch(text+"\n"+string(w.HTML), -1) {
			if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(m[1]))); err != nil {
				bad = append(bad, fmt.Sprintf("%s: links to %s, which is not in the repo", w.Slug, m[1]))
			}
		}
		for _, s := range w.Sources {
			if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(s))); err != nil {
				bad = append(bad, fmt.Sprintf("%s: source %s does not exist", w.Slug, s))
			}
		}
	}
	if len(bad) > 0 {
		return fmt.Errorf("writeup check failed:\n  %s", strings.Join(bad, "\n  "))
	}
	return nil
}

// Lead is the writeups shown first, in this order, ahead of the rest in file order: the site's primary reader is the
// Go engineer embedding a model (task-site-2026-09.md, amendment 2026-09-30), so the struct guarantee comes first and
// the tool-call guarantee second. Reordering here rather than renumbering the files keeps every writeup's URL.
var Lead = []string{"04-a-go-struct-the-model-cant-break", "03-tool-calls-that-cant-be-malformed"}

// Visible is the writeups a build shows: every reviewed one, plus the drafts when previewing, with Lead first. Each is
// numbered among them and linked to its neighbours.
func Visible(ws []*Writeup, drafts bool) []*Writeup {
	var shown []*Writeup
	for _, w := range ws {
		if !w.Draft() || drafts {
			shown = append(shown, w)
		}
	}
	var v []*Writeup
	led := map[string]bool{}
	for _, slug := range Lead {
		for _, w := range shown {
			if w.Slug == slug {
				v = append(v, w)
				led[slug] = true
			}
		}
	}
	for _, w := range shown {
		if !led[w.Slug] {
			v = append(v, w)
		}
	}
	for i, w := range v {
		w.Pos, w.Of = i+1, len(v)
		w.Prev, w.Next = nil, nil
		if i > 0 {
			w.Prev = v[i-1]
		}
		if i+1 < len(v) {
			w.Next = v[i+1]
		}
	}
	return v
}
