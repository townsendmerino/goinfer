package site

import (
	"bytes"
	"embed"
	"fmt"
	"html/template"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

//go:embed templates/*.html
var templateFS embed.FS

//go:embed assets/*
var assetFS embed.FS

// Config is what a build is told, apart from the repo.
type Config struct {
	BookURL   string // where the book lives: /book/ on this same site (the workflow builds it into the output)
	GitHubURL string
	Generated string // the date shown on generated pages
	Drafts    bool   // also build the unreviewed writeups (a preview; never for a deploy)
}

// DefaultConfig is the site as it is today.
func DefaultConfig() Config {
	return Config{
		BookURL:   "/book/",
		GitHubURL: "https://github.com/townsendmerino/goinfer",
	}
}

func gb(b int64) string {
	g := float64(b) / 1e9
	if g < 1 {
		return strconv.FormatFloat(g, 'f', 2, 64) + " GB"
	}
	return strconv.FormatFloat(g, 'f', 1, 64) + " GB"
}

func capFirst(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}

// firstClause is the part of a good-for line before its first ":", ";" or "(", capitalised, for a card.
func firstClause(s string) string {
	if i := strings.IndexAny(s, ":;("); i >= 0 {
		s = s[:i]
	}
	return capFirst(s)
}

func num(p *float64) string {
	if p == nil {
		return ""
	}
	return strconv.FormatFloat(*p, 'f', -1, 64)
}

func taskLabel(t string) string {
	if t == "vision" {
		return "images"
	}
	return t
}

// vsText is how a speed's verdict reads: a ratio against Ollama when there is one, otherwise the plain reason.
func vsText(s *Speed) string {
	if s.Verdict == "ahead" || s.Verdict == "behind" {
		return s.Ratio + " Ollama"
	}
	return s.Ratio
}

func barWidth(v *float64, max float64) string {
	if v == nil || max <= 0 {
		return "0"
	}
	return strconv.FormatFloat(*v/max*100, 'f', 1, 64)
}

// speedMax is the scale of a checkpoint's speed bars, common to every machine so they compare.
func speedMax(c *Checkpoint) float64 {
	max := 1.0
	for _, s := range c.Speed {
		if s.Tok != nil && *s.Tok > max {
			max = *s.Tok
		}
		if s.Peer != nil && *s.Peer > max {
			max = *s.Peer
		}
	}
	return max * 1.05
}

func meterCos(c *float64) string {
	if c == nil {
		return "0"
	}
	w := (*c - 0.95) / 0.05 * 100
	if w < 0 {
		w = 0
	}
	return strconv.FormatFloat(w, 'f', 1, 64)
}

func (m *Model) funcs(cfg Config) template.FuncMap {
	return template.FuncMap{
		"gb": gb, "cap": capFirst, "first": firstClause, "num": num, "task": taskLabel, "vs": vsText,
		"bar": barWidth, "smax": speedMax, "cosw": meterCos,
		"join":       strings.Join,
		"pad":        func(n int) string { return fmt.Sprintf("%02d", n) },
		"parityWord": func(p string) string { w, _, _ := strings.Cut(p, " "); return strings.TrimSuffix(w, ":") },
		"factval":    factVal,
		"machines":   func() []Machine { return m.Machines },
		"cfg":        func() Config { return cfg },
		"tierText":   func(t string) string { return TierText[t] },
		"b2s": func(b bool) string {
			if b {
				return "1"
			}
			return "0"
		},
		"hasTask": func(f *Family, t string) bool {
			for _, x := range f.Tasks {
				if x == t {
					return true
				}
			}
			return false
		},
		"lt100": func(p *float64) bool { return p != nil && *p < 100 },
		"lt99":  func(p *float64) bool { return p != nil && *p < 0.99 },
		"fmt1":  func(p *float64) string { return strconv.FormatFloat(*p, 'f', 1, 64) },
		"fmt5":  func(p *float64) string { return strconv.FormatFloat(*p, 'f', 5, 64) },
		"sha16": func(s string) string {
			if s == "" {
				return "none on file"
			}
			return s[:16] + "…"
		},
		"via": func(f *Family) string {
			if v := m.ByName[f.P.Via]; v != nil {
				return v.DisplayName
			}
			return f.P.Via
		},
		"gaps": func(f *Family, c *Checkpoint) map[string]any {
			common, by := m.Gaps(f, c)
			return map[string]any{"Common": common, "ByMachine": by}
		},
		"dict2":    func(f *Family, c *Checkpoint) map[string]any { return map[string]any{"F": f, "C": c} },
		"otherGap": func() string { return "Never measured on a machine like yours, so no speed is shown." },
		"anySpeed": func(c *Checkpoint) bool { return len(c.Speed) > 0 },
		"speedOf":  func(c *Checkpoint, key string) *Speed { return c.Speed[key] },
		"fitOf":    func(c *Checkpoint, key string) Fit { return c.Fit[key] },
	}
}

// page is the data every template sees.
type page struct {
	Title, Desc, Path, Nav string
	M                      *Model
	Family                 *Family
	Cfg                    Config
	Writeup                *Writeup  // a writeup page
	Download               *Download // the download page
	Doc                    *DocPage  // a docs page (nil on the docs index)
	DocGroups              []DocGroup
	Writeups               []*Writeup // the visible writeups, in order
	Styles, Scripts        []string   // extra stylesheets and scripts for this page
	NoIndex                bool
}

// factVal escapes text and turns `backticked` spans into <code>, for a fact strip or a limit's text.
func factVal(s string) template.HTML {
	parts := strings.Split(s, "`")
	var b strings.Builder
	for i, p := range parts {
		if i%2 == 1 && i < len(parts)-1 {
			b.WriteString("<code>" + template.HTMLEscapeString(p) + "</code>")
		} else if i%2 == 1 {
			b.WriteString("`" + template.HTMLEscapeString(p))
		} else {
			b.WriteString(template.HTMLEscapeString(p))
		}
	}
	return template.HTML(b.String())
}

// Report says what a build wrote, for the gates and the CI log.
type Report struct {
	Files    []string
	Families int
}

func write(out, rel string, b []byte, r *Report) error {
	p := filepath.Join(out, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	r.Files = append(r.Files, rel)
	return os.WriteFile(p, b, 0o644)
}

// Build renders the whole site into out.
func Build(root, out string, cfg Config) (*Report, error) {
	in, err := LoadInputs(root)
	if err != nil {
		return nil, err
	}
	if err := in.Validate(); err != nil {
		return nil, err
	}
	if err := CheckClaims(root, in); err != nil {
		return nil, err
	}
	if err := CheckOllama(root, in); err != nil {
		return nil, err
	}
	m, err := Derive(in)
	if err != nil {
		return nil, err
	}
	if m.Book, err = LoadBook(root); err != nil {
		return nil, err
	}
	if m.Docs, err = LoadDocs(root, cfg.GitHubURL); err != nil {
		return nil, err
	}
	rel, err := LoadRelease(root)
	if err != nil {
		return nil, err
	}
	if m.Download, err = BuildDownload(rel); err != nil {
		return nil, err
	}
	all, err := LoadWriteups(root)
	if err != nil {
		return nil, err
	}
	if err := CheckWriteups(root, all); err != nil {
		return nil, err
	}
	ws := Visible(all, cfg.Drafts)
	rep, err := m.Render(out, cfg, ws)
	if err != nil {
		return nil, err
	}
	if err := Verify(out, m, ws, cfg.Drafts); err != nil {
		return nil, err
	}
	if err := CheckLinks(out); err != nil {
		return nil, err
	}
	return rep, nil
}

// Render writes the pages of an already-derived model.
func (m *Model) Render(out string, cfg Config, ws []*Writeup) (*Report, error) {
	tpl, err := template.New("site").Funcs(m.funcs(cfg)).ParseFS(templateFS, "templates/*.html")
	if err != nil {
		return nil, fmt.Errorf("templates: %w", err)
	}
	r := &Report{}
	exec := func(name string, p page) ([]byte, error) {
		var b bytes.Buffer
		if err := tpl.ExecuteTemplate(&b, name, p); err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		return b.Bytes(), nil
	}
	b, err := exec("models.html", page{Title: "Models · goinfer", Desc: "Every model goinfer runs, and how sure we are about each one.", Path: "/models/", Nav: "models", M: m, Cfg: cfg})
	if err != nil {
		return nil, err
	}
	if err := write(out, "models/index.html", b, r); err != nil {
		return nil, err
	}
	for _, f := range m.Families {
		b, err := exec("family.html", page{Title: f.DisplayName + " · goinfer", Desc: f.Summary, Path: "/models/" + f.Name + "/", Nav: "models", M: m, Family: f, Cfg: cfg})
		if err != nil {
			return nil, err
		}
		if err := write(out, "models/"+f.Name+"/index.html", b, r); err != nil {
			return nil, err
		}
		r.Families++
	}
	groups := GroupDocs(m.Docs)
	b, err = exec("docs.html", page{Title: "Docs · goinfer", Desc: "How to run goinfer, connect tools to it, and choose a quantization.", Path: "/docs/", Nav: "docs", M: m, Cfg: cfg, DocGroups: groups, Styles: []string{"/assets/writeup.css", "/assets/docs.css"}})
	if err != nil {
		return nil, err
	}
	if err := write(out, "docs/index.html", b, r); err != nil {
		return nil, err
	}
	for _, d := range m.Docs {
		b, err := exec("docs.html", page{Title: d.Title + " · goinfer docs", Desc: "", Path: "/docs/" + d.Slug + "/", Nav: "docs", M: m, Cfg: cfg, Doc: d, DocGroups: groups, Styles: []string{"/assets/writeup.css", "/assets/docs.css"}})
		if err != nil {
			return nil, err
		}
		if err := write(out, "docs/"+d.Slug+"/index.html", b, r); err != nil {
			return nil, err
		}
	}
	b, err = exec("download.html", page{Title: "Download · goinfer", Desc: "goinfer binaries for macOS, Linux and Windows, with checksums.", Path: "/download/", Nav: "download", M: m, Cfg: cfg, Download: m.Download, Styles: []string{"/assets/home.css"}, Scripts: []string{"/assets/download.js"}})
	if err != nil {
		return nil, err
	}
	if err := write(out, "download/index.html", b, r); err != nil {
		return nil, err
	}
	b, err = exec("different.html", page{Title: "What's different · goinfer", Desc: "Things goinfer does that are unusual, each with what it doesn't do.", Path: "/different/", Nav: "different", M: m, Cfg: cfg, Writeups: ws, Styles: []string{"/assets/writeup.css", "/assets/home.css"}})
	if err != nil {
		return nil, err
	}
	if err := write(out, "different/index.html", b, r); err != nil {
		return nil, err
	}
	for _, w := range ws {
		styles, scripts := []string{"/assets/writeup.css"}, []string{"/assets/writeup.js"}
		for _, f := range w.CSS {
			styles = append(styles, "/different/"+w.Slug+"/"+f)
		}
		for _, f := range w.JS {
			scripts = append(scripts, "/different/"+w.Slug+"/"+f)
		}
		b, err := exec("writeup.html", page{Title: w.Title + " · goinfer", Desc: w.Summary, Path: "/different/" + w.Slug + "/", Nav: "different", M: m, Cfg: cfg, Writeup: w, Writeups: ws, Styles: styles, Scripts: scripts, NoIndex: w.Draft()})
		if err != nil {
			return nil, err
		}
		if err := write(out, "different/"+w.Slug+"/index.html", b, r); err != nil {
			return nil, err
		}
		for name, fb := range w.Files {
			if err := write(out, "different/"+w.Slug+"/"+name, fb, r); err != nil {
				return nil, err
			}
		}
	}
	hw := ws
	if len(hw) > 6 {
		hw = hw[:6]
	}
	for _, n := range []struct {
		tpl, file string
		p         page
	}{
		{"home.html", "index.html", page{Title: "goinfer · what will run well on your machine", Desc: "A local LLM engine in pure Go. Pick your machine and see what fits and how fast it measured.", Path: "/", Nav: "home", M: m, Cfg: cfg, Writeups: hw, Styles: []string{"/assets/home.css"}}},
		{"notfound.html", "404.html", page{Title: "goinfer", Path: "/", M: m, Cfg: cfg}},
	} {
		b, err := exec(n.tpl, n.p)
		if err != nil {
			return nil, err
		}
		if err := write(out, n.file, b, r); err != nil {
			return nil, err
		}
	}
	names, _ := fs.Glob(assetFS, "assets/*")
	sort.Strings(names)
	for _, n := range names {
		b, err := assetFS.ReadFile(n)
		if err != nil {
			return nil, err
		}
		if err := write(out, n, b, r); err != nil {
			return nil, err
		}
	}
	return r, nil
}
