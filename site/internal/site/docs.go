package site

import (
	"bytes"
	"encoding/json"
	"fmt"
	"html/template"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
)

// DocPage is one repo document rendered under /docs/.
type DocPage struct {
	Slug, Src, Group, Title string
	HTML                    template.HTML
}

// DocGroup is a heading of the docs index and rail, with its pages.
type DocGroup struct {
	Name  string
	Pages []*DocPage
}

var (
	hrefRE  = regexp.MustCompile(`href="([^"]+)"`)
	titleRE = regexp.MustCompile(`(?m)^#\s+(.+?)\s*$`)
)

// LoadDocs reads site/data/docs.json and renders each listed document. githubURL is where a link that leaves the shell goes.
func LoadDocs(root, githubURL string) ([]*DocPage, error) {
	var cfg struct {
		Docs []struct{ Slug, Src, Group string } `json:"docs"`
	}
	b, err := os.ReadFile(filepath.Join(root, "site", "data", "docs.json"))
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		return nil, fmt.Errorf("site/data/docs.json: %w", err)
	}
	bySrc := map[string]string{}
	for _, d := range cfg.Docs {
		bySrc[d.Src] = d.Slug
	}
	var out []*DocPage
	for _, d := range cfg.Docs {
		src, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(d.Src)))
		if err != nil {
			return nil, fmt.Errorf("docs: %s: %w", d.Src, err)
		}
		p := &DocPage{Slug: d.Slug, Src: d.Src, Group: d.Group}
		if m := titleRE.FindSubmatch(src); m != nil {
			p.Title = strings.TrimSpace(string(m[1]))
		} else {
			return nil, fmt.Errorf("docs: %s has no heading to title it", d.Src)
		}
		var buf bytes.Buffer
		if err := md.Convert(src, &buf); err != nil {
			return nil, fmt.Errorf("docs: %s: %w", d.Src, err)
		}
		dir := path.Dir(d.Src)
		var bad []string
		html := hrefRE.ReplaceAllStringFunc(buf.String(), func(m string) string {
			href := hrefRE.FindStringSubmatch(m)[1]
			switch {
			case strings.HasPrefix(href, "#"), strings.HasPrefix(href, "/"), strings.Contains(href, "://"), strings.HasPrefix(href, "mailto:"):
				return m
			}
			target, frag, _ := strings.Cut(href, "#")
			if target == "" {
				return m
			}
			repo := path.Clean(path.Join(dir, target))
			if strings.HasPrefix(repo, "..") {
				bad = append(bad, href)
				return m
			}
			if slug, ok := bySrc[repo]; ok {
				u := "/docs/" + slug + "/"
				if frag != "" {
					u += "#" + frag
				}
				return `href="` + u + `"`
			}
			u := githubURL + "/blob/main/" + repo
			if frag != "" {
				u += "#" + frag
			}
			return `href="` + u + `"`
		})
		if len(bad) > 0 {
			return nil, fmt.Errorf("docs: %s links outside the repo: %v", d.Src, bad)
		}
		p.HTML = template.HTML(html)
		out = append(out, p)
	}
	return out, nil
}

// GroupDocs arranges pages under their group headings, in the order groups first appear.
func GroupDocs(pages []*DocPage) []DocGroup {
	var gs []DocGroup
	idx := map[string]int{}
	for _, p := range pages {
		i, ok := idx[p.Group]
		if !ok {
			i = len(gs)
			idx[p.Group] = i
			gs = append(gs, DocGroup{Name: p.Group})
		}
		gs[i].Pages = append(gs[i].Pages, p)
	}
	return gs
}
