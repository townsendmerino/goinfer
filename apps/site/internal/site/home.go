package site

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// BookChapter is one chapter of the primer (docs/book), for the Home page's list.
type BookChapter struct {
	Num   int
	Title string
	File  string // the built page, relative to the book's root
}

var chapterRE = regexp.MustCompile(`^(\d{2})-.+\.md$`)
var chapterTitleRE = regexp.MustCompile(`^#\s+Chapter\s+(\d+)\s+—\s+(.+)$`)

// LoadBook lists the book's numbered chapters from docs/book, in order. The glossary is not a chapter.
func LoadBook(root string) ([]BookChapter, error) {
	dir := filepath.Join(root, "docs", "book")
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var out []BookChapter
	for _, e := range ents {
		m := chapterRE.FindStringSubmatch(e.Name())
		if m == nil {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			return nil, err
		}
		first, _, _ := strings.Cut(string(b), "\n")
		t := chapterTitleRE.FindStringSubmatch(first)
		if t == nil {
			continue // the glossary and any page that is not "Chapter N — Title"
		}
		var n int
		fmt.Sscanf(t[1], "%d", &n)
		out = append(out, BookChapter{Num: n, Title: strings.TrimSpace(t[2]), File: strings.TrimSuffix(e.Name(), ".md") + ".html"})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Num < out[j].Num })
	if len(out) == 0 {
		return nil, fmt.Errorf("docs/book has no chapters")
	}
	return out, nil
}

// MachineNote is the line under a machine's answer on Home, derived from the claims so it cannot say more than they do.
func (m *Model) MachineNote(key string) string {
	var ahead, behind, graded int
	first, last := "", ""
	for _, f := range m.Families {
		for _, c := range f.Checkpoints {
			s := c.Speed[key]
			if s == nil || s.Tok == nil {
				continue
			}
			if first == "" || s.Date < first {
				first = s.Date
			}
			if s.Date > last {
				last = s.Date
			}
			switch s.Verdict {
			case "ahead":
				ahead++
				graded++
			case "behind":
				behind++
				graded++
			}
		}
	}
	var mach Machine
	for _, x := range m.Machines {
		if x.Key == key {
			mach = x
		}
	}
	if first == "" {
		return mach.Path + ": no speed measured on this machine."
	}
	when := "Measured " + first
	if last != first {
		when = "Measured " + first + " to " + last
	}
	note := fmt.Sprintf("%s decode. %s %s.", mach.Path, m.Method, when)
	if graded > 0 {
		note += fmt.Sprintf(" Against Ollama here goinfer was ahead in %d of the %d graded cells and behind in %d.", ahead, graded, behind)
	}
	return note
}
