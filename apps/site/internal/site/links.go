package site

import (
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// internalRefRE finds a root-relative href or src in a built page.
var internalRefRE = regexp.MustCompile(`(?:href|src)="(/[^"]*)"`)

// CheckLinks is the output's own broken-link check: every root-relative link on every built page must name a file the build
// wrote. /book/ is excluded (the workflow builds the book with Jekyll after this runs, and gates it there). A link to a directory
// needs its index.html. A writeup cross-link to a draft that a deploy build leaves out fails here, which is the point.
func CheckLinks(out string) error {
	var bad []string
	err := filepath.WalkDir(out, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(p, ".html") {
			return err
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(out, p)
		for _, m := range internalRefRE.FindAllStringSubmatch(string(b), -1) {
			ref := m[1]
			if i := strings.IndexAny(ref, "?#"); i >= 0 {
				ref = ref[:i]
			}
			if ref == "/" || strings.HasPrefix(ref, "/book/") || ref == "/book" {
				continue
			}
			target := filepath.Join(out, filepath.FromSlash(path.Clean(ref)))
			if strings.HasSuffix(ref, "/") {
				target = filepath.Join(target, "index.html")
			}
			if _, err := os.Stat(target); err != nil {
				bad = append(bad, fmt.Sprintf("%s -> %s", filepath.ToSlash(rel), m[1]))
			}
		}
		return nil
	})
	if err != nil {
		return err
	}
	if len(bad) > 0 {
		sort.Strings(bad)
		return fmt.Errorf("broken internal links:\n  %s", strings.Join(bad, "\n  "))
	}
	return nil
}
