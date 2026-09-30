package site

import (
	"bufio"
	"fmt"
	"html"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// The install-line gate. A `go get` that looks right can fail: the bare module
// (`go get github.com/townsendmerino/goinfer`) records the requirement but does not fetch enough to build against, and the
// next build fails with "missing go.sum entry". The README documents that, and scripts/readme_smoke.sh runs the README's
// `<!-- smoke -->` lines from an empty directory, so those lines are the ones known to work. The site printed the bare
// form on two pages for a release cycle, because nothing compared the two.
//
// So every `go get` the built site prints must be one of the README's smoke-marked `go get` lines, and every `go install`
// must be a command line in one of the README's fenced blocks (the GPU entrypoints cannot be smoke-run in CI, so they are
// not smoke-marked, but they are still the README's own text). Whitespace is collapsed and a trailing `# comment` dropped
// on both sides.

var (
	goCmdRE = regexp.MustCompile(`go (?:get|install) [^\n#]*github\.com/townsendmerino/goinfer[^\n#]*`)
	codeRE  = regexp.MustCompile(`(?s)<(pre|code)[^>]*>(.*?)</(?:pre|code)>`)
	tagRE   = regexp.MustCompile(`<[^>]+>`)
	// a command wrapped to fit a narrow box: a path broken after "/", or a shell continuation
	wrapRE   = regexp.MustCompile(`(/|\\)\n\s*`)
	spacesRE = regexp.MustCompile(`\s+`)
)

func normCmd(s string) string {
	if i := strings.Index(s, "#"); i >= 0 {
		s = s[:i]
	}
	return strings.TrimSpace(spacesRE.ReplaceAllString(s, " "))
}

// ReadmeInstallLines returns the README's smoke-marked `go get` lines and every `go install` line in its fenced blocks.
func ReadmeInstallLines(readme string) (goGet, goInstall map[string]bool, err error) {
	f, err := os.Open(readme)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	goGet, goInstall = map[string]bool{}, map[string]bool{}
	sc := bufio.NewScanner(f)
	fenced, smoke := false, false
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(strings.TrimSpace(line), "```") {
			fenced, smoke = !fenced, false
			continue
		}
		if !fenced {
			continue
		}
		t := strings.TrimSpace(line)
		if strings.HasPrefix(t, "#") {
			smoke = strings.Contains(t, "<!-- smoke -->")
			continue
		}
		switch c := normCmd(t); {
		case strings.HasPrefix(c, "go get ") && smoke:
			goGet[c] = true
		case strings.HasPrefix(c, "go install "):
			goInstall[c] = true
		}
		smoke = false
	}
	if err := sc.Err(); err != nil {
		return nil, nil, err
	}
	if len(goGet) == 0 {
		return nil, nil, fmt.Errorf("%s has no smoke-marked go get line to check the site against", readme)
	}
	return goGet, goInstall, nil
}

// SiteInstallLines finds every `go get` / `go install` of this module inside a <pre> or <code> element of a page.
func SiteInstallLines(page string) []string {
	var out []string
	for _, m := range codeRE.FindAllStringSubmatch(page, -1) {
		text := html.UnescapeString(tagRE.ReplaceAllString(m[2], ""))
		text = wrapRE.ReplaceAllStringFunc(text, func(w string) string {
			if w[0] == '/' {
				return "/"
			}
			return " "
		})
		for _, c := range goCmdRE.FindAllString(text, -1) {
			out = append(out, normCmd(c))
		}
	}
	return out
}

// CheckInstallLines is the gate: it reads every page under out and refuses any install line the README does not carry.
func CheckInstallLines(root, out string) error {
	goGet, goInstall, err := ReadmeInstallLines(filepath.Join(root, "README.md"))
	if err != nil {
		return err
	}
	var bad []string
	err = filepath.WalkDir(out, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(p, ".html") {
			return err
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(out, p)
		for _, c := range SiteInstallLines(string(b)) {
			ok := goInstall[c]
			if strings.HasPrefix(c, "go get ") {
				ok = goGet[c]
			}
			if !ok {
				bad = append(bad, fmt.Sprintf("%s: %q", filepath.ToSlash(rel), c))
			}
		}
		return nil
	})
	if err != nil {
		return err
	}
	if len(bad) > 0 {
		sort.Strings(bad)
		return fmt.Errorf("the site prints %d install line(s) the README does not (a go get must be a README <!-- smoke --> line):\n  %s",
			len(bad), strings.Join(bad, "\n  "))
	}
	return nil
}
