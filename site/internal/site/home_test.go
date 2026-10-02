package site

import (
	"html"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// The home page's struct example is hand-placed in the template, so nothing would notice it drifting from code that
// compiles. Every line of it must be a line of examples/structured/main.go (indentation aside), which CI builds.
func TestHome_heroCodeIsFromTheExample(t *testing.T) {
	needRepo(t)
	tpl, err := templateFS.ReadFile("templates/home.html")
	if err != nil {
		t.Fatal(err)
	}
	m := regexp.MustCompile(`(?s)<figure class="herocode">\s*<pre><code>(.*?)</code></pre>`).FindSubmatch(tpl)
	if m == nil {
		t.Fatal("home.html has no herocode block")
	}
	src, err := os.ReadFile(filepath.Join(repoRoot, "examples", "structured", "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	have := map[string]bool{}
	for _, l := range strings.Split(string(src), "\n") {
		have[strings.TrimSpace(l)] = true
	}
	n := 0
	for _, l := range strings.Split(html.UnescapeString(string(m[1])), "\n") {
		if l = strings.TrimSpace(l); l == "" {
			continue
		}
		n++
		if !have[l] {
			t.Errorf("home.html's struct example has a line examples/structured/main.go does not: %q", l)
		}
	}
	if n < 8 {
		t.Errorf("the struct example has %d lines; expected the struct and the constrain/generate/unmarshal lines", n)
	}
}
