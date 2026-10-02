package site

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const readmeGoGet = "go get github.com/townsendmerino/goinfer/decoder@latest github.com/townsendmerino/goinfer/tokenizer@latest"

func TestReadmeInstallLines(t *testing.T) {
	needRepo(t)
	goGet, goInstall, err := ReadmeInstallLines(filepath.Join(repoRoot, "README.md"))
	if err != nil {
		t.Fatal(err)
	}
	if !goGet[readmeGoGet] {
		t.Errorf("the README's smoke-marked go get line is not %q; got %v", readmeGoGet, goGet)
	}
	if goGet["go get github.com/townsendmerino/goinfer"] {
		t.Error("the bare-module go get counts as a smoke-marked README line")
	}
	if !goInstall["go install github.com/townsendmerino/goinfer/cmd/serve@latest"] {
		t.Errorf("the CPU go install is missing: %v", goInstall)
	}
}

func TestSiteInstallLines(t *testing.T) {
	page := `<p>go get github.com/townsendmerino/goinfer in prose is not a command</p>
<pre>chmod +x x
go get github.com/
  townsendmerino/goinfer</pre>
<code>go install -tags cuda  github.com/townsendmerino/goinfer/cuda/cmd/serve@latest    # Linux + NVIDIA</code>
<code>CGO_ENABLED=0 go install github.com/townsendmerino/goinfer/cmd/serve@latest</code>`
	got := SiteInstallLines(page)
	want := []string{
		"go get github.com/townsendmerino/goinfer",
		"go install -tags cuda github.com/townsendmerino/goinfer/cuda/cmd/serve@latest",
		"go install github.com/townsendmerino/goinfer/cmd/serve@latest",
	}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Errorf("got %q\nwant %q", got, want)
	}
}

// The mutation: put back each wrong form the site has printed or could print, and the gate must go red; the README's
// own lines must pass.
func TestCheckInstallLines_goesRed(t *testing.T) {
	needRepo(t)
	for _, c := range []struct {
		page string
		ok   bool
	}{
		{"<pre>" + readmeGoGet + "</pre>", true},
		{"<pre>go install github.com/townsendmerino/goinfer/metal/cmd/serve@latest</pre>", true},
		{"<pre>go get github.com/townsendmerino/goinfer</pre>", false},                // the download page until 2026-09-30
		{"<pre>go get github.com/\n  townsendmerino/goinfer</pre>", false},            // the home page's wrapped copy
		{"<pre>go get github.com/townsendmerino/goinfer/decoder@latest</pre>", false}, // not the README's line
		{"<pre>go install github.com/townsendmerino/goinfer/cmd/chat@latest</pre>", false},
	} {
		out := t.TempDir()
		if err := os.WriteFile(filepath.Join(out, "index.html"), []byte(c.page), 0o644); err != nil {
			t.Fatal(err)
		}
		err := CheckInstallLines(repoRoot, out)
		if (err == nil) != c.ok {
			t.Errorf("%q: ok=%v, got err %v", c.page, c.ok, err)
		}
	}
}
