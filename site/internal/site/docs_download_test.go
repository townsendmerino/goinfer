package site

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func docsRoot(t *testing.T, files map[string]string, cfg string) string {
	t.Helper()
	root := t.TempDir()
	files["site/data/docs.json"] = cfg
	for n, c := range files {
		p := filepath.Join(root, filepath.FromSlash(n))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(c), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

const docsCfg = `{"docs":[{"slug":"a","src":"docs/a.md","group":"G"},{"slug":"b","src":"docs/sub/b.md","group":"G"}]}`

func TestLoadDocs_rewritesRelativeLinks(t *testing.T) {
	root := docsRoot(t, map[string]string{
		"docs/a.md":     "# Doc A\n\n[to b](sub/b.md#part), [to code](../cmd/x/main.go), [ext](https://example.com/x), [self](#top)\n",
		"docs/sub/b.md": "# Doc B\n\n[back](../a.md) and [a record](../measurements/r.md)\n",
	}, docsCfg)
	ds, err := LoadDocs(root, "https://gh.example/repo")
	if err != nil {
		t.Fatal(err)
	}
	a, b := string(ds[0].HTML), string(ds[1].HTML)
	for _, want := range []string{`href="/docs/b/#part"`, `href="https://gh.example/repo/blob/main/cmd/x/main.go"`, `href="https://example.com/x"`, `href="#top"`} {
		if !strings.Contains(a, want) {
			t.Errorf("doc A missing %s in %s", want, a)
		}
	}
	for _, want := range []string{`href="/docs/a/"`, `href="https://gh.example/repo/blob/main/docs/measurements/r.md"`} {
		if !strings.Contains(b, want) {
			t.Errorf("doc B missing %s in %s", want, b)
		}
	}
	if ds[0].Title != "Doc A" {
		t.Errorf("title = %q", ds[0].Title)
	}
}

func TestLoadDocs_refusesWhatCannotBeRendered(t *testing.T) {
	root := docsRoot(t, map[string]string{"docs/a.md": "# A\n\n[out](../../../etc/passwd)\n", "docs/sub/b.md": "# B\n"}, docsCfg)
	if _, err := LoadDocs(root, "g"); err == nil || !strings.Contains(err.Error(), "outside the repo") {
		t.Errorf("a link out of the repo must be refused, got %v", err)
	}
	root = docsRoot(t, map[string]string{"docs/a.md": "no heading here\n", "docs/sub/b.md": "# B\n"}, docsCfg)
	if _, err := LoadDocs(root, "g"); err == nil || !strings.Contains(err.Error(), "no heading") {
		t.Errorf("a doc with no heading must be refused, got %v", err)
	}
	root = docsRoot(t, map[string]string{"docs/sub/b.md": "# B\n"}, docsCfg)
	if _, err := LoadDocs(root, "g"); err == nil {
		t.Error("a listed doc that does not exist must be refused")
	}
}

func fullRelease() *Release {
	r := &Release{Tag: "v9.9.9", Published: "2026-09-29", URL: "https://x/r", Checksums: map[string]string{}}
	add := func(name string, size int64) {
		r.Assets = append(r.Assets, ReleaseAsset{Name: name, Size: size, URL: "https://x/" + name})
		r.Checksums[name] = strings.Repeat("a", 64)
	}
	for _, k := range []string{"goinfer-serve", "goinfer-chat", "goinfer-chat-0.5b", "goinfer-chat-1.5b"} {
		for _, p := range []string{"darwin-arm64", "linux-amd64", "windows-amd64"} {
			n := k + "-" + p
			if strings.HasPrefix(p, "windows") {
				n += ".exe"
			}
			add(n, 5<<20)
		}
	}
	r.Assets = append(r.Assets, ReleaseAsset{Name: "checksums.txt", Size: 100, URL: "https://x/checksums.txt"})
	return r
}

func TestBuildDownload_groupsBinariesByKindAndPlatform(t *testing.T) {
	d, err := BuildDownload(fullRelease())
	if err != nil {
		t.Fatal(err)
	}
	if len(d.Kinds) != 4 || d.Kinds[0].Files["darwin-arm64"] == nil || d.Kinds[0].Files["windows-amd64"].Name != "goinfer-serve-windows-amd64.exe" {
		t.Errorf("grid not built: %+v", d.Kinds[0].Files)
	}
	if d.Sample == nil || !strings.HasPrefix(d.Sample.Name, "goinfer-chat-0.5b-") || strings.HasSuffix(d.Sample.Shell(), ".exe") {
		t.Errorf("the sample command must use a real 0.5B file, without .exe: %+v", d.Sample)
	}
	if len(d.Others) != 1 || d.Others[0].Name != "checksums.txt" {
		t.Errorf("checksums.txt is the one non-binary asset here: %+v", d.Others)
	}
}

func TestBuildDownload_refusesAnIncompleteRelease(t *testing.T) {
	r := fullRelease()
	delete(r.Checksums, "goinfer-serve-darwin-arm64")
	if _, err := BuildDownload(r); err == nil || !strings.Contains(err.Error(), "no checksum") {
		t.Errorf("a binary with no checksum must be refused, got %v", err)
	}
	r = fullRelease()
	var kept []ReleaseAsset
	for _, a := range r.Assets {
		if !strings.HasPrefix(a.Name, "goinfer-chat-1.5b") {
			kept = append(kept, a)
		}
	}
	r.Assets = kept
	if _, err := BuildDownload(r); err == nil || !strings.Contains(err.Error(), "goinfer-chat-1.5b") {
		t.Errorf("a release with no 1.5B binary must be refused, got %v", err)
	}
	if d, err := BuildDownload(nil); d != nil || err != nil {
		t.Error("no release data is not an error; the page says so")
	}
}

func TestLoadBook_titlesTheChaptersNotTheGlossary(t *testing.T) {
	needRepo(t)
	ch, err := LoadBook(repoRoot)
	if err != nil {
		t.Fatal(err)
	}
	if len(ch) < 11 || ch[0].Num != 1 || ch[0].Title != "Text becomes numbers" || ch[0].File != "01-text-becomes-numbers.html" {
		t.Errorf("chapters: %+v", ch[:2])
	}
	for _, c := range ch {
		if strings.Contains(strings.ToLower(c.Title), "glossary") {
			t.Error("the glossary is not a chapter")
		}
	}
}

// The download page's example is the Apple silicon (darwin-arm64) file whenever the release has one, wherever it is
// listed, not whichever asset comes first.
func TestBuildDownload_sampleIsDarwinArm64(t *testing.T) {
	r := fullRelease()
	r.Assets = append([]ReleaseAsset{{Name: "goinfer-chat-0.5b-darwin-amd64", Size: 5 << 20, URL: "https://x/a"}}, r.Assets...)
	r.Checksums["goinfer-chat-0.5b-darwin-amd64"] = strings.Repeat("b", 64)
	d, err := BuildDownload(r)
	if err != nil {
		t.Fatal(err)
	}
	if d.Sample == nil || d.Sample.Name != "goinfer-chat-0.5b-darwin-arm64" {
		t.Errorf("sample %+v, want goinfer-chat-0.5b-darwin-arm64", d.Sample)
	}
}
