package pull

import (
	"path/filepath"
	"strings"
	"testing"
)

// A model reference can come from an HTTP request (the web UI's pull), and its repo goes into a URL path and is joined under the
// cache directory, so whatever ParseRef accepts must be one owner and one name that stay under that directory.
func FuzzParseRef(f *testing.F) {
	for _, s := range []string{
		"owner/repo", "owner/repo:q4_k_m", "owner/repo:file.gguf", "owner/repo:safetensors", "owner/repo:../../evil.gguf", " owner/repo ", "demo:nope",
		"../..", "../../etc:x.gguf", "./repo", "owner/..", "..%2f../x", "owner/repo/extra", "owner\\repo", "owner/repo?x=1", "/abs/path", "", " ", ":", "::", "a/b:", "\x00",
	} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, s string) {
		r, err := ParseRef(s)
		if err != nil {
			return
		}
		if !filepath.IsLocal(r.Repo) || strings.Count(r.Repo, "/") != 1 || strings.ContainsAny(r.Repo, "\\?#%: \x00") {
			t.Fatalf("ParseRef(%q) accepted repo %q", s, r.Repo)
		}
	})
}
