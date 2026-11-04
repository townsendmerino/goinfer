//go:build goinfer_testhooks

package decoder

import (
	"os"
	"path/filepath"
	"testing"
)

// TestProseSeedAbove: the prefill gate names its prose seeds "../testdata/prefill-gate-prose-<label>/<file>", which holds from decoder/. A
// backend's test runs two directories down, where that path names nothing, so the seed is found from the level above instead
// (backend/cuda's TestKV16 and the gates beside it read their seeds through the same function). A seed that exists nowhere is not found.
func TestProseSeedAbove(t *testing.T) {
	root := t.TempDir()
	seed := filepath.Join("testdata", "prefill-gate-prose-a", "seed.md")
	if err := os.MkdirAll(filepath.Join(root, "testdata", "prefill-gate-prose-a"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, seed), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		dir  string
		want string // "" = not found
	}{
		{"backend/cuda", filepath.Join("..", "..", seed)},
		{"decoder", ""}, // one level: the first attempt, the caller's own, already names it; this function only looks higher
		{"backend/cuda/kernels", filepath.Join("..", "..", "..", seed)},
	} {
		wd := filepath.Join(root, filepath.FromSlash(c.dir))
		if err := os.MkdirAll(wd, 0o755); err != nil {
			t.Fatal(err)
		}
		t.Chdir(wd)
		got, ok := proseSeedAbove("../" + filepath.ToSlash(seed))
		if c.want == "" {
			// from decoder/ the plain "../testdata/..." works and is the caller's first try; the helper starts at two levels, so it finds nothing there
			if ok && c.dir == "decoder" {
				t.Errorf("%s: found %q, but the seed is one level up and the caller's own path reaches it", c.dir, got)
			}
			continue
		}
		if !ok || got != c.want {
			t.Errorf("%s: found %q (%v), want %q", c.dir, got, ok, c.want)
		}
	}
	t.Chdir(filepath.Join(root, "backend", "cuda"))
	if got, ok := proseSeedAbove("../testdata/prefill-gate-prose-a/nothing-here.md"); ok {
		t.Errorf("a seed that exists nowhere was found: %q", got)
	}
	if got, ok := proseSeedAbove("testdata/not-a-dotdot-path.md"); ok {
		t.Errorf("a path that does not begin with ../ was resolved: %q", got)
	}
}
