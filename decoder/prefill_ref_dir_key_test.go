//go:build goinfer_testhooks

package decoder

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A safetensors checkpoint directory keys a reference by its files: the key is stable, ignores goinfer's own sidecars and
// hidden files, and changes when a weight byte does.
func TestCheckpointSHA256_directory(t *testing.T) {
	t.Setenv("GOINFER_PREFILL_REF_CACHE", t.TempDir())
	dir := t.TempDir()
	write := func(rel, body string) {
		t.Helper()
		p := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("config.json", "{}")
	write("model.safetensors", "weights-a")
	write("1_Pooling/config.json", "{}")
	h1, err := checkpointSHA256(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(h1, "dir:") {
		t.Fatalf("directory hash %q lacks its dir: prefix", h1)
	}
	write("model.int8.metal.giw", "sidecar")
	write("model.int8.metal.giw.verified", "")
	write(".cache/x", "y")
	if h2, _ := checkpointSHA256(dir); h2 != h1 {
		t.Fatalf("a sidecar or a hidden file changed the key: %s -> %s", h1, h2)
	}
	write("model.safetensors", "weights-b")
	if h3, _ := checkpointSHA256(dir); h3 == h1 {
		t.Fatal("changing the weights did not change the key")
	}
}

// TestCheckpointSHA256_sameSizeRewriteAtTheSameMtime: a file's hash is not served from the cache after a same-size rewrite whose
// modification time is held at the first write's (a coarse clock does that: two writes in one tick share it). The stamp must also
// read what the clock cannot hide, or not trust a stamp the clock could not have moved.
func TestCheckpointSHA256_sameSizeRewriteAtTheSameMtime(t *testing.T) {
	t.Setenv("GOINFER_PREFILL_REF_CACHE", t.TempDir())
	p := filepath.Join(t.TempDir(), "model.safetensors")
	if err := os.WriteFile(p, []byte("weights-a"), 0o644); err != nil {
		t.Fatal(err)
	}
	st, err := os.Stat(p)
	if err != nil {
		t.Fatal(err)
	}
	h1, err := checkpointSHA256(p)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte("weights-b"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(p, st.ModTime(), st.ModTime()); err != nil {
		t.Fatal(err)
	}
	if h2, err := checkpointSHA256(p); err != nil || h2 == h1 {
		t.Fatalf("a same-size rewrite at the same mtime kept the hash %s (err %v): the cache served a stale key", h1, err)
	}
}
