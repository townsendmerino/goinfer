package pull

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestCacheEntries is P8's listing gate (task-checkpoint-fetch-2026-09.md): every shape a pull leaves on disk reads as one
// entry with the right path, kind, size and completeness, a model's .giw sidecars are counted against it, and the files
// that are not models (digest sidecars, verified markers, the checkpoint marker) are not listed as models.
func TestCacheEntries(t *testing.T) {
	withCacheRoot(t)
	root, err := CacheRoot()
	if err != nil {
		t.Fatal(err)
	}
	write := func(p string, n int) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, make([]byte, n), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	gg := filepath.Join(root, "o", "gguf-repo")
	write(filepath.Join(gg, "m-Q4_K_M.gguf"), 100)
	write(filepath.Join(gg, "m-Q4_K_M.int4.metal.giw"), 40) // a sidecar built from it
	write(filepath.Join(gg, "m-Q4_K_M.int4.metal.giw.verified"), 1)
	write(filepath.Join(gg, ".m-Q4_K_M.gguf.sha256"), 1) // the digest sidecar
	write(filepath.Join(gg, "m-Q8_0.gguf.part"), 30)     // an interrupted pull
	for i := 1; i <= 3; i++ {
		write(filepath.Join(gg, fmt.Sprintf("big-Q8_0-%05d-of-00003.gguf", i)), 10*i)
	}
	write(filepath.Join(gg, "big-Q8_0-00001-of-00003.int4.metal.giw"), 7)
	write(filepath.Join(gg, "half-Q6_K-00001-of-00002.gguf"), 5) // a split set missing shard 2

	ck := filepath.Join(root, "o", "ckpt")
	marker := func(dir string, files []File) {
		b, _ := json.Marshal(markerDoc{Repo: "o/x", Files: files})
		write(filepath.Join(dir, checkpointMarker), 0)
		if err := os.WriteFile(filepath.Join(dir, checkpointMarker), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write(filepath.Join(ck, "config.json"), 3)
	write(filepath.Join(ck, "model.safetensors"), 50)
	write(filepath.Join(ck, "1_Pooling", "config.json"), 2)
	marker(ck, []File{{Path: "1_Pooling/config.json", Size: 2}, {Path: "config.json", Size: 3}, {Path: "model.safetensors", Size: 50}})
	bad := filepath.Join(root, "o", "ckpt-bad")
	write(filepath.Join(bad, "model.safetensors"), 10) // the marker says 50
	marker(bad, []File{{Path: "model.safetensors", Size: 50}})
	write(filepath.Join(root, "o", "staging.partial", "model.safetensors"), 9)

	got, err := CacheEntries()
	if err != nil {
		t.Fatal(err)
	}
	var lines []string
	for _, e := range got {
		rel, _ := filepath.Rel(root, e.Path)
		lines = append(lines, fmt.Sprintf("%s %s %s bytes=%d shards=%d complete=%v sidecars=%d", e.Repo, filepath.ToSlash(rel), e.Kind, e.Bytes, e.Shards, e.Complete, e.Sidecars))
	}
	want := []string{
		"o/ckpt o/ckpt checkpoint bytes=" + fmt.Sprint(dirBytes(ck)) + " shards=0 complete=true sidecars=0",
		"o/ckpt-bad o/ckpt-bad checkpoint bytes=" + fmt.Sprint(dirBytes(bad)) + " shards=0 complete=false sidecars=0",
		"o/gguf-repo o/gguf-repo/big-Q8_0-00001-of-00003.gguf split bytes=60 shards=3 complete=true sidecars=7",
		"o/gguf-repo o/gguf-repo/half-Q6_K-00001-of-00002.gguf split bytes=5 shards=2 complete=false sidecars=0",
		"o/gguf-repo o/gguf-repo/m-Q4_K_M.gguf gguf bytes=100 shards=0 complete=true sidecars=40",
		"o/gguf-repo o/gguf-repo/m-Q8_0.gguf gguf bytes=30 shards=0 complete=false sidecars=0",
		"o/staging o/staging checkpoint bytes=9 shards=0 complete=false sidecars=0",
	}
	if strings.Join(lines, "\n") != strings.Join(want, "\n") {
		t.Fatalf("cache entries:\n%s\nwant:\n%s", strings.Join(lines, "\n"), strings.Join(want, "\n"))
	}
}

// An empty or missing cache is an empty list, not an error: a fresh install has none.
func TestCacheEntries_noCache(t *testing.T) {
	withCacheRoot(t)
	root, _ := CacheRoot()
	_ = os.RemoveAll(root)
	got, err := CacheEntries()
	if err != nil || len(got) != 0 {
		t.Fatalf("no cache: %v, %v; want none and no error", got, err)
	}
}
