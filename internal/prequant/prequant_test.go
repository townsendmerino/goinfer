package prequant

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestStreamCachePath(t *testing.T) {
	cases := []struct{ gguf, quant, want string }{
		{"/m/foo.gguf", "int8int8", "/m/foo.int8int8.giw"},
		{"/m/foo.gguf", "int4", "/m/foo.int4.giw"},
		{"/m/foo.gguf", "", "/m/foo.f32.giw"}, // "" → f32 label
		{"bar.gguf", "int8", "bar.int8.giw"},
	}
	for _, c := range cases {
		if got := streamCachePath(c.gguf, c.quant); got != c.want {
			t.Errorf("streamCachePath(%q,%q) = %q, want %q", c.gguf, c.quant, got, c.want)
		}
	}
}

// TestCacheFresh: a cache is fresh only when it exists and is newer than the source
// — so replacing/re-touching the GGUF invalidates a stale cache.
func TestCacheNewer(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "model.gguf")
	cache := filepath.Join(dir, "model.int8.giw")
	write := func(p string) {
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	write(src)
	if cacheNewer(cache, src) {
		t.Error("no cache file → must not be fresh")
	}

	// cache written after src → fresh
	time.Sleep(10 * time.Millisecond)
	write(cache)
	if !cacheNewer(cache, src) {
		t.Error("cache newer than src → must be fresh")
	}

	// src re-touched after cache → stale
	time.Sleep(10 * time.Millisecond)
	now := time.Now()
	if err := os.Chtimes(src, now, now); err != nil {
		t.Fatal(err)
	}
	if cacheNewer(cache, src) {
		t.Error("src newer than cache → must be stale")
	}
}
