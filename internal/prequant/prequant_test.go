package prequant

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

func TestStreamCachePath(t *testing.T) {
	cases := []struct {
		gguf, quant string
		target      decoder.GIWTarget
		want        string
	}{
		{"/m/foo.gguf", "int8int8", decoder.GIWTargetCPUArm64, "/m/foo.int8int8.cpu-arm64.giw"},
		{"/m/foo.gguf", "int4", decoder.GIWTargetCPUArm64, "/m/foo.int4.cpu-arm64.giw"},
		{"/m/foo.gguf", "", decoder.GIWTargetCPUArm64, "/m/foo.f32.cpu-arm64.giw"}, // "" → f32 label
		{"bar.gguf", "int8", decoder.GIWTargetMetal, "bar.int8.metal.giw"},
		// GIWTargetNone (unknown/multi-consumer) spells as "canonical" in the cache key, not a
		// blank segment — so the path stays unambiguous and never collides with a real target.
		{"bar.gguf", "int8", decoder.GIWTargetNone, "bar.int8.canonical.giw"},
	}
	for _, c := range cases {
		if got := streamCachePath(c.gguf, c.quant, c.target); got != c.want {
			t.Errorf("streamCachePath(%q,%q,%q) = %q, want %q", c.gguf, c.quant, c.target, got, c.want)
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
