package prequant

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

func TestStreamCachePath(t *testing.T) {
	cases := []struct {
		gguf, quant string
		embedInt4   bool
		target      decoder.GIWTarget
		want        string
	}{
		{"/m/foo.gguf", "int8int8", false, decoder.GIWTargetCPUArm64, "/m/foo.int8int8.cpu-arm64.v16.giw"},
		{"/m/foo.gguf", "int4", false, decoder.GIWTargetCPUArm64, "/m/foo.int4.cpu-arm64.v16.giw"},
		{"/m/foo.gguf", "", false, decoder.GIWTargetCPUArm64, "/m/foo.f32.cpu-arm64.v16.giw"}, // "" → f32 label
		{"bar.gguf", "int8", false, decoder.GIWTargetMetal, "bar.int8.metal.v16.giw"},
		// GIWTargetNone (unknown/multi-consumer) spells as "canonical" in the cache key, not a
		// blank segment — so the path stays unambiguous and never collides with a real target.
		{"bar.gguf", "int8", false, decoder.GIWTargetNone, "bar.int8.canonical.v16.giw"},
		// embedInt4 folds an "e4h" segment into the key: a plain-head and an embed-int4 sidecar of
		// the same source and quant must never collide or be reused for each other.
		{"/m/foo.gguf", "int4", true, decoder.GIWTargetCPUArm64, "/m/foo.int4.e4h.cpu-arm64.v16.giw"},
		{"bar.gguf", "int8", true, decoder.GIWTargetNone, "bar.int8.e4h.canonical.v16.giw"},
	}
	for _, c := range cases {
		want := strings.Replace(c.want, ".v16.giw", fmt.Sprintf(".v%d.giw", decoder.GIWFormatVersion), 1) // the cases spell today's v16; a bump moves the name, not these rows
		if got := streamCachePath(c.gguf, c.quant, c.embedInt4, c.target); got != want {
			t.Errorf("streamCachePath(%q,%q,%v,%q) = %q, want %q", c.gguf, c.quant, c.embedInt4, c.target, got, want)
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
