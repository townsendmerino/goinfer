package prequant

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// task-never-swap-2026-09.md S1: sidecar .giw by default on darwin. These gates cover the three
// pieces this session added — SidecarPathIfFresh (fit's reuse-only-if-fresh path),
// DefaultToSidecar (the platform policy), and the disk-space guard in EnsureCachedGIW — separate
// from stream_test.go's existing Transcode/cacheFresh coverage, which these build on rather than
// duplicate.

func copyFixture(t *testing.T, src, dst string) {
	t.Helper()
	b, err := os.ReadFile(src)
	if err != nil {
		t.Fatalf("read fixture %s: %v", src, err)
	}
	if err := os.WriteFile(dst, b, 0o644); err != nil {
		t.Fatalf("write %s: %v", dst, err)
	}
}

// realTokenizerGGUF finds a small, tokenizer-bearing GGUF for tests that need a Transcode to
// actually SUCCEED — glm-tiny.gguf (giwFixture) has no embedded tokenizer (Transcode refuses
// before it ever reaches weight-streaming), so a real successful transcode needs a different
// fixture. Same candidate list and heavy-gate as TestTranscode_realGGUFSucceedsAndPublishedBundleLoads,
// this file's own established pattern for the same constraint.
func realTokenizerGGUF(t *testing.T) string {
	t.Helper()
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1 to opt in (needs a real, tokenizer-bearing GGUF)")
	}
	for _, cand := range []string{
		filepath.Join("..", "..", "testdata", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"),
		filepath.Join(os.Getenv("HOME"), "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"),
	} {
		if fi, err := os.Stat(cand); err == nil && fi.Size() > 1<<20 {
			return cand
		}
	}
	t.Skip("no small tokenizer-bearing GGUF found (testdata/ or ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf)")
	return ""
}

func TestDefaultToSidecar(t *testing.T) {
	if got := DefaultToSidecar(true); got {
		t.Error("directLoad=true must always return false, regardless of platform")
	}
	got := DefaultToSidecar(false)
	want := runtime.GOOS == "darwin" || runtime.GOOS == "linux"
	if got != want {
		t.Errorf("directLoad=false on GOOS=%s: got %v, want %v", runtime.GOOS, got, want)
	}
}

func TestSidecarPathIfFresh(t *testing.T) {
	gguf := giwFixture(t)

	t.Run("no cache yet — never transcodes", func(t *testing.T) {
		dir := t.TempDir()
		src := filepath.Join(dir, "model.gguf")
		copyFixture(t, gguf, src)
		if _, ok := SidecarPathIfFresh(src, "int8int8", ""); ok {
			t.Fatal("no cache exists, want ok=false")
		}
		cache := streamCachePath(src, "int8int8", decoder.GIWTargetNone)
		if _, err := os.Stat(cache); err == nil {
			t.Fatal("SidecarPathIfFresh built a cache — it must only ever REUSE one, never build it (fit's whole point is staying cheap)")
		}
	})

	t.Run("fresh cache exists — returned, not rebuilt", func(t *testing.T) {
		dir := t.TempDir()
		src := filepath.Join(dir, "model.gguf")
		copyFixture(t, realTokenizerGGUF(t), src)
		cache := streamCachePath(src, "int8int8", decoder.GIWTargetNone)
		if err := Transcode(context.Background(), src, cache, "int8int8", false, decoder.GIWTargetNone); err != nil {
			t.Fatalf("Transcode: %v", err)
		}
		builtAt, err := os.Stat(cache)
		if err != nil {
			t.Fatal(err)
		}
		got, ok := SidecarPathIfFresh(src, "int8int8", "")
		if !ok || got != cache {
			t.Fatalf("got (%q, %v), want (%q, true)", got, ok, cache)
		}
		after, err := os.Stat(cache)
		if err != nil {
			t.Fatal(err)
		}
		if !after.ModTime().Equal(builtAt.ModTime()) {
			t.Error("the cache's mtime changed — SidecarPathIfFresh rebuilt a bundle that was already fresh")
		}
	})

	t.Run("stale cache (source touched after) — refuses, never rebuilds", func(t *testing.T) {
		dir := t.TempDir()
		src := filepath.Join(dir, "model.gguf")
		copyFixture(t, realTokenizerGGUF(t), src)
		cache := streamCachePath(src, "int8int8", decoder.GIWTargetNone)
		if err := Transcode(context.Background(), src, cache, "int8int8", false, decoder.GIWTargetNone); err != nil {
			t.Fatalf("Transcode: %v", err)
		}
		future := time.Now().Add(time.Hour)
		if err := os.Chtimes(src, future, future); err != nil {
			t.Fatal(err)
		}
		if _, ok := SidecarPathIfFresh(src, "int8int8", ""); ok {
			t.Fatal("source is newer than the cache, want ok=false")
		}
	})
}

func TestEnsureCachedGIW_refusesOnInsufficientDisk(t *testing.T) {
	gguf := giwFixture(t)
	dir := t.TempDir()
	src := filepath.Join(dir, "model.gguf")
	copyFixture(t, gguf, src)

	orig := freeDiskBytes
	defer func() { freeDiskBytes = orig }()
	freeDiskBytes = func(string) (int64, bool) { return 1, true } // 1 byte free: always less than any real source

	_, err := EnsureCachedGIW(context.Background(), src, "int8int8", "")
	if err == nil {
		t.Fatal("want a refusal when free disk is far below the projected sidecar size")
	}
	if !strings.Contains(err.Error(), "free on this disk") {
		t.Errorf("refusal %q does not name the disk-space reason", err.Error())
	}
	cache := streamCachePath(src, "int8int8", decoder.GIWTargetNone)
	if _, statErr := os.Stat(cache); statErr == nil {
		t.Error("a disk-space refusal must not leave a half-written sidecar (or any sidecar) behind")
	}
	tmp := strings.TrimSuffix(cache, ".giw") + ".tmp.giw"
	if _, statErr := os.Stat(tmp); statErr == nil {
		t.Error("a disk-space refusal must not leave a .tmp.giw behind either — it must refuse BEFORE Transcode starts")
	}
}

func TestEnsureCachedGIW_proceedsWhenDiskProbeUnknown(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "model.gguf")
	copyFixture(t, realTokenizerGGUF(t), src)

	orig := freeDiskBytes
	defer func() { freeDiskBytes = orig }()
	freeDiskBytes = func(string) (int64, bool) { return 0, false } // unknown — every unknown proceeds

	cache, err := EnsureCachedGIW(context.Background(), src, "int8int8", "")
	if err != nil {
		t.Fatalf("an unknown disk probe must not block a load that would otherwise have worked: %v", err)
	}
	if _, statErr := os.Stat(cache); statErr != nil {
		t.Fatalf("expected a sidecar to have been built: %v", statErr)
	}
}
