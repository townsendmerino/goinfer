package prequant

import (
	"context"
	"encoding/binary"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/giw"
)

// captureStderr runs fn and returns what it wrote to os.Stderr.
func captureStderr(t *testing.T, fn func()) string {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	saved := os.Stderr
	os.Stderr = w
	done := make(chan []byte)
	go func() { b, _ := io.ReadAll(r); done <- b }()
	fn()
	os.Stderr = saved
	w.Close()
	return string(<-done)
}

// stampWeightsVersion overwrites the format version of the weights blob inside the bundle at path, in place and
// without touching its mtime order against the source: a sidecar exactly as an earlier build left it, as far as the
// reader's first check can tell.
func stampWeightsVersion(t *testing.T, path string, v uint32) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	blob, _, err := giw.Read(data)
	if err != nil {
		t.Fatal(err)
	}
	const magic = "GINFW"
	if string(blob[:len(magic)]) != magic {
		t.Fatalf("no weights magic in %s", path)
	}
	binary.LittleEndian.PutUint32(blob[len(magic):], v) // blob aliases data
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, fi.ModTime(), fi.ModTime()); err != nil {
		t.Fatal(err)
	}
}

// TestEnsureCachedGIW_olderFormatIsRebuiltOnce: a sidecar in a format this build does not read is newer than its
// source, so mtime alone would keep it forever. EnsureCachedGIW must refuse it for its version, rebuild it, and then
// find the rebuilt file fresh, or every load would transcode.
func TestEnsureCachedGIW_olderFormatIsRebuiltOnce(t *testing.T) {
	// A tracked tiny checkpoint directory that transcodes without a tokenizer-bearing GGUF, copied so the sidecar is
	// written beside the copy and not into testdata.
	fixture := filepath.Join("..", "..", "testdata", "qwen3moe-tiny")
	src := filepath.Join(t.TempDir(), "qwen3moe-tiny")
	if err := os.Mkdir(src, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"config.json", "generation_config.json", "model.safetensors"} {
		copyFixture(t, filepath.Join(fixture, name), filepath.Join(src, name))
	}
	ctx := context.Background()

	cache, err := EnsureCachedGIW(ctx, src, "int4", "", false)
	if err != nil {
		t.Fatalf("first EnsureCachedGIW: %v", err)
	}
	cur, err := giw.WeightsVersionFile(cache)
	if err != nil {
		t.Fatal(err)
	}
	stampWeightsVersion(t, cache, cur-1)
	if _, err := decoder.Load(cache, decoder.Options{}); err == nil || !strings.Contains(err.Error(), "delete the sidecar") {
		t.Fatalf("a v%d sidecar loaded, or was refused without naming the fix: %v", cur-1, err)
	}

	var again string
	msg := captureStderr(t, func() { again, err = EnsureCachedGIW(ctx, src, "int4", "", false) })
	if err != nil {
		t.Fatalf("EnsureCachedGIW over a v%d sidecar: %v", cur-1, err)
	}
	if again != cache {
		t.Errorf("rebuilt into %s, want the same sidecar %s", again, cache)
	}
	if !strings.Contains(msg, "does not load") || !strings.Contains(msg, "transcoding") {
		t.Errorf("the older sidecar was not refused and rebuilt (stderr %q)", msg)
	}
	if v, err := giw.WeightsVersionFile(cache); err != nil || v != cur {
		t.Errorf("after the rebuild the sidecar is v%d (%v), want v%d", v, err, cur)
	}
	m, err := decoder.Load(cache, decoder.Options{})
	if err != nil {
		t.Fatalf("the rebuilt sidecar does not load: %v", err)
	}
	m.Close()

	// Once: the rebuilt sidecar is fresh, so a third call transcodes nothing.
	msg = captureStderr(t, func() { _, err = EnsureCachedGIW(ctx, src, "int4", "", false) })
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(msg, "transcoding") {
		t.Errorf("the rebuilt sidecar was transcoded again (stderr %q): every load would rebuild", msg)
	}
}
