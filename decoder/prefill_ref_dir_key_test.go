//go:build goinfer_testhooks

package decoder

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A safetensors checkpoint directory keys a reference by its files (D-B01's first directory cell failed with "is a
// directory"): the key is stable, ignores goinfer's own sidecars and hidden files, and changes when a weight byte does.
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
