package decoder

import (
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/giw"
)

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
	fixture := filepath.Join("..", "testdata", "qwen3moe-tiny")
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
	if _, err := Load(cache, Options{}); err == nil || !strings.Contains(err.Error(), "delete the sidecar") {
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
	m, err := Load(cache, Options{})
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

// tinySource copies the tiny MoE checkpoint into a fresh directory, so the sidecars written beside it do not land in testdata.
func tinySource(t *testing.T) string {
	t.Helper()
	fixture := filepath.Join("..", "testdata", "qwen3moe-tiny")
	src := filepath.Join(t.TempDir(), "qwen3moe-tiny")
	if err := os.Mkdir(src, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"config.json", "generation_config.json", "model.safetensors"} {
		copyFixture(t, filepath.Join(fixture, name), filepath.Join(src, name))
	}
	return src
}

// The name carries the format version: "<base>.<quant>.<target>.v<N>.giw".
func TestSidecarName_carriesTheFormatVersion(t *testing.T) {
	src := tinySource(t)
	cache, err := EnsureCachedGIW(context.Background(), src, "int4", "", false)
	if err != nil {
		t.Fatal(err)
	}
	if want := fmt.Sprintf(".v%d.giw", GIWFormatVersion); !strings.HasSuffix(cache, want) {
		t.Fatalf("sidecar %s does not end in %s", cache, want)
	}
	if v, err := giw.WeightsVersionFile(cache); err != nil || v != GIWFormatVersion {
		t.Fatalf("the file named for v%d holds v%d (%v)", GIWFormatVersion, v, err)
	}
}

// A sidecar an earlier build wrote under the unversioned name, already in this build's format, is this build's sidecar: it is
// hard-linked to the versioned name, with no transcode, and the old name stays for the builds that still read it.
func TestEnsureCachedGIW_adoptsAnUnversionedSidecarInThisFormat(t *testing.T) {
	src := tinySource(t)
	ctx := context.Background()
	cache, err := EnsureCachedGIW(ctx, src, "int4", "", false)
	if err != nil {
		t.Fatal(err)
	}
	legacy := legacyCachePath(src, "int4", false, GIWTargetForBackend(""))
	if err := os.Rename(cache, legacy); err != nil { // as if a build before the change had written it
		t.Fatal(err)
	}
	var got string
	msg := captureStderr(t, func() { got, err = EnsureCachedGIW(ctx, src, "int4", "", false) })
	if err != nil {
		t.Fatal(err)
	}
	if got != cache {
		t.Fatalf("returned %s, want the versioned name %s", got, cache)
	}
	if strings.Contains(msg, "transcoding") {
		t.Errorf("a loadable sidecar under the old name was transcoded again (stderr %q)", msg)
	}
	a, errA := os.Stat(cache)
	b, errB := os.Stat(legacy)
	if errA != nil || errB != nil || !os.SameFile(a, b) {
		t.Fatalf("the versioned name is not a link to the old one (%v, %v)", errA, errB)
	}
	// SidecarPathIfFresh, the cheap probe fit uses, finds it too.
	if p, ok := SidecarPathIfFresh(src, "int4", "", false); !ok || p != cache {
		t.Errorf("SidecarPathIfFresh = %q, %v; want %q, true", p, ok, cache)
	}
}

// Two builds, one box: a build from before the change rewriting the unversioned name (by temp and rename, as it does) does not
// disturb this build's sidecar, and this build never touches that name. Neither transcodes again.
func TestEnsureCachedGIW_buildsAtDifferentFormatsKeepSeparateSidecars(t *testing.T) {
	src := tinySource(t)
	ctx := context.Background()
	cache, err := EnsureCachedGIW(ctx, src, "int4", "", false)
	if err != nil {
		t.Fatal(err)
	}
	legacy := legacyCachePath(src, "int4", false, GIWTargetForBackend(""))
	// the other build's file: a different format, newer than the source
	copyFixture(t, cache, legacy)
	stampWeightsVersion(t, legacy, GIWFormatVersion+1)
	before, err := os.ReadFile(legacy)
	if err != nil {
		t.Fatal(err)
	}
	var again string
	msg := captureStderr(t, func() { again, err = EnsureCachedGIW(ctx, src, "int4", "", false) })
	if err != nil || again != cache {
		t.Fatalf("EnsureCachedGIW = %q, %v; want %q", again, err, cache)
	}
	if strings.Contains(msg, "transcoding") {
		t.Errorf("this build transcoded because the other build's file sits beside its own (stderr %q)", msg)
	}
	after, _ := os.ReadFile(legacy)
	if string(before) != string(after) {
		t.Error("this build rewrote the unversioned sidecar, which belongs to the other build")
	}
}

// An unversioned sidecar in another format is not adopted, is left in place, and the build that finds none of its own transcodes
// once and says the old file is there.
func TestEnsureCachedGIW_leavesAnUnversionedSidecarOfAnotherFormat(t *testing.T) {
	src := tinySource(t)
	ctx := context.Background()
	cache, err := EnsureCachedGIW(ctx, src, "int4", "", false)
	if err != nil {
		t.Fatal(err)
	}
	legacy := legacyCachePath(src, "int4", false, GIWTargetForBackend(""))
	if err := os.Rename(cache, legacy); err != nil {
		t.Fatal(err)
	}
	stampWeightsVersion(t, legacy, GIWFormatVersion-1)
	var got string
	msg := captureStderr(t, func() { got, err = EnsureCachedGIW(ctx, src, "int4", "", false) })
	if err != nil || got != cache {
		t.Fatalf("EnsureCachedGIW = %q, %v; want %q", got, err, cache)
	}
	if !strings.Contains(msg, "transcoding") || !strings.Contains(msg, "unversioned") {
		t.Errorf("expected one transcode and a note about the old name (stderr %q)", msg)
	}
	if v, err := giw.WeightsVersionFile(legacy); err != nil || v != GIWFormatVersion-1 {
		t.Errorf("the old file was changed: v%d (%v)", v, err)
	}
	if v, err := giw.WeightsVersionFile(cache); err != nil || v != GIWFormatVersion {
		t.Errorf("the new sidecar is v%d (%v)", v, err)
	}
}
