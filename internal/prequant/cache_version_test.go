package prequant

import (
	"bytes"
	"context"
	"encoding/binary"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/giw"
)

// fakeBundle writes a v3 bundle whose weights blob is only a "GINFW" header at version v: enough for
// giw.WeightsVersionFile, never enough to load.
func fakeBundle(t *testing.T, path string, v uint32) {
	t.Helper()
	blob := binary.LittleEndian.AppendUint32([]byte("GINFW"), v)
	if err := os.WriteFile(path, giw.Write(blob, []byte("tok")), 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestCacheLayoutCurrent is the rule: an int4-bearing sidecar older than v15 is not current (it loads
// only by converting its scales), anything at v15 or later is, and a non-int4 quant is never rebuilt for
// it — v15 changed nothing but the int4 kinds.
func TestCacheLayoutCurrent(t *testing.T) {
	dir := t.TempDir()
	old, cur := filepath.Join(dir, "old.giw"), filepath.Join(dir, "cur.giw")
	fakeBundle(t, old, 12)
	fakeBundle(t, cur, minInt4CacheGIWVersion)
	for _, c := range []struct {
		path, quant string
		want        bool
	}{
		{old, "int4", false},
		{old, "int4mix", false},
		{cur, "int4", true},
		{cur, "int4mix", true},
		{old, "int8int8", true},
		{old, "int8", true},
		{old, "", true},
		{filepath.Join(dir, "missing.giw"), "int4", true}, // unreadable: selfCheck names the failure
	} {
		if _, got := cacheLayoutCurrent(c.path, c.quant); got != c.want {
			t.Errorf("cacheLayoutCurrent(%s, %q) = %v, want %v", filepath.Base(c.path), c.quant, got, c.want)
		}
	}
}

// TestCacheFresh_pre15Int4Rebuilds pins that cacheFresh CONSULTS the rule, not just that the rule exists:
// a pre-v15 int4 sidecar newer than its source is refused with the rebuild message. The fake bundle
// would fail selfCheck too, so the assertion is on the message, which only the version branch prints.
func TestCacheFresh_pre15Int4Rebuilds(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(src, []byte("gguf"), 0o644); err != nil {
		t.Fatal(err)
	}
	cache := filepath.Join(dir, "model.int4.cpu-amd64.giw")
	fakeBundle(t, cache, 12)
	later := time.Now().Add(time.Minute)
	if err := os.Chtimes(cache, later, later); err != nil {
		t.Fatal(err)
	}
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	stderr := os.Stderr
	os.Stderr = w
	fresh := cacheFresh(cache, src, "int4")
	os.Stderr = stderr
	w.Close()
	msg, _ := io.ReadAll(r)
	if fresh {
		t.Fatal("a v12 int4 sidecar was judged fresh — it would be kept, converting its scales on every load")
	}
	if !strings.Contains(string(msg), "is format v12") || !strings.Contains(string(msg), "rebuilding once") {
		t.Errorf("refused, but not by the version rule (stderr %q)", msg)
	}
}

// TestCacheLayoutCurrent_currentWriterPasses is the loop guard: a sidecar the CURRENT transcode writes must
// pass the rule, or every load would rebuild it. Drives decoder.StreamTranscodeGGUF, the exact writer
// Transcode uses, at int4 for both CPU targets.
func TestCacheLayoutCurrent_currentWriterPasses(t *testing.T) {
	gguf := giwFixture(t)
	for _, target := range []decoder.GIWTarget{decoder.GIWTargetCPUAmd64, decoder.GIWTargetCPUArm64} {
		var blob bytes.Buffer
		if _, err := decoder.StreamTranscodeGGUF(context.Background(), gguf, &blob, "int4", false, target, "glm-tiny.gguf"); err != nil {
			t.Fatalf("%s: StreamTranscodeGGUF: %v", target, err)
		}
		p := filepath.Join(t.TempDir(), "model.int4."+string(target)+".giw")
		if err := os.WriteFile(p, giw.Write(blob.Bytes(), []byte("tok")), 0o644); err != nil {
			t.Fatal(err)
		}
		v, ok := cacheLayoutCurrent(p, "int4")
		if !ok {
			t.Errorf("%s: the current writer emits v%d, below minInt4CacheGIWVersion %d — every load would rebuild", target, v, minInt4CacheGIWVersion)
		}
		t.Logf("%s: current writer emits v%d", target, v)
	}
}
