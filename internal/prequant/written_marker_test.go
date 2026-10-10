package prequant

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// flipLastPayloadByte flips the byte before a bundle's trailing CRC word and restores the file's mtime, so the file
// keeps the size and mtime its verified-marker names while no longer matching its own CRC.
func flipLastPayloadByte(t *testing.T, path string) {
	t.Helper()
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(path, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	b := make([]byte, 1)
	off := fi.Size() - 5
	if _, err := f.ReadAt(b, off); err != nil {
		t.Fatal(err)
	}
	b[0] ^= 0xff
	if _, err := f.WriteAt(b, off); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, fi.ModTime(), fi.ModTime()); err != nil {
		t.Fatal(err)
	}
}

// TestTranscode_selfCheckRunsNoCRCPass: the transcode marks the bundle it wrote, so its self-check loads the bundle
// without reading it back for the CRC, and the marker follows the bundle to its final name. The proof is a payload
// byte flipped between the write and the self-check: the transcode still succeeds and the sidecar still loads, and the
// same sidecar under GOINFER_GIW_VERIFY=always is refused, which shows the flip is one a CRC pass catches.
func TestTranscode_selfCheckRunsNoCRCPass(t *testing.T) {
	if v, was := os.LookupEnv("GOINFER_GIW_VERIFY"); was {
		os.Unsetenv("GOINFER_GIW_VERIFY")
		t.Cleanup(func() { os.Setenv("GOINFER_GIW_VERIFY", v) })
	}
	hooked := 0
	afterWriteForTest = func(tmp string) {
		hooked++
		if _, err := os.Stat(decoder.GIWVerifiedMarkerPath(tmp)); err != nil {
			t.Errorf("the written bundle has no marker before its self-check: %v", err)
		}
		flipLastPayloadByte(t, tmp)
	}
	t.Cleanup(func() { afterWriteForTest = nil })

	src := filepath.Join("..", "..", "testdata", "qwen3moe-tiny")
	out := filepath.Join(t.TempDir(), "p.giw")
	if err := Transcode(context.Background(), src, out, "int4", false, decoder.GIWTargetNone); err != nil {
		t.Fatalf("the self-check ran a CRC pass over a bundle the transcode had marked: %v", err)
	}
	if hooked != 1 {
		t.Fatalf("the hook ran %d times, want 1", hooked)
	}
	if _, err := os.Stat(decoder.GIWVerifiedMarkerPath(out)); err != nil {
		t.Fatalf("the marker did not follow the bundle to its final name: %v", err)
	}
	m, err := decoder.Load(out, decoder.Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("the first load of the sidecar ran a CRC pass: %v", err)
	}
	m.Close()

	t.Setenv("GOINFER_GIW_VERIFY", "always")
	if _, err := decoder.Load(out, decoder.Options{Backend: "cpu"}); err == nil || !strings.Contains(err.Error(), "CRC") {
		t.Fatalf("GOINFER_GIW_VERIFY=always must run the CRC and refuse the flipped bundle, got: %v", err)
	}
}

// TestTranscode_selfCheckStillRefusesAnUnloadableBundle: the self-check still loads the bundle, so one that does not
// deserialize never becomes the sidecar. The bundle's header is overwritten between the write and the self-check, at
// the same size and mtime, so the marker still holds and the refusal is the load's own and not the CRC's.
func TestTranscode_selfCheckStillRefusesAnUnloadableBundle(t *testing.T) {
	afterWriteForTest = func(tmp string) {
		fi, err := os.Stat(tmp)
		if err != nil {
			t.Fatal(err)
		}
		f, err := os.OpenFile(tmp, os.O_RDWR, 0)
		if err != nil {
			t.Fatal(err)
		}
		junk := make([]byte, 64)
		for i := range junk {
			junk[i] = 0xff
		}
		if _, err := f.WriteAt(junk, 8); err != nil {
			t.Fatal(err)
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(tmp, fi.ModTime(), fi.ModTime()); err != nil {
			t.Fatal(err)
		}
	}
	t.Cleanup(func() { afterWriteForTest = nil })

	src := filepath.Join("..", "..", "testdata", "qwen3moe-tiny")
	dir := t.TempDir()
	out := filepath.Join(dir, "p.giw")
	err := Transcode(context.Background(), src, out, "int4", false, decoder.GIWTargetNone)
	if err == nil {
		t.Fatal("a bundle with an overwritten header passed the self-check")
	}
	if strings.Contains(err.Error(), "CRC") {
		t.Fatalf("the refusal came from a CRC pass, not from the load: %v", err)
	}
	left, _ := os.ReadDir(dir)
	if len(left) != 0 {
		t.Fatalf("a failed transcode left %d files behind (%s first)", len(left), left[0].Name())
	}
}
