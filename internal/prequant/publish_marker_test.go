package prequant

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// The temp bundle is self-checked (full CRC, marker written) and then renamed into place. The
// marker must follow the rename — otherwise every freshly built sidecar pays a second whole-file
// CRC on its first real load — and must never be left behind under the temp name.
func TestCarryVerifiedMarker_followsTheRename(t *testing.T) {
	dir := t.TempDir()
	tmp, out := filepath.Join(dir, "m.tmp.giw"), filepath.Join(dir, "m.giw")
	if err := os.WriteFile(tmp, []byte("bundle"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(decoder.GIWVerifiedMarkerPath(tmp), []byte("marker"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(tmp, out); err != nil {
		t.Fatal(err)
	}
	carryVerifiedMarker(tmp, out)
	if b, err := os.ReadFile(decoder.GIWVerifiedMarkerPath(out)); err != nil || string(b) != "marker" {
		t.Fatalf("marker did not follow the rename to %s: %v %q", out, err, b)
	}
	if _, err := os.Stat(decoder.GIWVerifiedMarkerPath(tmp)); err == nil {
		t.Error("marker left behind under the temp name")
	}
}

// A publish with no marker (e.g. the self-check ran where the marker could not be written) is not
// an error.
func TestCarryVerifiedMarker_noMarkerIsFine(t *testing.T) {
	dir := t.TempDir()
	tmp, out := filepath.Join(dir, "m.tmp.giw"), filepath.Join(dir, "m.giw")
	if err := os.WriteFile(tmp, []byte("bundle"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(tmp, out); err != nil {
		t.Fatal(err)
	}
	carryVerifiedMarker(tmp, out) // must not panic or leave anything behind
	if _, err := os.Stat(decoder.GIWVerifiedMarkerPath(out)); err == nil {
		t.Error("a marker appeared from nowhere")
	}
}

func TestRemoveTempGIW_removesMarkerToo(t *testing.T) {
	dir := t.TempDir()
	tmp := filepath.Join(dir, "m.tmp.giw")
	for _, p := range []string{tmp, decoder.GIWVerifiedMarkerPath(tmp)} {
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	removeTempGIW(tmp)
	for _, p := range []string{tmp, decoder.GIWVerifiedMarkerPath(tmp)} {
		if _, err := os.Stat(p); err == nil {
			t.Errorf("%s survived removeTempGIW", p)
		}
	}
}
