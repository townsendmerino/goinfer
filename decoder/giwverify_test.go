package decoder

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/internal/giw"
)

// writeTinyGIW writes a small real .giw bundle and returns its path.
func writeTinyGIW(t *testing.T) string {
	t.Helper()
	raw, _, _, _ := tinyNormRopeGGUF("llama")
	w := loadTinyGGUFWeights(t, raw, "llama")
	out := filepath.Join(t.TempDir(), "verify.giw")
	f, err := os.Create(out)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	werr := giw.WriteStream(f, nil, func(dst io.Writer) (int64, error) {
		return SerializeWeightsToForTarget(dst, w, "verify-fixture", GIWTargetNone)
	})
	if cerr := f.Close(); werr == nil {
		werr = cerr
	}
	if werr != nil {
		t.Fatalf("write bundle: %v", werr)
	}
	return out
}

// corruptWeightByte flips one byte deep inside the payload without changing the file's size, and
// restores its mtime so the file looks untouched to any (size, mtime) check.
func corruptWeightByte(t *testing.T, path string) {
	t.Helper()
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(path, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	off := fi.Size() * 3 / 4
	var b [1]byte
	if _, err := f.ReadAt(b[:], off); err != nil {
		t.Fatal(err)
	}
	b[0] ^= 0xFF
	if _, err := f.WriteAt(b[:], off); err != nil {
		t.Fatal(err)
	}
	f.Close()
	if err := os.Chtimes(path, fi.ModTime(), fi.ModTime()); err != nil {
		t.Fatal(err)
	}
}

func unsetVerifyEnv(t *testing.T) {
	t.Helper()
	orig, was := os.LookupEnv(giwVerifyEnv)
	os.Unsetenv(giwVerifyEnv)
	t.Cleanup(func() {
		if was {
			os.Setenv(giwVerifyEnv, orig)
		}
	})
}

// TestGIWVerified_firstLoadVerifiesThenSkips is the gate for the once-per-file CRC. The proof that
// a later load SKIPS the pass is indirect on purpose: flip a payload byte without changing size or
// mtime and the file loads anyway; the same file under GOINFER_GIW_VERIFY=always is refused, which
// also proves the flip is a real corruption the CRC would have caught.
func TestGIWVerified_firstLoadVerifiesThenSkips(t *testing.T) {
	unsetVerifyEnv(t)
	path := writeTinyGIW(t)
	marker := GIWVerifiedMarkerPath(path)

	if _, err := os.Stat(marker); err == nil {
		t.Fatal("marker exists before any load")
	}
	m, err := Load(path, Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("first Load: %v", err)
	}
	m.Close()
	if _, err := os.Stat(marker); err != nil {
		t.Fatalf("first Load wrote no marker: %v", err)
	}

	corruptWeightByte(t, path)

	m, err = Load(path, Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("second Load of a verified, unchanged-stat file re-ran the CRC (corruption was detected, so the pass was NOT skipped): %v", err)
	}
	m.Close()

	t.Setenv(giwVerifyEnv, "always")
	if _, err := Load(path, Options{Backend: "cpu"}); err == nil || !strings.Contains(err.Error(), "CRC") {
		t.Fatalf("GOINFER_GIW_VERIFY=always must force the full CRC and refuse the corrupted file, got: %v", err)
	}
}

// A changed mtime or size invalidates the marker, so a replaced or truncated file is re-verified.
func TestGIWVerified_changedFileIsReverified(t *testing.T) {
	unsetVerifyEnv(t)
	for _, tc := range []struct {
		name   string
		mutate func(t *testing.T, path string)
	}{
		{"mtime changed", func(t *testing.T, path string) {
			later := time.Now().Add(2 * time.Hour)
			if err := os.Chtimes(path, later, later); err != nil {
				t.Fatal(err)
			}
		}},
		{"size changed (truncated)", func(t *testing.T, path string) {
			fi, _ := os.Stat(path)
			if err := os.Truncate(path, fi.Size()-8); err != nil {
				t.Fatal(err)
			}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := writeTinyGIW(t)
			m, err := Load(path, Options{Backend: "cpu"})
			if err != nil {
				t.Fatalf("first Load: %v", err)
			}
			m.Close()
			corruptWeightByte(t, path)
			tc.mutate(t, path)
			if _, err := Load(path, Options{Backend: "cpu"}); err == nil {
				t.Fatal("a file whose size/mtime changed after verification loaded without a fresh CRC pass")
			}
		})
	}
}

// A marker that cannot be written must never fail the load — it only costs the next load a CRC.
func TestGIWVerified_unwritableMarkerDirIsNotAnError(t *testing.T) {
	unsetVerifyEnv(t)
	path := writeTinyGIW(t)
	dir := filepath.Dir(path)
	if err := os.Chmod(dir, 0o555); err != nil {
		t.Skipf("cannot make dir read-only: %v", err)
	}
	t.Cleanup(func() { os.Chmod(dir, 0o755) })
	if f, err := os.Create(filepath.Join(dir, "probe")); err == nil {
		f.Close()
		t.Skip("directory is still writable (running as root?) — cannot exercise the read-only case")
	}
	m, err := Load(path, Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("Load must succeed when the marker cannot be written: %v", err)
	}
	m.Close()
	if _, err := os.Stat(GIWVerifiedMarkerPath(path)); err == nil {
		t.Fatal("marker appeared in a read-only directory")
	}
}

// A malformed marker is treated as absent.
func TestGIWVerified_malformedMarkerIgnored(t *testing.T) {
	unsetVerifyEnv(t)
	path := writeTinyGIW(t)
	fi, _ := os.Stat(path)
	for _, body := range []string{"", "garbage", "giwv1 1 2\n", "giwv1 " + "x y\n"} {
		if err := os.WriteFile(GIWVerifiedMarkerPath(path), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		if giwVerified(path, fi) {
			t.Errorf("marker %q accepted", body)
		}
	}
	if err := os.WriteFile(GIWVerifiedMarkerPath(path), []byte(giwMarkerBody(fi)), 0o644); err != nil {
		t.Fatal(err)
	}
	if !giwVerified(path, fi) {
		t.Error("a correct marker was rejected")
	}
}
