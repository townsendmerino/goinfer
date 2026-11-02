package pull

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// diskTestFiles is two files of a set, served by a counting CDN, with the disk's free space under the test's control.
// asked records each directory the free-space probe was given; a probe of a directory that does not exist fails the test.
type diskTestFiles struct {
	set   []File
	hits  atomic.Int32
	asked []string
}

func newDiskTestFiles(t *testing.T, free int64, known bool) *diskTestFiles {
	t.Helper()
	d := &diskTestFiles{}
	bodies := map[string][]byte{"a.gguf": bytes.Repeat([]byte("a"), 1000), "b.gguf": bytes.Repeat([]byte("b"), 500)}
	for _, p := range []string{"a.gguf", "b.gguf"} {
		sum := sha256.Sum256(bodies[p])
		d.set = append(d.set, File{Path: p, Size: int64(len(bodies[p])), SHA256: hex.EncodeToString(sum[:])})
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		d.hits.Add(1)
		name := filepath.Base(r.URL.Path)
		http.ServeContent(w, r, name, time.Unix(0, 0), bytes.NewReader(bodies[name]))
	}))
	t.Cleanup(srv.Close)
	oc, of := hfCDN, freeDiskBytes
	hfCDN = srv.URL
	freeDiskBytes = func(dir string) (int64, bool) {
		if _, err := os.Stat(dir); err != nil {
			t.Errorf("free space asked of %s, which does not exist: %v", dir, err)
		}
		d.asked = append(d.asked, dir)
		return free, known
	}
	t.Cleanup(func() { hfCDN, freeDiskBytes = oc, of })
	return d
}

func TestDownloadSet_refusesWhatWillNotFit_beforeAnyByte(t *testing.T) {
	d := newDiskTestFiles(t, 1499, true)
	dir := filepath.Join(t.TempDir(), "not", "yet")
	_, err := DownloadSet(context.Background(), "o/r", d.set, dir, nil)
	if err == nil {
		t.Fatal("1,499 bytes free for a 1,500-byte set: want a refusal")
	}
	for _, want := range []string{"o/r", humanBytes(1500), humanBytes(1499), dir} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("refusal %q does not name %q", err, want)
		}
	}
	if n := d.hits.Load(); n != 0 {
		t.Errorf("%d requests made before refusing, want 0", n)
	}
	if ents, _ := os.ReadDir(dir); len(ents) != 0 {
		t.Errorf("a refusal left %d files behind", len(ents))
	}
}

func TestDownloadSet_fitsAtExactlyItsBytes(t *testing.T) {
	d := newDiskTestFiles(t, 1500, true)
	dir := t.TempDir()
	path, err := DownloadSet(context.Background(), "o/r", d.set, dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	if path != filepath.Join(dir, "a.gguf") {
		t.Errorf("path %s", path)
	}
	if len(d.asked) != 1 || d.asked[0] != dir {
		t.Errorf("free space asked of %v, want once, of %s", d.asked, dir)
	}
}

func TestDownloadSet_unknownFreeSpaceProceeds(t *testing.T) {
	d := newDiskTestFiles(t, 0, false)
	if _, err := DownloadSet(context.Background(), "o/r", d.set, t.TempDir(), nil); err != nil {
		t.Fatalf("an unknown reading must not refuse: %v", err)
	}
}

// A resume needs only what is left: a file already at its final path costs nothing, and a .part costs its tail.
func TestDownloadSet_countsOnlyWhatIsLeft(t *testing.T) {
	for _, tc := range []struct {
		name string
		free int64
		ok   bool
	}{
		{"one byte short of the tail", 199, false},
		{"exactly the tail", 200, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d := newDiskTestFiles(t, tc.free, true)
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "a.gguf"), bytes.Repeat([]byte("a"), 1000), 0o644); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, "b.gguf.part"), bytes.Repeat([]byte("b"), 300), 0o644); err != nil {
				t.Fatal(err)
			}
			_, err := DownloadSet(context.Background(), "o/r", d.set, dir, nil)
			if (err == nil) != tc.ok {
				t.Fatalf("free %d: err %v, want ok=%v", tc.free, err, tc.ok)
			}
			if !tc.ok && !strings.Contains(err.Error(), humanBytes(200)) {
				t.Errorf("refusal %q does not name the %s left", err, humanBytes(200))
			}
			if tc.ok {
				if got := d.hits.Load(); got != 1 {
					t.Errorf("%d requests, want 1 (the tail of b.gguf)", got)
				}
			}
		})
	}
}

func TestDownloadCheckpoint_refusesWhatWillNotFit_beforeAnyByte(t *testing.T) {
	d := newDiskTestFiles(t, 1499, true)
	dest := filepath.Join(t.TempDir(), "ckpt")
	p := Plan{Repo: "o/r", Files: d.set, Bytes: SetBytes(d.set)}
	_, err := DownloadCheckpoint(context.Background(), p, dest, nil)
	if err == nil {
		t.Fatal("1,499 bytes free for a 1,500-byte checkpoint: want a refusal")
	}
	if n := d.hits.Load(); n != 0 {
		t.Errorf("%d requests made before refusing, want 0", n)
	}
	if want := dest + ".partial"; len(d.asked) != 1 || d.asked[0] != want {
		t.Errorf("free space asked of %v, want once, of %s", d.asked, want)
	}
}

// A checkpoint's files live in subdirectories of its staging directory; a partial in one still counts.
func TestDownloadCheckpoint_countsPartialsInSubdirectories(t *testing.T) {
	d := newDiskTestFiles(t, 1001, true)
	d.set[1].Path = "sub/b.gguf"
	dest := filepath.Join(t.TempDir(), "ckpt")
	sub := filepath.Join(dest+".partial", "sub")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(sub, "b.gguf.part"), bytes.Repeat([]byte("b"), 499), 0o644); err != nil {
		t.Fatal(err)
	}
	p := Plan{Repo: "o/r", Files: d.set, Bytes: SetBytes(d.set)}
	if _, err := DownloadCheckpoint(context.Background(), p, dest, nil); err != nil {
		t.Fatalf("1,001 bytes left and 1,001 free: %v", err)
	}
}
