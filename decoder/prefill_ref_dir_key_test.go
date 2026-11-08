//go:build goinfer_testhooks

package decoder

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// A safetensors checkpoint directory keys a reference by its files: the key is stable, ignores goinfer's own sidecars and
// hidden files, and changes when a weight byte does.
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

// TestCheckpointSHA256_sameSizeRewriteAtTheSameMtime: a settled file's hash is cached under its stamp, and a same-size rewrite whose
// modification time is held at the first write's (a coarse clock does that: two writes in one tick share it) is not served that
// entry. The clock is held a minute ahead, so both reads are settled and the second can only miss the entry through the stamp.
func TestCheckpointSHA256_sameSizeRewriteAtTheSameMtime(t *testing.T) {
	cache := t.TempDir()
	t.Setenv("GOINFER_PREFILL_REF_CACHE", cache)
	defer func(now func() time.Time) { checkpointNow = now }(checkpointNow)
	checkpointNow = func() time.Time { return time.Now().Add(time.Minute) }
	p := filepath.Join(t.TempDir(), "model.safetensors")
	if err := os.WriteFile(p, []byte("weights-a"), 0o644); err != nil {
		t.Fatal(err)
	}
	st, err := os.Stat(p)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, _, ok := checkpointChange(st); !ok {
		t.Skip("this OS reports no change time, so nothing is cached (TestCheckpointStamp_vouchesOnlyForASettledFile holds that)")
	}
	h1, err := checkpointSHA256(p)
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(cache, "filehash.json"))
	if err != nil {
		t.Fatalf("the settled file's hash was not cached: %v", err)
	}
	var idx map[string]string
	if err := json.Unmarshal(b, &idx); err != nil || len(idx) != 1 {
		t.Fatalf("the cache holds %v (err %v), want the one entry for the file", idx, err)
	}
	for _, h := range idx {
		if h != h1 {
			t.Fatalf("the cache holds %s for the file, which hashed to %s", h, h1)
		}
	}
	if err := os.WriteFile(p, []byte("weights-b"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(p, st.ModTime(), st.ModTime()); err != nil {
		t.Fatal(err)
	}
	if h2, err := checkpointSHA256(p); err != nil || h2 == h1 {
		t.Fatalf("a same-size rewrite at the same mtime kept the hash %s (err %v): the cache served a stale key", h1, err)
	}
}

// fakeStat is an os.FileInfo with a set size and mtime; checkpointChange is stubbed beside it for the rest.
type fakeStat struct {
	size  int64
	mtime time.Time
}

func (f fakeStat) Name() string       { return "model.safetensors" }
func (f fakeStat) Size() int64        { return f.size }
func (f fakeStat) Mode() os.FileMode  { return 0o644 }
func (f fakeStat) ModTime() time.Time { return f.mtime }
func (f fakeStat) IsDir() bool        { return false }
func (f fakeStat) Sys() any           { return nil }

// TestCheckpointStamp_vouchesOnlyForASettledFile: the stamp vouches for a file only once its last change, the later of its mtime
// and change time, is checkpointRaceWindow before now; it moves with the change time, the inode and the device; and where the OS
// reports no change time (Windows) it vouches for nothing. The clock and the change time are given, not read.
func TestCheckpointStamp_vouchesOnlyForASettledFile(t *testing.T) {
	defer func(c func(os.FileInfo) (time.Time, uint64, uint64, bool)) { checkpointChange = c }(checkpointChange)
	now := time.Date(2026, 11, 8, 12, 0, 0, 0, time.UTC)
	mtime := now.Add(-time.Hour)
	st := fakeStat{size: 9, mtime: mtime}
	change := func(ctime time.Time, ino, dev uint64, ok bool) {
		checkpointChange = func(os.FileInfo) (time.Time, uint64, uint64, bool) { return ctime, ino, dev, ok }
	}

	change(time.Time{}, 0, 0, false)
	if _, settled := checkpointStamp("/m", st, now); settled {
		t.Error("a stamp with no change time to read vouched for the file")
	}
	change(now.Add(-checkpointRaceWindow+time.Millisecond), 7, 1, true)
	if _, settled := checkpointStamp("/m", st, now); settled {
		t.Error("a file whose change time is inside the race window is stamped as settled")
	}
	change(now.Add(-checkpointRaceWindow), 7, 1, true)
	if _, settled := checkpointStamp("/m", st, now); !settled {
		t.Error("a file changed exactly the race window ago is not stamped")
	}
	change(now.Add(-time.Minute), 7, 1, true)
	s1, settled := checkpointStamp("/m", st, now)
	if !settled {
		t.Fatal("a file settled a minute ago is not stamped")
	}
	if _, settled := checkpointStamp("/m", fakeStat{size: 9, mtime: now.Add(-time.Millisecond)}, now); settled {
		t.Error("a file whose mtime is inside the race window is stamped as settled")
	}
	for _, c := range []struct {
		what     string
		ctime    time.Time
		ino, dev uint64
	}{
		{"a later change time (a same-size rewrite with its mtime held)", now.Add(-30 * time.Second), 7, 1},
		{"another inode", now.Add(-time.Minute), 8, 1},
		{"another device", now.Add(-time.Minute), 7, 2},
	} {
		change(c.ctime, c.ino, c.dev, true)
		if s, _ := checkpointStamp("/m", st, now); s == s1 {
			t.Errorf("%s kept the stamp %s", c.what, s1)
		}
	}
}
