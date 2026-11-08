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

// TestCheckpointSHA256_sameSizeRewriteAtTheSameMtime: a same-size rewrite with its modification time held at the first write's (as
// a coarse clock, or a backdating copy, leaves it) is never served the first write's hash. The test gives the clock and the change
// time, so neither depends on a real clock advancing:
//   - the change time moves (the rewrite lands a tick later): the first read is settled and cached, and the rewrite's new stamp
//     misses that entry;
//   - the change time does not move (the rewrite lands in the same tick): the stamp cannot tell the two apart, so the first read,
//     within checkpointRaceWindow of the change, must cache nothing, and the second, settled, hashes the new bytes.
func TestCheckpointSHA256_sameSizeRewriteAtTheSameMtime(t *testing.T) {
	defer func(now func() time.Time, c func(os.FileInfo) (time.Time, uint64, uint64, bool)) {
		checkpointNow, checkpointChange = now, c
	}(checkpointNow, checkpointChange)
	for _, c := range []struct {
		name          string
		ctimeMoves    bool
		firstRead     time.Duration // after the change time
		firstCachedTo int           // filehash.json entries after the first read
	}{
		{"the change time moves", true, time.Minute, 1},
		{"the same tick: the change time does not move", false, time.Second, 0},
	} {
		t.Run(c.name, func(t *testing.T) {
			cache := t.TempDir()
			t.Setenv("GOINFER_PREFILL_REF_CACHE", cache)
			p := filepath.Join(t.TempDir(), "model.safetensors")
			if err := os.WriteFile(p, []byte("weights-a"), 0o644); err != nil {
				t.Fatal(err)
			}
			st, err := os.Stat(p)
			if err != nil {
				t.Fatal(err)
			}
			mtime := st.ModTime()
			ctime := mtime
			checkpointChange = func(os.FileInfo) (time.Time, uint64, uint64, bool) { return ctime, 7, 1, true }
			read := func(now time.Time) string {
				t.Helper()
				checkpointNow = func() time.Time { return now }
				h, err := checkpointSHA256(p)
				if err != nil {
					t.Fatal(err)
				}
				return h
			}
			entries := func() map[string]string {
				t.Helper()
				b, err := os.ReadFile(filepath.Join(cache, "filehash.json"))
				if os.IsNotExist(err) {
					return nil
				}
				var idx map[string]string
				if err != nil || json.Unmarshal(b, &idx) != nil {
					t.Fatalf("filehash.json: %v", err)
				}
				return idx
			}
			h1 := read(ctime.Add(c.firstRead))
			if idx := entries(); len(idx) != c.firstCachedTo {
				t.Fatalf("after the first read the cache holds %v, want %d entries", idx, c.firstCachedTo)
			}
			if err := os.WriteFile(p, []byte("weights-b"), 0o644); err != nil {
				t.Fatal(err)
			}
			if err := os.Chtimes(p, mtime, mtime); err != nil {
				t.Fatal(err)
			}
			if c.ctimeMoves {
				ctime = mtime.Add(30 * time.Second)
			}
			if h2 := read(ctime.Add(2 * time.Minute)); h2 == h1 {
				t.Fatalf("a same-size rewrite at the same mtime kept the hash %s: the cache served a stale key", h1)
			}
		})
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
