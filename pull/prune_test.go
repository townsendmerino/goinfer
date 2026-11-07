package pull

import (
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
)

// resolved is p with every symlink resolved (on macOS t.TempDir is under /var, a symlink to /private/var; on Windows
// it may carry an 8.3 short name), the form StaleSidecars reports paths in.
func resolved(t *testing.T, p string) string {
	t.Helper()
	r, err := filepath.EvalSymlinks(p)
	if err != nil {
		t.Fatalf("EvalSymlinks(%s): %v", p, err)
	}
	return r
}

// resolvedAll is resolved over ps.
func resolvedAll(t *testing.T, ps []string) []string {
	t.Helper()
	out := make([]string, len(ps))
	for i, p := range ps {
		out[i] = resolved(t, p)
	}
	return out
}

// listStale is StaleSidecars that fails the test on an error or an unreadable directory.
func listStale(t *testing.T, root string, current int) []StaleSidecar {
	t.Helper()
	l, err := StaleSidecars(root, current)
	if err != nil {
		t.Fatal(err)
	}
	if len(l.Unreadable) != 0 {
		t.Fatalf("StaleSidecars(%s): unreadable %v", root, l.Unreadable)
	}
	return l.Stale
}

// pruneFixture lays out a cache holding every name prune-sidecars must tell apart, at current version 17, and returns the
// stale sidecars' paths.
func pruneFixture(t *testing.T, root string) (stale, kept []string) {
	t.Helper()
	write := func(rel string, n int) string {
		t.Helper()
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, make([]byte, n), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	kept = []string{
		write("o/r/m-Q4_K_M.gguf", 100),
		write("o/r/m-Q4_K_M.int4.cpu.v17.giw", 40),             // the current version
		write("o/r/m-Q4_K_M.int4.cpu.v17.giw.verified", 1),     // its marker
		write("o/r/m-Q4_K_M.int4.cpu.v18.giw", 41),             // a newer build's
		write("o/r/m-Q4_K_M.int4.cpu.giw", 39),                 // a legacy unversioned name, which an older build still reads
		write("o/r/m-Q4_K_M.int4.cpu.v16.tmp.giw", 5),          // a transcode in progress
		write("o/r/m-Q4_K_M.int4.cpu.v15.giw.replaced-1-2", 5), // moved aside by a publish
		write("o/ckpt.int8.canonical.v17.giw", 30),             // a checkpoint directory's sidecar, current
	}
	stale = []string{
		write("o/r/m-Q4_K_M.int4.cpu.v16.giw", 38),
		write("o/r/m-Q4_K_M.int4.e4h.metal.v9.giw", 37),
		write("o/ckpt.int8.canonical.v14.giw", 29),
	}
	write("o/r/m-Q4_K_M.int4.cpu.v16.giw.verified", 1) // removed with its sidecar
	return stale, kept
}

// TestStaleSidecars lists exactly the versioned sidecars older than the current version, with their sizes and markers,
// and none of the names a prune must never touch.
func TestStaleSidecars(t *testing.T) {
	root := t.TempDir()
	stale, _ := pruneFixture(t, root)
	got := listStale(t, root, 17)
	var paths []string
	for _, s := range got {
		paths = append(paths, resolved(t, s.Path))
	}
	stale = resolvedAll(t, stale)
	want := slices.Clone(stale)
	slices.Sort(want)
	if !slices.Equal(paths, want) {
		t.Fatalf("StaleSidecars = %v, want %v", paths, want)
	}
	for _, s := range got {
		if resolved(t, s.Path) == stale[0] && (s.Version != 16 || s.Bytes != 38 || resolved(t, s.Marker) != stale[0]+".verified" || s.Freed() != 38) {
			t.Errorf("v16 sidecar: %+v, want version 16, 38 bytes, its marker, 38 freed", s)
		}
	}
}

// TestRemoveStaleSidecar_keepsEverythingElse is the -apply gate: removing every listed sidecar deletes those and their
// markers, and leaves the current version, a newer one, the legacy name, a temp transcode and the source as they were.
func TestRemoveStaleSidecar_keepsEverythingElse(t *testing.T) {
	root := t.TempDir()
	stale, kept := pruneFixture(t, root)
	got := listStale(t, root, 17)
	for _, s := range got {
		if _, err := RemoveStaleSidecar(root, s, 17); err != nil {
			t.Fatalf("RemoveStaleSidecar(%s): %v", s.Path, err)
		}
	}
	for _, p := range append(slices.Clone(stale), stale[0]+".verified") {
		if _, err := os.Stat(p); !os.IsNotExist(err) {
			t.Errorf("%s survived the prune (stat err %v)", p, err)
		}
	}
	for _, p := range kept {
		if _, err := os.Stat(p); err != nil {
			t.Errorf("%s was touched by the prune: %v", p, err)
		}
	}
	if again := listStale(t, root, 17); len(again) != 0 {
		t.Errorf("after the prune, still stale: %+v", again)
	}
}

// TestRemoveStaleSidecar_refusesTheCurrentVersion: a StaleSidecar naming a current or newer file (a caller's mistake, or a
// listing older than the build) is refused, and the file stays.
func TestRemoveStaleSidecar_refusesTheCurrentVersion(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"m.int4.cpu.v17.giw", "m.int4.cpu.v18.giw", "m.int4.cpu.giw"} {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := RemoveStaleSidecar(dir, StaleSidecar{Path: p, Version: 1}, 17); err == nil {
			t.Errorf("RemoveStaleSidecar(%s) at current 17 succeeded, want a refusal", name)
		}
		if _, err := os.Stat(p); err != nil {
			t.Errorf("%s removed: %v", name, err)
		}
	}
}

// TestStaleSidecars_hardLinkFreesNothing: a stale sidecar that is the same file as a name that stays (a legacy name a
// build adopted by hard link) is listed, and its Freed is 0.
func TestStaleSidecars_hardLinkFreesNothing(t *testing.T) {
	dir := t.TempDir()
	legacy := filepath.Join(dir, "m.int4.cpu.giw")
	old := filepath.Join(dir, "m.int4.cpu.v16.giw")
	if err := os.WriteFile(legacy, make([]byte, 50), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Link(legacy, old); err != nil {
		t.Skipf("no hard links here: %v", err)
	}
	got := listStale(t, dir, 17)
	if len(got) != 1 || got[0].SharedWith != "m.int4.cpu.giw" || got[0].Freed() != 0 || Reclaimed(got, got) != 0 {
		t.Fatalf("StaleSidecars = %+v, want the v16 name, shared with the legacy one, freeing 0", got)
	}
}

// TestStaleSidecars_noRoot: a cache that does not exist yet is an empty list.
func TestStaleSidecars_noRoot(t *testing.T) {
	got, err := StaleSidecars(filepath.Join(t.TempDir(), "absent"), 17)
	if err != nil || len(got.Stale) != 0 || len(got.Unreadable) != 0 {
		t.Fatalf("StaleSidecars(absent) = %+v, %v; want empty, nil", got, err)
	}
}

// TestStaleSidecars_rootIsAFile: a root that is a regular file (a -dir naming a model file, say) is an error, not an
// empty listing that reads as "nothing to prune".
func TestStaleSidecars_rootIsAFile(t *testing.T) {
	f := filepath.Join(t.TempDir(), "m.int4.cpu.v16.giw")
	if err := os.WriteFile(f, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got, err := StaleSidecars(f, 17); err == nil || !strings.Contains(err.Error(), "not a directory") {
		t.Fatalf("StaleSidecars(<file>) = %+v, %v; want a not-a-directory error", got, err)
	}
}

// TestStaleSidecars_rootUnreadable: a root directory that cannot be read is an error.
func TestStaleSidecars_rootUnreadable(t *testing.T) {
	skipWithoutPermissions(t)
	root := t.TempDir()
	if err := os.Chmod(root, 0); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(root, 0o755) })
	if got, err := StaleSidecars(root, 17); err == nil {
		t.Fatalf("StaleSidecars(<unreadable>) = %+v, nil; want an error", got)
	}
}

// TestStaleSidecars_unreadableSubdirReported: a directory below the root that cannot be read is named in Unreadable, and
// what the rest of the walk found is still listed.
func TestStaleSidecars_unreadableSubdirReported(t *testing.T) {
	skipWithoutPermissions(t)
	root := t.TempDir()
	stale, _ := pruneFixture(t, root)
	locked := filepath.Join(root, "locked")
	if err := os.MkdirAll(locked, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(locked, "m.int4.cpu.v16.giw"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(locked, 0); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(locked, 0o755) })
	l, err := StaleSidecars(root, 17)
	if err != nil {
		t.Fatal(err)
	}
	if len(l.Stale) != len(stale) {
		t.Errorf("listed %d stale, want %d (the readable ones)", len(l.Stale), len(stale))
	}
	if len(l.Unreadable) != 1 || !strings.Contains(l.Unreadable[0].Error(), "locked") {
		t.Errorf("Unreadable = %v, want the one locked directory", l.Unreadable)
	}
}

// skipWithoutPermissions skips where a mode-0 directory stays readable: Windows ignores the mode, and root reads anything.
func skipWithoutPermissions(t *testing.T) {
	t.Helper()
	if runtime.GOOS == "windows" || os.Geteuid() == 0 {
		t.Skip("directory permissions do not deny reads here")
	}
}

// TestStaleSidecars_symlinkedRoot: a root that is a symlink to the cache is walked as the cache, not skipped as one entry.
func TestStaleSidecars_symlinkedRoot(t *testing.T) {
	base := t.TempDir()
	real := filepath.Join(base, "real")
	stale, _ := pruneFixture(t, real)
	link := filepath.Join(base, "link")
	if err := os.Symlink(real, link); err != nil {
		t.Skipf("no symlinks here: %v", err)
	}
	got := listStale(t, link, 17)
	if len(got) != len(stale) {
		t.Fatalf("StaleSidecars(symlink) listed %d, want %d: %+v", len(got), len(stale), got)
	}
	for _, s := range got {
		if _, err := RemoveStaleSidecar(link, s, 17); err != nil {
			t.Errorf("RemoveStaleSidecar(%s) through the symlinked root: %v", s.Path, err)
		}
	}
}

// TestRemoveStaleSidecar_refusesABadMarker: a Marker that is not <Path>.verified (another file, or one outside the root)
// is refused before anything is removed.
func TestRemoveStaleSidecar_refusesABadMarker(t *testing.T) {
	base := t.TempDir()
	root := filepath.Join(base, "cache")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(root, "m.int4.cpu.v16.giw")
	other := filepath.Join(root, "m.gguf")
	outside := filepath.Join(base, "precious")
	for _, f := range []string{p, other, outside} {
		if err := os.WriteFile(f, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for _, marker := range []string{other, outside, p + ".verified.bak"} {
		if _, err := RemoveStaleSidecar(root, StaleSidecar{Path: p, Version: 16, Marker: marker}, 17); err == nil {
			t.Errorf("RemoveStaleSidecar with Marker %s succeeded, want a refusal", marker)
		}
	}
	for _, f := range []string{p, other, outside} {
		if _, err := os.Stat(f); err != nil {
			t.Errorf("%s removed: %v", f, err)
		}
	}
}

// TestRemoveStaleSidecar_refusesOutsideTheRoot: a stale-named sidecar outside the root it is given is refused.
func TestRemoveStaleSidecar_refusesOutsideTheRoot(t *testing.T) {
	base := t.TempDir()
	root := filepath.Join(base, "cache")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(base, "m.int4.cpu.v16.giw")
	if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := RemoveStaleSidecar(root, StaleSidecar{Path: p, Version: 16}, 17); err == nil {
		t.Error("RemoveStaleSidecar outside its root succeeded, want a refusal")
	}
	if _, err := os.Stat(p); err != nil {
		t.Errorf("%s removed: %v", p, err)
	}
}

// TestRemoveStaleSidecar_refusesAReplacedPath: a listed path that is no longer a regular file when removal comes (here a
// symlink put in its place since the listing) is refused, and the symlink's target stays.
func TestRemoveStaleSidecar_refusesAReplacedPath(t *testing.T) {
	root := t.TempDir()
	p := filepath.Join(root, "m.int4.cpu.v16.giw")
	if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	got := listStale(t, root, 17)
	if len(got) != 1 {
		t.Fatalf("StaleSidecars = %+v", got)
	}
	target := filepath.Join(root, "m.gguf")
	if err := os.WriteFile(target, []byte("model"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(p); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, p); err != nil {
		t.Skipf("no symlinks here: %v", err)
	}
	if _, err := RemoveStaleSidecar(root, got[0], 17); err == nil {
		t.Error("RemoveStaleSidecar of a path now a symlink succeeded, want a refusal")
	}
	if _, err := os.Lstat(p); err != nil {
		t.Errorf("the symlink was removed: %v", err)
	}
}

// TestReclaimed_staleHardLinksCountOnce: two stale names that are one file (hard-linked to each other, no kept name
// sharing it) reclaim its bytes once, and only when both are removed.
func TestReclaimed_staleHardLinksCountOnce(t *testing.T) {
	dir := t.TempDir()
	a := filepath.Join(dir, "m.int4.cpu.v15.giw")
	b := filepath.Join(dir, "m.int4.cpu.v16.giw")
	if err := os.WriteFile(a, make([]byte, 50), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Link(a, b); err != nil {
		t.Skipf("no hard links here: %v", err)
	}
	got := listStale(t, dir, 17)
	if len(got) != 2 {
		t.Fatalf("StaleSidecars = %+v, want both names", got)
	}
	if n := Reclaimed(got, got); n != 50 {
		t.Errorf("Reclaimed(both removed) = %d, want 50 (one file)", n)
	}
	if n := Reclaimed(got, got[:1]); n != 0 {
		t.Errorf("Reclaimed(one name removed) = %d, want 0 (the other name keeps the file)", n)
	}
}

// TestRemoveStaleSidecar_reportsWhatWasRemoved: when the marker's removal fails after the sidecar is gone, the result
// names the sidecar as removed and returns the marker's error.
func TestRemoveStaleSidecar_reportsWhatWasRemoved(t *testing.T) {
	root := t.TempDir()
	p := filepath.Join(root, "m.int4.cpu.v16.giw")
	if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	// A marker that is a directory: listed (it exists), and refused at removal.
	if err := os.MkdirAll(filepath.Join(p+".verified", "x"), 0o755); err != nil {
		t.Fatal(err)
	}
	got := listStale(t, root, 17)
	if len(got) != 1 || got[0].Marker == "" {
		t.Fatalf("StaleSidecars = %+v, want the sidecar with its marker", got)
	}
	removed, err := RemoveStaleSidecar(root, got[0], 17)
	if err == nil {
		t.Fatal("RemoveStaleSidecar with an unremovable marker returned nil, want the marker's error")
	}
	if !slices.Equal(removed, []string{got[0].Path}) {
		t.Errorf("removed = %v, want exactly the sidecar %s", removed, got[0].Path)
	}
	if _, err := os.Lstat(p); !os.IsNotExist(err) {
		t.Errorf("the sidecar is still there (lstat err %v)", err)
	}
}
