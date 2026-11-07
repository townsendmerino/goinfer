package pull

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

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
	got, err := StaleSidecars(root, 17)
	if err != nil {
		t.Fatal(err)
	}
	var paths []string
	for _, s := range got {
		paths = append(paths, s.Path)
	}
	want := slices.Clone(stale)
	slices.Sort(want)
	if !slices.Equal(paths, want) {
		t.Fatalf("StaleSidecars = %v, want %v", paths, want)
	}
	for _, s := range got {
		if s.Path == stale[0] && (s.Version != 16 || s.Bytes != 38 || s.Marker != stale[0]+".verified" || s.Freed() != 38) {
			t.Errorf("v16 sidecar: %+v, want version 16, 38 bytes, its marker, 38 freed", s)
		}
	}
}

// TestRemoveStaleSidecar_keepsEverythingElse is the -apply gate: removing every listed sidecar deletes those and their
// markers, and leaves the current version, a newer one, the legacy name, a temp transcode and the source as they were.
func TestRemoveStaleSidecar_keepsEverythingElse(t *testing.T) {
	root := t.TempDir()
	stale, kept := pruneFixture(t, root)
	got, err := StaleSidecars(root, 17)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range got {
		if err := RemoveStaleSidecar(s, 17); err != nil {
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
	if again, _ := StaleSidecars(root, 17); len(again) != 0 {
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
		if err := RemoveStaleSidecar(StaleSidecar{Path: p, Version: 1}, 17); err == nil {
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
	got, err := StaleSidecars(dir, 17)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].SharedWith != "m.int4.cpu.giw" || got[0].Freed() != 0 {
		t.Fatalf("StaleSidecars = %+v, want the v16 name, shared with the legacy one, freeing 0", got)
	}
}

// TestStaleSidecars_noRoot: a cache that does not exist yet is an empty list.
func TestStaleSidecars_noRoot(t *testing.T) {
	got, err := StaleSidecars(filepath.Join(t.TempDir(), "absent"), 17)
	if err != nil || len(got) != 0 {
		t.Fatalf("StaleSidecars(absent) = %v, %v; want empty, nil", got, err)
	}
}
