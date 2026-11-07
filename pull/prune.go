package pull

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// StaleSidecar is a versioned .giw sidecar in a format older than the one this build reads, which no load of this build
// opens again: the build that wrote it either reads only its own version, or rebuilt the sidecar under the newer name.
type StaleSidecar struct {
	Path    string
	Version int   // the vN in its name
	Bytes   int64 // its size
	// SharedWith names another file in the same directory, not itself stale, that is the same file (a hard link, made
	// when a build adopts a legacy-named sidecar): removing this name frees nothing while that one stays.
	SharedWith string
	Marker     string // its verified marker (<path>.verified), "" when it has none

	info os.FileInfo // from the listing, for Reclaimed's same-file test (device and inode on Unix)
}

// Freed is the disk a removal of s alone returns: its size, or nothing when a kept name still holds the file. Two stale
// names of one file each report its size here; Reclaimed counts them once.
func (s StaleSidecar) Freed() int64 {
	if s.SharedWith != "" {
		return 0
	}
	return s.Bytes
}

// Listing is what StaleSidecars found under Root (the root it was given, symlinks resolved).
type Listing struct {
	Root  string
	Stale []StaleSidecar // sorted by path
	// Unreadable holds one error per directory or entry below Root that could not be read. The walk went on past each,
	// so Stale may be missing sidecars inside them: a caller reports these rather than calling the listing complete.
	Unreadable []error
}

// versionedSidecar matches a sidecar's versioned name, "<base>.<quant>[.e4h][.ple4].<target>.v<N>.giw" (decoder's
// streamCachePath). A transcode's temp file ends ".v<N>.tmp.giw" and a moved-aside one carries ".replaced-" after
// ".giw", so neither matches: a transcode in progress is never touched.
var versionedSidecar = regexp.MustCompile(`\.v(\d+)\.giw$`)

// StaleSidecars lists the versioned .giw sidecars under root older than current (decoder.GIWFormatVersion). A sidecar at
// current or newer (a newer build's, on the same box) is never listed, nor is a legacy unversioned name, which an older
// build still reads and rewrites. It reads names and sizes only. The root is resolved through symlinks first and every
// path listed is under the resolved one. A missing root is an empty listing; a root that is not a directory, or cannot be
// read, is an error. Anything unreadable below the root is in Listing.Unreadable.
func StaleSidecars(root string, current int) (Listing, error) {
	resolved, err := filepath.EvalSymlinks(root)
	if os.IsNotExist(err) {
		return Listing{Root: root}, nil
	}
	if err != nil {
		return Listing{}, err
	}
	fi, err := os.Stat(resolved)
	if err != nil {
		return Listing{}, err
	}
	if !fi.IsDir() {
		return Listing{}, fmt.Errorf("%s is not a directory", root)
	}
	l := Listing{Root: resolved}
	err = filepath.WalkDir(resolved, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			if p == resolved {
				return err
			}
			l.Unreadable = append(l.Unreadable, err)
			return nil
		}
		if !d.Type().IsRegular() {
			return nil
		}
		m := versionedSidecar.FindStringSubmatch(d.Name())
		if m == nil {
			return nil
		}
		v, err := strconv.Atoi(m[1])
		if err != nil || v >= current {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			l.Unreadable = append(l.Unreadable, err)
			return nil
		}
		s := StaleSidecar{Path: p, Version: v, Bytes: info.Size(), info: info}
		if _, err := os.Stat(p + ".verified"); err == nil {
			s.Marker = p + ".verified"
		}
		l.Stale = append(l.Stale, s)
		return nil
	})
	if err != nil {
		return Listing{}, err
	}
	stale := map[string]bool{}
	for _, s := range l.Stale {
		stale[s.Path] = true
	}
	for i := range l.Stale {
		l.Stale[i].SharedWith = sameFileIn(l.Stale[i].Path, stale)
	}
	sort.Slice(l.Stale, func(i, j int) bool { return l.Stale[i].Path < l.Stale[j].Path })
	return l, nil
}

// Reclaimed is the disk that removing removed (a subset of listed, the listing it came from) returns. Each file counts
// once however many stale names it has, and only when every listed name of it is in removed and no kept name holds it
// (SharedWith). Same file means os.SameFile on the listing's FileInfo: device and inode on Unix.
func Reclaimed(listed, removed []StaleSidecar) int64 {
	gone := map[string]bool{}
	for _, s := range removed {
		gone[s.Path] = true
	}
	var total int64
	counted := make([]bool, len(listed))
	for i, s := range listed {
		if counted[i] {
			continue
		}
		all, kept := gone[s.Path], s.SharedWith != ""
		for j := i + 1; j < len(listed); j++ {
			if !counted[j] && sameInfo(s, listed[j]) {
				counted[j] = true
				all = all && gone[listed[j].Path]
				kept = kept || listed[j].SharedWith != ""
			}
		}
		if all && !kept {
			total += s.Bytes
		}
	}
	return total
}

func sameInfo(a, b StaleSidecar) bool {
	return a.info != nil && b.info != nil && os.SameFile(a.info, b.info)
}

// sameFileIn is the name of a file beside p, not itself stale, that is the same file as p ("" for none).
func sameFileIn(p string, stale map[string]bool) string {
	pi, err := os.Stat(p)
	if err != nil {
		return ""
	}
	dir := filepath.Dir(p)
	ents, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	for _, e := range ents {
		q := filepath.Join(dir, e.Name())
		if q == p || stale[q] || !e.Type().IsRegular() {
			continue
		}
		if qi, err := os.Stat(q); err == nil && os.SameFile(pi, qi) {
			return e.Name()
		}
	}
	return ""
}

// RemoveStaleSidecar deletes s and its verified marker, and returns the paths it removed, in order, even when it also
// returns an error: a marker that fails after the sidecar is gone leaves the sidecar in the list. It refuses (removing
// nothing) unless s.Path is a stale versioned sidecar name inside root (both resolved through symlinks) and s.Marker is ""
// or s.Path+".verified". Just before each removal it re-reads the name with Lstat and refuses one that is no longer a
// regular file (replaced by a symlink or a directory since the listing). That narrows the window between listing and
// removal; it does not close it, since a rename between the Lstat and the remove is still possible. Run it only on a
// cache no load is writing.
func RemoveStaleSidecar(root string, s StaleSidecar, current int) ([]string, error) {
	refuse := func(p string) error { return &os.PathError{Op: "prune", Path: p, Err: os.ErrInvalid} }
	m := versionedSidecar.FindStringSubmatch(filepath.Base(s.Path))
	if m == nil {
		return nil, refuse(s.Path)
	}
	if v, err := strconv.Atoi(m[1]); err != nil || v >= current {
		return nil, refuse(s.Path)
	}
	if s.Marker != "" && s.Marker != s.Path+".verified" {
		return nil, refuse(s.Marker)
	}
	if !insideRoot(root, s.Path) {
		return nil, refuse(s.Path)
	}
	if err := removeRegular(s.Path); err != nil {
		return nil, err
	}
	removed := []string{s.Path}
	if s.Marker != "" {
		err := removeRegular(s.Marker)
		if err == nil {
			removed = append(removed, s.Marker)
		} else if !errors.Is(err, fs.ErrNotExist) {
			return removed, err
		}
	}
	return removed, nil
}

// insideRoot reports whether p's directory, resolved through symlinks, is root or below it.
func insideRoot(root, p string) bool {
	r, err := filepath.EvalSymlinks(root)
	if err != nil {
		return false
	}
	d, err := filepath.EvalSymlinks(filepath.Dir(p))
	if err != nil {
		return false
	}
	rel, err := filepath.Rel(r, d)
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

// removeRegular removes p when it is a regular file now, and refuses anything else.
func removeRegular(p string) error {
	fi, err := os.Lstat(p)
	if err != nil {
		return err
	}
	if !fi.Mode().IsRegular() {
		return &os.PathError{Op: "prune", Path: p, Err: os.ErrInvalid}
	}
	return os.Remove(p)
}
