package pull

import (
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
	// SharedWith names another file in the same directory that is the same file (a hard link, made when a build adopts
	// a legacy-named sidecar): removing this name frees nothing while that one stays.
	SharedWith string
	Marker     string // its verified marker (<path>.verified), "" when it has none
}

// Freed is the disk a removal of s returns: its size, or nothing when another name still holds the file.
func (s StaleSidecar) Freed() int64 {
	if s.SharedWith != "" {
		return 0
	}
	return s.Bytes
}

// versionedSidecar matches a sidecar's versioned name, "<base>.<quant>[.e4h][.ple4].<target>.v<N>.giw" (decoder's
// streamCachePath). A transcode's temp file ends ".v<N>.tmp.giw" and a moved-aside one carries ".replaced-" after
// ".giw", so neither matches: a transcode in progress is never touched.
var versionedSidecar = regexp.MustCompile(`\.v(\d+)\.giw$`)

// StaleSidecars lists the versioned .giw sidecars under root older than current (decoder.GIWFormatVersion), sorted by
// path. A sidecar at current or newer (a newer build's, on the same box) is never listed, nor is a legacy unversioned
// name, which an older build still reads and rewrites. It reads names and sizes only. A symlinked root is resolved first
// and the paths listed are under the resolved one. A missing root is an empty list; any other error reaching the root is
// returned. An unreadable directory below the root is skipped: it holds nothing this can prune.
func StaleSidecars(root string, current int) ([]StaleSidecar, error) {
	resolved, err := filepath.EvalSymlinks(root)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	root = resolved
	var out []StaleSidecar
	err = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			if p == root {
				return err
			}
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
			return nil
		}
		s := StaleSidecar{Path: p, Version: v, Bytes: info.Size()}
		if _, err := os.Stat(p + ".verified"); err == nil {
			s.Marker = p + ".verified"
		}
		out = append(out, s)
		return nil
	})
	if err != nil {
		return nil, err
	}
	stale := map[string]bool{}
	for _, s := range out {
		stale[s.Path] = true
	}
	for i := range out {
		out[i].SharedWith = sameFileIn(out[i].Path, stale)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out, nil
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

// RemoveStaleSidecar deletes s and its verified marker, and refuses (removing nothing) unless s.Path is a stale versioned
// sidecar name inside root (both resolved through symlinks) and s.Marker is "" or s.Path+".verified". Just before each
// removal it re-reads the name with Lstat and refuses one that is no longer a regular file (replaced by a symlink or a
// directory since the listing). That narrows the window between listing and removal; it does not close it, since a
// rename between the Lstat and the remove is still possible. Run it only on a cache no load is writing.
func RemoveStaleSidecar(root string, s StaleSidecar, current int) error {
	refuse := func(p string) error { return &os.PathError{Op: "prune", Path: p, Err: os.ErrInvalid} }
	m := versionedSidecar.FindStringSubmatch(filepath.Base(s.Path))
	if m == nil {
		return refuse(s.Path)
	}
	if v, err := strconv.Atoi(m[1]); err != nil || v >= current {
		return refuse(s.Path)
	}
	if s.Marker != "" && s.Marker != s.Path+".verified" {
		return refuse(s.Marker)
	}
	if !insideRoot(root, s.Path) {
		return refuse(s.Path)
	}
	if err := removeRegular(s.Path); err != nil {
		return err
	}
	if s.Marker != "" {
		if err := removeRegular(s.Marker); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	return nil
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
