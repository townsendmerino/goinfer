package pull

import (
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
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
// name, which an older build still reads and rewrites. It reads names and sizes only. A missing root is an empty list.
func StaleSidecars(root string, current int) ([]StaleSidecar, error) {
	var out []StaleSidecar
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			if p == root && os.IsNotExist(err) {
				return filepath.SkipAll
			}
			return nil // an unreadable subtree holds nothing this can prune
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

// RemoveStaleSidecar deletes s and its verified marker. It re-checks the name first, so a path that is no longer a stale
// versioned sidecar (renamed, or replaced since the listing) is refused rather than removed.
func RemoveStaleSidecar(s StaleSidecar, current int) error {
	m := versionedSidecar.FindStringSubmatch(filepath.Base(s.Path))
	if m == nil {
		return &os.PathError{Op: "prune", Path: s.Path, Err: os.ErrInvalid}
	}
	if v, err := strconv.Atoi(m[1]); err != nil || v >= current {
		return &os.PathError{Op: "prune", Path: s.Path, Err: os.ErrInvalid}
	}
	if err := os.Remove(s.Path); err != nil {
		return err
	}
	if s.Marker != "" {
		if err := os.Remove(s.Marker); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	return nil
}
