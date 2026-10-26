package paritykey

import (
	"fmt"
	"sort"
	"strings"
)

// Side is one end of a push, as Compare needs it: each family's two keys and dependency files, the content digest of
// every dependency path, the aikit pin, and the kernel fingerprints' digest.
type Side struct {
	Pin     string
	Digest  string
	Today   map[string]string
	Shadow  map[string]string
	Files   map[string][]string
	Content map[string]string
}

// LoadSide reads a Side from src. digest is the kernel fingerprints' digest for src's aikit pin.
func LoadSide(src Source, digest string) (Side, error) {
	m, err := ReadManifest(src)
	if err != nil {
		return Side{}, err
	}
	pin, err := AikitPin(src)
	if err != nil {
		return Side{}, err
	}
	s := Side{Pin: pin, Digest: digest, Today: map[string]string{}, Shadow: map[string]string{},
		Files: map[string][]string{}, Content: map[string]string{}}
	for name, fam := range m.Families {
		if s.Today[name], err = Key(src, m, fam, TodayTail(pin)); err != nil {
			return Side{}, fmt.Errorf("%s: %w", name, err)
		}
		if s.Shadow[name], err = Key(src, m, fam, ShadowTail(digest)); err != nil {
			return Side{}, fmt.Errorf("%s: %w", name, err)
		}
		s.Files[name] = DepFiles(m, fam)
		for _, p := range s.Files[name] {
			if _, ok := s.Content[p]; !ok {
				if s.Content[p], err = ContentDigest(src, p); err != nil {
					return Side{}, err
				}
			}
		}
	}
	return s, nil
}

// Class is how today's key and the shadow key judged one family across a push.
type Class string

const (
	AgreeStale Class = "agree-stale"
	AgreeFresh Class = "agree-fresh"
	OldOnly    Class = "old-only" // today's key re-stales, the shadow key does not
	NewOnly    Class = "new-only" // the shadow key re-stales, today's key does not
	Added      Class = "added"
	Removed    Class = "removed"
)

// Verdict is one family's class, the inputs of its keys that moved, and whether a disagreement is explained.
type Verdict struct {
	Family    string   `json:"family"`
	Class     Class    `json:"class"`
	Moved     []string `json:"moved,omitempty"`
	Explained bool     `json:"explained"`
	Reason    string   `json:"reason,omitempty"`
}

// Compare classes every family across a push from before to after. An old-only verdict is explained only when the aikit
// pin is the one input of today's key that moved and the kernel fingerprints did not; a new-only verdict is never
// explained here, because with one suite and one toolchain the shadow key cannot move while today's does not unless an
// input neither key names moved.
func Compare(before, after Side) []Verdict {
	var out []Verdict
	for name, ak := range after.Today {
		v := Verdict{Family: name, Explained: true}
		bk, ok := before.Today[name]
		if !ok {
			v.Class = Added
			out = append(out, v)
			continue
		}
		v.Moved = moved(before, after, name)
		oldStale, newStale := bk != ak, before.Shadow[name] != after.Shadow[name]
		switch {
		case oldStale && newStale:
			v.Class = AgreeStale
		case !oldStale && !newStale:
			v.Class = AgreeFresh
		case oldStale:
			v.Class = OldOnly
			// The pin is the only moved input; a moved fingerprint digest would be a second entry in v.Moved.
			v.Explained = len(v.Moved) == 1 && strings.HasPrefix(v.Moved[0], "aikit ")
			if v.Explained {
				v.Reason = fmt.Sprintf("aikit %s -> %s changed no kernel fingerprint", before.Pin, after.Pin)
			} else {
				v.Reason = "today's key moved on inputs the shadow key also reads, or the fingerprints moved"
			}
		default:
			v.Class = NewOnly
			v.Explained = false
			v.Reason = "the shadow key moved while today's did not"
		}
		out = append(out, v)
	}
	for name := range before.Today {
		if _, ok := after.Today[name]; !ok {
			out = append(out, Verdict{Family: name, Class: Removed, Explained: true})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Family < out[j].Family })
	return out
}

// moved names the inputs of a family's keys that differ between the sides: a dependency path, the file list, the aikit
// pin, the kernel fingerprints.
func moved(before, after Side, name string) []string {
	var out []string
	if strings.Join(before.Files[name], "\x00") != strings.Join(after.Files[name], "\x00") {
		out = append(out, "file list")
	}
	for _, p := range after.Files[name] {
		if b, ok := before.Content[p]; ok && b != after.Content[p] {
			out = append(out, p)
		}
	}
	if before.Pin != after.Pin {
		out = append(out, "aikit "+before.Pin+" -> "+after.Pin)
	}
	if before.Digest != after.Digest {
		out = append(out, "kernel fingerprints")
	}
	return out
}
