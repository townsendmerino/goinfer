package paritykey

import (
	"fmt"
	"sort"
	"strings"
)

// Side is one end of a push, as Compare needs it: each family's two keys, dependency files, program list and programs
// digest, the content digest of every dependency path, and today's tail.
type Side struct {
	Tail     string
	Today    map[string]string
	Shadow   map[string]string
	Files    map[string][]string
	Programs map[string][]string
	Digest   map[string]string // the family's ProgramsDigest; "" for a family with no programs
	Content  map[string]string
}

// LoadSide reads a Side from src.
func LoadSide(src Source) (Side, error) {
	m, err := ReadManifest(src)
	if err != nil {
		return Side{}, err
	}
	tail, err := TodayTail(src)
	if err != nil {
		return Side{}, err
	}
	progs, err := ReadProgramTable(src)
	if err != nil {
		return Side{}, err
	}
	s := Side{Tail: tail, Today: map[string]string{}, Shadow: map[string]string{}, Files: map[string][]string{},
		Programs: map[string][]string{}, Digest: map[string]string{}, Content: map[string]string{}}
	for name, fam := range m.Families {
		if s.Today[name], err = Key(src, m, fam, tail); err != nil {
			return Side{}, fmt.Errorf("%s: %w", name, err)
		}
		if s.Shadow[name], err = ShadowKey(src, m, fam, tail, progs); err != nil {
			return Side{}, fmt.Errorf("%s: %w", name, err)
		}
		s.Files[name] = DepFiles(m, fam)
		s.Programs[name] = append([]string(nil), fam.Programs...)
		sort.Strings(s.Programs[name])
		if len(fam.Programs) > 0 {
			if s.Digest[name], err = ProgramsDigest(fam.Programs, progs); err != nil {
				return Side{}, fmt.Errorf("%s: %w", name, err)
			}
		}
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

// Compare classes every family across a push from before to after. An old-only verdict is explained when every moved
// input is a builder file (BuilderFile), which the shadow key replaces by the family's programs; a builder change on a
// path no fixture exercises reads the same, so a grade reads those pushes' builder diffs. A new-only verdict is
// explained when the only moved input is the family's program list. Every other disagreement is unexplained.
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
			v.Explained = len(v.Moved) > 0
			for _, m := range v.Moved {
				if !BuilderFile(m) {
					v.Explained = false
				}
			}
			if v.Explained {
				v.Reason = "only builder files moved, and no program of the family did: a refactor, or a change on a path no fixture exercises"
			} else {
				v.Reason = "today's key moved on an input the shadow key also reads"
			}
		default:
			v.Class = NewOnly
			v.Explained = len(v.Moved) == 1 && v.Moved[0] == "program list"
			if v.Explained {
				v.Reason = "the family's program list changed"
			} else {
				v.Reason = "the shadow key moved while today's did not: a program moved from a file outside the family's dependencies"
			}
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

// moved names the inputs of a family's keys that differ between the sides: the file list, a dependency path, today's
// tail, the program list, the programs.
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
	if before.Tail != after.Tail {
		out = append(out, "today's tail")
	}
	if strings.Join(before.Programs[name], "\x00") != strings.Join(after.Programs[name], "\x00") {
		out = append(out, "program list")
	} else if before.Digest[name] != after.Digest[name] {
		out = append(out, "programs")
	}
	return out
}
