package paritykey

import (
	"fmt"
	"sort"
	"strings"
)

// Side is one end of a push, as Compare needs it: each family's two keys and dependency files, the content digest of
// every dependency path, the aikit pin, today's tail, the kernel table's entries, and this machine's configuration and
// kernel fingerprints' digest.
type Side struct {
	Pin     string
	Tail    string
	Table   map[string]map[string]string // nil in a tree from before the table
	Config  string
	Digest  string
	Today   map[string]string
	Shadow  map[string]string
	Files   map[string][]string
	Content map[string]string
}

// LoadSide reads a Side from src. config and digest are this machine's dispatch configuration and its kernel
// fingerprints' digest for src's aikit pin.
func LoadSide(src Source, config, digest string) (Side, error) {
	m, err := ReadManifest(src)
	if err != nil {
		return Side{}, err
	}
	pin, err := AikitPin(src)
	if err != nil {
		return Side{}, err
	}
	tail, err := TodayTail(src)
	if err != nil {
		return Side{}, err
	}
	s := Side{Pin: pin, Tail: tail, Config: config, Digest: digest, Today: map[string]string{}, Shadow: map[string]string{},
		Files: map[string][]string{}, Content: map[string]string{}}
	if strings.HasPrefix(tail, KernelTailPrefix) {
		if s.Table, err = ReadKernelTable(src); err != nil {
			return Side{}, err
		}
	}
	for name, fam := range m.Families {
		if s.Today[name], err = Key(src, m, fam, tail); err != nil {
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

// Compare classes every family across a push from before to after. An old-only verdict is explained when the one input
// of today's key that moved is the aikit pin (the fingerprints did not move), the key's definition (the switch from the
// pin to the kernel table), or the kernel table's entries for configurations other than after.Config, which this machine
// cannot check. A new-only verdict is never explained here: with one suite and one toolchain the shadow key cannot move
// while today's does not unless the table missed a kernel change, or an input neither key names moved.
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
			// One moved input only; a moved fingerprint digest would be a second entry in v.Moved.
			v.Explained = len(v.Moved) == 1
			switch m := v.Moved[0]; {
			case !v.Explained:
				v.Reason = "today's key moved on inputs the shadow key also reads, or the fingerprints moved"
			case strings.HasPrefix(m, "aikit "):
				v.Reason = fmt.Sprintf("aikit %s -> %s changed no kernel fingerprint", before.Pin, after.Pin)
			case strings.HasPrefix(m, "key definition "):
				v.Reason = "the key's definition changed (" + strings.TrimPrefix(m, "key definition ") + "); no input it reads moved"
			case strings.HasPrefix(m, "kernel table: ") && !tableMoved(before, after, after.Config):
				v.Reason = "the kernel table moved for configurations this machine does not run (" + strings.TrimPrefix(m, "kernel table: ") + ")"
			default:
				v.Explained = false
				v.Reason = "today's key moved on an input the shadow key should have seen move"
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

// moved names the inputs of a family's keys that differ between the sides: a dependency path, the file list, today's
// tail (the aikit pin, the kernel table's moved configurations, or the key's definition), the kernel fingerprints.
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
		bk, ak := tailKind(before.Tail), tailKind(after.Tail)
		switch {
		case bk != ak:
			out = append(out, "key definition "+bk+" -> "+ak)
		case ak == tailKind(AikitTailPrefix):
			out = append(out, "aikit "+before.Pin+" -> "+after.Pin)
		default:
			var cfgs []string
			for _, c := range sortedKeys(union(before.Table, after.Table)) {
				if tableMoved(before, after, c) {
					cfgs = append(cfgs, c)
				}
			}
			out = append(out, "kernel table: "+strings.Join(cfgs, ", "))
		}
	}
	if before.Digest != after.Digest {
		out = append(out, "kernel fingerprints")
	}
	return out
}

// tailKind names the definition a tail belongs to: "aikit_version" or "kernel_fingerprints".
func tailKind(tail string) string {
	if strings.HasPrefix(tail, KernelTailPrefix) {
		return strings.TrimSuffix(KernelTailPrefix, "=")
	}
	return strings.TrimSuffix(AikitTailPrefix, "=")
}

// tableMoved reports whether the kernel table's entry for cfg differs between the sides (added and removed included).
func tableMoved(before, after Side, cfg string) bool {
	b, bok := before.Table[cfg]
	a, aok := after.Table[cfg]
	return bok != aok || kernelLines(b) != kernelLines(a)
}

func union(a, b map[string]map[string]string) map[string]bool {
	out := map[string]bool{}
	for k := range a {
		out[k] = true
	}
	for k := range b {
		out[k] = true
	}
	return out
}
