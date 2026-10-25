//go:build goinfer_testhooks

package decoder

// Source witnesses for the reference cache. The key hashes all of the reference path's source, so any edit under decoder/ re-keys
// every reference although most edits leave its numbers unchanged. A reference computed under one source stands in for another's
// once a witness shows the two agree: one prompt of the same scope (every key part but the prompt and the source), computed under
// both, byte-identical. A witness that differs is recorded too, and that pair of sources is never adopted. Only a direct pair of
// NATIVELY computed references counts, never a chain through a third source: an adopted reference carries a <bin>.adopted marker
// (one adopted before markers existed is the newer of two entries sharing one file), and it is never a sibling, a backfill input or
// either side of a witness. The records live in the cache dir's source-witness.json.
//
// A limit, stated: one agreeing prompt adopts the rest, so a source change whose effect depends on the prompt (a branch only some
// token sequences reach) could agree on the witness and differ elsewhere.

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

// prefillRefWitness is one comparison of two sources' references in one scope.
type prefillRefWitness struct {
	Scope  string `json:"scope"` // PrefillRefKeyParts.scopeKey
	Prompt string `json:"prompt_sha256"`
	A      string `json:"source_a"` // the lesser source hash
	B      string `json:"source_b"`
	Same   bool   `json:"same"`
}

var refWitnessMu sync.Mutex

// scopeKey is Key over the parts two sources' references must share for one to stand in for the other.
func (p PrefillRefKeyParts) scopeKey() string {
	q := p
	q.PromptSHA256, q.SourceSHA256 = "", ""
	return q.Key()
}

func refWitnessPath() (string, error) {
	dir, err := PrefillRefCacheDirForTest()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "source-witness.json"), nil
}

func readRefWitnesses() []prefillRefWitness {
	p, err := refWitnessPath()
	if err != nil {
		return nil
	}
	var ws []prefillRefWitness
	if b, err := os.ReadFile(p); err == nil {
		_ = json.Unmarshal(b, &ws)
	}
	return ws
}

func orderedSources(a, b string) (string, string) {
	if b < a {
		return b, a
	}
	return a, b
}

// PrefillRefSourcesAgreeForTest reports whether a witness has shown parts' source and other to agree in parts' scope (agree), and
// whether any witness of the pair exists (known). One differing witness makes the pair disagree.
func PrefillRefSourcesAgreeForTest(parts PrefillRefKeyParts, other string) (agree, known bool) {
	scope := parts.scopeKey()
	a, b := orderedSources(parts.SourceSHA256, other)
	refWitnessMu.Lock()
	defer refWitnessMu.Unlock()
	for _, w := range readRefWitnesses() {
		if w.Scope == scope && w.A == a && w.B == b {
			if !w.Same {
				return false, true
			}
			agree, known = true, true
		}
	}
	return agree, known
}

func recordRefWitness(parts PrefillRefKeyParts, other string, same bool) error {
	p, err := refWitnessPath()
	if err != nil {
		return err
	}
	a, b := orderedSources(parts.SourceSHA256, other)
	refWitnessMu.Lock()
	defer refWitnessMu.Unlock()
	ws := append(readRefWitnesses(), prefillRefWitness{Scope: parts.scopeKey(), Prompt: parts.PromptSHA256, A: a, B: b, Same: same})
	out, _ := json.MarshalIndent(ws, "", " ")
	return writeFileAtomic(p, out)
}

// refWitnessedPrompt reports whether parts' prompt already witnesses the pair (parts' source, other).
func refWitnessedPrompt(parts PrefillRefKeyParts, other string) bool {
	scope := parts.scopeKey()
	a, b := orderedSources(parts.SourceSHA256, other)
	refWitnessMu.Lock()
	defer refWitnessMu.Unlock()
	for _, w := range readRefWitnesses() {
		if w.Scope == scope && w.Prompt == parts.PromptSHA256 && w.A == a && w.B == b {
			return true
		}
	}
	return false
}

// refEntry is one complete cache entry.
type refEntry struct {
	parts   PrefillRefKeyParts
	adopted bool // a sibling's file under this source, not computed by it
}

// refEntries lists every complete cache entry by its .bin path. An entry is adopted when it carries a .adopted marker, or when it
// shares its file with another source's entry for the same scope and prompt and its sidecar is the newer: an adoption from before
// the marker existed, which hard-linked the sibling.
func refEntries() map[string]refEntry {
	dir, err := PrefillRefCacheDirForTest()
	if err != nil {
		return nil
	}
	sidecars, _ := filepath.Glob(filepath.Join(dir, "*", "*.bin.key.json"))
	out := map[string]refEntry{}
	type group struct{ scope, prompt string }
	by := map[group][]string{}
	for _, s := range sidecars {
		b, err := os.ReadFile(s)
		if err != nil {
			continue
		}
		var q PrefillRefKeyParts
		if json.Unmarshal(b, &q) != nil {
			continue
		}
		bin := strings.TrimSuffix(s, ".key.json")
		if _, err := os.Stat(bin); err != nil {
			continue
		}
		_, marked := os.Stat(bin + ".adopted")
		out[bin] = refEntry{parts: q, adopted: marked == nil}
		g := group{q.scopeKey(), q.PromptSHA256}
		by[g] = append(by[g], bin)
	}
	for _, bins := range by {
		for i := range bins {
			for j := i + 1; j < len(bins); j++ {
				a, b := bins[i], bins[j]
				fa, ea := os.Stat(a)
				fb, eb := os.Stat(b)
				if ea != nil || eb != nil || !os.SameFile(fa, fb) {
					continue
				}
				sa, _ := os.Stat(a + ".key.json")
				sb, _ := os.Stat(b + ".key.json")
				newer := b
				if sa != nil && sb != nil && sa.ModTime().After(sb.ModTime()) {
					newer = a
				}
				e := out[newer]
				e.adopted = true
				out[newer] = e
			}
		}
	}
	return out
}

// cachedRefParts lists every natively computed cache entry: its .bin path and its parts.
func cachedRefParts() map[string]PrefillRefKeyParts {
	out := map[string]PrefillRefKeyParts{}
	for bin, e := range refEntries() {
		if !e.adopted {
			out[bin] = e.parts
		}
	}
	return out
}

// pruneRefWitnesses drops every witness record one of whose two entries, for its scope and prompt, is adopted: such a record
// compared a file with its own link, or named a source that never computed the bytes. It returns how many it dropped.
func pruneRefWitnesses() (int, error) {
	p, err := refWitnessPath()
	if err != nil {
		return 0, err
	}
	adopted := map[[3]string]bool{} // scope, prompt, source
	for _, e := range refEntries() {
		if e.adopted {
			adopted[[3]string{e.parts.scopeKey(), e.parts.PromptSHA256, e.parts.SourceSHA256}] = true
		}
	}
	refWitnessMu.Lock()
	defer refWitnessMu.Unlock()
	ws := readRefWitnesses()
	keep := ws[:0]
	for _, w := range ws {
		if adopted[[3]string{w.Scope, w.Prompt, w.A}] || adopted[[3]string{w.Scope, w.Prompt, w.B}] {
			continue
		}
		keep = append(keep, w)
	}
	dropped := len(ws) - len(keep)
	if dropped == 0 {
		return 0, nil
	}
	out, _ := json.MarshalIndent(keep, "", " ")
	return dropped, writeFileAtomic(p, out)
}

// PrefillRefSiblingForTest returns a cached reference for parts' scope and prompt computed under another source, and that
// source. It prefers one a witness has shown to agree, then the source holding the most references in the scope, so the prompts
// of one cell share a source and one witness covers them.
func PrefillRefSiblingForTest(parts PrefillRefKeyParts) (path, source string, ok bool) {
	scope := parts.scopeKey()
	var bins []string
	all := cachedRefParts()
	held := map[string]int{} // references per source in this scope
	for bin, q := range all {
		if q.scopeKey() != scope {
			continue
		}
		held[q.SourceSHA256]++
		if q.PromptSHA256 == parts.PromptSHA256 && q.SourceSHA256 != parts.SourceSHA256 {
			bins = append(bins, bin)
		}
	}
	sort.Slice(bins, func(i, j int) bool {
		si, sj := all[bins[i]].SourceSHA256, all[bins[j]].SourceSHA256
		if held[si] != held[sj] {
			return held[si] > held[sj]
		}
		return si < sj
	})
	untested := ""
	for _, bin := range bins {
		switch agree, known := PrefillRefSourcesAgreeForTest(parts, all[bin].SourceSHA256); {
		case agree:
			return bin, all[bin].SourceSHA256, true
		case !known && untested == "":
			untested = bin // a source a witness has shown to disagree is never offered
		}
	}
	if untested == "" {
		return "", "", false
	}
	return untested, all[untested].SourceSHA256, true
}

// refAdoptedFrom reports whether the cache entry bin (whose parts are parts) is an adoption, and the source it was adopted from:
// its .adopted marker's, or for an adoption from before the marker, the source of the entry it shares its file with.
func refAdoptedFrom(bin string, parts PrefillRefKeyParts) (string, bool) {
	if b, err := os.ReadFile(bin + ".adopted"); err == nil {
		var m struct {
			From string `json:"adopted_from_source"`
		}
		if json.Unmarshal(b, &m) == nil && m.From != "" {
			return m.From, true
		}
	}
	e, ok := refEntries()[bin]
	if !ok || !e.adopted {
		return "", false
	}
	fi, err := os.Stat(bin)
	if err != nil {
		return "", false
	}
	for other, o := range refEntries() {
		if other == bin || o.parts.SourceSHA256 == parts.SourceSHA256 || o.parts.PromptSHA256 != parts.PromptSHA256 || o.parts.scopeKey() != parts.scopeKey() {
			continue
		}
		if fo, err := os.Stat(other); err == nil && os.SameFile(fi, fo) {
			return o.parts.SourceSHA256, true
		}
	}
	return "", false
}

// WitnessPrefillRefForTest compares the reference just stored at computed (under parts) with sibling (the same prompt under
// source other), records the result, and reports whether the two are byte-identical.
func WitnessPrefillRefForTest(parts PrefillRefKeyParts, computed, sibling, other string) (bool, error) {
	a, err := os.ReadFile(computed)
	if err != nil {
		return false, err
	}
	b, err := os.ReadFile(sibling)
	if err != nil {
		return false, err
	}
	same := bytes.Equal(a, b)
	return same, recordRefWitness(parts, other, same)
}

// AdoptPrefillRefForTest stores sibling's reference (computed under source from) under parts, a hard link, else a copy, with a
// .adopted marker naming from, and returns the cache path. The caller adopts only across sources a witness has shown to agree.
func AdoptPrefillRefForTest(sibling string, parts PrefillRefKeyParts, from string) (string, error) {
	p, err := prefillRefCachePath(parts.Key())
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return "", err
	}
	tmp := p + ".tmp"
	_ = os.Remove(tmp)
	if err := os.Link(sibling, tmp); err != nil {
		if err := copyFile(sibling, tmp); err != nil {
			return "", err
		}
	}
	if err := os.Rename(tmp, p); err != nil {
		return "", err
	}
	mark, _ := json.Marshal(struct {
		From string `json:"adopted_from_source"`
	}{from})
	if err := writeFileAtomic(p+".adopted", mark); err != nil {
		return "", err
	}
	b, _ := json.MarshalIndent(parts, "", " ")
	if err := writeFileAtomic(p+".key.json", b); err != nil {
		return "", err
	}
	return p, nil
}

// BackfillPrefillRefWitnessesForTest records a witness for every pair the cache already holds that no record covers: two
// references of one scope and prompt under different sources. It returns how many it recorded and how many of those differ.
//
// It first drops the records an adopted entry produced (pruneRefWitnesses).
func BackfillPrefillRefWitnessesForTest() (recorded, differ int, err error) {
	if _, err := pruneRefWitnesses(); err != nil {
		return 0, 0, err
	}
	type group struct{ scope, prompt string }
	by := map[group]map[string]string{} // source -> bin
	parts := map[group]PrefillRefKeyParts{}
	for bin, q := range cachedRefParts() {
		g := group{q.scopeKey(), q.PromptSHA256}
		if by[g] == nil {
			by[g] = map[string]string{}
		}
		by[g][q.SourceSHA256] = bin
		parts[g] = q
	}
	for g, srcs := range by {
		var names []string
		for s := range srcs {
			names = append(names, s)
		}
		sort.Strings(names)
		for i := range names {
			for j := i + 1; j < len(names); j++ {
				q := parts[g]
				q.SourceSHA256 = names[i]
				if refWitnessedPrompt(q, names[j]) {
					continue
				}
				same, werr := WitnessPrefillRefForTest(q, srcs[names[i]], srcs[names[j]], names[j])
				if werr != nil {
					return recorded, differ, werr
				}
				recorded++
				if !same {
					differ++
				}
			}
		}
	}
	return recorded, differ, nil
}
