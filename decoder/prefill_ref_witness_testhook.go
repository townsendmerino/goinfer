//go:build goinfer_testhooks

package decoder

// Source witnesses for the reference cache. The key hashes all of the reference path's source, so any edit under decoder/ re-keys
// every reference although most edits leave its numbers unchanged. A reference computed under one source stands in for another's
// once a witness shows the two agree: one prompt of the same scope (every key part but the prompt and the source), computed under
// both, byte-identical. A witness that differs is recorded too, and that pair of sources is never adopted. Only a direct pair
// counts, never a chain through a third source. The records live in the cache dir's source-witness.json.

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

// cachedRefParts lists every complete cache entry: its .bin path and its parts.
func cachedRefParts() map[string]PrefillRefKeyParts {
	dir, err := PrefillRefCacheDirForTest()
	if err != nil {
		return nil
	}
	sidecars, _ := filepath.Glob(filepath.Join(dir, "*", "*.bin.key.json"))
	out := map[string]PrefillRefKeyParts{}
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
		if _, err := os.Stat(bin); err == nil {
			out[bin] = q
		}
	}
	return out
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
	for _, bin := range bins {
		if agree, _ := PrefillRefSourcesAgreeForTest(parts, all[bin].SourceSHA256); agree {
			return bin, all[bin].SourceSHA256, true
		}
	}
	if len(bins) == 0 {
		return "", "", false
	}
	return bins[0], all[bins[0]].SourceSHA256, true
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

// AdoptPrefillRefForTest stores sibling's reference under parts (a hard link, else a copy) and returns the cache path. The
// caller adopts only across sources a witness has shown to agree.
func AdoptPrefillRefForTest(sibling string, parts PrefillRefKeyParts) (string, error) {
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
	b, _ := json.MarshalIndent(parts, "", " ")
	if err := writeFileAtomic(p+".key.json", b); err != nil {
		return "", err
	}
	return p, nil
}

// BackfillPrefillRefWitnessesForTest records a witness for every pair the cache already holds that no record covers: two
// references of one scope and prompt under different sources. It returns how many it recorded and how many of those differ.
func BackfillPrefillRefWitnessesForTest() (recorded, differ int, err error) {
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
