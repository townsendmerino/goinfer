package pull

import (
	"encoding/json"
	"os"
	"regexp"
	"strings"
	"testing"
)

// THE COPY MUST NOT DRIFT. pull/capability-matrix.json is a byte copy of docs/capability-matrix.json
// because go:embed cannot reach outside a package directory. A copy is only defensible if
// divergence is impossible, so this compares them byte for byte. Without it there would be two
// sources of truth, which is the one thing this registry was required not to create.
func TestRegistry_embeddedMatrixMatchesTheDoc(t *testing.T) {
	doc, err := os.ReadFile("../docs/capability-matrix.json")
	if err != nil {
		t.Fatalf("read the canonical matrix: %v", err)
	}
	if string(doc) != string(capabilityMatrixJSON) {
		t.Fatal("pull/capability-matrix.json has drifted from docs/capability-matrix.json — " +
			"the doc is canonical; re-copy it (cp docs/capability-matrix.json pull/)")
	}
}

// Every entry traces to a family the matrix actually has. An entry naming a family that is not
// there would be a recommendation with nothing behind it.
func TestRegistry_everyEntryTracesToItsFamily(t *testing.T) {
	var rows []struct {
		Name   string `json:"name"`
		Parity string `json:"parity"`
	}
	if err := json.Unmarshal(capabilityMatrixJSON, &rows); err != nil {
		t.Fatal(err)
	}
	known := map[string]string{}
	for _, r := range rows {
		known[r.Name] = r.Parity
	}
	all := RecommendedAll()
	if len(all) == 0 {
		t.Fatal("the registry is empty: this gate would pass having checked nothing")
	}
	for _, c := range all {
		parity, ok := known[c.Family]
		if !ok {
			t.Errorf("%s claims family %q, which is not in the capability matrix", c.Name, c.Family)
			continue
		}
		if parity != c.Parity {
			t.Errorf("%s carries parity %q, matrix says %q", c.Name, c.Parity, parity)
		}
	}
}

// NO ENTRY MAY CLAIM SUPPORT THE MATRIX DOES NOT BACK. A family whose parity row is still
// scaffolded or unvalidated must not be recommended to a first-time user, whatever else is true
// of it — that is the difference between "we can read this" and "we stand behind this".
func TestRegistry_noEntryOutrunsItsParity(t *testing.T) {
	for _, c := range RecommendedAll() {
		p := strings.ToLower(c.Parity)
		switch {
		case strings.HasPrefix(p, "full-oracle"), strings.HasPrefix(p, "real-oracle"):
			// Both are T3: validated against a real reference implementation.
		default:
			t.Errorf("%s (family %s) is recommended but its parity is %q — only full-oracle or "+
				"real-oracle families may appear here", c.Name, c.Family, c.Parity)
		}
	}
}

// A digest and a size are the whole substitute for hosting the file ourselves.
func TestRegistry_everyEntryIsVerifiable(t *testing.T) {
	sha := regexp.MustCompile(`^[0-9a-f]{64}$`)
	for _, c := range RecommendedAll() {
		if !sha.MatchString(c.SHA256) {
			t.Errorf("%s: sha256 %q is not a 64-char hex digest", c.Name, c.SHA256)
		}
		if c.Bytes <= 0 {
			t.Errorf("%s: bytes = %d", c.Name, c.Bytes)
		}
		if c.Repo == "" || c.File == "" {
			t.Errorf("%s: repo/file incomplete (%q, %q)", c.Name, c.Repo, c.File)
		}
		if !strings.HasSuffix(c.File, ".gguf") {
			t.Errorf("%s: file %q is not a .gguf", c.Name, c.File)
		}
		// The prose fields are the reason a name registry beats a repo path; an entry without
		// them is a row a user still has to go and research.
		if c.GoodFor == "" || c.Needs == "" {
			t.Errorf("%s: missing good_for/needs — the point of a name is that it comes with both", c.Name)
		}
		// The short name must be resolvable, and Ref must reconstruct something ParseRef takes.
		if _, ok := Recommended(c.Name); !ok {
			t.Errorf("%s is listed but does not resolve", c.Name)
		}
		if _, err := ParseRef(c.Ref()); err != nil {
			t.Errorf("%s: Ref() = %q, which ParseRef rejects: %v", c.Name, c.Ref(), err)
		}
	}
}

// Listing order is smallest first: a first-time user wants the cheapest thing to try, not the
// alphabetically first.
func TestRegistry_listedSmallestFirst(t *testing.T) {
	all := RecommendedAll()
	for i := 1; i < len(all); i++ {
		if all[i-1].Bytes > all[i].Bytes {
			t.Errorf("entry %d (%s, %d B) is larger than %d (%s, %d B)",
				i-1, all[i-1].Name, all[i-1].Bytes, i, all[i].Name, all[i].Bytes)
		}
	}
}

// The registry must not shadow the `demo:` tiers, which pin the models EMBEDDED in release
// binaries and answer a different question. Overlapping short names would make it ambiguous which
// list a name came from.
func TestRegistry_doesNotCollideWithDemoTiers(t *testing.T) {
	for _, n := range RecommendedNames() {
		if _, clash := Curated()[n]; clash {
			t.Errorf("%q is both a recommended checkpoint and a demo tier", n)
		}
	}
}
