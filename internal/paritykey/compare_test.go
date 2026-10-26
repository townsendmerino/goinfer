package paritykey

import (
	"os"
	"testing"
)

func side(pin, digest string, today, shadow string, content string) Side {
	return Side{Pin: pin, Digest: digest,
		Today: map[string]string{"f": today}, Shadow: map[string]string{"f": shadow},
		Files: map[string][]string{"f": {"decoder/model.go"}}, Content: map[string]string{"decoder/model.go": content}}
}

// TestCompare_classes is P2: every class and reason, from sides built by hand.
func TestCompare_classes(t *testing.T) {
	base := side("v1", "d1", "t1", "s1", "c1")
	cases := []struct {
		name      string
		after     Side
		class     Class
		explained bool
	}{
		{"nothing moved", side("v1", "d1", "t1", "s1", "c1"), AgreeFresh, true},
		{"a dependency moved", side("v1", "d1", "t2", "s2", "c2"), AgreeStale, true},
		{"aikit moved, kernels moved", side("v2", "d2", "t2", "s2", "c1"), AgreeStale, true},
		{"aikit moved, kernels did not", side("v2", "d1", "t2", "s1", "c1"), OldOnly, true},
		{"today moved on a file, shadow did not", side("v1", "d1", "t2", "s1", "c2"), OldOnly, false},
		{"aikit and a file moved, shadow did not", side("v2", "d1", "t2", "s1", "c2"), OldOnly, false},
		{"kernels moved, aikit did not", side("v1", "d2", "t1", "s2", "c1"), NewOnly, false},
		// Not producible by real keys (a moved digest moves the shadow key); it pins that the fingerprints must not have moved.
		{"aikit and kernels moved, shadow did not", side("v2", "d2", "t2", "s1", "c1"), OldOnly, false},
	}
	for _, c := range cases {
		v := Compare(base, c.after)
		if len(v) != 1 || v[0].Class != c.class || v[0].Explained != c.explained {
			t.Errorf("%s: got %+v, want class %s explained %v", c.name, v, c.class, c.explained)
		}
	}
	after := base
	after.Today = map[string]string{"g": "t"}
	got := Compare(base, after)
	if len(got) != 2 || got[0].Family != "f" || got[0].Class != Removed || got[1].Family != "g" || got[1].Class != Added {
		t.Errorf("added and removed: got %+v", got)
	}
}

// TestKey_onTheTree hashes every family of the working tree with both tails, so a manifest the package cannot read
// fails here rather than in the shadow job.
func TestKey_onTheTree(t *testing.T) {
	src := Dir("../..")
	if _, err := os.Stat("../../" + ManifestPath); err != nil {
		t.Fatal(err)
	}
	s, err := LoadSide(src, "digest")
	if err != nil {
		t.Fatal(err)
	}
	if len(s.Today) == 0 {
		t.Fatal("no families")
	}
	for name := range s.Today {
		if s.Today[name] == s.Shadow[name] {
			t.Errorf("%s: today's key equals the shadow key", name)
		}
	}
}
